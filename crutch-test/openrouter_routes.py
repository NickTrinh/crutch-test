"""Per-model OpenRouter settings, applied to every chat call EduardoRL makes.

EduardoRL's clients send vLLM-style options and no provider pin. Through
OpenRouter that would (1) let the student land on a cheaper fp8 host, which
changes its weights, (2) leave the Qwen judge thinking, because OpenRouter
ignores chat_template_kwargs, and (3) send sampling fields Claude 5.5 models do
not take. Importing this module patches the openai client so each model gets
the settings below, logs which provider actually served each call, and keeps a
ledger of tokens and dollars per model (OpenRouter reports cost on every call).
A call rejected with 429 waits and tries again for up to CRUTCH_429_PATIENCE
seconds (default 30 minutes) before it fails.

It also serves the scripted tutors in-process. ``crutch/null-tutor`` (arm N)
never reaches the network and only answers "Keep going. Show your next step."
``crutch/try-first`` (arm S) asks for an attempt, replays T's first turn, then
hands over to the live tutor; ``crutch/wrong-solution`` (arm W) gives a
pre-checked wrong solution, then N's filler.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import logging
import os
import random
import time

import openai
import openai.resources.chat.completions as _cc
from openai.types.chat import ChatCompletion

logger = logging.getLogger("crutch.routes")

STUDENT = "meta-llama/llama-3.1-8b-instruct"
JUDGE = "qwen/qwen3.6-27b"

ROUTES: dict[str, dict] = {
    # bf16 only: the dataset's pre-test rates were measured with bf16 vLLM.
    STUDENT: {
        "extra_body": {
            "provider": {
                "order": ["coreweave"],
                "quantizations": ["bf16"],
                "allow_fallbacks": False,
            }
        },
    },
    # Pinned host; thinking off so the judge answers in bare JSON.
    JUDGE: {
        "extra_body": {
            "provider": {"order": ["chutes"], "allow_fallbacks": False},
            "reasoning": {"enabled": False},
        },
    },
}
# Claude 5.5 models take no temperature or top_p; they run at API defaults
# with low effort, as the proposal specifies.
CLAUDE_PREFIX = "anthropic/"
CLAUDE_EXTRA_BODY = {"reasoning": {"effort": "low"}}
# The proposal's tutor. pilot.sh passes its own TUTOR as CRUTCH_LIVE_TUTOR, the
# model that takes over S's dialogs after T's replayed solution.
LIVE_TUTOR = "anthropic/claude-haiku-5.5"

# --- No-AI control tutor ---
NULL_TUTOR = "crutch/null-tutor"
NULL_TEXT = "Keep going. Show your next step."
# make_arms.py writes this prefix plus the problem as the N arm's system prompt.
NULL_SYSTEM_PREFIX = "No-AI control. The student practices this problem without help:\n"
END_OF_CONVERSATION = "<end_of_conversation>"


# --- Arm S: the student tries first, then sees the answer tutor's solution ---
# Turn 1 asks for an attempt. Turn 2 replays T's first turn on the same problem
# (CRUTCH_T_FIRST_TURNS, written by first_turns.py), so S and T show the same
# solution and S - T isolates the attempt. Later turns go to the live tutor
# (CRUTCH_LIVE_TUTOR); S's system prompt is T's.
TRY_FIRST_TUTOR = "crutch/try-first"
TRY_FIRST_TEXT = ("Before I show you the solution, try the problem yourself. Write out your attempt "
                  "step by step and give your answer, even if you are unsure.")
# T's and S's GPT Base prompt (make_arms.py) ends with this phrase and the
# problem, then the end rule; S's tutor reads the problem from between them.
BASE_PROBLEM_ANCHOR = "Now you can help with this problem: "
END_RULE_ANCHOR = "\n\nDiscuss only this problem:"

# --- Arm W: a confidently wrong solution, the positive control ---
# Turn 1 is a solution checked wrong by wrong_solutions.py
# (CRUTCH_WRONG_SOLUTIONS); later turns are N's filler with N's turn count,
# so W differs from N only in that first turn.
WRONG_TUTOR = "crutch/wrong-solution"
WRONG_SYSTEM_PREFIX = "Positive control. The tutor gives a confidently wrong solution to this problem:\n"


def problem_key(problem: str) -> str:
    return hashlib.sha1(problem.strip().encode("utf-8")).hexdigest()[:12]


def problem_of(messages: list[dict]) -> str:
    """The problem a scripted tutor's system prompt is about (see make_arms.py)."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    for prefix in (NULL_SYSTEM_PREFIX, WRONG_SYSTEM_PREFIX):
        if system.startswith(prefix):
            return system[len(prefix):]
    if BASE_PROBLEM_ANCHOR in system:
        return system.split(BASE_PROBLEM_ANCHOR, 1)[1].split(END_RULE_ANCHOR, 1)[0]
    return system


_null_turns: dict[str, int] | None = None


def _null_target(messages: list[dict]) -> int:
    """Tutor turns the H arm used on this problem (CRUTCH_NULL_TURNS map)."""
    global _null_turns
    if _null_turns is None:
        path = os.environ.get("CRUTCH_NULL_TURNS")
        _null_turns = json.load(open(path)) if path else {}
    default = int(os.environ.get("CRUTCH_NULL_DEFAULT_TURNS", "6"))
    return int(_null_turns.get(problem_key(problem_of(messages)), default))


def _local_reply(model: str, turn: int, text: str) -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": f"{model.split('/')[-1]}-{turn}", "object": "chat.completion", "created": int(time.time()),
        "model": model, "provider": "local",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


def _null_reply(model: str, messages: list[dict]) -> ChatCompletion:
    turn = 1 + sum(1 for m in messages if m["role"] == "assistant")
    text = NULL_TEXT
    if turn >= _null_target(messages):
        text = f"{NULL_TEXT} {END_OF_CONVERSATION}"
    return _local_reply(model, turn, text)


_maps: dict[str, dict] = {}  # environment variable -> its JSON file, read once per process
_replayed: collections.Counter = collections.Counter()  # problem key -> S dialogs given T's turn so far


def _map(env: str) -> dict:
    if env not in _maps:
        path = os.environ.get(env)
        _maps[env] = json.load(open(path, encoding="utf-8")) if path else {}
    return _maps[env]


def _scripted_reply(model: str, messages: list[dict]) -> ChatCompletion | None:
    """The S and W tutors' local turns; None hands the turn to the live tutor.

    A missing replay text returns an empty turn: EduardoRL then marks the
    dialog teacher_failed, and pilot.sh stops the run instead of scoring it.
    """
    turn = 1 + sum(1 for m in messages if m["role"] == "assistant")
    key = problem_key(problem_of(messages))
    if model == TRY_FIRST_TUTOR:
        if turn == 1:
            return _local_reply(model, turn, TRY_FIRST_TEXT)
        if turn > 2:
            return None
        texts = _map("CRUTCH_T_FIRST_TURNS").get(key)
        if not texts:
            logger.error("S: no T first turn for problem %s; this dialog fails", key)
            return _local_reply(model, turn, "")
        # The tutor cannot tell a problem's dialogs apart (same messages so far
        # but the attempt), so they take T's dialogs' first turns in order.
        text = texts[_replayed[key] % len(texts)]
        _replayed[key] += 1
        return _local_reply(model, turn, text)
    if turn > 1:
        return _null_reply(model, messages)
    solution = _map("CRUTCH_WRONG_SOLUTIONS").get("solutions", {}).get(key)
    if not solution:
        logger.error("W: no checked wrong solution for problem %s; this dialog fails", key)
    return _local_reply(model, turn, solution["text"] if solution else "")


# --- Rate limits ---
# At busy times CoreWeave's shared pool rejects most student calls with 429 for
# an hour or more, whoever sends them. A call that gives up ends its dialog
# early or drops a test answer, which biases that arm down, so calls wait the
# rejections out. The concurrency slot stays held while waiting, which also
# slows the run's own request rate.
RATE_LIMIT_PATIENCE = float(os.environ.get("CRUTCH_429_PATIENCE", "1800"))  # seconds per call
RATE_LIMIT_MAX_DELAY = 60.0

# --- Bookkeeping ---
served_by: collections.Counter = collections.Counter()
rate_limited: collections.Counter = collections.Counter()  # 429s waited out, per model
# Responses that carried reasoning text; should stay 0 for the judge.
reasoned: collections.Counter = collections.Counter()
ledger: dict[str, dict[str, float]] = collections.defaultdict(
    lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
)


def _record(model: str, resp) -> None:
    entry = ledger[model]
    entry["calls"] += 1
    usage = getattr(resp, "usage", None)
    if usage is not None:
        entry["prompt_tokens"] += usage.prompt_tokens or 0
        entry["completion_tokens"] += usage.completion_tokens or 0
        entry["cost_usd"] += float(getattr(usage, "cost", 0) or 0)


def dump_ledger(path: str, tag: str) -> None:
    """Append this process's per-model usage to a JSONL ledger."""
    row = {
        "tag": tag,
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": dict(ledger),
        "served_by": [{"model": m, "provider": p, "calls": n} for (m, p), n in sorted(served_by.items())],
        "reasoned": dict(reasoned),
        "rate_limited": dict(rate_limited),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    total = sum(e["cost_usd"] for e in ledger.values())
    logger.info("ledger %s: $%.4f over %d calls -> %s",
                tag, total, sum(e["calls"] for e in ledger.values()), path)


_original_create = _cc.AsyncCompletions.create


async def _routed_create(self, *args, **kwargs):
    model = kwargs.get("model", "")
    if model in (NULL_TUTOR, TRY_FIRST_TUTOR, WRONG_TUTOR):
        messages = kwargs.get("messages") or []
        resp = _null_reply(model, messages) if model == NULL_TUTOR else _scripted_reply(model, messages)
        if resp is None:  # S after T's solution: the live tutor carries on, with T's prompt
            kwargs["model"] = os.environ.get("CRUTCH_LIVE_TUTOR", LIVE_TUTOR)
            return await _routed_create(self, *args, **kwargs)
        served_by[(model, "local")] += 1
        _record(model, resp)
        return resp
    is_claude = model.startswith(CLAUDE_PREFIX)
    if model in ROUTES or is_claude:
        extra = dict(kwargs.get("extra_body") or {})
        extra.pop("chat_template_kwargs", None)  # vLLM-only; OpenRouter ignores it
        extra.update(CLAUDE_EXTRA_BODY if is_claude else ROUTES[model]["extra_body"])
        kwargs["extra_body"] = extra
    if is_claude:
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)
    waited, attempt = 0.0, 0
    while True:
        try:
            resp = await _original_create(self, *args, **kwargs)
            break
        except openai.RateLimitError:
            rate_limited[model] += 1
            if waited >= RATE_LIMIT_PATIENCE:
                raise
            if rate_limited[model] % 500 == 1:
                logger.warning("%s: %d calls rejected with 429 so far; waiting them out",
                               model, rate_limited[model])
            delay = min(5.0 * 2 ** attempt, RATE_LIMIT_MAX_DELAY) * random.uniform(0.5, 1.5)
            attempt += 1
            waited += delay
            await asyncio.sleep(delay)
    provider = getattr(resp, "provider", None) or "unknown"
    if resp.choices and getattr(resp.choices[0].message, "reasoning", None):
        reasoned[model] += 1
    key = (model, provider)
    if key not in served_by:
        logger.info("first %s response served by provider=%s", model, provider)
    served_by[key] += 1
    _record(model, resp)
    return resp


_cc.AsyncCompletions.create = _routed_create
