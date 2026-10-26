"""Sample and grade unaided post-test answers, logging every answer.

Shared by run_eduardo_eval.py, which swaps it in for EduardoRL's
compute_fraction, and by views.py. With CRUTCH_ANSWERS set, every answer is
appended to that JSONL file with its text, finish reason, length and grade,
so cut-off and unboxed answers can be checked per arm and view.
"""

from __future__ import annotations

import asyncio
import json
import os

from eduardo.utils import call_openai_with_retries, grade_math_answer

# EduardoRL caps post-test answers at 800 tokens. In PILOT-50 v1, at 1,024,
# 21% of the no-AI arm's answers and 13-14% of the tutored arms' answers had
# no gradable \boxed{} answer, so the cap was cutting off long solutions.
MAX_TOKENS = int(os.environ.get("CRUTCH_POSTTEST_MAX_TOKENS", "2048"))

# EduardoRL's same-problem post-test only says "create a step by step complete
# solution", trusting the dialog to say which problem; every view restates it.
SAME_PROBLEM_PROMPT = (
    "The conversation with the teacher has ended. You are now tested on the original "
    "problem from this conversation, to check whether you can solve it on your own.\n\n"
    "Original problem:\n{problem}\n\n"
    "It is time you create a step by step complete solution such that it can be graded "
    "without the past conversation.\n"
    "Always format your final answer in perfect LaTeX \\boxed{final_answer} format.\n"
)


async def _sample(client, model, msgs, temperature, max_tokens):
    resp = await client.chat.completions.create(
        model=model, messages=msgs, temperature=temperature, max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    usage = getattr(resp, "usage", None)
    return {
        "text": choice.message.content or "",
        "finish": choice.finish_reason,
        "tokens": getattr(usage, "completion_tokens", None),
    }


async def sample_and_grade(client, model, msgs, ground_truth, k, *, temperature=0.6,
                           max_tokens=MAX_TOKENS, semaphore=None, retry_kwargs=None,
                           log: dict | None = None):
    """Return (fraction correct among answered calls or None, n_unboxed, n_failed)."""
    outs = await asyncio.gather(*[
        call_openai_with_retries(
            _sample, client, model, msgs, temperature, max_tokens,
            label="posttest", use_fallback=True, fallback=None, semaphore=semaphore,
            **(retry_kwargs or {}),
        )
        for _ in range(k)
    ])
    n_correct = n_unboxed = n_failed = n_scored = 0
    rows = []
    for i, out in enumerate(outs):
        if out is None:
            n_failed += 1
            continue
        grade = grade_math_answer(out["text"], ground_truth)
        n_scored += 1
        n_correct += int(grade["acc"])
        n_unboxed += int(grade["incorrect_format"])
        if log is not None:
            rows.append({**log, "i": i, "cap": max_tokens, "finish": out["finish"], "tokens": out["tokens"],
                         "pred": grade["pred"], "acc": int(grade["acc"]),
                         "unboxed": int(grade["incorrect_format"]), "text": out["text"]})
    if rows and os.environ.get("CRUTCH_ANSWERS"):
        with open(os.environ["CRUTCH_ANSWERS"], "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return (n_correct / n_scored if n_scored else None), n_unboxed, n_failed
