"""Offline stand-in for the OpenRouter chat API, for testing the pipeline at $0.

Answers by model name (tutor, judge or student) and appends each request's
non-message fields to a JSONL log, so tests can check what would be sent.
--reject-student rejects that share of student calls with 429, like
CoreWeave's shared pool at busy times. The tutor answers wrong_solutions.py's
request with a boxed wrong answer; --wrong-drop makes that share of problems
give away the mistake instead, so W leaves them out.

Usage: python crutch-test/mock_openrouter.py --port 8099 --log requests.jsonl
"""

import argparse
import hashlib
import json
import random
import time

from aiohttp import web

_TUTOR_TURNS = [
    "What is the problem asking you to find?",
    "Good. Which quantity do you know, and which one do you need?",
    "Try writing one equation that links them.",
    "Check your result by substituting it back. <end_of_conversation>",
]
_WRONG_REQUEST = "plausible mistake"  # a phrase of wrong_solutions.PROMPT


def _wrong_solution(prompt: str, drop_share: float) -> str:
    # The same problems fail on every try, so the drop path is reproducible.
    problem = prompt.rsplit("Problem: ", 1)[-1]
    if int(hashlib.sha1(problem.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < drop_share:
        return "First, add the two terms. Wait, that was a mistake.\n\nFinal answer: $\\boxed{1}$"
    return "First, combine the given quantities.\n\n$$\nx = 7777777\n$$\n\nFinal answer: $\\boxed{7777777}$"


def reply(model: str, messages: list[dict], wrong_drop: float = 0.0) -> str:
    if model.startswith("anthropic/"):
        if _WRONG_REQUEST in messages[-1]["content"]:
            return _wrong_solution(messages[-1]["content"], wrong_drop)
        n = sum(1 for m in messages if m["role"] == "assistant")
        return _TUTOR_TURNS[min(n, len(_TUTOR_TURNS) - 1)]
    if model.startswith("qwen/"):
        return '{"reasoning": "stub", "decision": "OK"}'
    last = messages[-1]["content"]
    if "boxed" in last:
        return "Working it through, the answer is \\boxed{0}."
    return "I think I should start by writing down what I know."


async def chat(request: web.Request) -> web.Response:
    body = await request.json()
    if body["model"].startswith("meta-llama/") and request.app["rng"].random() < request.app["reject"]:
        return web.json_response({"error": {"message": "rate-limited upstream (mock)", "code": 429}}, status=429)
    row = {k: v for k, v in body.items() if k != "messages"}
    if body["model"].startswith("meta-llama/"):
        # In the student's view the tutor speaks as "user"; the last user turn
        # is the final-attempt prompt. Lets tests check what the student saw.
        tutor = [m["content"] for m in body["messages"] if m["role"] == "user"][:-1]
        row["tutor_turns_seen"] = {
            "text": sum(1 for t in tutor if t.strip() and t != "(hidden)"),
            "hidden": sum(1 for t in tutor if t == "(hidden)"),
            "empty": sum(1 for t in tutor if not t.strip()),
        }
        row["final_attempt"] = "boxed" in body["messages"][-1]["content"]
        row["final_prompt_head"] = body["messages"][-1]["content"][:90]
    with open(request.app["log"], "a") as f:
        f.write(json.dumps(row) + "\n")
    content = reply(body["model"], body["messages"], request.app["wrong_drop"])
    return web.json_response({
        "id": "mock", "object": "chat.completion", "created": int(time.time()),
        "model": body["model"], "provider": "Mock",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--log", default="mock_requests.jsonl")
    ap.add_argument("--reject-student", type=float, default=0.0, help="share of student calls answered with 429")
    ap.add_argument("--wrong-drop", type=float, default=0.0,
                    help="share of problems whose wrong solution gives the mistake away")
    args = ap.parse_args()
    app = web.Application()
    app["log"] = args.log
    app["reject"] = args.reject_student
    app["wrong_drop"] = args.wrong_drop
    app["rng"] = random.Random(0)
    app.router.add_post("/v1/chat/completions", chat)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
