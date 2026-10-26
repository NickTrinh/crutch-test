"""Check that CoreWeave's shared pool is serving the student model; optionally wait for it.

At busy times OpenRouter's shared CoreWeave pool, the only bf16 host of
Llama-3.1-8B, rejects most calls with 429 for an hour or more, whoever sends
them. A run started then spends its time waiting on retries. This sends a few
tiny student calls (about $0.00001 each) and exits 0 once at least --need of
--probes succeed. With --wait it checks again every --every minutes, for up to
--max-hours, and exits 1 if the pool never recovers. Never prints the key.

Usage: python check_pool.py [--wait]
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).resolve().parent))
import openrouter_routes  # noqa: E402  (routes the probe to CoreWeave bf16)

openrouter_routes.RATE_LIMIT_PATIENCE = 0  # a probe reports a 429 instead of waiting it out


async def probe(client: openai.AsyncOpenAI, n: int) -> int:
    ok = 0
    for i in range(n):
        try:
            await client.chat.completions.create(
                model=openrouter_routes.STUDENT, max_tokens=3,
                messages=[{"role": "user", "content": "Say ok."}])
            ok += 1
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError):
            pass  # rejected, or a network blip: either way not served
        if i < n - 1:
            await asyncio.sleep(3)
    return ok


async def check(args) -> bool:
    # One event loop for the whole wait: the client's connections belong to it.
    client = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                                api_key=os.environ["OPENROUTER_API_KEY"], max_retries=0)
    deadline = time.time() + args.max_hours * 3600
    while True:
        ok = await probe(client, args.probes)
        stamp = time.strftime("%H:%M UTC", time.gmtime())
        if ok >= args.need:
            print(f"[pool] {stamp}: {ok} of {args.probes} student calls served; pool is usable", flush=True)
            return True
        print(f"[pool] {stamp}: only {ok} of {args.probes} student calls served (429 from CoreWeave's shared pool)",
              flush=True)
        if not args.wait or time.time() + args.every * 60 > deadline:
            return False
        await asyncio.sleep(args.every * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probes", type=int, default=6)
    ap.add_argument("--need", type=int, default=5)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--every", type=float, default=10, help="minutes between checks")
    ap.add_argument("--max-hours", type=float, default=8)
    args = ap.parse_args()
    if not asyncio.run(check(args)):
        sys.exit(1)


if __name__ == "__main__":
    main()
