"""Re-measure the student's cold solve rates on the pilot problems.

EduardoRL measured each problem's untutored solve rate when it built the
dataset (vLLM bf16, one user message, 16 samples, temperature 0.6, 1,024
tokens). This repeats that measurement with EduardoRL's own code through the
pinned OpenRouter route. If the mean moves by more than 0.05, the API student
is not the student the dataset was built with, and comparisons against the
stored rates are suspect.

Usage: python pretest.py <EduardoRL checkout> --parquet .data/arms/dev_N.parquet \
           --out runs/pilot/pretest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
GATE = 0.05


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("eduardo", help="EduardoRL checkout")
    ap.add_argument("--parquet", required=True, help="any arm's parquet (same problems)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--num-problems", type=int, default=-1)
    ap.add_argument("--num-retries", type=int, default=8)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.eduardo).resolve()))
    sys.path.insert(0, str(HERE))
    import functools

    import openrouter_routes
    from eduardo import create_transfer_dataset
    from eduardo.create_transfer_dataset import _solve_rate
    from eduardo.utils import get_async_openai_client

    # The shared CoreWeave pool returns bursts of 429s; three retries lose calls.
    create_transfer_dataset.call_openai_with_retries = functools.partial(
        create_transfer_dataset.call_openai_with_retries, num_retries=args.num_retries
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    extras = list(pd.read_parquet(args.parquet)["extra_info"])
    if args.num_problems > 0:
        extras = extras[: args.num_problems]
    # Same settings create_transfer_dataset.py used for the stored rates.
    settings = argparse.Namespace(
        student_model=openrouter_routes.STUDENT, student_temperature=0.6,
        student_max_tokens=1024, k=args.k,
    )

    async def run():
        client = get_async_openai_client(args.api_base, args.api_key_env)
        sem = asyncio.Semaphore(args.concurrency)
        jobs = []
        for e in extras:
            jobs.append(_solve_rate(e["problem"], str(e["answer"]), client, settings, sem))
            jobs.append(_solve_rate(e["transfer_problem"], str(e["transfer_answer"]), client, settings, sem))
        return await asyncio.gather(*jobs)

    results = asyncio.run(run())
    rows = []
    for i, e in enumerate(extras):
        (src, src_n, src_fail), (var, var_n, var_fail) = results[2 * i], results[2 * i + 1]
        rows.append({
            "key": openrouter_routes.problem_key(e["problem"]),
            "stored_source": float(e["pre_solve_rate"]),
            "measured_source": src,
            "stored_variant": float(e["pre_solve_rate_transfer_problem"]),
            "measured_variant": var,
            "n_scored": src_n + var_n,
            "n_failed": src_fail + var_fail,
        })

    def mean(k):
        return statistics.fmean(r[k] for r in rows)

    summary = {
        "problems": len(rows),
        "k": args.k,
        "stored_source": mean("stored_source"),
        "measured_source": mean("measured_source"),
        "stored_variant": mean("stored_variant"),
        "measured_variant": mean("measured_variant"),
        "failed_calls": sum(r["n_failed"] for r in rows),
    }
    summary["variant_gap"] = summary["measured_variant"] - summary["stored_variant"]
    summary["source_gap"] = summary["measured_source"] - summary["stored_source"]
    summary["gate_pass"] = abs(summary["variant_gap"]) <= GATE
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "problems": rows}, indent=2))

    print("=" * 60)
    print(f"Pre-test on {len(rows)} problems, {args.k} samples each")
    print(f"  source problem : stored {summary['stored_source']:.3f}  measured {summary['measured_source']:.3f}"
          f"  gap {summary['source_gap']:+.3f}")
    print(f"  variant        : stored {summary['stored_variant']:.3f}  measured {summary['measured_variant']:.3f}"
          f"  gap {summary['variant_gap']:+.3f}")
    print(f"  failed calls   : {summary['failed_calls']}")
    print(f"  GATE (|variant gap| <= {GATE}): {'PASS' if summary['gate_pass'] else 'FAIL'}")
    print(f"Written to {args.out}")
    if os.environ.get("CRUTCH_LEDGER"):
        openrouter_routes.dump_ledger(os.environ["CRUTCH_LEDGER"], "pretest")


if __name__ == "__main__":
    main()
