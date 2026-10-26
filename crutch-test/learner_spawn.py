"""Start the learning-student run on the deployed Modal app, then fetch its result.

A spawned call on a deployed app keeps running when this computer drops its
connection or sleeps. An ephemeral `modal run` does not: on Oct 26 the full run
died three times with its local client (a session restart, a network drop, and a
cancelled detached call). From the repo root:

    modal deploy crutch-test/learner_modal.py                       # after any code change
    python crutch-test/learner_spawn.py start --run pilot50v2       # prints the call id
    python crutch-test/learner_spawn.py fetch <call id>             # writes the outputs once done

`fetch` exits 0 when it wrote results/learner_outputs_<run>.json.gz, 2 while the
call is still running, 3 when Modal cannot be reached (retry), and fails with the
remote error if the run failed. The
pipeline also saves the same file in the crutch-test-cache volume under outputs/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import learner_modal as lm  # noqa: E402  (input building and output writing are shared with `modal run`)

APP = "crutch-test-learner"  # must match modal.App(...) in learner_modal.py


def start(run: str, limit: int, lr: float | None) -> str:
    """Spawn the pipeline on the deployed app and return the function call id."""
    call = modal.Function.from_name(APP, "pipeline").spawn(lm.load_inputs(run, limit, lr))
    extras = ", ".join(x for x in (f"limit {limit}" if limit else "", f"lr {lr:g}" if lr is not None else "") if x)
    print(f"spawned {run}{f' ({extras})' if extras else ''}: {call.object_id}")
    print(f"fetch with: python crutch-test/learner_spawn.py fetch {call.object_id}")
    return call.object_id


def fetch(call_id: str, wait: float) -> int:
    """Write the outputs if the call has finished; 2 means it is still running."""
    try:
        out = modal.FunctionCall.from_id(call_id).get(timeout=wait)
    except (TimeoutError, modal.exception.TimeoutError):  # modal 1.6 raises the builtin one
        print("still running")
        return 2
    except modal.exception.ConnectionError as e:  # this laptop's network drops now and then
        print(f"could not reach Modal ({e}); try again")
        return 3
    lm.write_outputs(out)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--run", default="pilot50v2")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--lr", type=float, default=None, help="fix the learning rate instead of picking it")
    f = sub.add_parser("fetch")
    f.add_argument("call_id")
    f.add_argument("--wait", type=float, default=0.0, help="seconds to wait for the result")
    args = ap.parse_args()
    if args.cmd == "start":
        start(args.run, args.limit, args.lr)
    else:
        sys.exit(fetch(args.call_id, args.wait))


if __name__ == "__main__":
    main()
