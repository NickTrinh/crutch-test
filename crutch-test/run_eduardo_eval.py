"""Run EduardoRL's environment eval with every role served through OpenRouter.

Usage: python crutch-test/run_eduardo_eval.py <EduardoRL checkout> [eval args...]

Set CRUTCH_LEDGER to a file path to append this run's tokens and dollars per
model (the --tag value labels the row).
"""

import logging
import os
import sys
from pathlib import Path

repo = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(repo))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import functools  # noqa: E402

import openrouter_routes  # noqa: E402  (patches the openai client)
import posttest  # noqa: E402
from eduardo import utils as eduardo_utils  # noqa: E402
from eduardo.metrics import correctness  # noqa: E402
from eval import run_env_test_eval  # noqa: E402

# EduardoRL's post-test builder strips tutor turns with assume_thinking=True,
# which blanks every turn that has no <think> tag. API tutors rarely send one,
# so with --no-omit-teacher-turns the student would see empty tutor turns. All
# tutors here are API or scripted, so untagged text is the visible reply.
correctness.strip_teacher_controls = functools.partial(
    eduardo_utils.strip_teacher_controls, assume_thinking=False
)

# EduardoRL's same-problem post-test only says "create a step by step complete
# solution", trusting the dialog to say which problem. When a tutor moves on to
# new practice problems, the student solves the last one discussed and is
# graded against the original answer. Restate the original problem, as the
# variant post-test already does.
_build_final_attempt_messages = correctness._build_final_attempt_messages


def _build_with_restated_problem(messages, problem, transfer_problem=None, omit_teacher_turns=False):
    out = _build_final_attempt_messages(
        messages, problem, transfer_problem, omit_teacher_turns=omit_teacher_turns
    )
    if not transfer_problem:
        out[-1] = {"role": "user", "content": posttest.SAME_PROBLEM_PROMPT.replace("{problem}", problem)}
    return out


correctness._build_final_attempt_messages = _build_with_restated_problem

def _tag() -> str:
    argv = sys.argv
    return argv[argv.index("--tag") + 1] if "--tag" in argv else "untagged"


# Post-tests go through posttest.py: a 2,048-token cap instead of EduardoRL's
# 800 (long solutions were cut off before their \boxed{} answer), and every
# answer is logged to CRUTCH_ANSWERS. Same messages, sampling and grading.
async def _compute_fraction_logged(messages, ground_truth, problem, client, model,
                                   num_attempts=8, temperature=0.6, max_tokens=800,
                                   retry_kwargs=None, semaphore=None,
                                   transfer_problem=None, omit_teacher_turns=False):
    msgs = correctness._build_final_attempt_messages(
        messages, problem, transfer_problem, omit_teacher_turns=omit_teacher_turns
    )
    log = {"tag": _tag(), "problem": openrouter_routes.problem_key(problem),
           "test": "variant" if transfer_problem else "same", "hidden": omit_teacher_turns}
    frac, n_unboxed, n_failed = await posttest.sample_and_grade(
        client, model, msgs, ground_truth, num_attempts, temperature=temperature,
        semaphore=semaphore, retry_kwargs=retry_kwargs, log=log,
    )
    # EduardoRL scores a problem with no answered calls as 0.0; the report
    # marks such runs invalid, and the answer log has the real counts.
    return (0.0 if frac is None else frac), n_unboxed, n_failed


correctness.compute_fraction = _compute_fraction_logged


try:
    run_env_test_eval.main()
finally:
    log = logging.getLogger("crutch.routes")
    for (model, provider), n in sorted(openrouter_routes.served_by.items()):
        log.info("served %5d calls: %s via %s (%d with reasoning text)",
                 n, model, provider, openrouter_routes.reasoned[model])
    if os.environ.get("CRUTCH_LEDGER"):
        openrouter_routes.dump_ledger(os.environ["CRUTCH_LEDGER"], _tag())
