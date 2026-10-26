#!/usr/bin/env bash
# The Crutch Test runner: pre-test, tutor dialogs, unaided post-tests, report.
# PILOT-50 ran it on the 50 dev problems; SPLIT=heldout runs the 300 held-out ones.
#
#   crutch-test/pilot.sh [run-name]                 real run; needs OPENROUTER_API_KEY
#   SPLIT=heldout crutch-test/pilot.sh heldout300   the 300 held-out problems (default: dev)
#   DIALOGS=1 crutch-test/pilot.sh ...              dialogs per problem and arm (default 2, as proposed;
#                                                   PILOT-50 had 1)
#   ARMS="N H T" crutch-test/pilot.sh ...           arms to run (default "N H T S W"; N, H and T are required)
#   N_PROBLEMS=3 crutch-test/pilot.sh smoke         first 3 problems only (about $0.40 for N, H, T)
#   CRUTCH_MOCK=1 crutch-test/pilot.sh dry          $0 dry run against a local stand-in
#   CRUTCH_MOCK=1 MOCK_REJECT=0.5 ...               same, with half the student calls rate-limited
#   CRUTCH_MOCK=1 MOCK_WRONG_DROP=0.5 ...           same, with about half the problems left out of W
#
# Arms: N no AI, H hints, T answers, S tries first and then sees T's solution,
# W gets a confidently wrong solution (the positive control). On the dev split,
# ARMS="N H T" DIALOGS=1 reproduces PILOT-50. Finished steps are skipped on a
# rerun, so a crash costs only the unfinished step. FORCE=1 redoes everything.
# Run setup.sh once first.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/.venv/bin/python"
EDU="$HERE/.deps/EduardoRL"
ARMS_DIR="$HERE/.data/arms"
RUN="$HERE/runs/${1:-pilot50}"
SPLIT=${SPLIT:-dev}  # make_arms.py's seeded split: dev (50 problems) or heldout (300)
DIALOGS=${DIALOGS:-2}  # analyze.py averages a problem's dialogs before bootstrapping over problems
ARMS=${ARMS:-N H T S W}
N=${N_PROBLEMS:--1}
K=${K_SAMPLES:-16}
CONC=${CONCURRENCY:-16}  # CoreWeave's shared pool rate-limits bursts; retries absorb the rest
RETRIES=${RETRIES:-8}
BASE=https://openrouter.ai/api/v1
TUTOR=anthropic/claude-haiku-5.5
NULL=crutch/null-tutor
TRY_FIRST=crutch/try-first
WRONG=crutch/wrong-solution

[[ $SPLIT == dev || $SPLIT == heldout ]] || { echo "SPLIT must be dev or heldout, not '$SPLIT'." >&2; exit 1; }
[[ $DIALOGS =~ ^[1-9][0-9]*$ ]] || { echo "DIALOGS must be a positive integer, not '$DIALOGS'." >&2; exit 1; }
for a in $ARMS; do [[ $a =~ ^[NHTSW]$ ]] || { echo "ARMS takes N, H, T, S and W, not '$a'." >&2; exit 1; }; done
for a in N H T; do
  [[ " $ARMS " == *" $a "* ]] || { echo "ARMS must include N, H and T: N takes H's turn counts, S replays T." >&2; exit 1; }
done
canon=""  # fixed order and no repeats, so the run-folder check below compares like with like
for a in N H T S W; do
  if [[ " $ARMS " == *" $a "* ]]; then canon+="$a "; fi
done
ARMS=${canon% }
has() { [[ " $ARMS " == *" $1 "* ]]; }
[[ -x "$PY" && -d "$EDU" ]] || { echo "Run crutch-test/setup.sh first." >&2; exit 1; }
for a in $ARMS; do
  [[ -f "$ARMS_DIR/${SPLIT}_$a.parquet" ]] || { echo "No ${SPLIT}_$a.parquet: run crutch-test/setup.sh first." >&2; exit 1; }
done
mkdir -p "$RUN/eval"

# Finished steps are skipped on a rerun, so one run folder must keep one design:
# a rerun with another split would keep the dev pre-test for held-out dialogs.
# Folders with results but no config predate this check: dev, one dialog, N H T.
DESIGN="split=$SPLIT dialogs=$DIALOGS arms=${ARMS// /,}"
if [[ ! -f "$RUN/config" ]]; then
  if [[ -f "$RUN/pretest.json" ]] || compgen -G "$RUN/eval/transfer_*" > /dev/null; then
    echo "split=dev dialogs=1 arms=N,H,T" > "$RUN/config"
  else
    echo "$DESIGN" > "$RUN/config"
  fi
fi
if [[ $(< "$RUN/config") != "$DESIGN" ]]; then
  echo "$RUN was run with '$(< "$RUN/config")', not '$DESIGN'. Use a new run name." >&2
  exit 1
fi

if [[ ${CRUTCH_MOCK:-0} == 1 ]]; then
  BASE=http://127.0.0.1:8099/v1
  export OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-dummy}
  "$PY" "$HERE/mock_openrouter.py" --port 8099 --log "$RUN/mock_requests.jsonl" \
    --reject-student "${MOCK_REJECT:-0}" --wrong-drop "${MOCK_WRONG_DROP:-0}" &
  MOCK_PID=$!
  trap 'kill $MOCK_PID 2>/dev/null' EXIT
  for _ in $(seq 20); do curl -s -o /dev/null "$BASE/../" 2>/dev/null && break; sleep 0.3; done
fi
[[ -n ${OPENROUTER_API_KEY:-} ]] || { echo "Set OPENROUTER_API_KEY first (see README)." >&2; exit 1; }
if [[ ${CRUTCH_MOCK:-0} != 1 ]]; then
  "$PY" "$HERE/check_key.py"
  # Don't start into a saturated pool; calls also wait out 429s (openrouter_routes.py).
  "$PY" "$HERE/check_pool.py" --wait --max-hours "${POOL_WAIT_HOURS:-8}"
fi
export CRUTCH_LEDGER="$RUN/ledger.jsonl"
export CRUTCH_ANSWERS="$RUN/answers.jsonl"  # every post-test answer, for cut-off and format checks

COMMON=(
  --use-openrouter --teacher-api-base "$BASE"
  --student-api-base "$BASE" --student-model meta-llama/llama-3.1-8b-instruct
  --student-api-key-env OPENROUTER_API_KEY
  --judge-api-base "$BASE" --judge-model qwen/qwen3.6-27b --judge-api-key-env OPENROUTER_API_KEY
  --num-problems "$N" --num-samples-per-problem "$DIALOGS" --max-teacher-turns 8 --min-teacher-turns-before-eoc 1
  --student-max-tokens 300 --num-retries "$RETRIES"
  --num-student-attempts "$K" --student-concurrency "$CONC" --output-dir "$RUN/eval"
)
edu() {  # EduardoRL's eval, minus the per-request log lines; keeps its exit code
  local rc
  set +e
  "$PY" "$HERE/run_eduardo_eval.py" "$EDU" "$@" 2>&1 | grep --line-buffered -v -E "HTTP Request|^\s*$"
  rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}
done_() { [[ ${FORCE:-0} != 1 ]] && compgen -G "$RUN/eval/$1_*" > /dev/null; }
latest() { ls -t "$RUN"/eval/transfer_conversations_"$1"_*.jsonl | head -1; }
step() { echo; echo "=== $* ==="; }
tutor_of() { case $1 in N) echo "$NULL" ;; S) echo "$TRY_FIRST" ;; W) echo "$WRONG" ;; *) echo "$TUTOR" ;; esac; }
# A rollout with dialogs that failed after retries must not count as done or
# be scored: move it to eval/failed/ and stop, so a rerun redoes that arm.
check_rollout() {
  local f bad
  f=$(latest "$1")
  bad=$("$PY" -c 'import json, sys
print(sum("failed" in str(json.loads(l).get("end_reason")) for l in open(sys.argv[1])))' "$f")
  if [[ $bad != 0 ]]; then
    mkdir -p "$RUN/eval/failed" && mv "$f" "$RUN/eval/failed/"
    echo "$1: $bad dialogs failed after retries; moved $(basename "$f") to eval/failed/. Rerun later." >&2
    exit 1
  fi
}

step "1/4 Pre-test: re-measure cold solve rates"
if [[ ${FORCE:-0} == 1 || ! -f "$RUN/pretest.json" ]]; then
  "$PY" "$HERE/pretest.py" "$EDU" --parquet "$ARMS_DIR/${SPLIT}_N.parquet" --out "$RUN/pretest.json" \
    --api-base "$BASE" --k "$K" --num-problems "$N" --concurrency "$CONC" --num-retries "$RETRIES"
else echo "done already"; fi

step "2/4 Dialogs: H and T first (N and W take H's turn counts, S replays T's first turns)"
for arm in H T; do
  if done_ "transfer_conversations_$arm"; then echo "$arm: done already"; continue; fi
  edu --stage rollout --model "$TUTOR" --test-parquet "$ARMS_DIR/${SPLIT}_$arm.parquet" --tag "$arm" "${COMMON[@]}"
  check_rollout "$arm"
done
"$PY" "$HERE/match_turns.py" "$(latest H)" "$RUN/null_turns.json"
if done_ "transfer_conversations_N"; then echo "N: done already"; else
  CRUTCH_NULL_TURNS="$RUN/null_turns.json" edu --stage rollout --model "$NULL" \
    --test-parquet "$ARMS_DIR/${SPLIT}_N.parquet" --tag N "${COMMON[@]}"
  check_rollout N
fi
if has S; then
  "$PY" "$HERE/first_turns.py" "$EDU" "$(latest T)" "$RUN/t_first_turns.json"
  if done_ "transfer_conversations_S"; then echo "S: done already"; else
    CRUTCH_T_FIRST_TURNS="$RUN/t_first_turns.json" CRUTCH_LIVE_TUTOR="$TUTOR" edu --stage rollout \
      --model "$TRY_FIRST" --test-parquet "$ARMS_DIR/${SPLIT}_S.parquet" --tag S "${COMMON[@]}"
    check_rollout S
  fi
fi
if has W; then
  # Written once per run; W runs on the problems that got a checked wrong solution.
  if [[ ${FORCE:-0} == 1 || ! -f "$RUN/W.parquet" ]]; then
    "$PY" "$HERE/wrong_solutions.py" "$EDU" --parquet "$ARMS_DIR/${SPLIT}_W.parquet" \
      --out "$RUN/wrong_solutions.json" --w-parquet "$RUN/W.parquet" --model "$TUTOR" --api-base "$BASE" \
      --num-problems "$N" --concurrency "$CONC" --num-retries "$RETRIES"
  fi
  if done_ "transfer_conversations_W"; then echo "W: done already"; else
    CRUTCH_WRONG_SOLUTIONS="$RUN/wrong_solutions.json" CRUTCH_NULL_TURNS="$RUN/null_turns.json" \
      edu --stage rollout --model "$WRONG" --test-parquet "$RUN/W.parquet" --tag W "${COMMON[@]}"
    check_rollout W
  fi
fi

step "3/4 Post-tests: tutor turns hidden (M), then shown (P, U)"
for arm in $ARMS; do
  model=$(tutor_of "$arm")
  if ! done_ "transfer_eval_${arm}_masked"; then
    edu --stage score --model "$model" --conversations "$(latest $arm)" --tag "${arm}_masked" "${COMMON[@]}"
  fi
  if ! done_ "transfer_eval_${arm}_open"; then
    # The leak verdict comes from the masked pass; one vote here keeps the judge cheap.
    edu --stage score --model "$model" --conversations "$(latest $arm)" --tag "${arm}_open" \
      --no-omit-teacher-turns --judge-votes 1 "${COMMON[@]}"
  fi
done

step "4/4 Report"
REPORT=()  # real runs land in results/ (tracked); dry runs stay in runs/
[[ ${CRUTCH_MOCK:-0} == 1 ]] || REPORT=(--report "$HERE/results/$(basename "$RUN").md")
"$PY" "$HERE/analyze.py" "$RUN" ${REPORT[@]+"${REPORT[@]}"}
