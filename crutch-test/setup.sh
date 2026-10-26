#!/usr/bin/env bash
# One-time setup: Python env with EduardoRL's pinned packages, EduardoRL at the
# commit the proposal cites, the human trial's log (the tutor prompts are cut
# from it), and the pilot's problem split. About 2 minutes.
# Works on a laptop, Colab or a Claude cloud session; no GPU needed.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
EDU_COMMIT=e8c89b7
TRIAL_COMMIT=2f63dae1a01d51453826fe07ef5cf6678e339588  # obastani/GenAICanHarmLearning
TRIAL_LOG="$HERE/.deps/GenAICanHarmLearning/valid_student_data.csv"
PKGS=(openai==2.33.0 math-verify==0.8.0 antlr4-python3-runtime==4.9.3 latex2sympy2_extended==1.10.2
      sympy==1.14.0 mpmath==1.3.0 pandas==2.3.3 pyarrow==22.0.0 datasets==4.8.5 numpy aiohttp)
PY=${PYTHON:-$(command -v python3.12 || command -v python3)}

if [[ ! -x "$HERE/.venv/bin/python" ]]; then
  echo "[setup] Python env ($("$PY" --version))"
  if command -v uv > /dev/null; then
    uv venv -q -p "$PY" "$HERE/.venv"
    VIRTUAL_ENV="$HERE/.venv" uv pip install -q "${PKGS[@]}"
  else
    "$PY" -m venv "$HERE/.venv"
    "$HERE/.venv/bin/pip" install -q "${PKGS[@]}"
  fi
fi

if [[ ! -d "$HERE/.deps/EduardoRL" ]]; then
  echo "[setup] EduardoRL @ $EDU_COMMIT"
  git clone -q https://github.com/eth-lre/EduardoRL "$HERE/.deps/EduardoRL"
  git -C "$HERE/.deps/EduardoRL" checkout -q "$EDU_COMMIT"
fi

if [[ ! -f "$TRIAL_LOG" ]]; then
  echo "[setup] human trial's log @ ${TRIAL_COMMIT:0:7} (40 MB), for the tutor prompts"
  mkdir -p "$(dirname "$TRIAL_LOG")"
  # Download beside the target, then rename, so a broken download is never mistaken for the file.
  curl -fsSL -o "$TRIAL_LOG.part" "https://raw.githubusercontent.com/obastani/GenAICanHarmLearning/$TRIAL_COMMIT/text_analysis/data/raw/valid_student_data.csv"
  mv "$TRIAL_LOG.part" "$TRIAL_LOG"
fi

if [[ ! -f "$HERE/.data/test.parquet" ]]; then
  echo "[setup] dataset (dmacjam/eduardoRL-transfer-dataset)"
  (cd "$HERE/.deps/EduardoRL" && "$HERE/.venv/bin/python" -m eduardo.process_dataset \
     --num-train 16 --num-test 16 --num-holdout 494 --out-dir "$HERE/.data" 2>&1 \
     | grep -E "^\[data\]" || true)
fi

"$HERE/.venv/bin/python" "$HERE/make_arms.py" --test-parquet "$HERE/.data/test.parquet" --out-dir "$HERE/.data/arms"
# The proposal's split; a different hash means the dataset changed upstream.
EXPECTED=0d1333aed481da89
GOT=$("$HERE/.venv/bin/python" -c "import json; print(json.load(open('$HERE/.data/arms/split.json'))['content_hash'])")
if [[ "$GOT" != "$EXPECTED" ]]; then
  echo "[setup] WARNING: split hash $GOT != $EXPECTED; the dataset changed since the proposal." >&2
fi
echo "[setup] done. Next: crutch-test/pilot.sh (see crutch-test/README.md)"
