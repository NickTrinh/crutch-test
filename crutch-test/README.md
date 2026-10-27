# Crutch Test: runner

Runs the experiments of the Crutch Test ([write-up](report/)): PILOT-50, the go/no-go run on 50 dev problems, and the 300-problem held-out run.

A simulated student (Llama-3.1-8B-Instruct) works through a math problem with one of five tutor arms. Then it takes an unaided post-test. The question is whether copied answers raise practice scores and lower test scores, as they did for humans (Bastani et al., PNAS 2025).

| Arm | Tutor | System prompt and turns |
|---|---|---|
| N | none (scripted "Keep going. Show your next step.") | turn count matched to H per problem |
| H | Claude Haiku 5.5 | Bastani's GPT Tutor prompt, reference answer as notes, plus the end rule |
| T | Claude Haiku 5.5 | Bastani's GPT Base prompt, plus the end rule |
| S | scripted, then Claude Haiku 5.5 | turn 1 asks for an attempt; turn 2 replays T's first turn on that problem; then the live tutor with T's prompt |
| W | scripted | turn 1 is a confidently wrong solution; then N's filler and N's turn count |

All arms open with the same student message, "What is the answer?". That was the most common first message in the trial's GPT Base arm. Only the tutor differs.

S and T show the same solution, so S - T isolates the student's own attempt before it (the generation effect). W is N with its first turn replaced by a wrong solution, so W - N is the positive control: the proposal says W must lower scores. `wrong_solutions.py` writes W's solutions with the tutor model and keeps one only if its boxed answer is graded wrong and its text does not hint at the mistake. A problem without one after 3 tries leaves W only, and the report counts it. Student turns are capped at 300 tokens in every arm, S's attempt included.

The end rule is added, word for word, to both tutor prompts: "Discuss only this problem: do not give the student new or practice problems. When the student has no further questions about this problem, end your reply with <end_of_conversation>." Dialogs may end from the second tutor turn and are capped at 8. Human students left when they were done; the simulated student cannot, and it asks for practice because its prompt says a test is coming.

Protocol history:
- First smoke run: no end rule. Every dialog ran 8 tutor turns, and both tutors filled the extra turns with new practice problems.
- PILOT-50 v1 (`results/pilot50.md`): end rule without the "Discuss only this problem" sentence. The answer tutor still set a new practice problem in 18 of 50 dialogs, and the hint tutor in 2.
- PILOT-50 v2 (`results/pilot50v2.md`, current): the full rule. In a 10-problem check, 0 of 20 dialogs set a new problem; in the run, 0 of 150. It ran on Oct 13 once CoreWeave's shared pool served calls again, after about 16 hours of rejecting most of them.

| View | What the student sees in the post-test |
|---|---|
| P | the same problem, with the full dialog (practice) |
| U | a new variant, with the full dialog |
| M | a new variant, with tutor turns hidden (EduardoRL's test) |
| Ms | the same problem, with tutor turns hidden |

## Run it

No GPU is needed. Every model call goes through OpenRouter.

1. **Setup.** Takes about 2 minutes on Linux, macOS, WSL or Colab.
   ```bash
   crutch-test/setup.sh
   ```
   If `python3 -m venv` is missing (WSL Ubuntu 26.04 ships Python 3.14 without it), install [uv](https://docs.astral.sh/uv/) in user space and build on Python 3.12:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 sh
   export PATH="$HOME/.local/bin:$PATH" && uv python install 3.12
   PYTHON=$(uv python find 3.12) crutch-test/setup.sh
   ```
   On Windows, run the scripts from WSL.
2. **Dry run.** Costs $0 and takes about 30 seconds. It checks the plumbing against a local stand-in server. The scores are meaningless because the stand-in student always answers 0. Add `SPLIT=heldout` for the held-out split. The unit tests need no server.
   ```bash
   CRUTCH_MOCK=1 N_PROBLEMS=5 crutch-test/pilot.sh dry
   crutch-test/.venv/bin/python crutch-test/tests/test_analyze.py
   crutch-test/.venv/bin/python crutch-test/tests/test_arms.py
   ```
3. **Key.** You need a regular OpenRouter API key from https://openrouter.ai/settings/keys, with credit; set a $25 limit on the key. A key from the Management Keys page can't call models: every call gets 401 "User not found.", and `pilot.sh` stops before running anything. Never commit the key or paste it into a chat.
   - Your own machine: `export OPENROUTER_API_KEY=...`
   - A Claude cloud session: add `OPENROUTER_API_KEY` in the environment settings. A new session picks it up.
4. **Smoke run.** Costs about $0.50 with `ARMS="N H T" DIALOGS=1`, and about $1.50 at the defaults (five arms, 2 dialogs). Takes a few minutes.
   ```bash
   N_PROBLEMS=3 crutch-test/pilot.sh smoke
   ```
   Check `crutch-test/results/smoke.md`:
   - The "Served by" line shows CoreWeave for the student and Chutes for the judge.
   - The decision line is not "INVALID RUN", and there are no blanked turns.
   - Read two dialogs per arm in `runs/smoke/eval/transfer_conversations_{T,H}_*.jsonl`. T should hand over the answer, and H should withhold it. Neither should move on to new problems.
   - S (`transfer_conversations_S_*.jsonl`): the attempt prompt, the student's attempt, T's solution, then live replies that stay on the problem.
   - W: read three solutions in `runs/smoke/wrong_solutions.json`. Each should be confident, wrong in one plausible step, and silent about it. The report's S and W section counts how many problems got one.
   - CoreWeave, the only bf16 host of the student, serves it from a pool shared by all OpenRouter users. At busy times the pool rejects most calls with 429 for an hour or more, whoever sends them. Three guards handle this:
     - `pilot.sh` first waits until the pool serves calls again (`check_pool.py`, up to `POOL_WAIT_HOURS=8`).
     - Each call waits out 429s for up to 30 minutes (`CRUTCH_429_PATIENCE`). The ledger counts the waits.
     - A rollout with any failed dialog moves to `eval/failed/`, and the run stops, so a rerun redoes that arm instead of scoring it.
5. **Pilot.** PILOT-50 v2 cost $6.54 and ran arms N, H and T with one dialog each:
   ```bash
   ARMS="N H T" DIALOGS=1 crutch-test/pilot.sh pilot50
   ```
   A rerun skips finished steps, so a crash only costs the unfinished step.
6. **Read `crutch-test/results/pilot50.md`** against the gates below and commit it. Raw runs stay in `runs/`, which git ignores.
7. **Held-out run.** The proposal's main run: 300 held-out problems, five arms, 2 dialogs per arm. Smoke it on 3 held-out problems first.
   ```bash
   SPLIT=heldout N_PROBLEMS=3 crutch-test/pilot.sh smoke-heldout
   SPLIT=heldout crutch-test/pilot.sh heldout300
   ```
   At the defaults that is 3,000 dialogs, 20 times PILOT-50 v2: about $155 on OpenRouter with `views.py`, and about 47 hours at PILOT-50 v2's pace.

| Setting | Default | What it does |
|---|---|---|
| `SPLIT` | `dev` | `dev` (50 problems) or `heldout` (300), from `make_arms.py`'s seeded split; the pre-test uses it too |
| `DIALOGS` | `2` | dialogs per problem and arm; `analyze.py` averages a problem's dialogs before bootstrapping over problems |
| `ARMS` | `N H T S W` | arms to run; N, H and T are required (N and W take H's turn counts, S replays T) |
| `N_PROBLEMS` | all | the first N problems of the split |
| `FORCE` | `0` | `1` redoes every step |
| `MOCK_REJECT`, `MOCK_WRONG_DROP` | `0` | dry runs only: the share of student calls rate-limited; the share of problems whose wrong solution gives the mistake away |

A run folder keeps one design. `runs/<run>/config` records the split, dialogs and arms, and a rerun with other values stops: finished steps are skipped, so a dev pre-test would otherwise be kept for held-out dialogs. Folders from before this check count as dev, 1 dialog, arms N, H, T.

## Go/no-go gates (proposal, Risks section)

| Gate | Pass |
|---|---|
| Practice goes up: P(T) - P(N) | ≥ +0.20 |
| The judge sees the handover: rate for T / H | ≥ 80% / ≤ 20% |
| The student matches the dataset: re-measured pre-test vs stored | within 0.05 |
| The 300-problem audit is affordable | ≤ 12 A100-hours projected |

A run where more than 1% of student calls or any dialog failed after retries is marked INVALID RUN. EduardoRL scores a problem whose attempts all failed as 0.0, which biases scores down.

If any gate fails, fix it for up to two days, then try the Qwen3-8B student. If that also fails, ship the minimum tier as an instrument audit.

The report also prints the first look at the research question. That is M: T - N, whether the masked student pays for copied answers:
- If T - N ≤ -0.05, the student already pays, and the pre-registered pivot applies.
- If T - N ≥ -0.02, the gap exists, and the project goes ahead as planned.

## Files

| File | Purpose |
|---|---|
| `setup.sh` | Builds the venv (EduardoRL's pinned packages), clones EduardoRL at e8c89b7, downloads the dataset and the human trial's log, and builds the split |
| `make_arms.py` | 355 eligible problems: seeded 50 dev and 300 held-out (seed 2026, hash `0d1333aed481da89`). Writes one parquet per arm and split |
| `pilot.sh` | Pre-test, then dialogs (H, T, then N, S and W), then post-tests (hidden, then shown), then the report |
| `pretest.py` | Re-measures cold solve rates with EduardoRL's own pre-test code |
| `match_turns.py` | Gives N (and W) the same number of tutor turns that H used on each problem, averaged over its dialogs |
| `first_turns.py` | Collects T's first turns, as T's student saw them, for S to replay |
| `wrong_solutions.py` | Writes W's confidently wrong solutions and keeps the ones graded wrong that do not hint at the mistake |
| `run_eduardo_eval.py` | Runs EduardoRL's offline eval with the runtime fixes below |
| `openrouter_routes.py` | Per-model routing, the in-process scripted tutors (N, S, W), and a cost ledger (OpenRouter reports dollars on every call) |
| `analyze.py` | Gates, paired contrasts with 95% cluster-bootstrap intervals, arms S and W, the pass rules with Holm's correction, dialog health, cost |
| `tests/test_analyze.py`, `tests/test_arms.py` | Plain assert scripts, no network: per-problem averaging, Holm, the scorecard, the S and W tutors, the split |
| `check_key.py` | Stops a real run unless the key is a regular OpenRouter key that can call models |
| `posttest.py` | Samples and grades post-test answers at a 2,048-token cap and logs every answer |
| `views.py` | Scores views M, C and K on a finished run's dialogs, S and W included when the run has them |
| `check_pool.py` | Probes CoreWeave's shared pool with tiny calls; `--wait` holds until it serves the student again |
| `learner_inputs.py` | Builds the learning student's training dialogs and tests from a finished run |
| `learner_core.py` | The learning student's training: label masking, LoRA, SGD, learning-rate pick |
| `learner_modal.py` | Trains the adapters and samples the tests on a Modal A100 (run from your computer) |
| `learner_grade.py` | Grades the Modal outputs into views L and base_L for `analyze.py` |
| `report/` | The write-up: LaTeX source and figures |
| `mock_openrouter.py` | $0 stand-in server; it also logs what the student saw at each post-test |
| `demo/build_data.py`, `demo/index.html` | The dialog explorer: the script builds `demo/data.json` from a graded run and checks its averages against the report; the page is served by GitHub Pages from the public repo's `docs/` |

## Fixes to EduardoRL, applied at runtime (no fork)

1. **Routing.**
   - The student is pinned to CoreWeave bf16 with no fallback. The fp8 hosts change its weights.
   - The judge is pinned to Chutes with thinking off. OpenRouter ignores `chat_template_kwargs`.
   - Claude gets no temperature or top_p, which Claude 5.5 models reject, and effort low.
   - A call rejected with 429 waits and retries for up to 30 minutes. A call that gave up would end its dialog early or drop a test answer, which biases its arm down.
2. **Blanked tutor turns.** EduardoRL's post-test builder strips tutor turns with `assume_thinking=True`. That rule turns any reply without a `<think>` tag into an empty string, which hides API tutors' answers in the P and U views. The wrapper sets it to False. The dry run checks this: every post-test in the shown pass has the tutor's text, and the hidden pass shows `(hidden)`.
3. **Same-problem post-test.** EduardoRL's prompt for the original problem says only "create a step by step complete solution", trusting the dialog to say which problem. When a tutor had moved on to a new problem, the student solved that one and was graded against the original answer. In the first smoke run, P for the answer tutor came out as 0.000 on all 3 problems. The wrapper restates the original problem, as the variant post-test already does.
4. **Post-test length cap and answer log.** EduardoRL caps post-test answers at 800 tokens. Llama-3.1-8B writes about 570 tokens per solution, and long ones were cut off before their `\boxed{}` answer. Cut-offs hit the no-AI arm more often (21% vs 13-14% unboxed at a 1,024 cap in PILOT-50 v1), because that student writes longer solutions, so they favoured the tutored arms. Post-tests now go through `posttest.py`: a 2,048-token cap (`CRUTCH_POSTTEST_MAX_TOKENS`), and every answer logged to `runs/<run>/answers.jsonl` with its text, finish reason and grade. The pre-test keeps the dataset's 1,024.

## More simulated students on the same dialogs

`views.py` scores further views on a finished run's dialogs, so no new dialogs are needed. About $2 for PILOT-50.

```bash
crutch-test/.venv/bin/python crutch-test/views.py crutch-test/.deps/EduardoRL crutch-test/runs/pilot50
crutch-test/.venv/bin/python crutch-test/analyze.py crutch-test/runs/pilot50 --report crutch-test/results/pilot50.md
```

| View | What reaches the test on the new variant | Samples |
|---|---|---|
| M | the student's turns, with tutor turns hidden (EduardoRL's test), rescored at the new cap | 16 |
| C | M minus every student sentence with half or more of its word 6-grams in a tutor turn. This removes what the leak judge looks for, so it is circular by construction and serves as a reference only | 8 |
| K | one note of at most 120 tokens, written by the student right after the dialog with the dialog in view | 16 |

The report gives each view's H - N and T - N and the proposal's pass rules:
- P2, test down: T - N ≤ -0.03, with its interval below 0.
- P3, hints safe: the lower end of H - N is above -0.03.

It also reports cut-off and unboxed answers per arm from the answer log.

### Pass rules with Holm's correction

The write-up calls these pass rules gates: P1–P4 here are G1–G4 there.

The report ends with a scorecard. It lists P1 (practice up), P2, P3 and P4 (L's drift check) for every test view (U, M, C, K, L), with the unadjusted and the Holm-corrected verdicts side by side, as the proposal promises.
- Each view is one family, corrected over P1-P3.
- The p-values come from the bootstrap's own resamples: twice the share of resampled means past the rule's boundary. So p < 0.05 matches the interval verdict, and no interval moves.
- P2 keeps its point condition (T - N ≤ -0.03).
- P4 is a point rule (within ±0.03), so it has no p-value and stays unadjusted. Inside Holm, a drift test would make passing easier, not harder.
- For the claim "L passes all four rules", an intersection-union test needs no correction, so Holm is conservative here. The unadjusted verdicts stand next to it.

S and W have their own section and stay out of the pass rules. It gives S - N, S - T and W - N per view, with their own random seed, so adding them moves no other interval. It also gives the manipulation checks:
- handover in S;
- the share of problems with a checked wrong solution;
- how often W's student repeats the planted wrong answer on the same problem with the dialog in view.

The teaching judge cannot check W, because it already rejects N's repeated filler, which W shares.

## The learning student (L)

The views above use a frozen student, so whatever it learned has to be in the text it carries into the test. L learns in its weights. A student who copied the answer wrote less of its own work, so it has less to learn from. That is the mechanism the frozen views cannot show. L needs a GPU, so it runs on [Modal](https://modal.com).

1. **Inputs (cloud session).** This writes `results/learner_inputs_<run>.json.gz`. Commit that file.
   ```bash
   crutch-test/.venv/bin/python crutch-test/learner_inputs.py crutch-test/.deps/EduardoRL crutch-test/runs/pilot50v2
   ```
2. **Train and sample (your computer).** You need a Modal account. The real run takes about an hour and $3-5 of credit. The first run also downloads the 16 GB model into a Modal volume. Start with the 15-minute check on 2 dialogs per arm. If it finishes cleanly, do the real run, then commit and push `results/learner_outputs_<run>.json.gz`.
   ```bash
   pip install modal
   python -m modal setup
   modal run crutch-test/learner_modal.py --run pilot50v2 --limit 2
   modal run crutch-test/learner_modal.py --run pilot50v2
   ```
3. **Grade and report (cloud session).**
   ```bash
   crutch-test/.venv/bin/python crutch-test/learner_grade.py crutch-test/.deps/EduardoRL crutch-test/runs/pilot50v2
   crutch-test/.venv/bin/python crutch-test/analyze.py crutch-test/runs/pilot50v2 --report crutch-test/results/pilot50v2.md
   ```

| Choice | Setting | Why |
|---|---|---|
| What it trains on | The student's own turns. Tutor turns are context with no loss, and the scripted opener is excluded | It learns from its own work, not from reading the tutor |
| Update | Plain SGD, 4 steps, on the summed token loss / 1,000 | More of its own tokens means a bigger update. Adam would rescale every update to the same size and cancel that |
| Adapter | LoRA rank 16, alpha 32, on all 7 projections; a fresh seeded adapter per dialog | Each dialog is one student |
| Learning rate | From 0.03, 0.1, 0.3, 1 and 3: the value whose median loss drop on 20 N and 20 H dialogs is closest to 25% | Fixed before any test. T dialogs never inform it |
| Tests | No transcript. The new variant (16 samples), the same problem restated (8), and an unrelated problem (8: another dialog's variant, the same one for every arm) | The unrelated problem checks for general drift |
| Model | `RedHatAI/Llama-3.1-8B-Instruct` at a pinned revision, in bf16, with vLLM 0.30.0 serving the adapters | An ungated copy of Meta's weights: the same SHA-256 for all 4 weight files |

On top of the views table, the report adds:
- L minus the frozen model, per arm, on each test. This checks that training changed anything.
- P4: T - N on the unrelated problem must stay within ±0.03.
- The correlation between trained tokens and the gain over the frozen model.

`tests/test_learner_cpu.py` runs the same training code on SmolLM2-135M on a CPU in about a minute. It needs torch, transformers 5.19 and peft 0.19.1.

## Not tested yet

- The learning student on 2 dialogs per problem, or on S and W. `learner_inputs.py` works per dialog and takes `--arms N,H,T` by default, and `analyze.py` averages L's dialogs per problem, but no run has tried it.
- S and W with the real tutor. Only the stand-in server has run them; the held-out smoke run is the first real check.

## Sources

- Tutor prompts: the system turns in `valid_student_data.csv` from [obastani/GenAICanHarmLearning](https://github.com/obastani/GenAICanHarmLearning) at commit 2f63dae. That repository has no license, so `setup.sh` downloads the file and `make_arms.py` cuts the prompts out of it, checked against the SHA-256 of the prompts the pilots ran with.
  - The CSV lost its escape characters, so `make_arms.py` restores the newlines; the original typos stay.
  - The instruction text matches all 3,375 GPT Base and 3,714 GPT Tutor conversations once the CSV's uneven escapes are normalized.
  - The topic slot is generic, because Big-Math has no lesson labels.
- Everything else (environment, dataset, judge and post-test): [EduardoRL](https://github.com/eth-lre/EduardoRL), CC BY-SA 4.0.
