# PILOT report: pilot50v2

**Decision:** NO-GO for now (failing: 300-problem audit <= 12 A100-hours). Fix it for up to two days, then try the Qwen3-8B student.

**Framing:** The masked student does not pay for copied answers (M: T-N >= -0.02), the gap the project set out to test.

## Go/no-go gates (proposal, Risks section)

| Gate | Value | Pass |
|---|---|---|
| practice: P(T) - P(N) >= +0.20 | +0.574 [+0.464, +0.681] | yes |
| judge: handover T >= 80% | 82% | yes |
| judge: handover H <= 20% | 0% | yes |
| pre-test within 0.05 of the dataset | -0.016 (measured 0.302 vs stored 0.319) | yes |
| 300-problem audit <= 12 A100-hours | 15.3 h (683 tokens per answer) | NO |

## Post-test solve rate by view (paired by problem vs N; 95% bootstrap CI)

| View | N | H | T | H - N | T - N |
|---|---|---|---|---|---|
| P | 0.285 | 0.814 | 0.859 | +0.529 [+0.435, +0.623] | +0.574 [+0.464, +0.681] |
| U | 0.325 | 0.579 | 0.640 | +0.254 [+0.151, +0.357] | +0.315 [+0.198, +0.428] |
| M | 0.331 | 0.464 | 0.454 | +0.133 [+0.040, +0.228] | +0.122 [+0.034, +0.212] |
| Ms | 0.285 | 0.608 | 0.557 | +0.323 [+0.230, +0.414] | +0.273 [+0.185, +0.360] |

Paired SD of T - N per problem under M: 0.320 (the power analysis assumed 0.18).

## Did the answer tutor set new problems?

Dialogs where the tutor set a new practice problem: N 0, H 0, T 0 (pattern match on tutor turns).

| T dialogs | M: T - N | U: T - N |
|---|---|---|
| T posed a new problem (n=0) | n/a | n/a |
| T did not (n=50) | +0.122 [+0.036, +0.210] | +0.315 [+0.201, +0.426] |

## More simulated students on the same dialogs (views.py)

Each view changes only what the student carries into its test on the new variant. Answers are capped at 2,048 tokens and all logged. Human target from Bastani's exam: T - N about -0.05 (GPT Base -17%), H - N about 0 (GPT Tutor).

| View | Carries | N | H | T | H - N | T - N | Test down (P2) | Hints safe (P3) |
|---|---|---|---|---|---|---|---|---|
| M | student turns; tutor turns hidden | 0.331 | 0.464 | 0.454 | +0.133 [+0.041, +0.225] | +0.122 [+0.036, +0.209] | fail | pass |
| C | M minus sentences copied from the tutor | 0.275 | 0.443 | 0.420 | +0.168 [+0.065, +0.270] | +0.145 [+0.052, +0.233] | fail | pass |
| K | one note of at most 120 tokens | 0.336 | 0.354 | 0.396 | +0.018 [-0.041, +0.079] | +0.060 [+0.003, +0.120] | fail | fail |
| L | weights trained on its own turns; no transcript | 0.356 | 0.361 | 0.334 | +0.005 [-0.039, +0.050] | -0.022 [-0.074, +0.028] | fail | fail |

P2 passes when T - N <= -0.03 with its interval below 0; P3 when the lower end of H - N is above -0.03.

| View | T - N where T set a new problem | T - N where it did not |
|---|---|---|
| M | n/a (n=0) | +0.122 [+0.035, +0.210] (n=50) |
| C | n/a (n=0) | +0.145 [+0.055, +0.233] (n=50) |
| K | n/a (n=0) | +0.060 [+0.004, +0.120] (n=50) |
| L | n/a (n=0) | -0.022 [-0.075, +0.026] (n=50) |

C removed N 0/1495 (0%), H 331/2225 (15%), T 167/1643 (10%) of student sentences as copied from the tutor.

K notes: N 71 words, 4% contain the original answer, H 78 words, 6% contain the original answer, T 75 words, 8% contain the original answer.

M here is the pilot's own hidden pass (same prompts, same 2,048-token cap), so views.py did not rescore it.

| View | Arm | Answers | Cut off at the cap | No boxed answer | Correct among boxed |
|---|---|---|---|---|---|
| M | N | 800 | 9.0% | 9.4% | 0.366 |
| M | H | 800 | 9.0% | 9.1% | 0.510 |
| M | T | 800 | 10.0% | 10.0% | 0.504 |
| C | N | 400 | 11.8% | 11.8% | 0.312 |
| C | H | 400 | 9.0% | 9.5% | 0.489 |
| C | T | 400 | 11.8% | 11.2% | 0.473 |
| K | N | 800 | 10.5% | 10.8% | 0.377 |
| K | H | 800 | 10.6% | 10.6% | 0.396 |
| K | T | 800 | 11.4% | 11.1% | 0.446 |
| L | N | 800 | 13.2% | 12.9% | 0.409 |
| L | H | 800 | 14.5% | 14.2% | 0.421 |
| L | T | 800 | 14.1% | 14.1% | 0.389 |

### The learning student (L)

Each dialog trains its own LoRA adapter on the student's own turns (plain SGD, lr 0.1, picked on N and H dialogs only); the test then shows no transcript. "Frozen" is the same model with no training, on the same tests.

| Arm | Trained tokens (median) | Loss drop (median) | Variant: L - frozen | Same problem: L - frozen | Unrelated: L - frozen |
|---|---|---|---|---|---|
| N | 535 | 40.2% | +0.016 [-0.024, +0.056] | -0.028 [-0.087, +0.030] | -0.043 [-0.110, +0.022] |
| H | 761 | 41.7% | +0.021 [-0.025, +0.066] | -0.058 [-0.113, -0.005] | -0.030 [-0.095, +0.033] |
| T | 570 | 37.6% | -0.006 [-0.048, +0.036] | -0.060 [-0.125, +0.005] | +0.022 [-0.030, +0.072] |

Drift check (P4): on an unrelated problem, T - N = +0.065 [-0.005, +0.133], fail (rule: within +/-0.03). A shift here means training changed the student in general, not what it practised.

Does practice volume matter? Spearman correlation across all dialogs between trained tokens and the variant gain over frozen: -0.03.

Learning-rate calibration (median loss drop on N and H): 0.03: 7.0%, 0.1: 42.5%, 0.3: -2612.6%, 1.0: -5486.4%, 3.0: -18023.8%. GPU time: train 22 min, sample 33 min.

Cost of these views: $1.16.

## Dialog health

| Arm | Dialogs | Tutor turns | Handover | Blanked turns | Failed student calls | Unboxed answers | End reasons |
|---|---|---|---|---|---|---|---|
| N | 50 | 5.70 | 0% | 0.0% | 0 of 3200 | 144 | {'end_of_conversation': 50} |
| H | 50 | 5.70 | 0% | 0.0% | 0 of 3200 | 119 | {'end_of_conversation': 39, 'max_teacher_turns': 11} |
| T | 50 | 5.60 | 82% | 0.0% | 0 of 3200 | 144 | {'end_of_conversation': 41, 'max_teacher_turns': 9} |

## Cost: $6.54

| Model | Calls | Input tokens | Output tokens | USD |
|---|---|---|---|---|
| anthropic/claude-haiku-5.5 | 565 | 1,320,251 | 350,725 | 0.31 |
| crutch/null-tutor | 285 | 0 | 0 | 0.00 |
| meta-llama/llama-3.1-8b-instruct | 10,300 | 15,127,263 | 6,655,403 | 4.79 |
| qwen/qwen3.6-27b | 1,200 | 2,752,496 | 430,155 | 1.44 |

Projected 300-problem audit: 15.3 A100-hours at 2,500 tokens/s, or about $97 of student calls on OpenRouter.

Served by: anthropic/claude-haiku-5.5 via Anthropic; anthropic/claude-haiku-5.5 via Azure; anthropic/claude-haiku-5.5 via Claude Platform on AWS; crutch/null-tutor via local; meta-llama/llama-3.1-8b-instruct via CoreWeave; qwen/qwen3.6-27b via Chutes
