# PILOT report: pilot50

**Decision:** NO-GO for now (failing: 300-problem audit <= 12 A100-hours). Fix it for up to two days, then try the Qwen3-8B student.

**Framing:** The masked student does not pay for copied answers (M: T-N >= -0.02), the gap the project set out to test.

## Go/no-go gates (proposal, Risks section)

| Gate | Value | Pass |
|---|---|---|
| practice: P(T) - P(N) >= +0.20 | +0.520 [+0.415, +0.619] | yes |
| judge: handover T >= 80% | 98% | yes |
| judge: handover H <= 20% | 6% | yes |
| pre-test within 0.05 of the dataset | -0.016 (measured 0.302 vs stored 0.319) | yes |
| 300-problem audit <= 12 A100-hours | 12.9 h (574 tokens per answer) | NO |

## Post-test solve rate by view (paired by problem vs N; 95% bootstrap CI)

| View | N | H | T | H - N | T - N |
|---|---|---|---|---|---|
| P | 0.304 | 0.739 | 0.824 | +0.435 [+0.316, +0.552] | +0.520 [+0.415, +0.619] |
| U | 0.269 | 0.511 | 0.544 | +0.242 [+0.145, +0.340] | +0.275 [+0.179, +0.372] |
| M | 0.264 | 0.414 | 0.433 | +0.150 [+0.056, +0.249] | +0.169 [+0.070, +0.269] |
| Ms | 0.270 | 0.578 | 0.496 | +0.308 [+0.198, +0.415] | +0.226 [+0.106, +0.343] |

Paired SD of T - N per problem under M: 0.373 (the power analysis assumed 0.18).

## Did the answer tutor set new problems?

Dialogs where the tutor set a new practice problem: N 0, H 2, T 18 (pattern match on tutor turns).

| T dialogs | M: T - N | U: T - N |
|---|---|---|
| T posed a new problem (n=18) | +0.275 [+0.098, +0.454] | +0.355 [+0.175, +0.521] |
| T did not (n=32) | +0.110 [-0.009, +0.233] | +0.230 [+0.123, +0.343] |

## More simulated students on the same dialogs (views.py)

Each view changes only what the student carries into its test on the new variant. Answers are capped at 2,048 tokens and all logged. Human target from Bastani's exam: T - N about -0.05 (GPT Base -17%), H - N about 0 (GPT Tutor).

| View | Carries | N | H | T | H - N | T - N | Test down (P2) | Hints safe (P3) |
|---|---|---|---|---|---|---|---|---|
| M | student turns; tutor turns hidden | 0.319 | 0.439 | 0.428 | +0.120 [+0.037, +0.207] | +0.110 [+0.021, +0.205] | fail | pass |
| C | M minus sentences copied from the tutor | 0.307 | 0.460 | 0.431 | +0.152 [+0.058, +0.245] | +0.124 [+0.039, +0.210] | fail | pass |
| K | one note of at most 120 tokens | 0.330 | 0.365 | 0.385 | +0.034 [-0.040, +0.109] | +0.053 [-0.017, +0.124] | fail | fail |

P2 passes when T - N <= -0.03 with its interval below 0; P3 when the lower end of H - N is above -0.03.

| View | T - N where T set a new problem | T - N where it did not |
|---|---|---|
| M | +0.210 [+0.057, +0.374] (n=18) | +0.053 [-0.054, +0.166] (n=32) |
| C | +0.188 [+0.069, +0.306] (n=18) | +0.088 [-0.027, +0.203] (n=32) |
| K | +0.037 [-0.053, +0.132] (n=18) | +0.062 [-0.037, +0.157] (n=31) |

C removed N 0/1424 (0%), H 375/2406 (16%), T 297/2406 (12%) of student sentences as copied from the tutor.

K notes: N 73 words, 4% contain the original answer, H 78 words, 4% contain the original answer, T 76 words, 6% contain the original answer.

| View | Arm | Answers | Cut off at the cap | No boxed answer | Correct among boxed |
|---|---|---|---|---|---|
| M | N | 796 | 10.9% | 11.1% | 0.357 |
| M | H | 800 | 8.0% | 8.2% | 0.478 |
| M | T | 799 | 7.0% | 7.3% | 0.462 |
| C | N | 400 | 11.0% | 10.8% | 0.345 |
| C | H | 400 | 8.0% | 8.2% | 0.501 |
| C | T | 399 | 7.5% | 7.8% | 0.467 |
| K | N | 780 | 11.0% | 11.3% | 0.371 |
| K | H | 799 | 10.1% | 10.1% | 0.407 |
| K | T | 800 | 11.8% | 11.8% | 0.436 |

Cost of these views: $2.17.

## Dialog health

| Arm | Dialogs | Tutor turns | Handover | Blanked turns | Failed student calls | Unboxed answers | End reasons |
|---|---|---|---|---|---|---|---|
| N | 50 | 5.74 | 0% | 0.0% | 3 of 3200 | 337 | {'end_of_conversation': 50} |
| H | 50 | 5.74 | 6% | 0.0% | 1 of 3200 | 206 | {'end_of_conversation': 41, 'max_teacher_turns': 9} |
| T | 50 | 6.48 | 98% | 0.0% | 4 of 3200 | 227 | {'end_of_conversation': 33, 'max_teacher_turns': 17} |

## Cost: $6.82

| Model | Calls | Input tokens | Output tokens | USD |
|---|---|---|---|---|
| anthropic/claude-haiku-5.5 | 611 | 1,483,107 | 384,853 | 0.34 |
| crutch/null-tutor | 287 | 0 | 0 | 0.00 |
| meta-llama/llama-3.1-8b-instruct | 11,939 | 16,626,043 | 6,562,626 | 5.10 |
| qwen/qwen3.6-27b | 1,200 | 2,957,472 | 408,425 | 1.38 |

Projected 300-problem audit: 12.9 A100-hours at 2,500 tokens/s, or about $98 of student calls on OpenRouter.

Served by: anthropic/claude-haiku-5.5 via Claude Platform on AWS; crutch/null-tutor via local; meta-llama/llama-3.1-8b-instruct via CoreWeave; qwen/qwen3.6-27b via Chutes
