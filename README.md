# The Crutch Test

Tutoring models built on large language models (LLMs) are increasingly trained with reinforcement learning against simulated students. In a randomized trial with about a thousand high-school students, unrestricted GPT-4 raised practice scores but lowered scores on a later exam taken without AI by 0.054 on a 0–1 scale, or 17% ([Bastani et al., PNAS 2025](https://doi.org/10.1073/pnas.2422633122)). A tutor that gave only hints did not lower them. We call this loss the crutch cost. If simulated students do not pay it, rewards built on them favor tutors that simply hand over answers.

The Crutch Test runs a student simulator through the trial's three conditions: no AI, hints, and answers. The simulated student then takes a test on a new variant of each problem, without help. Each condition is compared with no AI, problem by problem, and the differences are checked against pass gates set from the trial. The write-up is in [`crutch-test/report/`](crutch-test/report/).

**Demo:** the [dialog explorer](https://nicktrinh.github.io/crutch-test/) shows each pilot problem's three dialogs side by side, what each simulated student carried into the test, and its scores.

![Score on the new variant, answers minus no AI and hints minus no AI, for the human trial and each simulator](crutch-test/report/figures/effects.png)

## Pilot results

The pilot used 50 Big-Math problems, one dialog per condition, a Llama-3.1-8B-Instruct student, and Claude Haiku 5.5 tutors. Each cell is the difference from no AI in the score on the new variant, with a 95% interval.

| Student simulator | Answers minus no AI | Hints minus no AI |
|---|---|---|
| Human trial (Bastani et al.) | -0.054 [-0.097, -0.011] | -0.004 [-0.029, +0.021] |
| M: own turns, tutor hidden | +0.122 [+0.036, +0.209] | +0.133 [+0.041, +0.225] |
| C: M with copied text removed | +0.145 [+0.052, +0.233] | +0.168 [+0.065, +0.270] |
| K: one short note | +0.060 [+0.003, +0.120] | +0.018 [-0.041, +0.079] |
| L: learns in its weights, lr 0.1 | -0.022 [-0.074, +0.028] | +0.005 [-0.039, +0.050] |
| L: learns in its weights, lr 0.03 | -0.003 [-0.040, +0.036] | -0.035 [-0.071, +0.000] |

The three frozen simulators (M, C and K) gain from being told the answer, where students lost. The learning student (L) removes that gain at both learning rates. It does not show the human loss, and no setting passes every gate at this scale. Practice still rises as in the trial: with the dialog in view, answers raise the same-problem score by +0.574.

## Reproduce

These steps run on Linux, macOS or WSL and need no GPU and no API key. Setup builds a virtual environment and downloads EduardoRL, the dataset and the trial's log. It takes about 2 minutes. Where `python3 -m venv` is missing, the setup step in [`crutch-test/README.md`](crutch-test/README.md) shows how to use uv instead.

```bash
crutch-test/setup.sh
```

Next, rebuild the pilot report from the released logs. Grading adds to the run's answer log, so unpack a fresh copy for each report:

```bash
mkdir -p crutch-test/runs && tar -xzf crutch-test/results/pilot50v2_raw.tar.gz -C crutch-test/runs
crutch-test/.venv/bin/python crutch-test/learner_grade.py crutch-test/.deps/EduardoRL crutch-test/runs/pilot50v2 --outputs crutch-test/results/learner_outputs_pilot50v2.json.gz
crutch-test/.venv/bin/python crutch-test/analyze.py crutch-test/runs/pilot50v2 --report crutch-test/runs/pilot50v2.md
```

The numbers in `crutch-test/runs/pilot50v2.json` match `crutch-test/results/pilot50v2.json`. With `learner_outputs_pilot50v2_lr0.03.json.gz` instead, they match `pilot50v2_lr0.03.json`.

To run the whole pipeline for $0, point it at a local stand-in server. It takes about 30 seconds, and its scores are meaningless:

```bash
CRUTCH_MOCK=1 N_PROBLEMS=5 crutch-test/pilot.sh dry
```

New runs need an OpenRouter key. The pilot cost $6.54, and the learning student also needs a Modal account, at about $3–5. [`crutch-test/README.md`](crutch-test/README.md) has the steps, the protocol details, and the 300-problem held-out run.

## Data and prompts

- **Problems.** The problems come from the EduardoRL near-transfer dataset ([dmacjam/eduardoRL-transfer-dataset](https://huggingface.co/datasets/dmacjam/eduardoRL-transfer-dataset)), which pairs Big-Math problems with model-written variants. `setup.sh` downloads it.
- **Tutor prompts.** The tutors use the trial's GPT Base and GPT Tutor prompts. Their repository, [obastani/GenAICanHarmLearning](https://github.com/obastani/GenAICanHarmLearning), has no license, so this repository does not copy them. Instead, `setup.sh` downloads the trial's log, and `make_arms.py` cuts the prompts out of it and checks them by hash. In the released logs, a marker replaces the prompt text.
- **Models.** The student is Llama-3.1-8B-Instruct in bf16. It runs through OpenRouter on CoreWeave, and on Modal for the learning student. The tutor is Claude Haiku 5.5, and the judge is Qwen3.6-27B.
- **Environment.** `setup.sh` clones [EduardoRL](https://github.com/eth-lre/EduardoRL), licensed CC BY-SA 4.0, and patches it at runtime. This repository does not copy it.

## Citation

```bibtex
@misc{trinh2026crutch,
  title  = {The Crutch Test: Auditing the Cognitive Cost of Direct Answers in Simulated Student Environments},
  author = {{Trinh Van Minh Nhat}},
  year   = {2026},
  note   = {GStar Bootcamp 2026, New Turing Institute},
  url    = {https://github.com/NickTrinh/crutch-test}
}
```

## License

The code is under the MIT license ([LICENSE](LICENSE)). The write-up, figures and results (`crutch-test/report/` and `crutch-test/results/`) are under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party content inside the results keeps its own terms. This covers problem text from the EduardoRL dataset and its Big-Math sources (Apache-2.0), and model outputs under their providers' terms.
