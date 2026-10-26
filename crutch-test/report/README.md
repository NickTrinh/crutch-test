# Write-up

The Crutch Test write-up, in an ICLR-style template: Introduction, Related work, Methodology, Experiments, Conclusion.

| File | What |
|---|---|
| `main.tex` | The report |
| `iclr_bib.bib` | 26 references, generated from arXiv API and Crossref records |
| `figures/effects.{pdf,png}` | T − N and H − N for every simulator against the human trial, with 95% CIs |
| `figures/workflow.{pdf,png}` | The protocol, from problem to gates |
| `make_figures.py` | Rebuilds both figures from `results/pilot50v2.json` (`--extra-l` adds the lr 0.03 row) |

`main.tex` compiles with the template's `iclr_style.sty`, `iclr_bst.bst` and `math_commands.tex`, which are not included here.

## Regenerate the figures

The committed figures include the lr 0.03 row:

```bash
python crutch-test/report/make_figures.py --extra-l crutch-test/results/pilot50v2_lr0.03.json
```
