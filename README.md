# weight-space-programming

Weight Space Programming (WSP).

Reproduce the published Qwen3.5-0.8B edit, then compare layer 15, layer 19,
and both layers on disjoint, deterministic MATH-500 subsets.

Sources:
- https://rikiyat.github.io/blog/weight-space-programming/
- https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math

## Protocol

- Base revision: `2fc06364715b967f1860aea9cf38778875588b17`.
- Apply the published p50c diff and require original and edited tensor SHA256
  verification before generating any results.
- Split seed 20261116: 20 smoke problems, then 100 distinct held-out problems.
- Conditions: base, layer15, layer19, both, and base with thinking disabled.
- One sample per problem, seed 0. This is an exploratory ablation, not the
  original full MATH-500 evaluation with four samples per problem.
- All arms: temperature .6, top_p .95, top_k 20, presence penalty 1.5,
  32768 generated tokens. The no-thinking arm changes only the chat template;
  it is not a reproduction of the model card's separately tuned no-thinking profile.
- Original last-boxed-answer extraction and math-verify grading.
- Record accuracy, length termination rate, mean token count, and a distinct
  repetition proxy: more than 50% duplicate token 20-grams. This proxy is
  descriptive and does not establish a semantic reasoning loop.
- Raw outputs and token IDs are persisted before grading. All conditions use
  identical problem IDs. No edits are selected or tuned on the held-out split.

## Execution on Colab

Dedicated session: `weight-space`, A100. Existing sessions are left alone.

```sh
colab exec -s weight-space -f scripts/setup_colab.py --timeout 900
colab upload -s weight-space scripts/prepare.py /content/prepare_wsp.py
colab upload -s weight-space scripts/evaluate.py /content/evaluate_wsp.py
colab upload -s weight-space scripts/run_suite.py /content/run_wsp_suite.py
colab upload -s weight-space scripts/random_control.py /content/random_wsp_control.py
colab upload -s weight-space scripts/run_control.py /content/run_wsp_control.py
colab exec -s weight-space -f scripts/launch.py --timeout 120
colab exec -s weight-space -f scripts/launch_control.py --timeout 120
colab exec -s weight-space -f scripts/status.py --timeout 120
```

Run each condition in a fresh Python subprocess to release engine memory.
Artifacts live under `/content/weight-space/results/` and should be downloaded
before terminating the session. Scripts retain full generations for auditing.

## Status

Colab A100 pilot completed (2026-11-16), 20 problems, one sample each:

| Condition | Correct | Token-limit rate | Mean generated tokens |
|---|---:|---:|---:|
| Original, thinking on | 2/20 (10%) | 90% | 30,344.2 |
| Published edit, both layers | 13/20 (65%) | 20% | 20,203.2 |

Paired changes: 11 incorrect-to-correct, 0 correct-to-incorrect. These are new
pilot measurements, not the source's full benchmark numbers.

The five main arms and one random control on 100 separate problems are complete:

| Condition | Accuracy | Token-limit rate | Repetition proxy rate | Mean tokens |
|---|---:|---:|---:|---:|
| Original, thinking on | 8% | 92% | 88% | 30,587 |
| Layer 15 only | 41% | 42% | 22% | 22,563 |
| Layer 19 only | 31% | 66% | 62% | 26,209 |
| Both layers | 57% | 26% | 7% | 20,638 |
| Original, thinking off | 51% | 12% | 5% | 8,245 |
| Random direction, same columns and matched norms | 11% | 89% | 86% | 29,672 |

Both vs layer 15: 18 newly correct, 2 newly incorrect; +16 percentage points,
paired bootstrap exploratory 95% interval [+8, +24]. Both vs original thinking
on: +49 points, interval [+39, +59]. These intervals resample problems and do
not quantify generation-seed variation. The accuracy-scale interaction
`both - layer15 - layer19 + original` is -7 points, interval [-19, +5]; a
super-additive interaction is not established.

The published edit improves thinking-mode completion and accuracy. Turning
thinking off is also a strong practical baseline: 51% with approximately 40%
of the edited model's tokens. The experiment does not establish new reasoning
capabilities or an editor LLM's ability to independently discover the edit.

Both vs thinking off: +6 points, interval [-3, +15]; this sample does not
establish an accuracy advantage over thinking off. Both uses 2.50 times as many
generated tokens. The random control remains close to the original; one random
direction draw is not a general test of all possible perturbations.

All six conditions are complete. Full generations, token IDs, scores,
splits and weight audits are under `results/`; `scripts/summarize.py` recomputes
the summary and paired bootstrap comparisons.

Retrieve and verify artifacts before releasing the Colab VM:

```sh
colab exec -s weight-space -f scripts/provenance.py --timeout 120
colab exec -s weight-space -f scripts/collect.py --timeout 120
colab download -s weight-space /content/weight-space-results.tar.gz results/colab-results.tar.gz
tar -xzf results/colab-results.tar.gz -C results
python3 scripts/verify_results.py
python3 scripts/summarize.py
colab stop -s weight-space
```

Sources are pinned to `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.
The downloaded grading and diff files were byte-checked against that commit.
Package versions and source hashes are in `results/provenance.json`.

When the held-out edit improves accuracy or cut rate, `run_control.py` runs one
additional exploratory random-direction control. It uses the same edited
columns and matches per-column update norms before BF16 rounding, with a random
rank-one direction and random signs (seed 20261116). Actual update norms are
recorded. One random seed is a limited control, not a distributional comparison.

## New weight-editing experiments

A persistent CPU sandbox and budgeted GPU experiment runner are in
[`experiments/`](experiments/README.md). They support new direct weight edits,
10/25/50 practice forward passes and final full MATH-500 (4 samples) / GSM8K
(2 samples) evaluation. See that guide for setup, commands, isolation checks and
the differences from the complete published harness.
