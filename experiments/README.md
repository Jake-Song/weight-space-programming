# Weight editing sandbox and runner

This harness supports **new weight-editing experiments**, independently of the
previous published-edit ablation. An editor works in a persistent CPU Python
kernel; a trusted host controller validates requests, snapshots edits and spends
forward-pass budget on a separate GPU worker. No editor code runs in the host or
GPU worker. Use a separate run directory for each independent 10/25/50-pass run.

The source harness was not released. This implementation follows the public
[Experiment 8 setup](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md)
and [initial prompt](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/turns/t001/prompt.txt).
It is a tool runner: an editor (human or LLM) chooses each code block and forward
request. It does not automatically invoke an LLM or select the published edit.

## Setup

Requires Linux, Python 3.10+, Docker, NVIDIA Container Toolkit and an NVIDIA GPU
for forward passes. The prior pilot used an A100 40 GB. The editor has a 96 GiB
memory cap and 2 CPUs; its kernel initialization and code execute without a GPU.
Building images and preparing assets need internet; editing/inference use no
network. GPU workers run sequentially in fresh containers/subprocesses.

From the repository root:

```sh
python3 experiments/runner.py build
python3 experiments/runner.py init --budget 50
python3 experiments/runner.py prepare
python3 experiments/runner.py start
python3 experiments/runner.py python experiments/diagnose.py --explain 'Inspect the two MLPs before choosing a hypothesis'
python3 experiments/runner.py forward experiments/forward.example.json
python3 experiments/runner.py status
```

Default run location is `.sandbox/` (ignored by Git). For another independent run,
put `--root /absolute/path/to/run` **before** the subcommand on every command.
Assets include the pinned original model revision and the exact published
practice files: 7,469 GSM8K, 7,275 MATH and 714 AIME examples. Preparation verifies
counts, unique IDs and exact normalized overlap with the two tests. SHA256s of
all downloaded files and test dataset fingerprints are recorded; subsequent
workers reject changed assets. Benchmark datasets are snapshotted at preparation
time, not a claimed historical dataset revision.

The Python kernel provides `NAMES`, `load_original(name)`, `load(name)`,
`save_edit(name, tensor)`, `drop_edit(name)`, `edited()`, `tok`, `np`, `torch`.
Variables persist until `stop`, timeout, crash or OOM. `/work` persists across
kernel restarts. Use `start` after `stop` to restart a kernel. Give every code
execution an explanation; code, explanation and output are archived outside the
editor's writable mount. Each code execution costs one of 300 attempts, including
failed code blocks. A code call exceeding 900 seconds stops the kernel.

Use only weight analysis and justified direct edits. Do not train, enable
autograd, call backward/optimizers, fit weights to a loss or implement a model
forward in the editor. Autograd starts disabled. Docker enforces resource,
filesystem, network and GPU isolation; it cannot distinguish arbitrary numerical
analysis from a manually implemented forward or training loop. These semantic
rules require review of the archived code, as in the published experiment.
`save_edit` and the independent GPU validator reject non-block tensors (including
embedding, output and final norm), shape changes, non-finite tensors and
whole-tensor scaling. Edits are float32 `.npy` files and are cast to original
dtypes for evaluation; arbitrary editor Python is never loaded by the GPU worker.

## Forward calls

Choose IDs from `/practice/*.jsonl`. For generation:

```json
{"type":"generate", "model":"both", "problems":["REPLACE_WITH_PRACTICE_ID"], "n":1, "max_tokens":32768}
```

Models: `original`, `edited`, `both`. Optional `prompts` contains custom user
messages; these cannot be graded without a reference and are excluded from
accuracy. The cap is 512 total generations (256 per model with `both`). Paired
results use matching per-item seeds and report accuracy difference, standard
error over problems and correct/incorrect flips. Seed equality does not guarantee
identical random draws after sequences diverge. Full text and token IDs are saved
before grading with the source's last-box extraction and math-verify 0.9.0.

For teacher-forced analysis:

```json
{"type":"teacher_forced", "model":"edited", "texts":["Problem and reference solution..."], "return":{"token_logprobs":true,"hidden_states":[0,15,19,24],"mlp_act":[15,19]}}
```

Limits: 1,024 texts, 32,768 tokens per text, 524,288 total tokens, 6 GiB of output.
Arrays are per-text `.npy` files without padding. Hidden layer 0 is embeddings;
layer k is after block k-1. MLP activations are down-projection inputs (block IDs
0..23). Logprob entry j predicts input token j+1. Each successful request costs
one pass, regardless of size. Failed jobs remain for inspection and do not cost
a pass. Successful result markers recover accounting after controller restarts.
The host serializes mutations with a run lock. Returned artifacts are copied
into `/work/forward-*/`; trusted originals remain under `jobs/`.

## Submission and final evaluation

After exhausting the selected forward budget, make the final choice in
`/work/edited`, then:

```sh
python3 experiments/runner.py freeze
python3 experiments/runner.py evaluate
```

Freeze stops the editor and creates a submission snapshot and hash manifest.
Evaluation rejects changed submissions and runs original/edited models on the
**full** MATH-500 (4 samples per problem) and GSM8K test (2 samples per problem).
Thinking is on; temperature .6, top_p .95, top_k 20, presence penalty 1.5,
32,768 new tokens. Config/tokenizer/chat template are inherited unchanged.
Benchmark data and scores never mount into the editor. Results, scores, raw
outputs, weight audits, request, data hashes and package versions are archived in
`jobs/eval-*/`. Completed evaluations are skipped on retry; failed attempts are
preserved before rerunning. Final evaluation does not spend practice passes.

Current scope differences from the released experiment: no generation-time
activation capture, attention matrices, top-k teacher-forced alternatives or
chat-form teacher-forced requests; use raw texts for teacher forcing. The other
six final benchmarks and automatic Codex conversation/handoff orchestration are
not implemented. Unsupported fields are rejected instead of silently ignored.
This is the core two-benchmark weight-editing protocol, not a complete reproduction
of all published tables. The original editor's discovery is not guaranteed.

## Checks

```sh
python3 -m unittest discover -s tests -v
python3 tests/smoke_editor.py
```

The second command uses a real CPU Docker container with tiny synthetic weights.
It checks variable persistence, forbidden edits, read-only mounts, hidden test
data and absence of network/GPU access; it does not run the 0.8B model.

## Small headless diagnostic on Colab

When the host has no GPU, run a separate baseline pilot using an existing Colab
A100 session and the local authenticated Codex CLI:

```sh
colab new -s wsp-diagnostic --gpu A100
python3 experiments/diagnostic_colab.py --output .sandbox/diagnostics/astra-max-pilot
colab stop -s wsp-diagnostic
```

The planner and result interpreter both use `gpt-6-astra` with
`model_reasoning_effort=max`, noninteractively with structured JSON responses.
The planner sees only 32 sampled practice problems and chooses four GSM8K plus
four MATH items. The original model answers each once with the full 32,768-token
cap. No published edit or benchmark result is supplied to Astra. Shell tools,
web search, multi-agent mode and user-configured MCPs are disabled for these
editor calls. No editor-generated Python runs on Colab.

This is an independent infrastructure/diagnostic pilot; it does not spend any
passes from the local sandbox's 50-pass editing run. GPU inference runs through one-shot SSH as a
trusted native subprocess on Colab, rather than in Docker (Colab is the VM
boundary). The local Docker editor's isolation check remains a separate test.
The output folder records hypotheses, exact editor invocations, JSONL events,
requests, full responses/token IDs, repetition proxies, timing, scores, package
versions and Astra's interpretation. It does not automatically run the remaining
editing experiment. Stop the dedicated Colab session after retrieving results.
