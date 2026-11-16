# Weight editing sandbox and runner

This harness supports **new weight-editing experiments**, independently of the
previous published-edit ablation. An editor works in a persistent CPU Python
kernel; a trusted host controller validates requests, snapshots edits and spends
forward-pass budget on a separate GPU worker. No editor code runs in the host or
GPU worker. Use a separate run directory for each independent 10/25/50-pass run.

The source harness was not released. This implementation follows the public
[Experiment 8 setup](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md)
and [initial prompt](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/turns/t001/prompt.txt).
The core runner accepts explicit code blocks and forward requests. The optional
`agent.py` driver invokes headless Astra(max) for a bounded number of actions and
passes persistent hypotheses and results to each subsequent invocation. It does
not select or preload the published edit.

## Setup

Requires Linux, Python 3.12+, Docker, NVIDIA Container Toolkit and an NVIDIA GPU
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
six final benchmarks and the original harness's 40-turn conversation policy are
not implemented. The bounded `agent.py` driver uses fresh noninteractive turns
with persistent experiment context instead. Unsupported fields are rejected instead of silently ignored.
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


## 중단한 실험 이어가기

동일한 `--root`가 동일한 실험이다. 기본값은 `.sandbox/`이며, 중단했다가
다시 실행할 때 `init`으로 새 실험을 만들지 않는다. 사용한 예산은 완료된
GPU 작업에서 복구하며, 체크포인트 복원이나 새 대화로 돌아가지 않는다.

### 이전 결과를 읽고 몇 단계만 진행

모델·데이터를 준비하고 GPU 실행 환경을 갖춘 뒤 사용한다. 코드 변경 후에는
`python3 experiments/runner.py build`로 CPU/GPU 이미지를 갱신한다.

```sh
# 현재 예산·실험 상태, 재개 문서 확인
python3 experiments/runner.py status
python3 experiments/runner.py handoff

# 비대화형 gpt-6-astra / max로 최대 3개 작업만 수행
python3 experiments/agent.py --steps 3

# 다음 호출에서 같은 실험의 가설·결과·메모를 읽고 추가 2개 작업 수행
python3 experiments/agent.py --steps 2
```

여기서 단계는 Python 분석, GPU forward, 가설·메모 정리, 중단 중 하나다.
3단계가 GPU 호출 3회를 뜻하지는 않는다. 실제 forward 예산과 Python 예산은
각각 별도로 관리한다. Astra는 호스트 도구를 직접 사용하지 않으며, 반환한
코드는 CPU sandbox에서만, forward JSON은 신뢰된 GPU 작업기에서만 실행한다.
Astra가 `stop`을 선택하면 실험을 일시 중단한다. 이 경우 `resume` 후 계속한다.

자동 실행기는 현재 기본 Docker GPU backend를 사용한다. NVIDIA GPU와
NVIDIA Container Toolkit이 있는 실행 머신이 필요하다. Colab 소규모 진단의
결과는 가져올 수 있지만, `agent.py`가 Colab 세션을 자동 생성하거나 기존
진단 스크립트를 50회 편집 실행기로 바꾸지는 않는다.

### 멈추고 나중에 다시 시작

```sh
python3 experiments/runner.py pause --note '다음에는 출력 투영 행렬의 특이값 분포를 비교한다'

# 같은 머신, 새 터미널 또는 새 대화에서
python3 experiments/runner.py resume --start
python3 experiments/agent.py --steps 1
```

`pause`는 CPU 컨테이너를 종료하고 현재 편집본·노트·다음 할 일을 체크포인트에
저장한다. 이미 실행 중인 GPU 컨테이너를 취소하지는 않는다. `resume`은 결과
수집과 예산을 복구하고 `handoff.md`/`handoff.json`을 갱신한다. `--start`를
붙이면 CPU 커널도 시작한다. 실행 중인 정상 커널은 그대로 재사용한다.
Python 변수 자체는 직렬화하지 않는다. 코드는 `load()`/`load_original()`과
`/work`에 저장한 배열로 입력을 명시적으로 다시 불러와야 한다.

새 대화에서는 `.sandbox/handoff.md`를 읽고 이어가도록 지시하면 된다.
Astra 자동 실행기는 이와 같은 저장 기록을 매 호출에 전달한다. 가설·사용 예산·
완료 작업·이전 결과는 대화창의 기억에 의존하지 않는다.

### 이전 진단과 새 가설 연결

```sh
# 외부 연습 진단을 참고 자료로 가져오기: 본 예산에 포함되지 않음
python3 experiments/runner.py import-evidence results/diagnostics/astra_max_pilot_20261116

python3 experiments/runner.py hypothesis spectrum \
  --statement '일부 출력 투영 행렬에 에너지가 소수 방향으로 집중되어 있다' \
  --prediction '같은 shape의 다른 행렬보다 상위 특이값 에너지 비중이 크다'

# 직접 요청을 실행할 때도 가설 ID와 고정 작업 ID를 남길 수 있음
python3 experiments/runner.py forward request.json --id spectrum-01 --hypothesis spectrum

python3 experiments/runner.py hypothesis spectrum --status inconclusive \
  --conclusion '편중은 관찰했으나 반복과의 인과 관계는 아직 검증하지 않았다'
```

가설의 statement/prediction은 불변이며, 수정 가설은 새 ID와 `--parent`로
연결한다. 완료 결과는 해시로 봉인한다. 동일한 작업 ID를 다시 호출하면 기존
결과를 반환하며, 다른 요청 JSON을 같은 ID로 덮어쓸 수 없다. 재시도는 최초
요청 당시의 가중치 스냅샷을 사용한다. 외부 자료는 연습 진단만 허용하고,
최종 벤치마크 결과는 편집기 context에 넣지 않는다.

### 호출 도중 끊겼을 때

- 동일한 `forward --id ...` 또는 `agent.py --steps ...`를 다시 실행한다.
  저장된 결정과 요청을 재사용하므로 이미 완료된 작업을 다시 청구하지 않는다.
- Docker GPU 컨테이너가 계속 실행 중이면 그 작업이 끝나기를 기다려 결과를
  수집한다. 완료된 원본/편집 arm과 저장된 전체 생성 결과는 재사용한다.
  생성은 끝났지만 채점만 실패했다면 답변을 다시 생성하지 않는다.
- 아직 전체 응답을 저장하지 못한 생성은 토큰 중간부터 이어지지 않는다.
  해당 미완료 arm을 다시 실행한다. 성공하지 않은 전체 호출은 forward 예산에
  포함되지 않으며, 실패한 시도의 로그도 보존한다.
- Python 실행 결과를 호스트가 받기 전에 연결이 끊겨도 커널의 디스크 기록으로
  완료 여부를 확인한다. 코드가 중간에 중단되어 편집 여부가 불명확하면 자동
  재실행하지 않는다. 실행 전 체크포인트 복원 후 기존 에이전트 작업을 명시적으로
  포기하고 새 작업으로 진행한다.

```sh
# action의 실행 전 checkpoint ID는 agent/action-000001/state.json에 있음
python3 experiments/runner.py resume --checkpoint CHECKPOINT_ID --start
python3 experiments/agent.py --abandon action-000001 --reason '실행 결과 불명확; 실행 전 체크포인트로 복구함'
python3 experiments/agent.py --steps 1
```

수동 Python 호출도 `python FILE --explain '...' --id analysis-01`로 고정 ID를
사용할 수 있다. 같은 ID와 코드는 완료 결과를 재사용한다. 코드가 바뀌면 새 ID가
필요하다. 체크포인트를 복원해도 이미 쓴 Python/forward 예산이나 실행 ID는
되돌리지 않는다.

### 다른 머신으로 옮기거나 장기간 보관

```sh
python3 experiments/runner.py pause --note '다른 GPU 머신에서 이어서 실행'
python3 experiments/runner.py export /tmp/wsp-run.tar.gz

# 새 머신: root는 아직 존재하지 않는 경로여야 함
python3 experiments/runner.py --root /workspace/wsp-run restore /tmp/wsp-run.tar.gz
python3 experiments/runner.py --root /workspace/wsp-run build
python3 experiments/runner.py --root /workspace/wsp-run prepare
python3 experiments/runner.py --root /workspace/wsp-run resume --start
python3 experiments/agent.py --root /workspace/wsp-run --steps 2
```

기본 백업에는 가설·결정·노트·편집본·체크포인트·전체 결과·모델/데이터 manifest와
비가중치 assets가 들어간다. 원본 `.safetensors`는 용량 때문에 생략하고,
`prepare`가 고정된 revision에서 누락된 원본 가중치만 다시 다운로드한 뒤
해시를 확인한다. `export ... --include-weights`로 원본 가중치도 포함할 수 있다.
원본 생성 결과와 배열이 많으면 백업도 커진다. `.sandbox/`와 `results/`는 Git에서
제외되므로 실제 파일 보관·이동에는 백업이 필요하다.

진행 중인 GPU 작업은 같은 ID로 다시 접속해 결과를 회수한 뒤 내보낸다.
실행 중인 GPU 컨테이너가 확인되면 내보내기를 거부한다. 복원은 새 root에만
가능하고, 파일 해시와 archive 경로를 검증하며 기존 실험을 덮어쓰지 않는다.
실험 ID와 예산은 보존하지만 CPU 컨테이너 ID는 새로 만든다.

### Colab 진단 스크립트 자체 재개

`diagnostic_colab.py --output SAME_FOLDER`를 다시 실행하면 완료된 진단은 저장된
결과만 반환한다. GPU 결과를 이미 다운로드했고 보고서만 남았다면 Colab VM이
종료된 뒤에도 로컬에서 보고서를 마무리한다. 계획이 저장됐다면 문제를 다시
선택하지 않고, 미완료 원격 작업은 동일한 저장 경로를 재사용한다.

아직 받지 못한 GPU 결과가 있는 상태에서 Colab VM 자체가 삭제되면 그 VM의
임시 파일은 복구할 수 없다. 로컬에 내려받은 결과·편집본·백업은 보존되며,
사라진 미완료 계산만 새 실행 환경에서 다시 수행해야 한다.
