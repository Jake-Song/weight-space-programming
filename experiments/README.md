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
`agent.py` driver invokes headless Astra(max), resumes the same Codex conversation for up to
40 turns, and hands off the task, full notebook, current edits and experiment
records to a fresh conversation every 40 turns or on driver restart. It does
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
editor's writable mount. Each code execution costs one attempt (default 300; configurable with
`init --python-budget N` or the run state `python_budget`), including
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
six final benchmarks are not implemented. The driver follows the 40-turn
conversation/handoff policy, but uses structured JSON actions with hypothesis
records instead of the original Markdown code-block interface. Explanations are
required from the first action rather than introduced partway through a run.
Infrastructure errors stop the driver with a resumable pending action; automatic
provider retries are not implemented. Unsupported fields are rejected explicitly.
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
Astra의 `stop`이나 예산 소진 전 `submit`은 실험을 종료하지 않는다.
명시적으로 일시 중단하려면 `runner.py pause`를 사용한다.

원실험처럼 최종 제출까지 계속 실행하려면 단계 제한을 생략한다:

```sh
python3 experiments/runner.py resume  # 일시정지 상태인 기존 실험
python3 experiments/agent.py
python3 experiments/runner.py evaluate
```

실행기 한 번의 실행 안에서는 같은 Codex 대화를 40턴까지 이어간다.
매 턴에는 직전 실행 결과와 남은 예산을 전달한다. 40턴 뒤 또는 실행기를
재시작하면 과제, 전체 노트, 현재 편집 목록과 실험 기록으로 새 대화를 시작한다.
`--steps N`은 운영상 선택할 수 있는 행동 수 제한이며, 예산을 초기화하지 않는다.
모든 forward 예산을 사용한 뒤에도 최종 편집 선택을 위한 Python 실행이 가능하다.
`submit`은 제출본과 해시를 고정하고 `final_answer.md`를 저장한다.
최종 벤치마크 평가는 별도 명령으로 실행하며 편집자에게 반환하지 않는다.

이 변경은 실행 루프를 맞춘 것이다. 기존 실행의 측정 기록은 유지되므로,
이미 진행한 실험을 이어가는 것은 처음부터 원설정으로 수행한 독립 재현이 아니다.
현재 파일의 `editor_memory` 같은 실행별 자원 설정도 그대로 적용된다.
원설정(96 GiB CPU 메모리, Python 300회, forward 10/25/50회)으로 새로
비교하려면 별도 `--root`에 `init`하고 자원을 준비한다.

자동 실행기는 기본적으로 Docker GPU backend를 사용한다. 로컬 GPU가 부족하면
CPU 편집기는 로컬 Docker에 유지하고 GPU 작업만 전용 Colab 세션으로 보낼 수 있다.
Colab에서 실행되는 코드는 신뢰된 작업기뿐이며, 편집기가 생성한 Python은
로컬 CPU 컨테이너에서만 실행한다. Colab GPU 작업의 격리 경계는 VM이다.

```sh
colab new -s wsp-ten-steps --gpu A100
docker build -f experiments/Dockerfile.editor -t wsp-editor:local .
python3 experiments/runner.py init --budget 10
python3 -m experiments.colab_backend setup --root .sandbox --session wsp-ten-steps
python3 experiments/agent.py --steps 10
python3 experiments/runner.py pause --note '10단계 실행 기록 보관'
colab stop -s wsp-ten-steps
colab sessions
```

`init`은 새 실험에만 사용한다. 각 GPU 작업의 원본 결과와 로그는 로컬 `jobs/`에
내려받고 해시 검증 후 예산에 반영한다. 세션 생성과 종료는 자동 실행기 밖에서
관리하며, 실패하거나 중단한 경우에도 결과를 회수한 뒤 해당 세션을 종료한다.
Colab 소규모 진단의 결과는 별도로 가져올 수 있다.

Colab VM이 사라졌다면 전용 세션을 새로 만든 뒤 같은 `--root`로 위의
`colab_backend setup`을 다시 실행한다. 로컬에 남은 원래 데이터와 manifest를
복원하고 누락된 고정 revision 가중치만 내려받아 검증한다. `init`은 반복하지
않는다. `agent.py --steps N`은 미완료 작업부터 이어가므로, 총 10단계가 목표라면
이미 완료한 단계 수를 빼고 N을 지정한다. 전송이 실패한 GPU 작업은 성공 결과가
복구될 때까지 forward 예산에 포함되지 않는다.

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


### 3-pass infrastructure smoke

`init --budget 3 --smoke` creates an independent smoke run (3 is rejected without
`--smoke`). For example, use `--root .sandbox/smoke-three-20261117` on each
runner/agent invocation. The editor is instructed to use at most two practice
items, 512 generated tokens, a short teacher-forced inspection, and a paired
comparison before submitting after three successful calls. These small request
sizes are prompt instructions, not additional validator limits. Smoke scores do
not measure benchmark performance. A local 5 GiB editor memory override is a
resource deviation from the original 96 GiB configuration.

The runner also accepts `init --budget 20` for a user-requested independent
20-pass run. This uses the normal protocol (`smoke=false`); 20 passes is an
additional budget condition, not one of the original 10/25/50 runs.

### 한국어 사실 갱신 실험 (Colab CLI)

로컬 CPU 편집기와 Colab GPU를 함께 사용한다. 전용 실행 디렉터리에 원본
Qwen3.5-0.8B를 준비하면 연습 문항만 편집기에 마운트된다. 수학용 편집본은
불러오지 않는다. 새 실행을 만들고 자동 편집기를 실행하는 예시는 다음과 같다.

```sh
docker build -f experiments/Dockerfile.editor -t wsp-editor:local .
python3 experiments/runner.py --root .sandbox/fact-ko init --task fact_update_ko
colab new -s wsp-fact-ko --gpu L4
python3 -m experiments.colab_backend setup --root .sandbox/fact-ko --session wsp-fact-ko
python3 experiments/agent.py --root .sandbox/fact-ko
python3 experiments/runner.py --root .sandbox/fact-ko status
```

연습 120쌍에서 변경 적용과 유지 문항을 모두 맞힌 비율을 측정한다. 원본이
95% 이상이면 첫 패스에서 종료한다. 그 외에는 활성값 진단 3회, 편집 후보
채점 12회를 수행한다. `{"type":"score_choices","model":"edited"}` 요청은
탐색 세트 전체의 thinking-off 응답 시작점에서 A/B 토큰 로그확률을 비교한다.
동점은 오답이다. 개별 기록은 작업별 `output/*_choices.json`에 저장한다.

탐색 직후 호스트가 상위 3개 스냅샷을 잠그고 선택 세트 80쌍에서 원본과
각 후보를 한 번씩 채점한다. 동률이면 제출한 가중치 변화의 L2 노름이 작은
후보를 택한다. 수동 실행에서는 16번째 패스 뒤 `freeze`가 이 단계를 수행한다.
중단 후 같은 디렉터리에서 재실행하면 봉인한 결과를 재사용한다.

`status`에 `frozen: true`가 기록되면 다음 명령으로 최종 평가한다.

```sh
python3 experiments/runner.py --root .sandbox/fact-ko evaluate
```

`fact-report.json`에는 세 분야의 합계 240쌍, 보존 대조 120문항에 대한
원본·선택 편집본·무작위 방향 5개의 점수가 담긴다. 각 대조는 변경 위치와
열별 변화량 노름을 원본 정밀도 변환 전에 일치시킨다. 난수 시드는 0–4다.
성공 기준은 쌍 정확도 5%p 이상 개선, 분야별 문장 틀을 묶은 paired bootstrap
95% 구간 하한이 0 초과, 보존 정확도 하락 3%p 이하다. 원본이 탐색 종료 기준을
넘거나 최종 보고서를 로컬에서 확인한 뒤 `colab stop -s wsp-fact-ko`로 세션을 종료한다.


영어 독립 편집 실험은 같은 명령에서 `--task fact_update_en`을 사용하고 별도
실행 경로와 Colab 세션을 지정한다. 탐색·선택·최종 세트의 언어만 바꾸고
각 사례의 ID, 상태, 정답, 보기와 기록 필드 순서, 패스 예산, 채점 방법을
유지한다. 편집기는 원본 모델과 영어 탐색 세트에서 시작한다.

### 한국어 폐쇄형 지식 주입 실험

`knowledge_write.py`는 가상 회사 4곳에 같은 등록 코드 4개를 서로 다르게
배정한 두 편집본을 만든다. 편집자는 배정표를 읽으며, 모델이 받는 질문에는
회사명만 들어간다. 기존 Qwen3.5-0.8B의 마지막 MLP 출력 가중치 한 개에 회사별
활성 방향과 코드 토큰의 출력 방향의 외적을 더한다. 최적화기와 역전파는 쓰지
않는다. 편집 강도는 회사당 보정용 질문 2개에서 선택한 뒤 고정한다.

```sh
python3 experiments/knowledge_write.py /path/to/new/results
```

transformers 5.20.2와 CUDA가 필요하다. 결과의 `both_maps_correct`는 서로
다른 두 편집본이 동일한 새 질문에 각자 배정된 코드를 모두 답한 횟수다
(회사 4곳 × 새 표현 4개). 디코딩은 thinking-off, greedy, 최대 16토큰이며
응답 전체에서 앞뒤 공백과 대소문자를 정규화해 채점한다. 근접 회사 16문항과
사실·속성 대조 12문항, 두 개의 노름 일치 무작위 편집을 함께 보고한다.
`protocol.json`, `selection.json`, `factors.safetensors`와 편집본별 응답이
기록되어 같은 원본 가중치에 편집을 재현할 수 있다.

후속 실험은 첫 실행의 결과 디렉터리를 입력으로 사용한다. 시작할 때 두 편집본의
가중치 해시를 검증한다. 서로 다른 두 출력 디렉터리로 다음과 같이 실행한다.

```sh
python3 experiments/knowledge_write_ten.py /path/to/first/results /path/to/new/ten-pass-results
```

추가 예산 10회(활성 진단 2회, 후보 비교 6회, 최종 평가 2회)를 사용한 실행에서
두 편집본이 각각 배정된 코드를 모두 맞힌 횟수는 기존 질문 10/16에서 16/16으로
올랐다. 새 질문은 13/16이었다. 다른 회사에 주입 코드가 나온 횟수는 α에서
5/16→9/16, β에서 5/16→6/16이었다. 토큰 생성까지 센 모델 호출은 73회였다.
