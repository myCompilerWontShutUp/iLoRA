# RUNBOOK: VESSL 실행 순서와 명령

재구현(`rebuild/`)의 실험을 VESSL workspace(Linux, A100 80GB 1장)에서 돌리는 순서와 정확한 명령이다. 근거는 `rebuild/DESIGN.md` 18.6.6(M14)이고, 시험·smoke 판정은 17장(M13), 실험 목록과 프로파일은 18.6.1–18.6.5에 있다. `rebuild/tests/test_m14_runbook.py`가 이 문서의 bash 블록을 실제 CLI(`rebuild/main.py`, `rebuild/tools/*.py`)와 설정 schema로 검사한다.

## 규칙

- 모든 명령은 저장소 루트(`~/iLoRA`)에서, conda env `ilora-rebuild`, `CUDA_VISIBLE_DEVICES=0`으로 돌린다 [SPEC §0.12]. GPU 작업은 한 번에 하나만.
- 각 단계는 "명령 → 산출물 → 확인"이다. 확인이 실패하면 다음 단계로 가지 않고 원인을 기록한다(그 실행 폴더의 `meta.json` `error`, `run.log`). 결과 수치는 측정한 뒤에만 적는다 [SPEC §0.11]. 시간·메모리·비용은 측정 전 TBD.
- 실행 폴더는 그것을 만든 명령 뒤에 `&&`로 이어 `latest <프로파일>`로 잡고, `done_ok <폴더> <명령>`으로 `meta.json` 상태가 `completed`이고 `command`가 그 명령인지 확인한 뒤 `rec`로 `runs/rebuild/runbook_paths.sh`에 덧붙인다. 명령이 실패하면(폴더를 만들기 전의 `ConfigError`, 종료 코드 2 포함) 사슬이 멈추므로 앞 단계의 폴더를 이 단계의 것으로 기록하지 않는다.
- 단계 사이의 경로는 `rec`가 기록한 `ILORA_*` 이름으로만 넘긴다: R8 이후의 명령은 `$ILORA_RUN_E2`처럼 그 이름을 직접 읽고, 같은 줄 앞에서 `need <이름>...`으로 기록되어 있는지 확인한다. 그래서 셸이 끊겨도 R0을 다시 하면 `source`로 경로가 돌아온다. `E2`, `R`, `CL` 같은 짧은 변수는 한 줄 안에서 `latest`의 결과를 `rec`에 넘기는 임시 이름일 뿐이다.
- 실행 폴더는 덮어쓰지 않는다. 실패한 폴더도 지우지 않고 기록으로 둔다(다시 돌리면 새 폴더가 생김). 이 문서에는 삭제 명령이 없다.
- 완료 판단은 완료 표시로만 한다: `meta.json` 상태, 학습 checkpoint의 `COMPLETE`, 평가 폴더의 `metrics.json`, smoke의 `smoke_report.json` [SPEC §17]. 파일이 있다는 것만으로 단계를 건너뛰지 않는다.
- 비밀값: Hugging Face token은 셸 환경 변수 `HF_TOKEN`으로만 쓰고 파일·명령 기록에 적지 않는다. 재구현 코드는 token을 읽지도 기록하지도 않는다(DESIGN 4.2, 4.6.3). `LLM_PATH`는 LLaMA 폴더 경로다.
- workspace Pause·Stop은 container 파일 시스템(저장소, env, 가중치, 실행 폴더)을 지운다(`experiments/VESSL_GUIDE.md`). 실행 중에는 Pause하지 않고, Pause 전에는 R16으로 묶어 내려받는다.
- 긴 실행(R5–R15)은 tmux 안에서 한다: `tmux new -s rebuild` 뒤 R0. 분리는 `Ctrl-b d`, 다시 붙기는 `tmux attach -t rebuild`. SSH나 셸이 끊겨도 tmux 안의 실행은 계속된다: 다시 접속하면 먼저 `tmux attach -t rebuild`로 붙어 실행이 끝났는지 본다(재개는 R7의 조건에서만).

## 단계 한눈에 보기

| 단계 | 내용 | 기록하는 경로 변수 | 확인 시험 (`outputs <선택어>`) |
|---|---|---|---|
| R0 | 셸 준비 (매 셸) | - | - |
| R1 | clone, env `ilora-rebuild` (workspace마다 한 번) | - | - |
| R2 | 코드 기록, SASRec 변환 (한 번) | - | - |
| R3 | 시험 묶음 (CPU, GPU) | - | pytest 실패 0 |
| R4 | 환경 점검 `check-env` | `ILORA_RUN_CHECK_ENV` | 종료 코드 0 |
| R5 | smoke 2단계 (GPU 100명) | `ILORA_RUN_SMOKE_GPU` | `smoke_gpu` |
| R6 | smoke 3단계 (과적합) | `ILORA_RUN_OVERFIT` | `overfit` |
| R7 | E2 학습 | `ILORA_RUN_E2` | `e2_run` |
| R8 | E2 재평가 2회 | `ILORA_RUN_E2_REEVAL`, `ILORA_RUN_E2_REEVAL_BI` | `e2_reeval` |
| R9 | E3 (text-only 평가) | `ILORA_RUN_E3` | summary·기록 점검 |
| R10 | E2 export, gate 분석 | `ILORA_RUN_E2_EXPORT`, `ILORA_E2_BEST` | `e2_export` |
| R11 | 군집 (train만 fit) | `ILORA_RUN_CLUSTER` | `cluster` |
| R12 | E4 학습, export | `ILORA_RUN_E4`, `ILORA_RUN_E4_EXPORT` | `e4` |
| R13 | E1 학습, export | `ILORA_RUN_E1`, `ILORA_RUN_E1_EXPORT` | `e1` |
| R14 | 비교표 | `ILORA_RUN_COMPARE` | `done_ok` |
| R15 | 변형 (선택, 예산이 있을 때) | `ILORA_RUN_V_<이름>`, `ILORA_RUN_CLUSTER_USER` | `done_ok` |
| R16 | 보관 (Pause 전) | - | 내려받기 |

E2를 먼저 하는 이유: 주 재현이고 E4가 E2 export와 군집에 의존한다(DESIGN 18.7).

## R0. 매 셸 준비

workspace에서 처음이면 R1로 clone과 env를 먼저 만든다.

```bash
cd ~/iLoRA                                         # repository root (cloned in R1)
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ilora-rebuild
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0
export LLM_PATH=/root/models/Llama-2-7b-hf         # the real location of Llama-2-7b-hf (fp16)
export ILORA_TEST_TOKENIZER="$LLM_PATH"
mkdir -p runs/rebuild && touch runs/rebuild/runbook_paths.sh && source runs/rebuild/runbook_paths.sh
latest() { ls -1d runs/rebuild/"$1"_[0-9]*Z 2>/dev/null | sort | tail -n 1; }
done_ok() { python -I -c 'import json,sys; m=json.load(open(sys.argv[1]+"/meta.json")); print(sys.argv[1], m["status"], m.get("command")); sys.exit(m["status"]!="completed" or m.get("command")!=sys.argv[2])' "$1" "$2"; }
rec() { echo "export $1=$2" >> runs/rebuild/runbook_paths.sh; export "$1=$2"; }
need() { for n in "$@"; do test -n "${!n:-}" || { echo "need: $n is not recorded (do its step first)" >&2; return 1; }; done; }
outputs() { ILORA_VESSL=1 python -I -m pytest rebuild/tests/vessl/test_run_outputs.py -q -k "$1"; }
```

- `latest <프로파일>`: 그 프로파일의 가장 최근 실행 폴더(`runs/rebuild/<프로파일>_<UTC>`). `main.py`는 `--config` 파일 이름을 프로파일 이름으로 쓰므로 eval·export·cluster 실행도 `paper_spec_<UTC>` 같은 이름을 가진다. 그래서 `latest`는 폴더를 만든 명령 바로 뒤에 `&&`로 잇는다(명령이 폴더를 만들기 전에 실패했을 때 `latest`가 돌려주는 앞 단계의 폴더를 쓰지 않게).
- `done_ok <폴더> <명령>`: `meta.json` 상태가 `completed`가 아니거나 `command`가 `<명령>`(`train`, `eval`, `export`, `cluster`, `smoke`, `check-env`, `compare`)이 아니면 0이 아닌 종료 코드(뒤의 `&&`가 멈춤).
- `rec <이름> <폴더>`: 경로를 기록하고 현재 셸에도 export. `rebuild/tests/vessl/test_run_outputs.py`는 `ILORA_RUN_<이름>`으로 실행 폴더를 찾는다(DESIGN 17.6.2).
- `need <이름>...`: 그 `ILORA_*` 변수가 비어 있으면 이름을 알리고 0이 아닌 종료 코드(같은 줄의 명령을 돌리지 않음). 빈 경로가 `--set`에 들어가 엉뚱한 폴더를 만들거나 비교표가 모두 NOT_RUN이 되는 일을 막는다.
- `outputs <선택어>`: 그 단계의 실행 폴더 검사(`test_run_outputs.py -k <선택어>`, DESIGN 17.6.1). 측정값은 `runs/rebuild/vessl_tests_<UTC>/measurements.json`에 남는다(판정 없는 값은 TBD).
- 확인: `python --version`이 3.10, `echo "$CONDA_DEFAULT_ENV"`가 `ilora-rebuild`.

## R1. clone과 환경 (workspace마다 한 번)

```bash
git clone https://github.com/myCompilerWontShutUp/iLoRA.git ~/iLoRA     # full history: never --depth
cd ~/iLoRA && git checkout exp/gate-cluster
conda create -n ilora-rebuild -y -q -c conda-forge --override-channels python=3.10
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ilora-rebuild
export PYTHONNOUSERSITE=1
python -m pip install -r requirements.txt          # repo pins: U requirements + safetensors 0.3.1 (fork)
python -m pip install pytest==9.2.0
python -I -c "import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)"
```

- 새 env `ilora-rebuild`만 쓴다. fork의 `ilora` env는 `setup_vessl.sh`가 transformers의 `generation/utils.py`·`models/llama/modeling_llama.py`를 `debug/` 사본으로 바꿔 두었으므로 쓰지 않고, `setup_vessl.sh`·`bootstrap_vessl.sh`도 이 env에서 돌리지 않는다(D13, DESIGN 18.5, 18.6.12 불일치 1).
- `conda create`는 conda-forge만 쓴다(`--override-channels`): Anaconda 기본 channel의 이용 약관 확인 창이 뜨지 않음.
- 확인: 마지막 줄이 `2.0.0+cu117 11.7 4.28.0`. transformers 파일 무결성은 R4 check-env가 wheel RECORD와 대조한다.
- 실행할 commit이 `rebuild/`를 포함해야 한다(로컬에서 commit·push한 뒤 clone). R2의 `git status --porcelain`이 비어 있어야 `meta.json`의 commit이 실행한 코드를 대표한다.

실패하면:
- 이미 clone이 있으면 clone 대신 `cd ~/iLoRA && git fetch origin && git checkout exp/gate-cluster && git pull --ff-only`. 얕은 clone이면 `git fetch --unshallow` (R2가 확인).
- `conda`가 없으면 Miniforge를 `$HOME/miniforge3`에 설치하고 `export PATH="$HOME/miniforge3/bin:$PATH"` 뒤 다시 (fork `bootstrap_vessl.sh` 2단계와 같은 방법: 설치 파일 `https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh`를 `bash <파일> -b -p "$HOME/miniforge3"`).
- `pip install`이 실패하면 `python -c "import sys; print(sys.executable)"`가 env 안의 Python 3.10인지 확인한다. 다른 env의 패키지로 우회하지 않는다.
- `tmux`가 없으면 `apt-get install -y tmux` (root).

LLaMA 가중치가 `$LLM_PATH`에 없을 때만 (Llama-2-7b-hf, 접근 승인된 계정의 `HF_TOKEN`을 셸에서 export한 상태). 내려받기용 별도 env를 써서 `ilora-rebuild`의 고정 판을 건드리지 않는다. 처음 workspace에서는 아직 R0 전이므로 첫 줄에서 `LLM_PATH`를 R0과 같은 값으로 export한다:

```bash
export LLM_PATH=/root/models/Llama-2-7b-hf         # the same value as R0 (R0 comes after R1 on a fresh workspace)
conda create -n hf-download -y -q -c conda-forge --override-channels python=3.10
conda activate hf-download
python -m pip install -q "huggingface_hub>=0.20,<1.0"
python -I -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id="meta-llama/Llama-2-7b-hf", local_dir=os.environ["LLM_PATH"], allow_patterns=["*.json", "*.safetensors", "tokenizer.model"], token=os.environ["HF_TOKEN"])'
conda activate ilora-rebuild
```

- 확인은 R4 check-env가 한다: `config.json`의 `torch_dtype`이 `float16`, shard와 `tokenizer.model`이 있음(D9), `tokenizer.model` sha256 `9e556afd…d347`.

## R2. 코드 기록과 SASRec 변환 (한 번, VESSL 기계에서)

```bash
git rev-parse HEAD && git status --porcelain       # also recorded in every meta.json
test "$(git rev-parse --is-shallow-repository)" = false
git cat-file -e 28e47371192121b6ced44401d9e149fedfae13de^{commit}
python -I rebuild/tools/convert_sasrec.py --repo . --out runs/rebuild/assets/sasrec
```

- 산출물: `runs/rebuild/assets/sasrec/{sasrec_movielens_state.pt, sasrec_movielens_reference.pt, manifest.json}` (DESIGN 7.6.1). CPU만, 수십 초.
- 확인: 종료 코드 0, `manifest.json`의 digest = 프로파일 `sasrec.state_digest` (`6a78697b…1dae`). R4 check-env가 다시 대조한다.
- 변환은 이 VESSL 기계에서 한다: CPU 기준 출력의 bitwise 일치는 같은 기계에서 만든 기준 파일에서만 보장된다(DESIGN 7.6.5). 다른 기계의 변환본을 복사해 쓰지 않는다.

실패하면:
- `is-shallow-repository`가 `true`이거나 `cat-file`이 실패하면 `git fetch --unshallow` 뒤 다시 (도구가 upstream commit `28e4737`의 원본 파일을 git에서 꺼냄).
- 출력 폴더가 이미 있으면 도구가 거부한다(덮어쓰기 금지). 같은 기계에서 앞서 만든 변환본이면 그대로 쓰고 R4가 digest를 확인한다. 군집(R11) 이후에는 변환을 다시 하지 않는다: E4는 군집 산출물에 기록된 state 파일 sha256과 같은 파일을 요구한다(M15).

## R3. 시험

```bash
python -I -m pytest rebuild/tests -q                                   # CPU bundle (Linux, real tokenizer)
ILORA_VESSL=1 python -I -m pytest rebuild/tests/vessl -q -k "not run_outputs"
```

- 산출물: pytest 요약, `runs/rebuild/vessl_tests_<UTC>/measurements.json` (DESIGN 17.6.2).
- 확인: 두 묶음 모두 실패 0. skip은 문서화된 이유(`tests/conftest.py`, `tests/vessl/conftest.py`의 `SKIP_*`, `VESSL_REQUIRED`)뿐이다. CPU 묶음은 `tests/vessl`도 모으지만 그 명령에는 `ILORA_VESSL`이 없으므로 그 시험은 모두 `VESSL_REQUIRED`로 skip된다(정상: 둘째 명령이 돌린다). 자산(R2 변환본, `ILORA_TEST_TOKENIZER`)이 있으므로 CPU 묶음에서 `VESSL_REQUIRED` 밖의 skip은 0이어야 한다: 요약 끝 `SKIPPED` 줄(`pytest.ini`의 `-ra`)의 이유가 모두 `VESSL_REQUIRED`인지 본다. VESSL 묶음의 skip은 0 (`SKIP_NO_PL`이면 그 이유가 `measurements.json`에 남음) (DESIGN 17.6.1, 17.8 #2, #7).
- CPU 묶음에는 느린 시험(`slow`, smoke 1단계 포함)이 들어 있다. 빠른 묶음만 다시 볼 때는 `-m "not slow"`.

실패하면: R4 이후를 시작하지 않는다. 실패한 시험 이름과 메시지를 기록하고, 원인이 환경(판, 경로, 자산)이면 고친 뒤 R3을 다시 한다. 시험을 건너뛰거나 기준을 바꾸지 않는다.

## R4. 환경 점검

```bash
python -I rebuild/main.py check-env --config rebuild/configs/paper_spec.yaml \
  && C=$(latest paper_spec) && done_ok "$C" check-env && rec ILORA_RUN_CHECK_ENV "$C" && python -m pip freeze > "$C/pip_freeze.txt"
```

- 산출물: 실행 폴더의 `check_env.json`, `meta.json`, 그리고 env 판 기록 `pip_freeze.txt` (DESIGN 18.8 #9).
- 확인: 종료 코드 0, 상태 `completed`. 내용: GPU 1장·75 GiB 이상, 판 고정(torch 2.0.0, transformers 4.28.0, numpy 1.24.2, pandas 1.5.3, sentencepiece 0.1.97, scikit-learn 1.2.2), transformers 파일 무결성(wheel RECORD), `LLM_PATH`·`torch_dtype` float16, 입력 해시, SASRec digest·기준 대조(CUDA ≤ 1e-4), tokenizer sha256, `candidates.self_check()` (DESIGN 4.6.5).

실패하면: 폴더는 콘솔 첫 줄의 `run_dir=` (또는 `latest paper_spec`). `check_env.json`의 `checks`와 `meta.json` `error`가 실패 항목을 모두 나열한다. 판이 다르면 R1 env를 다시 만들고(패치·수동 교체 금지), GPU가 2장 이상 보이면 `CUDA_VISIBLE_DEVICES=0`을 확인한다. 고친 뒤 R4를 다시 돌리면 새 실행 폴더가 생긴다.

## R5. smoke 2단계 (GPU 100명)

```bash
python -I rebuild/main.py smoke --config rebuild/configs/smoke_gpu_100.yaml \
  && S=$(latest smoke_gpu_100) && done_ok "$S" smoke && rec ILORA_RUN_SMOKE_GPU "$S" && outputs smoke_gpu
```

- 산출물: `smoke_report.json`, `steps/` (G0 check-env, G1 학습 2 epoch, G2 재평가, G3 export, G4 군집, G5 E4 경로 1 epoch, G6 E1 경로 1 epoch; DESIGN 17.6.7).
- 확인: `smoke_report.json`의 `passed` true(하나라도 실패하면 `main.py smoke`가 0이 아닌 종료 코드, 상태 `failed`), `outputs smoke_gpu` 통과. 기록 수: micro 1,806, step 114, epoch 2, `gate_calls` 합 1,806, `n_hybrid` 합 888, E4 경로 `cluster_lookups` 7,224 (DESIGN 17.6.7, 18.6.5). 메모리·시간은 기록만(TBD).

실패하면: E1·E2·E4 학습(R7 이후)을 시작하지 않는다. 폴더(`latest smoke_gpu_100`)의 `smoke_report.json` `checks`에서 실패한 항목과 `steps/<단계>/meta.json` `error`를 기록한다.

## R6. smoke 3단계 (과적합)

```bash
python -I rebuild/main.py smoke --config rebuild/configs/overfit_gpu.yaml \
  && O=$(latest overfit_gpu) && done_ok "$O" smoke && rec ILORA_RUN_OVERFIT "$O" && outputs overfit
```

- 산출물: `smoke_report.json`, `steps/` (O0 check-env, O1 학습 25 epoch, O2 과적합 생성 점검). checkpoint 25개, 약 10.35 GiB (DESIGN 18.6.11).
- 확인: `passed` true: 마지막 epoch 평균 loss ≤ 첫 epoch × 0.1, 16/16 첫 줄이 정답 제목과 일치(`match_norm`), micro 100행 모두 hybrid (DESIGN 17.6.8). loss 비와 일치 수는 기록만.

실패하면: R7 이후를 시작하지 않는다. loss가 낮은데 첫 줄이 맞지 않으면 학습과 생성 경로가 어긋난 것(padding·position, label 정렬 D18, 자리 대응)으로 보고 원인을 찾는다.

## R7. E2 학습

tmux 안에서 (`tmux new -s rebuild` 뒤 R0).

```bash
python -I rebuild/main.py train --config rebuild/configs/paper_spec.yaml \
  && E2=$(latest paper_spec) && done_ok "$E2" train && rec ILORA_RUN_E2 "$E2" && outputs e2_run
```

- 산출물: `checkpoints/epoch0..4/` (각 `COMPLETE`), `eval/epoch0..4/val/`, `eval/epoch{a,b}/test/` (a = best_val, b = last), `eval/selection.json`, `records/train/` (DESIGN 14.6.6–14.6.8, 16.6.2).
- 확인: `outputs e2_run` (`test_e2_run_*`: M3-13, M6-9, M7-14, M10-15, M11-18·22, M12-15) 통과. 기록 수 micro 41,325, step 2,585 (epoch 끝 `n_micro` 9), epoch 5 (DESIGN 18.6.5).

중단되면: 먼저 학습 프로세스가 정말 끝났는지 본다. SSH나 셸이 끊긴 것만으로는 tmux 안의 학습이 멈추지 않고 `meta.json`도 계속 `running`이다. 다시 접속해 `tmux attach -t rebuild`로 붙고, 학습이 아직 돌고 있으면 그대로 둔다(재개하지 않는다). 재개는 학습 프로세스가 끝났을 때만 한다: 예외로 끝나면 상태 `failed`, 강제 종료(OOM killer, `kill -9`, tmux 세션 종료)로 끝나면 `running`이 남는다. 아래 블록의 앞 두 명령으로 확인한다: `pgrep`은 아무것도 출력하지 않고(살아 있는 `rebuild/main.py` 프로세스 없음), `nvidia-smi`의 사용 메모리는 수 MiB뿐이어야 한다(GPU를 쥔 프로세스 없음). `train --resume`도 그 폴더의 `run.lock`을 살아 있는 프로세스가 쥐고 있으면 거부하고 아무것도 바꾸지 않는다(`cannot resume ...: ... is in use by a live process`). `failed`·`running` 상태만 열리고 `--config`·`--set`은 주지 않는다(폴더의 `config.yaml`을 씀, DESIGN 14.6.10). 강제 종료된 프로세스의 `train.segments` 항목은 재개 때 `interrupted`로 표시된다(끝난 시각은 측정되지 않았으므로 `ended_utc` null). 재개 뒤 같은 확인을 한다.

```bash
pgrep -af 'rebuild/main.py'                                    # must print nothing: no rebuild command is alive
nvidia-smi --query-gpu=memory.used --format=csv,noheader       # must be a few MiB: no process holds the GPU
R=$(latest paper_spec) && python -I rebuild/main.py train --resume "$R" && done_ok "$R" train && rec ILORA_RUN_E2 "$R" && outputs e2_run
```

실패하면: `meta.json` `error`와 `run.log`의 마지막 `train step` 줄을 기록한다. GradScaler skip이 연속 16회(`train.max_consecutive_skips`)면 학습이 멈춘다(DESIGN 14.6.8). fp16이 이 규칙에 걸리거나 parameter가 비유한일 때만 bf16 대안(`variant_bf16.yaml`)을 쓰고 편차로 기록한다(D9, DESIGN 18.6.4). 그때는 E3·E4·E1과 변형도 같은 precision으로 돌려야 짝 비교가 같은 조건이므로, 문서 끝 "bf16 대안" 절의 명령으로 E2부터 다시 하고 그 절이 바꾸는 단계는 그 절의 명령을 쓴다.

## R8. E2 재평가 2회 (결정성, batch 무관성)

```bash
need ILORA_RUN_E2 && python -I rebuild/main.py eval --config rebuild/configs/paper_spec.yaml --set eval.source_run="$ILORA_RUN_E2" \
  && R=$(latest paper_spec) && done_ok "$R" eval && rec ILORA_RUN_E2_REEVAL "$R"
need ILORA_RUN_E2 && python -I rebuild/main.py eval --config rebuild/configs/paper_spec.yaml --set eval.source_run="$ILORA_RUN_E2" \
  --set eval.batch_invariance_check=true && R=$(latest paper_spec) && done_ok "$R" eval && rec ILORA_RUN_E2_REEVAL_BI "$R" \
  && outputs e2_reeval
```

- 산출물: 각 실행의 `eval/epoch{a,b}/{val,test}/`와 `eval/summary.json` (DESIGN 15.6.11). 둘째 실행은 평가 폴더마다 `batch_invariance.json`.
- 확인: `outputs e2_reeval`: 두 실행의 `records.jsonl`이 학습 중 기록과 sha256이 같고 `summary.json`의 `reproduction`이 모두 `identical` (M10-17, M11-19). batch 8 대 batch 1 일치율은 기록만(M11-20, TBD).

실패하면: 다르면 R9 이후의 결과를 확정하지 않는다. 시험이 출력하는 다른 표본과 처음 다른 생성 단계를 기록하고 원인을 밝힌다(DESIGN 15.8 #19). R9 이후의 실행은 계속할 수 있지만 그 결과는 원인이 밝혀질 때까지 확정하지 않는다.

## R9. E3 (E2 checkpoint의 text-only 평가)

```bash
need ILORA_RUN_E2 && python -I rebuild/main.py eval --config rebuild/configs/variant_text_only_eval.yaml --set eval.source_run="$ILORA_RUN_E2" \
  && E3=$(latest variant_text_only_eval) && done_ok "$E3" eval && rec ILORA_RUN_E3 "$E3"
need ILORA_RUN_E3 && python -I -c 'import json,sys; r=json.load(open(sys.argv[1]+"/eval/summary.json"))["results"]; s={k: v["reproduction"]["status"] for k, v in r.items()}; print(s); sys.exit(set(s.values())!={"not_comparable"})' "$ILORA_RUN_E3"
need ILORA_RUN_E3 && python -I -c 'import glob,json,sys; h=[json.loads(x)["hybrid"] for p in glob.glob(sys.argv[1]+"/eval/*/*/records.jsonl") for x in open(p, encoding="utf-8")]; print(len(h), "records, hybrid values", set(h)); sys.exit(not h or any(h))' "$ILORA_RUN_E3"
```

- 산출물: `eval/epoch{a,b}/{val,test}/`, `eval/summary.json` (평가 prompt만 text_only, D4).
- 확인: 상태 `completed`, 기록의 `hybrid`가 모두 false, `reproduction`이 모두 `not_comparable` (prompt가 원 실행과 달라 재현 비교 대상이 아님, DESIGN 15.6.10). 두 확인 명령이 0으로 끝남.

## R10. E2 export와 gate 분석

```bash
need ILORA_RUN_E2 && python -I rebuild/main.py export --config rebuild/configs/paper_spec.yaml --set records.source_run="$ILORA_RUN_E2" \
  && E2X=$(latest paper_spec) && done_ok "$E2X" export && rec ILORA_RUN_E2_EXPORT "$E2X" \
  && A=$(python -I -c 'import json,sys; print(json.load(open(sys.argv[1]+"/eval/selection.json"))["best_val_epoch"])' "$ILORA_RUN_E2") \
  && rec ILORA_E2_BEST "$A" && outputs e2_export
```

- 산출물: `records/export/epoch{a,b}/` (NPZ 3개, `gates_{validation,test}.csv`, `crosscheck.json`, `manifest.json`), `records/analysis/epoch{a,b}/{test,val,train}/` (요약, 해석 문장, 그림; DESIGN 16.6.5–16.6.7).
- 확인: `outputs e2_export`: 대조 통과(|Δω| ≤ 1e-5), 파일 sha256이 manifest와 같음, 그림·요약 존재 (M12-16).

실패하면: R11을 시작하지 않는다(군집의 dynamic 비교가 이 export를 읽음). 대조가 실패해도 `crosscheck.json`은 먼저 쓰인다: 실패한 export 폴더(`latest paper_spec`, 상태 `failed`)의 `records/export/epoch{e}/crosscheck.json`에서 `passed`가 false인 split의 `max_abs_diff_omega`·`max_abs_diff_omega_sample_id`·`argmax_mismatches`·`max_abs_diff_z`를 기록한다(오류 문구에도 같은 수가 있음). `max_abs_diff_z`가 1e-4 근처면 평가 z가 다른 것(CUDA 반올림·TF32), 훨씬 작으면 gate가 다른 것이다 (DESIGN 16.6.5 7단계, 16.7).

## R11. 군집 (train만 fit)

```bash
need ILORA_RUN_E2_EXPORT ILORA_E2_BEST && python -I rebuild/main.py cluster --config rebuild/configs/paper_spec.yaml \
  --set cluster.dynamic_export_dir="$ILORA_RUN_E2_EXPORT/records/export/epoch$ILORA_E2_BEST" \
  && CL=$(latest paper_spec) && done_ok "$CL" cluster && rec ILORA_RUN_CLUSTER "$CL" && outputs cluster
```

- 산출물: `cluster/` (`kmeans_k4_seed1234.joblib`, `centroids.npz`, 배정 CSV 3개, `cluster_summary.json`, `figures/{sasrec_pca_clusters, cluster_dynamic_gate_heatmap}.png`, `manifest.json`; DESIGN 19.2).
- 확인: `outputs cluster` (M15-14): manifest `fit_split` train, fit 66,126행, 빈 군집 없음, E2 export z와 대조 ≤ 1e-5, dynamic 비교 항목 존재. validation·test는 predict만 한다 [SPEC §0.8].

실패하면: R12를 시작하지 않는다. z 대조가 실패하면 R2의 변환본과 E2 export가 같은 SASRec state를 썼는지(`manifest.json`의 `sasrec`) 확인한다.

## R12. E4 학습과 export

```bash
need ILORA_RUN_CLUSTER && python -I rebuild/main.py train --config rebuild/configs/cluster_hard.yaml \
  --set cluster.artifact_dir="$ILORA_RUN_CLUSTER/cluster" \
  && E4=$(latest cluster_hard) && done_ok "$E4" train && rec ILORA_RUN_E4 "$E4"
need ILORA_RUN_CLUSTER ILORA_RUN_E4 && python -I rebuild/main.py export --config rebuild/configs/cluster_hard.yaml \
  --set cluster.artifact_dir="$ILORA_RUN_CLUSTER/cluster" --set records.source_run="$ILORA_RUN_E4" \
  && E4X=$(latest cluster_hard) && done_ok "$E4X" export && rec ILORA_RUN_E4_EXPORT "$E4X" && outputs e4
```

- E2 학습 가중치를 쓰지 않고 처음부터 학습한다 [SPEC §9, §25]. `cluster.artifact_dir`는 명령행으로만 준다(빠지면 `ConfigError`, DESIGN 18.6.4).
- 확인: `outputs e4`: 학습 시작 때 live z 대조 통과(M15-12, `meta.json` `cluster_live_check`), 학습 `gate_calls` 합 0·`cluster_lookups` 합 330,600, 평가 gate가 one-hot = mapping[배정] (M7-15, M15-13), export의 ω가 세 split 모두 mapping[배정]의 one-hot이고 대조 통과 (`test_e4_export`).
- 중단되면 R7의 조건(학습 프로세스가 끝났음: `pgrep`·`nvidia-smi` 확인)에서만 `R=$(latest cluster_hard) && python -I rebuild/main.py train --resume "$R" && done_ok "$R" train && rec ILORA_RUN_E4 "$R"`, 그다음 위 블록의 둘째 줄(export)부터 한다.

실패하면: 군집 산출물이 바뀌었거나(파일 sha256), SASRec state가 다르면 학습 전에 `ClusterError`로 멈춘다. 군집을 다시 fit하지 말고 원인(어느 파일인지 오류 메시지가 이름을 댐)을 기록한다.

## R13. E1 학습과 export

```bash
python -I rebuild/main.py train --config rebuild/configs/llara_baseline.yaml \
  && E1=$(latest llara_baseline) && done_ok "$E1" train && rec ILORA_RUN_E1 "$E1"
need ILORA_RUN_E1 && python -I rebuild/main.py export --config rebuild/configs/llara_baseline.yaml --set records.source_run="$ILORA_RUN_E1" \
  && E1X=$(latest llara_baseline) && done_ok "$E1X" export && rec ILORA_RUN_E1_EXPORT "$E1X" && outputs e1
```

- 확인: `outputs e1` (`test_e1_run_outputs`, M14 18.8 #8): optimizer 그룹 {projector, lora}, 평가 기록 `gate_0` = 1.0, 학습 기록 수가 E2와 같음(41,325·2,585·5), export는 gate 분석 없이 성공(M12-11).
- 중단되면 R7의 조건에서만 `R=$(latest llara_baseline) && python -I rebuild/main.py train --resume "$R" && done_ok "$R" train && rec ILORA_RUN_E1 "$R"`, 그다음 위 블록의 둘째 줄(export)부터 한다.

## R14. 비교표

```bash
need ILORA_RUN_E1 ILORA_RUN_E2 ILORA_RUN_E3 ILORA_RUN_E4 ILORA_RUN_E1_EXPORT ILORA_RUN_E2_EXPORT ILORA_RUN_E4_EXPORT ILORA_RUN_CLUSTER \
  && python -I rebuild/tools/compare_runs.py --run E1="$ILORA_RUN_E1" --run E2="$ILORA_RUN_E2" --run E3="$ILORA_RUN_E3" \
  --run E4="$ILORA_RUN_E4" --export E1="$ILORA_RUN_E1_EXPORT" --export E2="$ILORA_RUN_E2_EXPORT" \
  --export E4="$ILORA_RUN_E4_EXPORT" --cluster "$ILORA_RUN_CLUSTER" \
  && CMP=$(latest compare) && done_ok "$CMP" compare && rec ILORA_RUN_COMPARE "$CMP"
```

- 산출물: `runs/rebuild/compare_<UTC>/` 의 `comparison.csv`, `comparison.md` (표 A–F, D17 정의, 표 D, 한계), `spec20_index.json` (SPEC §20 파일 이름 → 실행 파일과 sha256), `summary.md` (SPEC §19 구성) (DESIGN 18.6.7–18.6.9).
- 확인: 상태 `completed`. 측정 칸은 모두 실행 파일 값이고(`source_path`, `source_sha256` 열), 없는 값은 `NOT_RUN`과 이유(`comparison.md` 끝). 논문 값은 표 D에만 있다. 저장소의 `results/`에는 쓰지 않는다.

실패하면: `compare error: ...`는 실행 파일이 스스로 모순될 때(지표와 개수가 1e-12 넘게 다름, `summary.json`의 sha256 불일치 등)만 난다. 그 파일을 고치지 말고 기록한다. 빠진 실행은 오류가 아니라 `NOT_RUN`이다: 실행 하나를 일부러 빼고 비교하려면 그 `--run`/`--export`와 `need`의 이름을 지우고 돌린다. precision이 float16 / fp16_amp가 아닌 실행(bf16 대안)이 있으면 `comparison.md`의 한계에 그 precision이 적힌다(D9).

## R15. 변형 (선택, 예산이 있을 때)

```bash
V=all_hybrid   # all_hybrid reverse_curriculum released_lr released_gate_lr released_label total_rank_scaling fixed_template fixed_titles
python -I rebuild/main.py train --config rebuild/configs/variant_$V.yaml \
  && VR=$(latest variant_$V) && done_ok "$VR" train && rec "ILORA_RUN_V_$V" "$VR"
# user-level cluster (D25): cluster with the user unit, then train with its artifacts
need ILORA_RUN_E2_EXPORT ILORA_E2_BEST && python -I rebuild/main.py cluster --config rebuild/configs/paper_spec.yaml \
  --set cluster.unit=user --set cluster.dynamic_export_dir="$ILORA_RUN_E2_EXPORT/records/export/epoch$ILORA_E2_BEST" \
  && CLU=$(latest paper_spec) && done_ok "$CLU" cluster && rec ILORA_RUN_CLUSTER_USER "$CLU"
need ILORA_RUN_CLUSTER_USER && python -I rebuild/main.py train --config rebuild/configs/variant_user_level_cluster.yaml \
  --set cluster.artifact_dir="$ILORA_RUN_CLUSTER_USER/cluster" \
  && VU=$(latest variant_user_level_cluster) && done_ok "$VU" train && rec ILORA_RUN_V_user_level_cluster "$VU"
```

- 변형마다 E2와 다른 키는 DESIGN 18.6.4 표와 같다(`extends: paper_spec.yaml`). 결과는 R14를 `--run <이름>=<폴더>`를 더해 다시 돌려 표에 넣는다. 돌리지 않은 변형은 NOT_RUN. 셸이 바뀐 뒤 비교만 다시 할 때는 먼저 `V=<변형 이름>`을 준다(경로는 `runbook_paths.sh`의 `ILORA_RUN_V_<이름>`).
- `released_lr`(lr 8e-4)는 고르기 전에 그 GPU 시간을 예산에 넣는다: fork는 같은 A100, fp16 AMP, 8e-4에서 첫 validation의 유효 예측 0, 둘째 epoch 내내 loss NaN이었다(fork `experiments/IMPLEMENTATION_NOTES.md` §11i). 같은 일이 생기면 GradScaler skip이 16번 이어진 뒤 `TrainingError`로 멈추고(상태 `failed`, 1 epoch 이상의 GPU 시간), `done_ok`에서 사슬이 멈춰 기록되지 않는다. 그것도 D5의 결과다: bf16으로 다시 돌리지 않고(D5와 D9가 섞임) 실패한 폴더를 비교에 그대로 넘긴다(아래 명령의 `V_$V` 대신 `--run V_released_lr="$(latest variant_released_lr)"`, 그 칸은 `NOT_RUN`과 `TrainingError` 이유).

```bash
need ILORA_RUN_E1 ILORA_RUN_E2 ILORA_RUN_E3 ILORA_RUN_E4 ILORA_RUN_E1_EXPORT ILORA_RUN_E2_EXPORT ILORA_RUN_E4_EXPORT ILORA_RUN_CLUSTER \
  "ILORA_RUN_V_$V" ILORA_RUN_V_user_level_cluster \
  && python -I rebuild/tools/compare_runs.py --run E1="$ILORA_RUN_E1" --run E2="$ILORA_RUN_E2" --run E3="$ILORA_RUN_E3" \
  --run E4="$ILORA_RUN_E4" --run V_$V="$(printenv ILORA_RUN_V_$V)" --run V_user_level_cluster="$ILORA_RUN_V_user_level_cluster" \
  --export E1="$ILORA_RUN_E1_EXPORT" --export E2="$ILORA_RUN_E2_EXPORT" --export E4="$ILORA_RUN_E4_EXPORT" \
  --cluster "$ILORA_RUN_CLUSTER" && CMP=$(latest compare) && done_ok "$CMP" compare && rec ILORA_RUN_COMPARE_VARIANTS "$CMP"
```

- `variant_total_rank_scaling`이 없으면 비교표의 한계 2번이 그 사실을 적는다 (DESIGN 18.6.10). `variant_bf16`은 R7의 실패 규칙에서만 쓴다(문서 끝 "bf16 대안" 절).

## R16. 보관 (Pause 전)

```bash
tar -czf ~/rebuild_runs_$(date -u +%Y%m%dT%H%M%SZ).tgz runs/rebuild
```

- 묶음에는 변환 자산, checkpoint(학습 실행 하나 약 2.07 GiB, DESIGN 18.6.11), 평가·export·군집·비교 폴더, `runbook_paths.sh`가 들어간다. 내려받은 뒤에만 Pause한다. 실행 중에는 Pause하지 않는다.
- 삭제 명령은 두지 않는다(사용자 판단).

## 경로 변수와 실행 폴더 검사 대응

| 변수 (`rec`) | 단계 | 쓰는 시험 (`rebuild/tests/vessl/test_run_outputs.py`) |
|---|---|---|
| `ILORA_RUN_SMOKE_GPU` | R5 | `test_smoke_gpu_*` (`outputs smoke_gpu`) |
| `ILORA_RUN_OVERFIT` | R6 | `test_overfit_*` (`outputs overfit`) |
| `ILORA_RUN_E2` | R7 | `test_e2_run_*` (`outputs e2_run`), `test_e2_reeval_reproduces` |
| `ILORA_RUN_E2_REEVAL`, `ILORA_RUN_E2_REEVAL_BI` | R8 | `test_e2_reeval_*` (`outputs e2_reeval`) |
| `ILORA_RUN_E2_EXPORT` | R10 | `test_e2_export` (`outputs e2_export`) |
| `ILORA_RUN_CLUSTER` | R11 | `test_cluster_dynamic_comparison` (`outputs cluster`), `test_e4_run_no_gate_calls_and_one_hot`, `test_e4_export` |
| `ILORA_RUN_E4`, `ILORA_RUN_E4_EXPORT` | R12 | `test_e4_run_*`, `test_e4_export` (`outputs e4`) |
| `ILORA_RUN_E1`, `ILORA_RUN_E1_EXPORT` | R13 | `test_e1_run_outputs` (`outputs e1`) |

- 단계 하나를 다시 할 때: 같은 명령을 다시 돌리면 새 실행 폴더가 생기고, 그 뒤의 `latest`·`done_ok`·`rec`이 새 경로를 기록한다(`runbook_paths.sh`에서는 나중 줄이 이김). 앞 단계 경로가 바뀌면 그 경로를 쓰는 뒤 단계도 다시 한다(E2 → R8–R14, 군집 → R12, R14).

## bf16 대안 (R7의 실패 규칙에서만)

fp16 E2가 R7의 실패 규칙(GradScaler skip 연속 16회, 비유한 parameter)에 걸렸을 때만 쓴다(D9, DESIGN 18.6.4 `variant_bf16`). E2를 `variant_bf16.yaml`로 처음부터 다시 학습하고, 짝 비교가 같은 조건이 되도록 E3·E4·E1과 R15의 변형도 bf16(`model.torch_dtype=bfloat16`, `train.precision=bf16_amp`)으로 돌린다. 실패한 fp16 E2 폴더는 기록으로 둔다(`rec`가 `ILORA_RUN_E2`를 새 폴더로 바꿈). 편차로 기록한다: M10-15의 fp16 항목(첫 step scale 65,536)은 bf16 실행에 해당하지 않으며 `outputs e2_run`은 GradScaler 없는 기록(scale 1.0, skip 없음)을 확인하고, `compare_runs.py`는 precision을 비교표의 한계에 적는다.

아래 블록의 줄은 주석이 가리키는 단계에서 그 단계의 명령 대신 하나씩 돌린다: R7(E2), R8, R9(첫 줄), R12(첫 줄), R13(첫 줄). 나머지(R9의 확인 두 줄, R10, R11, R12·R13의 export 줄, R14)는 본문 그대로이고 단계 순서도 본문과 같다(E4는 R10·R11 뒤). 중단되면 R7의 조건에서 `R=$(latest variant_bf16) && python -I rebuild/main.py train --resume "$R" && done_ok "$R" train && rec ILORA_RUN_E2 "$R" && outputs e2_run`.

```bash
# R7: E2 in bf16
python -I rebuild/main.py train --config rebuild/configs/variant_bf16.yaml \
  && E2=$(latest variant_bf16) && done_ok "$E2" train && rec ILORA_RUN_E2 "$E2" && outputs e2_run
# R8: the two re-evaluations with the training config
need ILORA_RUN_E2 && python -I rebuild/main.py eval --config rebuild/configs/variant_bf16.yaml --set eval.source_run="$ILORA_RUN_E2" \
  && R=$(latest variant_bf16) && done_ok "$R" eval && rec ILORA_RUN_E2_REEVAL "$R"
need ILORA_RUN_E2 && python -I rebuild/main.py eval --config rebuild/configs/variant_bf16.yaml --set eval.source_run="$ILORA_RUN_E2" \
  --set eval.batch_invariance_check=true && R=$(latest variant_bf16) && done_ok "$R" eval && rec ILORA_RUN_E2_REEVAL_BI "$R" \
  && outputs e2_reeval
# R9: first line (then the two R9 checks of the main text)
need ILORA_RUN_E2 && python -I rebuild/main.py eval --config rebuild/configs/variant_text_only_eval.yaml \
  --set model.torch_dtype=bfloat16 --set train.precision=bf16_amp --set eval.source_run="$ILORA_RUN_E2" \
  && E3=$(latest variant_text_only_eval) && done_ok "$E3" eval && rec ILORA_RUN_E3 "$E3"
# R12: first line, after R10 and R11 of the main text (then the R12 export line)
need ILORA_RUN_CLUSTER && python -I rebuild/main.py train --config rebuild/configs/cluster_hard.yaml \
  --set model.torch_dtype=bfloat16 --set train.precision=bf16_amp --set cluster.artifact_dir="$ILORA_RUN_CLUSTER/cluster" \
  && E4=$(latest cluster_hard) && done_ok "$E4" train && rec ILORA_RUN_E4 "$E4"
# R13: first line (then the R13 export line)
python -I rebuild/main.py train --config rebuild/configs/llara_baseline.yaml --set model.torch_dtype=bfloat16 --set train.precision=bf16_amp \
  && E1=$(latest llara_baseline) && done_ok "$E1" train && rec ILORA_RUN_E1 "$E1"
```

- R8의 재평가는 학습과 같은 설정이어야 하므로(`eval` 명령은 `eval`·`records` 구역과 `prompt.eval_prompt` 등 몇 키 밖의 설정 차이를 거부, DESIGN 15.6.11) `variant_bf16.yaml`을 쓰고, R9은 같은 이유로 두 `--set`을 더한다. R10의 export(`data`·`sasrec`·`router`·`lora`·`cluster` 구역만 비교, LLaMA를 만들지 않음)와 R11의 군집은 precision과 무관하므로 본문 명령 그대로다.
- R15의 변형: 본문의 `train` 명령 둘(`variant_$V.yaml`, `variant_user_level_cluster.yaml`)에 `--set model.torch_dtype=bfloat16 --set train.precision=bf16_amp`를 더한다(`variant_released_lr`도 같음: bf16 경로에서는 모든 실행이 bf16).
