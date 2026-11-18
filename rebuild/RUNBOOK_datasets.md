# RUNBOOK_datasets: Steam·LastFM 실행 순서와 명령

재구현(`rebuild/`)을 LastFM과 Steam으로 확장한 실험을 VESSL workspace(Linux, A100 80GB 1장)에서 돌리는 순서와 정확한 명령이다. 근거는 `rebuild/DESIGN.md` 23.6이고(결정 D27–D44), 규칙과 R0의 도움말 함수는 `rebuild/RUNBOOK.md`를 그대로 쓴다. 프로파일은 `rebuild/configs/datasets/<d>_<이름>.yaml`이다(`<d>` = `lastfm`, `steam`; D43). `rebuild/tests/test_ds_runbook.py`가 이 문서의 bash 블록을 실제 CLI(`rebuild/main.py`, `rebuild/tools/*.py`)와 프로파일로 검사한다. 필수 단계는 DS0–DS9이고 DS10–DS14는 선택이다(GPU 시간이 크므로 사용자 판단, D44, DESIGN 23.9).

## 규칙

- **54e5ae0 실행 보호 (DESIGN 23.1).** 구현 commit은 54e5ae0의 MovieLens 실행이 모두 끝난 뒤(또는 사용자가 중단한 뒤)에만 VESSL workspace에 올린다. 실행 중에는 workspace의 코드를 바꾸지 않는다: `meta.json`의 코드 기록이 실제 실행을 대표해야 한다. MovieLens는 다시 돌리지 않고, 이 문서의 어떤 명령도 `rebuild/configs/*.yaml`(MovieLens 프로파일)을 쓰지 않는다.
- 모든 명령은 저장소 루트(`~/iLoRA`)에서, conda env `ilora-rebuild`, `CUDA_VISIBLE_DEVICES=0`으로 돌린다 (R1의 env를 그대로 쓴다) [SPEC §0.12]. GPU 작업은 한 번에 하나만. 구현 commit을 올린 뒤의 코드 갱신은 R1 "실패하면"의 첫 항목(`git fetch origin`, `git checkout exp/gate-cluster`, `git pull --ff-only`)으로 한다.
- 각 단계는 "명령 → 산출물 → 확인"이다. 확인이 실패하면 아래 "실패 시 멈춤"대로 다음 단계로 가지 않고 원인을 기록한다(그 실행 폴더의 `meta.json` `error`, `run.log`). 결과 수치는 측정한 뒤에만 적는다 [SPEC §0.11]. 시간·메모리·비용은 측정 전 TBD다(DESIGN 23.9의 값은 추정).
- 실행 폴더는 그것을 만든 명령 뒤에 `&&`로 이어 `latest <프로파일>`로 잡고, `done_ok <폴더> <명령>`으로 `meta.json` 상태와 `command`를 확인한 뒤 `rec`로 기록한다(`RUNBOOK.md`와 같음). 명령이 실패하면(폴더를 만들기 전의 `ConfigError`, 종료 코드 2 포함) 사슬이 멈추므로 앞 단계의 폴더를 이 단계의 것으로 기록하지 않는다.
- 단계 사이의 경로는 `rec`가 기록한 `ILORA_RUN_<DATASET>_<단계>`로만 넘긴다(`<DATASET>` = `LASTFM`, `STEAM`; 예: `ILORA_RUN_STEAM_E2`). 같은 `runs/rebuild/runbook_paths.sh`에 쓰지만 이름이 달라 MovieLens 경로 기록(`ILORA_RUN_E2` 등)을 덮어쓰지 않는다. 뒤 단계의 명령은 그 이름을 직접 읽고, 같은 줄 앞에서 `need <이름>...`으로 기록되어 있는지 확인한다. 셸이 끊겨도 DS0을 다시 하면 `source`로 경로가 돌아온다.
- 실행 폴더는 덮어쓰지 않는다. 실패한 폴더도 지우지 않고 기록으로 둔다(다시 돌리면 새 폴더가 생김). 이 문서에는 삭제 명령이 없다.
- 완료 판단은 완료 표시로만 한다: `meta.json` 상태, 학습 checkpoint의 `COMPLETE`, 평가 폴더의 `metrics.json`, smoke의 `smoke_report.json` [SPEC §17]. 파일이 있다는 것만으로 단계를 건너뛰지 않는다.
- 비밀값: 이 문서의 명령은 token을 읽지도 쓰지도 않는다. Llama-2 내려받기와 `HF_TOKEN`은 `RUNBOOK.md` R1에만 있다.
- 긴 실행(DS4 이후)은 tmux 안에서 한다(`tmux new -s rebuild` 뒤 DS0). 재개 조건(R7), tmux, 보관(R16)은 `RUNBOOK.md`를 그대로 따른다. 보관 묶음은 `runs/rebuild` 전체이므로 이 문서의 실행 폴더도 함께 들어간다.
- 비교표는 데이터셋 안에서만 만든다: `tools/compare_runs.py`가 각 실행의 `prompt.template_path`로 데이터셋을 읽어 표 D와 한계 문구를 고르고, 데이터셋이 섞이면 거부한다(DESIGN 23.5 M14).

## 실행 순서와 실패 시 멈춤

순서 (DESIGN 23.6):

1. 54e5ae0의 MovieLens 실행이 끝난다(DESIGN 23.1).
2. 구현 commit을 VESSL에 올리고 DS1을 LastFM과 Steam 둘 다 돌린다(CPU, 몇 분).
3. DS2(시험 묶음)를 돌린다(`RUNBOOK.md`의 R2 → R3 순서와 같다).
4. LastFM DS3–DS9. 가장 작아 새 코드 경로를 싸게 확인한다(T 1,870).
5. Steam DS3–DS9는 LastFM DS3–DS6이 통과한 뒤 시작한다.

판정은 완료 표시와 기록 수(`done_ok`, `dsoutputs`)이지 정확도가 아니다. 선택 단계(DS10–DS14)는 각 데이터셋의 DS6·DS7 뒤에 사용자 판단으로 한다. DS12는 DS11, DS13은 DS12를 전제로 한다. 문서에는 필수 단계(DS3–DS9)를 LastFM, Steam 순서로 적고 선택 단계를 그 뒤에 적는다.

실패 시 멈춤 (DESIGN 23.6). `RUNBOOK.md` 규칙과 같다. 단계의 확인이 실패하면 그 데이터셋의 다음 단계를 돌리지 않고 원인을 기록한다(`meta.json` `error`, `run.log`). 어느 데이터셋의 DS1 또는 DS2가 실패하면 DS3 이후를 시작하지 않는다. LastFM DS3·DS4·DS5가 실패하면 Steam DS3 이후를 시작하지 않는다(공통 코드 경로). LastFM E2의 학습 실패가 데이터셋 고유 원인(예: lr 2e-4의 fp16 불안정, D9 규칙)이면 원인을 기록하고 Steam 진행은 사용자가 정한다. GradScaler skip 연속 16회, 비유한 parameter는 D9 규칙대로 멈춘다(DS6의 "실패하면", 문서 끝 "bf16 대안" 절).

## 단계 한눈에 보기

| 단계 | 내용 | 프로파일 (`<d>_` 뒤) | 기록하는 경로 변수 (`ILORA_RUN_<D>_` 뒤) | 확인 |
|---|---|---|---|---|
| DS0 | 셸 준비 (매 셸) | - | - | R0과 같음 |
| DS1 | SASRec 변환 (LastFM, Steam 둘 다, 그 기계에서 한 번) | - | - | 종료 코드 0, manifest digest = 프로파일 `sasrec.state_digest` |
| DS2 | 시험 묶음 (새 commit에서 한 번, DS1 뒤) | - | - | pytest 실패 0 |
| DS3 | 환경 점검 `check-env` | `paper_spec` | `CHECK_ENV` | `check_env.json` passed, SASRec 기준 대조 |
| DS4 | smoke 2단계 (GPU 100명) | `smoke_gpu_100` | `SMOKE_GPU` | `dsoutputs <d>_smoke_gpu` |
| DS5 | smoke 3단계 (과적합) | `overfit_gpu` | `OVERFIT` | `dsoutputs <d>_overfit` |
| DS6 | E2 학습 | `paper_spec` | `E2` | `dsoutputs <d>_e2_run` |
| DS7 | E2 재평가 2회 | `paper_spec` | `E2_REEVAL`, `E2_REEVAL_BI` | `dsoutputs <d>_e2_reeval` |
| DS8 | E1 학습 | `llara_baseline` | `E1` | `dsoutputs <d>_e1` |
| DS9 | 비교표 | `compare_runs.py` | `COMPARE` | `done_ok`, 빠진 실험 `NOT_RUN` |
| DS10 | E3 평가 (선택) | `variant_text_only_eval` | `E3` | R9와 같음 |
| DS11 | E2 export·gate 분석 (선택, DS12의 전제) | `paper_spec` | `E2_EXPORT`, `E2_BEST` | `dsoutputs <d>_e2_export` |
| DS12 | 군집 (선택, DS13의 전제) | `paper_spec` | `CLUSTER` | `dsoutputs <d>_cluster` |
| DS13 | E4 학습·export (선택) | `cluster_hard` | `E4`, `E4_EXPORT` | `dsoutputs <d>_e4` |
| DS14 | 변형, 비교표 다시 (선택) | `variant_released_lr`, `compare_runs.py` | `V_RELEASED_LR`, `COMPARE_ALL` | `done_ok` |

E2를 먼저 하는 이유: 주 재현이고 E1·재평가·E4가 E2에 의존하거나 비교된다(`RUNBOOK.md`의 E2 우선과 같음).

## DS0. 매 셸 준비

`RUNBOOK.md` R0과 같고 `dsoutputs`(새 시험 묶음)만 더한다. workspace에서 처음이면 R1로 clone과 env를 먼저 만든다.

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
dsoutputs() { ILORA_VESSL=1 python -I -m pytest rebuild/tests/vessl/test_ds_run_outputs.py -q -k "$1"; }
```

- `latest`, `done_ok`, `rec`, `need`, `outputs`의 뜻은 `RUNBOOK.md` R0과 같다. 프로파일 이름에 데이터셋이 들어 있으므로 `latest lastfm_paper_spec`의 폴더는 `runs/rebuild/lastfm_paper_spec_<UTC>`이고 MovieLens의 `latest paper_spec`과 겹치지 않는다(D43).
- `dsoutputs <데이터셋>_<단계>`: 그 단계의 실행 폴더 검사(`test_ds_run_outputs.py -k <데이터셋>_<단계>`, 시험 이름이 `<데이터셋>_<단계>`를 품는다, 예: `dsoutputs lastfm_e2_run`). `ILORA_RUN_<DATASET>_…`으로 실행 폴더를 찾는다. 측정값은 `runs/rebuild/vessl_tests_<UTC>/measurements.json`에 남는다(판정 없는 값은 TBD).
- 확인: `python --version`이 3.10, `echo "$CONDA_DEFAULT_ENV"`가 `ilora-rebuild`.

## DS1. SASRec 변환 (LastFM, Steam)

두 데이터셋 모두 이 기계에서 한 번, DS2 전에(시험 묶음이 변환본을 찾는다).

```bash
git rev-parse HEAD && git status --porcelain       # also recorded in every meta.json
test "$(git rev-parse --is-shallow-repository)" = false
git cat-file -e 28e47371192121b6ced44401d9e149fedfae13de^{commit}
python -I rebuild/tools/convert_sasrec.py --repo . --dataset lastfm --out runs/rebuild/assets/sasrec_lastfm
python -I rebuild/tools/convert_sasrec.py --repo . --dataset steam --out runs/rebuild/assets/sasrec_steam
```

- 산출물: `runs/rebuild/assets/sasrec_<d>/{sasrec_<d>_state.pt, sasrec_<d>_reference.pt, manifest.json}` (DESIGN 7.6.1, 23.5 M3). CPU만, 몇 분.
- 확인: 종료 코드 0, `manifest.json`의 digest = 프로파일 `sasrec.state_digest` (DESIGN 23.2: LastFM `6b0d1613…5d85`, Steam `d5d13fc0…3932`). DS3 check-env가 다시 대조한다. `git status --porcelain`이 비어 있어야 `meta.json`의 commit이 실행한 코드를 대표한다(R2와 같음).
- 변환은 이 VESSL 기계에서 한다: CPU 기준 출력의 bitwise 일치는 같은 기계에서 만든 기준 파일에서만 보장된다(DESIGN 7.6.5, R2와 같음). 출력 폴더가 이미 있으면 도구가 거부한다(덮어쓰기 금지): 같은 기계에서 앞서 만든 변환본이면 그대로 쓰고 DS3이 digest를 확인한다. 군집(DS12) 이후에는 변환을 다시 하지 않는다(M15: E4는 군집 산출물에 기록된 state 파일 sha256과 같은 파일을 요구한다).

실패하면: 어느 데이터셋이든 실패하면 DS3 이후를 시작하지 않는다. `is-shallow-repository`가 `true`이거나 `cat-file`이 실패하면 `git fetch --unshallow` 뒤 다시 한다(R2와 같음).

## DS2. 시험 묶음

새 commit에서 한 번, DS1 뒤에. `RUNBOOK.md` R3과 같은 두 명령이다.

```bash
python -I -m pytest rebuild/tests -q                                   # CPU bundle (Linux, real tokenizer)
ILORA_VESSL=1 python -I -m pytest rebuild/tests/vessl -q -k "not run_outputs"
```

- 산출물: pytest 요약, `runs/rebuild/vessl_tests_<UTC>/measurements.json` (DESIGN 17.6.2).
- 확인: 두 묶음 모두 실패 0. skip 규칙은 R3과 같다: 두 데이터셋의 DS1 변환본과 `ILORA_TEST_TOKENIZER`가 있으므로 CPU 묶음의 skip은 `VESSL_REQUIRED`뿐이고(요약 끝 `SKIPPED` 줄의 이유로 확인), VESSL 묶음의 skip은 0이다(`SKIP_NO_PL`이면 그 이유가 `measurements.json`에 남음). `-k "not run_outputs"`는 `test_run_outputs.py`와 `test_ds_run_outputs.py`를 둘 다 뺀다(실행 폴더 검사는 각 단계의 `dsoutputs`가 돌린다). MovieLens 시험은 R2의 변환본(`runs/rebuild/assets/sasrec`)을 쓰므로, R2를 하지 않은 새 workspace면 R1·R2를 먼저 한다(없으면 `SKIP_NO_SASREC`로 빠져 이 skip 규칙을 어긴다).

실패하면: DS3 이후를 시작하지 않는다. 실패한 시험 이름과 메시지를 기록하고, 원인이 환경(판, 경로, 자산)이면 고친 뒤 DS2를 다시 한다. 시험을 건너뛰거나 기준을 바꾸지 않는다.

## DS3 (LastFM). 환경 점검

```bash
python -I rebuild/main.py check-env --config rebuild/configs/datasets/lastfm_paper_spec.yaml \
  && C=$(latest lastfm_paper_spec) && done_ok "$C" check-env && rec ILORA_RUN_LASTFM_CHECK_ENV "$C" && python -m pip freeze > "$C/pip_freeze.txt"
```

- 산출물: 실행 폴더의 `check_env.json`, `meta.json`, env 판 기록 `pip_freeze.txt` (R4와 같음).
- 확인: 종료 코드 0, 상태 `completed`, `check_env.json`의 `passed` true. 내용은 R4와 같고(GPU 1장·75 GiB 이상, 판 고정, transformers 파일 무결성, `LLM_PATH`·`torch_dtype` float16, tokenizer sha256, `candidates.self_check()`), 입력 해시는 LastFM 표(`run.expected_inputs("lastfm")`), SASRec은 DS1 변환본의 digest `6b0d1613…5d85`와 기준 대조 CUDA ≤ 1e-4 (DESIGN 7.8 #12).

실패하면: DS4로 가지 않는다. `check_env.json`의 `checks`와 `meta.json` `error`가 실패 항목을 나열한다(R4와 같음). LastFM DS3·DS4·DS5가 실패하면 Steam DS3 이후도 시작하지 않는다(공통 코드 경로, DESIGN 23.6).

## DS4 (LastFM). smoke 2단계 (GPU 100명)

```bash
python -I rebuild/main.py smoke --config rebuild/configs/datasets/lastfm_smoke_gpu_100.yaml \
  && S=$(latest lastfm_smoke_gpu_100) && done_ok "$S" smoke && rec ILORA_RUN_LASTFM_SMOKE_GPU "$S" && dsoutputs lastfm_smoke_gpu
```

- 산출물: `smoke_report.json`, `steps/` (R5와 같음: G0 check-env, G1 학습 2 epoch, G2 재평가, G3 export, G4 군집, G5 E4 경로 1 epoch, G6 E1 경로 1 epoch; DESIGN 17.6.7).
- 확인: `smoke_report.json`의 `passed` true(하나라도 실패하면 `main.py smoke`가 0이 아닌 종료 코드, 상태 `failed`), `dsoutputs lastfm_smoke_gpu` 통과. 기록 수: train 표본 3,626 (DESIGN 23.6), micro 906, step 58 (epoch 끝 `n_micro` 5), epoch 2, `gate_calls` 합 906, E4 경로 `cluster_lookups` 3,624. 메모리·시간은 기록만(TBD).

실패하면: DS5 이후를 시작하지 않는다. 폴더(`latest lastfm_smoke_gpu_100`)의 `smoke_report.json` `checks`에서 실패한 항목과 `steps/<단계>/meta.json` `error`를 기록한다. LastFM DS3·DS4·DS5가 실패하면 Steam DS3 이후도 시작하지 않는다(공통 코드 경로, DESIGN 23.6).

## DS5 (LastFM). smoke 3단계 (과적합)

```bash
python -I rebuild/main.py smoke --config rebuild/configs/datasets/lastfm_overfit_gpu.yaml \
  && O=$(latest lastfm_overfit_gpu) && done_ok "$O" smoke && rec ILORA_RUN_LASTFM_OVERFIT "$O" && dsoutputs lastfm_overfit
```

- 산출물: `smoke_report.json`, `steps/` (R6과 같음: O0 check-env, O1 학습 25 epoch, O2 과적합 생성 점검). 표본은 DESIGN 23.6의 `data.train_rows` 16개(마지막 창의 이력 길이가 10인 앞 사용자 16명의 마지막 창).
- 확인: DESIGN 17.6.8과 같은 기준이다. `passed` true: 마지막 epoch 평균 loss ≤ 첫 epoch × 0.1, 16/16 첫 줄이 정답 제목과 일치(`match_norm`), micro 100행 모두 hybrid. `dsoutputs lastfm_overfit` 통과. loss 비와 일치 수는 기록만.

실패하면: DS6 이후를 시작하지 않는다(R6과 같음). LastFM DS3·DS4·DS5가 실패하면 Steam DS3 이후도 시작하지 않는다(공통 코드 경로, DESIGN 23.6).

## DS6 (LastFM). E2 학습

tmux 안에서 (`tmux new -s rebuild` 뒤 DS0).

```bash
python -I rebuild/main.py train --config rebuild/configs/datasets/lastfm_paper_spec.yaml \
  && E2=$(latest lastfm_paper_spec) && done_ok "$E2" train && rec ILORA_RUN_LASTFM_E2 "$E2" && dsoutputs lastfm_e2_run
```

- 산출물: `checkpoints/epoch0..4/` (각 `COMPLETE`), `eval/epoch0..4/val/`, `eval/epoch{a,b}/test/` (a = best_val, b = last), `eval/selection.json`, `records/train/` (R7과 같음).
- 확인: `dsoutputs lastfm_e2_run` (`test_ds_run_outputs.py`) 통과. 기록 수 micro 29,920, step 1,870 (epoch 끝 창 없음: 5,984 = 16 × 374), epoch 5 (DESIGN 23.2의 `step_counts`). 평가 N validation 122, test 122 (DESIGN 23.2), `candidates_sha256`가 DESIGN 23.2 표와 같고 `eval/selection.json`이 있음.
- 중단되면: R7의 조건(학습 프로세스가 끝났음: `pgrep`·`nvidia-smi` 확인, `run.lock`을 쥔 프로세스 없음)에서만 `R=$(latest lastfm_paper_spec) && python -I rebuild/main.py train --resume "$R" && done_ok "$R" train && rec ILORA_RUN_LASTFM_E2 "$R" && dsoutputs lastfm_e2_run`.

실패하면: `meta.json` `error`와 `run.log`의 마지막 `train step` 줄을 기록하고 DS7로 가지 않는다. GradScaler skip이 연속 16회(`train.max_consecutive_skips`)면 학습이 멈춘다(DESIGN 14.6.8). fp16이 이 규칙에 걸리거나 parameter가 비유한일 때만 문서 끝 "bf16 대안" 절의 LastFM 블록을 쓰고 편차로 기록한다(D9). LastFM lr 2e-4의 fp16 안정성은 측정 전이다(D9, DESIGN 23.3): 이 학습의 실패가 데이터셋 고유 원인이면 원인을 기록하고 Steam 진행은 사용자가 정한다(DESIGN 23.6).

## DS7 (LastFM). E2 재평가 2회 (결정성, batch 무관성)

`RUNBOOK.md` R8과 같은 형태다.

```bash
need ILORA_RUN_LASTFM_E2 && python -I rebuild/main.py eval --config rebuild/configs/datasets/lastfm_paper_spec.yaml --set eval.source_run="$ILORA_RUN_LASTFM_E2" \
  && R=$(latest lastfm_paper_spec) && done_ok "$R" eval && rec ILORA_RUN_LASTFM_E2_REEVAL "$R"
need ILORA_RUN_LASTFM_E2 && python -I rebuild/main.py eval --config rebuild/configs/datasets/lastfm_paper_spec.yaml --set eval.source_run="$ILORA_RUN_LASTFM_E2" \
  --set eval.batch_invariance_check=true && R=$(latest lastfm_paper_spec) && done_ok "$R" eval && rec ILORA_RUN_LASTFM_E2_REEVAL_BI "$R" \
  && dsoutputs lastfm_e2_reeval
```

- 산출물: 각 실행의 `eval/epoch{a,b}/{val,test}/`와 `eval/summary.json` (DESIGN 15.6.11). 둘째 실행은 평가 폴더마다 `batch_invariance.json`.
- 확인: `dsoutputs lastfm_e2_reeval`: 두 실행의 `records.jsonl`이 학습 중 기록과 sha256이 같고 `summary.json`의 `reproduction`이 모두 `identical`, 둘째 실행의 평가 폴더마다 `batch_invariance.json`. batch 8 대 batch 1 일치율은 기록만(TBD, D32).

실패하면: 다르면 DS8 이후의 결과를 확정하지 않는다. 시험이 출력하는 다른 표본과 처음 다른 생성 단계를 기록하고 원인을 밝힌다(R8과 같음).

## DS8 (LastFM). E1 학습

```bash
python -I rebuild/main.py train --config rebuild/configs/datasets/lastfm_llara_baseline.yaml \
  && E1=$(latest lastfm_llara_baseline) && done_ok "$E1" train && rec ILORA_RUN_LASTFM_E1 "$E1" && dsoutputs lastfm_e1
```

- 산출물: `checkpoints/epoch0..4/`, `eval/…`, `records/train/` (K = 1, 라우팅 없음. 이 데이터셋의 기준선이고 논문 표의 LLaRA 행과 짝을 이룬다, D41, D44).
- 확인: `dsoutputs lastfm_e1`: optimizer 그룹 {projector, lora}, 평가 기록 `gate_0` = 1.0, gate 호출 0, 학습 기록 수가 E2와 같음(micro 29,920, step 1,870, epoch 5), 평가 N은 DS6과 같음.
- 중단되면: R7의 조건에서만 `R=$(latest lastfm_llara_baseline) && python -I rebuild/main.py train --resume "$R" && done_ok "$R" train && rec ILORA_RUN_LASTFM_E1 "$R" && dsoutputs lastfm_e1`.

## DS9 (LastFM). 비교표

```bash
need ILORA_RUN_LASTFM_E1 ILORA_RUN_LASTFM_E2 \
  && python -I rebuild/tools/compare_runs.py --run E1="$ILORA_RUN_LASTFM_E1" --run E2="$ILORA_RUN_LASTFM_E2" \
  && CMP=$(latest compare) && done_ok "$CMP" compare && rec ILORA_RUN_LASTFM_COMPARE "$CMP"
```

- 산출물: `runs/rebuild/compare_<UTC>/` 의 `comparison.csv`, `comparison.md`, `spec20_index.json`, `summary.md` (R14와 같음). 표 D·정의·한계 문구는 LastFM 것이다(`compare_runs.py`가 실행의 `prompt.template_path`로 데이터셋을 읽음, 23.5 M14).
- 확인: 상태 `completed`. 측정 칸은 실행 파일 값이고, 빠진 실험(E3, E4, export, 군집)은 오류가 아니라 `NOT_RUN`과 이유(`comparison.md` 끝, R14와 같음). 선택 단계(DS10–DS14)를 돌렸으면 DS14의 둘째 블록으로 표를 다시 만든다.

실패하면: `compare error: ...`는 실행 파일이 스스로 모순될 때 나고(R14), 데이터셋이 섞이면 거부한다(DESIGN 23.5 M14). 그 파일을 고치지 말고 기록한다.

## DS3 (Steam). 환경 점검

시작 조건: LastFM DS3–DS6이 통과한 뒤에 시작한다(위 실행 순서, DESIGN 23.6). 이 단계들을 건너뛰지 않는다.

```bash
python -I rebuild/main.py check-env --config rebuild/configs/datasets/steam_paper_spec.yaml \
  && C=$(latest steam_paper_spec) && done_ok "$C" check-env && rec ILORA_RUN_STEAM_CHECK_ENV "$C" && python -m pip freeze > "$C/pip_freeze.txt"
```

- 산출물: 실행 폴더의 `check_env.json`, `meta.json`, env 판 기록 `pip_freeze.txt` (R4와 같음).
- 확인: 종료 코드 0, 상태 `completed`, `check_env.json`의 `passed` true. 내용은 R4와 같고(GPU 1장·75 GiB 이상, 판 고정, transformers 파일 무결성, `LLM_PATH`·`torch_dtype` float16, tokenizer sha256, `candidates.self_check()`), 입력 해시는 Steam 표(`run.expected_inputs("steam")`), SASRec은 DS1 변환본의 digest `d5d13fc0…3932`와 기준 대조 CUDA ≤ 1e-4 (DESIGN 7.8 #12).

실패하면: DS4로 가지 않는다. `check_env.json`의 `checks`와 `meta.json` `error`가 실패 항목을 나열한다(R4와 같음).

## DS4 (Steam). smoke 2단계 (GPU 100명)

```bash
python -I rebuild/main.py smoke --config rebuild/configs/datasets/steam_smoke_gpu_100.yaml \
  && S=$(latest steam_smoke_gpu_100) && done_ok "$S" smoke && rec ILORA_RUN_STEAM_SMOKE_GPU "$S" && dsoutputs steam_smoke_gpu
```

- 산출물: `smoke_report.json`, `steps/` (R5와 같음: G0 check-env, G1 학습 2 epoch, G2 재평가, G3 export, G4 군집, G5 E4 경로 1 epoch, G6 E1 경로 1 epoch; DESIGN 17.6.7).
- 확인: `smoke_report.json`의 `passed` true(하나라도 실패하면 `main.py smoke`가 0이 아닌 종료 코드, 상태 `failed`), `dsoutputs steam_smoke_gpu` 통과. 기록 수: train 표본 1,385 (DESIGN 23.6), micro 692, step 22 (epoch 끝 `n_micro` 26), epoch 2, `gate_calls` 합 692, E4 경로 `cluster_lookups` 1,384. 메모리·시간은 기록만(TBD).

실패하면: DS5 이후를 시작하지 않는다. 폴더(`latest steam_smoke_gpu_100`)의 `smoke_report.json` `checks`에서 실패한 항목과 `steps/<단계>/meta.json` `error`를 기록한다.

## DS5 (Steam). smoke 3단계 (과적합)

```bash
python -I rebuild/main.py smoke --config rebuild/configs/datasets/steam_overfit_gpu.yaml \
  && O=$(latest steam_overfit_gpu) && done_ok "$O" smoke && rec ILORA_RUN_STEAM_OVERFIT "$O" && dsoutputs steam_overfit
```

- 산출물: `smoke_report.json`, `steps/` (R6과 같음: O0 check-env, O1 학습 25 epoch, O2 과적합 생성 점검). 표본은 DESIGN 23.6의 `data.train_rows` 16개(마지막 창의 이력 길이가 10인 앞 사용자 16명의 마지막 창).
- 확인: DESIGN 17.6.8과 같은 기준이다. `passed` true: 마지막 epoch 평균 loss ≤ 첫 epoch × 0.1, 16/16 첫 줄이 정답 제목과 일치(`match_norm`), micro 100행 모두 hybrid. `dsoutputs steam_overfit` 통과. loss 비와 일치 수는 기록만.

실패하면: DS6 이후를 시작하지 않는다(R6과 같음).

## DS6 (Steam). E2 학습

tmux 안에서 (`tmux new -s rebuild` 뒤 DS0).

```bash
python -I rebuild/main.py train --config rebuild/configs/datasets/steam_paper_spec.yaml \
  && E2=$(latest steam_paper_spec) && done_ok "$E2" train && rec ILORA_RUN_STEAM_E2 "$E2" && dsoutputs steam_e2_run
```

- 산출물: `checkpoints/epoch0..4/` (각 `COMPLETE`), `eval/epoch0..4/val/`, `eval/epoch{a,b}/test/` (a = best_val, b = last), `eval/selection.json`, `records/train/` (R7과 같음).
- 확인: `dsoutputs steam_e2_run` (`test_ds_run_outputs.py`) 통과. 기록 수 micro 188,810, step 5,905 (epoch 끝 `n_micro` 2: 37,762 = 32 × 1,180 + 2), epoch 5 (DESIGN 23.2의 `step_counts`). 평가 N validation 1,184, test 1,186 (DESIGN 23.2), `candidates_sha256`가 DESIGN 23.2 표와 같고 `eval/selection.json`이 있음.
- 중단되면: R7의 조건(학습 프로세스가 끝났음: `pgrep`·`nvidia-smi` 확인, `run.lock`을 쥔 프로세스 없음)에서만 `R=$(latest steam_paper_spec) && python -I rebuild/main.py train --resume "$R" && done_ok "$R" train && rec ILORA_RUN_STEAM_E2 "$R" && dsoutputs steam_e2_run`.

실패하면: `meta.json` `error`와 `run.log`의 마지막 `train step` 줄을 기록하고 DS7로 가지 않는다. GradScaler skip이 연속 16회(`train.max_consecutive_skips`)면 학습이 멈춘다(DESIGN 14.6.8). fp16이 이 규칙에 걸리거나 parameter가 비유한일 때만 문서 끝 "bf16 대안" 절의 Steam 블록을 쓰고 편차로 기록한다(D9).

## DS7 (Steam). E2 재평가 2회 (결정성, batch 무관성)

`RUNBOOK.md` R8과 같은 형태다.

```bash
need ILORA_RUN_STEAM_E2 && python -I rebuild/main.py eval --config rebuild/configs/datasets/steam_paper_spec.yaml --set eval.source_run="$ILORA_RUN_STEAM_E2" \
  && R=$(latest steam_paper_spec) && done_ok "$R" eval && rec ILORA_RUN_STEAM_E2_REEVAL "$R"
need ILORA_RUN_STEAM_E2 && python -I rebuild/main.py eval --config rebuild/configs/datasets/steam_paper_spec.yaml --set eval.source_run="$ILORA_RUN_STEAM_E2" \
  --set eval.batch_invariance_check=true && R=$(latest steam_paper_spec) && done_ok "$R" eval && rec ILORA_RUN_STEAM_E2_REEVAL_BI "$R" \
  && dsoutputs steam_e2_reeval
```

- 산출물: 각 실행의 `eval/epoch{a,b}/{val,test}/`와 `eval/summary.json` (DESIGN 15.6.11). 둘째 실행은 평가 폴더마다 `batch_invariance.json`.
- 확인: `dsoutputs steam_e2_reeval`: 두 실행의 `records.jsonl`이 학습 중 기록과 sha256이 같고 `summary.json`의 `reproduction`이 모두 `identical`, 둘째 실행의 평가 폴더마다 `batch_invariance.json`. batch 4 대 batch 1 일치율은 기록만(TBD, D32).

실패하면: 다르면 DS8 이후의 결과를 확정하지 않는다. 시험이 출력하는 다른 표본과 처음 다른 생성 단계를 기록하고 원인을 밝힌다(R8과 같음).

## DS8 (Steam). E1 학습

```bash
python -I rebuild/main.py train --config rebuild/configs/datasets/steam_llara_baseline.yaml \
  && E1=$(latest steam_llara_baseline) && done_ok "$E1" train && rec ILORA_RUN_STEAM_E1 "$E1" && dsoutputs steam_e1
```

- 산출물: `checkpoints/epoch0..4/`, `eval/…`, `records/train/` (K = 1, 라우팅 없음. 이 데이터셋의 기준선이고 논문 표의 LLaRA 행과 짝을 이룬다, D41, D44).
- 확인: `dsoutputs steam_e1`: optimizer 그룹 {projector, lora}, 평가 기록 `gate_0` = 1.0, gate 호출 0, 학습 기록 수가 E2와 같음(micro 188,810, step 5,905, epoch 5), 평가 N은 DS6과 같음.
- 중단되면: R7의 조건에서만 `R=$(latest steam_llara_baseline) && python -I rebuild/main.py train --resume "$R" && done_ok "$R" train && rec ILORA_RUN_STEAM_E1 "$R" && dsoutputs steam_e1`.

## DS9 (Steam). 비교표

```bash
need ILORA_RUN_STEAM_E1 ILORA_RUN_STEAM_E2 \
  && python -I rebuild/tools/compare_runs.py --run E1="$ILORA_RUN_STEAM_E1" --run E2="$ILORA_RUN_STEAM_E2" \
  && CMP=$(latest compare) && done_ok "$CMP" compare && rec ILORA_RUN_STEAM_COMPARE "$CMP"
```

- 산출물: `runs/rebuild/compare_<UTC>/` 의 `comparison.csv`, `comparison.md`, `spec20_index.json`, `summary.md` (R14와 같음). 표 D·정의·한계 문구는 Steam 것이다(`compare_runs.py`가 실행의 `prompt.template_path`로 데이터셋을 읽음, 23.5 M14).
- 확인: 상태 `completed`. 측정 칸은 실행 파일 값이고, 빠진 실험(E3, E4, export, 군집)은 오류가 아니라 `NOT_RUN`과 이유(`comparison.md` 끝, R14와 같음). 선택 단계(DS10–DS14)를 돌렸으면 DS14의 둘째 블록으로 표를 다시 만든다.

실패하면: `compare error: ...`는 실행 파일이 스스로 모순될 때 나고(R14), 데이터셋이 섞이면 거부한다(DESIGN 23.5 M14). 그 파일을 고치지 말고 기록한다.

## DS10 (LastFM). E3 (E2 checkpoint의 text-only 평가, 선택)

```bash
need ILORA_RUN_LASTFM_E2 && python -I rebuild/main.py eval --config rebuild/configs/datasets/lastfm_variant_text_only_eval.yaml --set eval.source_run="$ILORA_RUN_LASTFM_E2" \
  && E3=$(latest lastfm_variant_text_only_eval) && done_ok "$E3" eval && rec ILORA_RUN_LASTFM_E3 "$E3"
need ILORA_RUN_LASTFM_E3 && python -I -c 'import json,sys; r=json.load(open(sys.argv[1]+"/eval/summary.json"))["results"]; s={k: v["reproduction"]["status"] for k, v in r.items()}; print(s); sys.exit(set(s.values())!={"not_comparable"})' "$ILORA_RUN_LASTFM_E3"
need ILORA_RUN_LASTFM_E3 && python -I -c 'import glob,json,sys; h=[json.loads(x)["hybrid"] for p in glob.glob(sys.argv[1]+"/eval/*/*/records.jsonl") for x in open(p, encoding="utf-8")]; print(len(h), "records, hybrid values", set(h)); sys.exit(not h or any(h))' "$ILORA_RUN_LASTFM_E3"
```

- 산출물: `eval/epoch{a,b}/{val,test}/`, `eval/summary.json` (평가 prompt만 text_only, D4).
- 확인: R9와 같음: 상태 `completed`, 기록의 `hybrid`가 모두 false, `reproduction`이 모두 `not_comparable`. 두 확인 명령이 0으로 끝남.

## DS11 (LastFM). E2 export와 gate 분석 (선택, DS12의 전제)

```bash
need ILORA_RUN_LASTFM_E2 && python -I rebuild/main.py export --config rebuild/configs/datasets/lastfm_paper_spec.yaml --set records.source_run="$ILORA_RUN_LASTFM_E2" \
  && E2X=$(latest lastfm_paper_spec) && done_ok "$E2X" export && rec ILORA_RUN_LASTFM_E2_EXPORT "$E2X" \
  && A=$(python -I -c 'import json,sys; print(json.load(open(sys.argv[1]+"/eval/selection.json"))["best_val_epoch"])' "$ILORA_RUN_LASTFM_E2") \
  && rec ILORA_RUN_LASTFM_E2_BEST "$A" && dsoutputs lastfm_e2_export
```

- 산출물: `records/export/epoch{a,b}/`, `records/analysis/epoch{a,b}/{test,val,train}/` (R10과 같음). `ILORA_RUN_LASTFM_E2_BEST`는 실행 폴더가 아니라 best_val epoch 번호이고 DS12가 읽는다(`RUNBOOK.md`의 `ILORA_E2_BEST`와 같은 값).
- 확인: `dsoutputs lastfm_e2_export`: 대조 통과(|Δω| ≤ 1e-5), 파일 sha256이 manifest와 같음, NPZ 행 수 train 47,872 · validation 122 · test 122, 그림·요약 존재 (R10과 같음).

실패하면: DS12를 시작하지 않는다(R10의 대조 항목을 그대로 기록한다).

## DS12 (LastFM). 군집 (train만 fit, 선택, DS13의 전제)

```bash
need ILORA_RUN_LASTFM_E2_EXPORT ILORA_RUN_LASTFM_E2_BEST && python -I rebuild/main.py cluster --config rebuild/configs/datasets/lastfm_paper_spec.yaml \
  --set cluster.dynamic_export_dir="$ILORA_RUN_LASTFM_E2_EXPORT/records/export/epoch$ILORA_RUN_LASTFM_E2_BEST" \
  && CL=$(latest lastfm_paper_spec) && done_ok "$CL" cluster && rec ILORA_RUN_LASTFM_CLUSTER "$CL" && dsoutputs lastfm_cluster
```

- 산출물: `cluster/` (`kmeans_k4_seed1234.joblib`, `centroids.npz`, 배정 CSV 3개, `cluster_summary.json`, `figures/`, `manifest.json`; R11과 같음).
- 확인: `dsoutputs lastfm_cluster`: manifest `fit_split` train, fit 47,872행, 빈 군집 없음, E2 export z와 대조 ≤ 1e-5, dynamic 비교 항목 존재. validation·test는 predict만 한다 [SPEC §0.8].

실패하면: DS13을 시작하지 않는다. z 대조가 실패하면 DS1의 변환본과 E2 export가 같은 SASRec state를 썼는지(`manifest.json`의 `sasrec`) 확인한다.

## DS13 (LastFM). E4 학습과 export (선택)

```bash
need ILORA_RUN_LASTFM_CLUSTER && python -I rebuild/main.py train --config rebuild/configs/datasets/lastfm_cluster_hard.yaml \
  --set cluster.artifact_dir="$ILORA_RUN_LASTFM_CLUSTER/cluster" \
  && E4=$(latest lastfm_cluster_hard) && done_ok "$E4" train && rec ILORA_RUN_LASTFM_E4 "$E4"
need ILORA_RUN_LASTFM_CLUSTER ILORA_RUN_LASTFM_E4 && python -I rebuild/main.py export --config rebuild/configs/datasets/lastfm_cluster_hard.yaml \
  --set cluster.artifact_dir="$ILORA_RUN_LASTFM_CLUSTER/cluster" --set records.source_run="$ILORA_RUN_LASTFM_E4" \
  && E4X=$(latest lastfm_cluster_hard) && done_ok "$E4X" export && rec ILORA_RUN_LASTFM_E4_EXPORT "$E4X" && dsoutputs lastfm_e4
```

- E2 학습 가중치를 쓰지 않고 처음부터 학습한다 [SPEC §9, §25]. `cluster.artifact_dir`는 명령행으로만 준다(빠지면 `ConfigError`, DESIGN 23.6).
- 확인: `dsoutputs lastfm_e4`: 학습 시작 때 live z 대조 통과(`meta.json` `cluster_live_check`), 학습 `gate_calls` 합 0·`cluster_lookups` 합 239,360 (= micro 29,920 × 8), 평가 gate가 one-hot = mapping[배정], export의 ω가 세 split 모두 mapping[배정]의 one-hot이고 대조 통과 (R12와 같음).
- 중단되면: R7의 조건에서만 `R=$(latest lastfm_cluster_hard) && python -I rebuild/main.py train --resume "$R" && done_ok "$R" train && rec ILORA_RUN_LASTFM_E4 "$R"`, 그다음 위 블록의 둘째 줄(export)부터 한다.

실패하면: 군집 산출물이 바뀌었거나(파일 sha256), SASRec state가 다르면 학습 전에 `ClusterError`로 멈춘다. 군집을 다시 fit하지 말고 원인을 기록한다(R12와 같음).

## DS14 (LastFM). 변형과 비교표 다시 (선택)

```bash
python -I rebuild/main.py train --config rebuild/configs/datasets/lastfm_variant_released_lr.yaml \
  && VR=$(latest lastfm_variant_released_lr) && done_ok "$VR" train && rec ILORA_RUN_LASTFM_V_RELEASED_LR "$VR"
```

- 변형은 공개 스크립트 lr(최대 7e-4, 시작 7e-6, 최솟값 7e-6, D29, D30)이고 부모(`lastfm_paper_spec`)와 다른 키는 `train.lr_peak`, `train.lr_warmup_start`, `train.lr_floor`뿐이다(DESIGN 23.6). 고르기 전에 그 GPU 시간을 예산에 넣는다. 같은 공개 스크립트 계열의 MovieLens lr 8e-4는 fork에서 발산했다(R15의 기록). 같은 일이 생기면 GradScaler skip 16번 뒤 `TrainingError`로 멈추고(상태 `failed`) `done_ok`에서 사슬이 멈춰 기록되지 않는다. 그것도 D29의 결과다: bf16으로 다시 돌리지 않고(D29와 D9가 섞임) 실패한 폴더를 비교에 그대로 넘긴다(아래 명령의 `need`에서 `ILORA_RUN_LASTFM_V_RELEASED_LR`을 빼고 `--run V_released_lr="$(latest lastfm_variant_released_lr)"`로 바꾼다. 그 칸은 `NOT_RUN`과 `TrainingError` 이유).
- 선택 단계를 돌렸으면 비교표를 다시 만든다(R14, R15와 같은 방식). 돌리지 않은 실험의 이름은 `need`와 `--run`/`--export`/`--cluster`에서 뺀다(빠진 실행은 `NOT_RUN`).

```bash
need ILORA_RUN_LASTFM_E1 ILORA_RUN_LASTFM_E2 ILORA_RUN_LASTFM_E3 ILORA_RUN_LASTFM_E4 ILORA_RUN_LASTFM_E2_EXPORT ILORA_RUN_LASTFM_E4_EXPORT ILORA_RUN_LASTFM_CLUSTER ILORA_RUN_LASTFM_V_RELEASED_LR \
  && python -I rebuild/tools/compare_runs.py --run E1="$ILORA_RUN_LASTFM_E1" --run E2="$ILORA_RUN_LASTFM_E2" --run E3="$ILORA_RUN_LASTFM_E3" \
  --run E4="$ILORA_RUN_LASTFM_E4" --run V_released_lr="$ILORA_RUN_LASTFM_V_RELEASED_LR" --export E2="$ILORA_RUN_LASTFM_E2_EXPORT" \
  --export E4="$ILORA_RUN_LASTFM_E4_EXPORT" --cluster "$ILORA_RUN_LASTFM_CLUSTER" \
  && CMP=$(latest compare) && done_ok "$CMP" compare && rec ILORA_RUN_LASTFM_COMPARE_ALL "$CMP"
```

## DS10 (Steam). E3 (E2 checkpoint의 text-only 평가, 선택)

```bash
need ILORA_RUN_STEAM_E2 && python -I rebuild/main.py eval --config rebuild/configs/datasets/steam_variant_text_only_eval.yaml --set eval.source_run="$ILORA_RUN_STEAM_E2" \
  && E3=$(latest steam_variant_text_only_eval) && done_ok "$E3" eval && rec ILORA_RUN_STEAM_E3 "$E3"
need ILORA_RUN_STEAM_E3 && python -I -c 'import json,sys; r=json.load(open(sys.argv[1]+"/eval/summary.json"))["results"]; s={k: v["reproduction"]["status"] for k, v in r.items()}; print(s); sys.exit(set(s.values())!={"not_comparable"})' "$ILORA_RUN_STEAM_E3"
need ILORA_RUN_STEAM_E3 && python -I -c 'import glob,json,sys; h=[json.loads(x)["hybrid"] for p in glob.glob(sys.argv[1]+"/eval/*/*/records.jsonl") for x in open(p, encoding="utf-8")]; print(len(h), "records, hybrid values", set(h)); sys.exit(not h or any(h))' "$ILORA_RUN_STEAM_E3"
```

- 산출물: `eval/epoch{a,b}/{val,test}/`, `eval/summary.json` (평가 prompt만 text_only, D4).
- 확인: R9와 같음: 상태 `completed`, 기록의 `hybrid`가 모두 false, `reproduction`이 모두 `not_comparable`. 두 확인 명령이 0으로 끝남.

## DS11 (Steam). E2 export와 gate 분석 (선택, DS12의 전제)

```bash
need ILORA_RUN_STEAM_E2 && python -I rebuild/main.py export --config rebuild/configs/datasets/steam_paper_spec.yaml --set records.source_run="$ILORA_RUN_STEAM_E2" \
  && E2X=$(latest steam_paper_spec) && done_ok "$E2X" export && rec ILORA_RUN_STEAM_E2_EXPORT "$E2X" \
  && A=$(python -I -c 'import json,sys; print(json.load(open(sys.argv[1]+"/eval/selection.json"))["best_val_epoch"])' "$ILORA_RUN_STEAM_E2") \
  && rec ILORA_RUN_STEAM_E2_BEST "$A" && dsoutputs steam_e2_export
```

- 산출물: `records/export/epoch{a,b}/`, `records/analysis/epoch{a,b}/{test,val,train}/` (R10과 같음). `ILORA_RUN_STEAM_E2_BEST`는 실행 폴더가 아니라 best_val epoch 번호이고 DS12가 읽는다(`RUNBOOK.md`의 `ILORA_E2_BEST`와 같은 값).
- 확인: `dsoutputs steam_e2_export`: 대조 통과(|Δω| ≤ 1e-5), 파일 sha256이 manifest와 같음, NPZ 행 수 train 151,048 · validation 1,184 · test 1,186, 그림·요약 존재 (R10과 같음).

실패하면: DS12를 시작하지 않는다(R10의 대조 항목을 그대로 기록한다).

## DS12 (Steam). 군집 (train만 fit, 선택, DS13의 전제)

```bash
need ILORA_RUN_STEAM_E2_EXPORT ILORA_RUN_STEAM_E2_BEST && python -I rebuild/main.py cluster --config rebuild/configs/datasets/steam_paper_spec.yaml \
  --set cluster.dynamic_export_dir="$ILORA_RUN_STEAM_E2_EXPORT/records/export/epoch$ILORA_RUN_STEAM_E2_BEST" \
  && CL=$(latest steam_paper_spec) && done_ok "$CL" cluster && rec ILORA_RUN_STEAM_CLUSTER "$CL" && dsoutputs steam_cluster
```

- 산출물: `cluster/` (`kmeans_k4_seed1234.joblib`, `centroids.npz`, 배정 CSV 3개, `cluster_summary.json`, `figures/`, `manifest.json`; R11과 같음).
- 확인: `dsoutputs steam_cluster`: manifest `fit_split` train, fit 151,048행, 빈 군집 없음, E2 export z와 대조 ≤ 1e-5, dynamic 비교 항목 존재. validation·test는 predict만 한다 [SPEC §0.8].

실패하면: DS13을 시작하지 않는다. z 대조가 실패하면 DS1의 변환본과 E2 export가 같은 SASRec state를 썼는지(`manifest.json`의 `sasrec`) 확인한다.

## DS13 (Steam). E4 학습과 export (선택)

```bash
need ILORA_RUN_STEAM_CLUSTER && python -I rebuild/main.py train --config rebuild/configs/datasets/steam_cluster_hard.yaml \
  --set cluster.artifact_dir="$ILORA_RUN_STEAM_CLUSTER/cluster" \
  && E4=$(latest steam_cluster_hard) && done_ok "$E4" train && rec ILORA_RUN_STEAM_E4 "$E4"
need ILORA_RUN_STEAM_CLUSTER ILORA_RUN_STEAM_E4 && python -I rebuild/main.py export --config rebuild/configs/datasets/steam_cluster_hard.yaml \
  --set cluster.artifact_dir="$ILORA_RUN_STEAM_CLUSTER/cluster" --set records.source_run="$ILORA_RUN_STEAM_E4" \
  && E4X=$(latest steam_cluster_hard) && done_ok "$E4X" export && rec ILORA_RUN_STEAM_E4_EXPORT "$E4X" && dsoutputs steam_e4
```

- E2 학습 가중치를 쓰지 않고 처음부터 학습한다 [SPEC §9, §25]. `cluster.artifact_dir`는 명령행으로만 준다(빠지면 `ConfigError`, DESIGN 23.6).
- 확인: `dsoutputs steam_e4`: 학습 시작 때 live z 대조 통과(`meta.json` `cluster_live_check`), 학습 `gate_calls` 합 0·`cluster_lookups` 합 755,240 (= micro 188,810 × 4), 평가 gate가 one-hot = mapping[배정], export의 ω가 세 split 모두 mapping[배정]의 one-hot이고 대조 통과 (R12와 같음).
- 중단되면: R7의 조건에서만 `R=$(latest steam_cluster_hard) && python -I rebuild/main.py train --resume "$R" && done_ok "$R" train && rec ILORA_RUN_STEAM_E4 "$R"`, 그다음 위 블록의 둘째 줄(export)부터 한다.

실패하면: 군집 산출물이 바뀌었거나(파일 sha256), SASRec state가 다르면 학습 전에 `ClusterError`로 멈춘다. 군집을 다시 fit하지 말고 원인을 기록한다(R12와 같음).

## DS14 (Steam). 변형과 비교표 다시 (선택)

```bash
python -I rebuild/main.py train --config rebuild/configs/datasets/steam_variant_released_lr.yaml \
  && VR=$(latest steam_variant_released_lr) && done_ok "$VR" train && rec ILORA_RUN_STEAM_V_RELEASED_LR "$VR"
```

- 변형은 공개 스크립트 lr(최대 5e-4, 시작 5e-6, 최솟값 5e-6, D29, D30)이고 부모(`steam_paper_spec`)와 다른 키는 `train.lr_peak`, `train.lr_warmup_start`, `train.lr_floor`뿐이다(DESIGN 23.6). 고르기 전에 그 GPU 시간을 예산에 넣는다. 같은 공개 스크립트 계열의 MovieLens lr 8e-4는 fork에서 발산했다(R15의 기록). 같은 일이 생기면 GradScaler skip 16번 뒤 `TrainingError`로 멈추고(상태 `failed`) `done_ok`에서 사슬이 멈춰 기록되지 않는다. 그것도 D29의 결과다: bf16으로 다시 돌리지 않고(D29와 D9가 섞임) 실패한 폴더를 비교에 그대로 넘긴다(아래 명령의 `need`에서 `ILORA_RUN_STEAM_V_RELEASED_LR`을 빼고 `--run V_released_lr="$(latest steam_variant_released_lr)"`로 바꾼다. 그 칸은 `NOT_RUN`과 `TrainingError` 이유).
- 선택 단계를 돌렸으면 비교표를 다시 만든다(R14, R15와 같은 방식). 돌리지 않은 실험의 이름은 `need`와 `--run`/`--export`/`--cluster`에서 뺀다(빠진 실행은 `NOT_RUN`).

```bash
need ILORA_RUN_STEAM_E1 ILORA_RUN_STEAM_E2 ILORA_RUN_STEAM_E3 ILORA_RUN_STEAM_E4 ILORA_RUN_STEAM_E2_EXPORT ILORA_RUN_STEAM_E4_EXPORT ILORA_RUN_STEAM_CLUSTER ILORA_RUN_STEAM_V_RELEASED_LR \
  && python -I rebuild/tools/compare_runs.py --run E1="$ILORA_RUN_STEAM_E1" --run E2="$ILORA_RUN_STEAM_E2" --run E3="$ILORA_RUN_STEAM_E3" \
  --run E4="$ILORA_RUN_STEAM_E4" --run V_released_lr="$ILORA_RUN_STEAM_V_RELEASED_LR" --export E2="$ILORA_RUN_STEAM_E2_EXPORT" \
  --export E4="$ILORA_RUN_STEAM_E4_EXPORT" --cluster "$ILORA_RUN_STEAM_CLUSTER" \
  && CMP=$(latest compare) && done_ok "$CMP" compare && rec ILORA_RUN_STEAM_COMPARE_ALL "$CMP"
```

## bf16 대안 (R7의 실패 규칙에서만)

fp16 E2가 R7의 실패 규칙(GradScaler skip 연속 16회, 비유한 parameter)에 걸렸을 때만 쓴다(D9, DESIGN 23.6). **`RUNBOOK.md`의 "bf16 대안" 절의 블록은 쓰지 않는다**: 그 명령은 MovieLens 프로파일과 `ILORA_RUN_E2`를 가리켜 MovieLens를 다시 학습하고 그 경로 기록을 덮어쓴다. 문제가 생긴 데이터셋의 아래 블록을 쓴다(다른 데이터셋은 fp16 그대로): E2를 `<d>_variant_bf16.yaml`로 처음부터 다시 학습하고, 짝 비교가 같은 조건이 되도록 E1·E3·E4와 변형도 bf16(`model.torch_dtype=bfloat16`, `train.precision=bf16_amp`)으로 돌린다. 기록 이름은 `ILORA_RUN_<D>_…` 그대로이고 실패한 fp16 E2 폴더는 기록으로 둔다(`rec`가 `ILORA_RUN_<D>_E2`를 새 폴더로 바꿈). 편차로 기록한다: M10-15의 fp16 항목(첫 step scale 65,536)은 bf16 실행에 해당하지 않으며 `dsoutputs <d>_e2_run`은 GradScaler 없는 기록(scale 1.0, skip 없음)을 확인하고, `compare_runs.py`는 precision을 비교표의 한계에 적는다.

블록의 줄은 주석이 가리키는 단계에서 그 단계의 명령 대신 하나씩 돌린다: DS6(E2), DS7(재평가 2회), DS8(E1), DS10(첫 줄), DS13(첫 줄), DS14(변형). 나머지(DS9, DS10의 확인 두 줄, DS11, DS12, DS13의 export 줄, DS14의 비교표)는 본문 그대로이고 단계 순서도 본문과 같다(E4는 DS11·DS12 뒤). DS7의 재평가는 학습과 같은 설정이어야 하므로 `<d>_variant_bf16.yaml`을 쓰고(DESIGN 15.6.11), DS11의 export와 DS12의 군집은 precision과 무관해 본문 명령 그대로다(R10·R11과 같음). 재개는 각 블록 끝의 명령이고 R7의 조건에서만 쓴다.

LastFM:

```bash
# DS6: E2 in bf16 (from scratch; the failed fp16 folder stays as a record)
python -I rebuild/main.py train --config rebuild/configs/datasets/lastfm_variant_bf16.yaml \
  && E2=$(latest lastfm_variant_bf16) && done_ok "$E2" train && rec ILORA_RUN_LASTFM_E2 "$E2" && dsoutputs lastfm_e2_run
# DS7: the two re-evaluations with the training config
need ILORA_RUN_LASTFM_E2 && python -I rebuild/main.py eval --config rebuild/configs/datasets/lastfm_variant_bf16.yaml --set eval.source_run="$ILORA_RUN_LASTFM_E2" \
  && R=$(latest lastfm_variant_bf16) && done_ok "$R" eval && rec ILORA_RUN_LASTFM_E2_REEVAL "$R"
need ILORA_RUN_LASTFM_E2 && python -I rebuild/main.py eval --config rebuild/configs/datasets/lastfm_variant_bf16.yaml --set eval.source_run="$ILORA_RUN_LASTFM_E2" \
  --set eval.batch_invariance_check=true && R=$(latest lastfm_variant_bf16) && done_ok "$R" eval && rec ILORA_RUN_LASTFM_E2_REEVAL_BI "$R" \
  && dsoutputs lastfm_e2_reeval
# DS8: E1 in bf16
python -I rebuild/main.py train --config rebuild/configs/datasets/lastfm_llara_baseline.yaml --set model.torch_dtype=bfloat16 --set train.precision=bf16_amp \
  && E1=$(latest lastfm_llara_baseline) && done_ok "$E1" train && rec ILORA_RUN_LASTFM_E1 "$E1" && dsoutputs lastfm_e1
# DS10: first line (then the two DS10 checks of the main text)
need ILORA_RUN_LASTFM_E2 && python -I rebuild/main.py eval --config rebuild/configs/datasets/lastfm_variant_text_only_eval.yaml \
  --set model.torch_dtype=bfloat16 --set train.precision=bf16_amp --set eval.source_run="$ILORA_RUN_LASTFM_E2" \
  && E3=$(latest lastfm_variant_text_only_eval) && done_ok "$E3" eval && rec ILORA_RUN_LASTFM_E3 "$E3"
# DS13: first line, after DS11 and DS12 (then the DS13 export line of the main text)
need ILORA_RUN_LASTFM_CLUSTER && python -I rebuild/main.py train --config rebuild/configs/datasets/lastfm_cluster_hard.yaml \
  --set model.torch_dtype=bfloat16 --set train.precision=bf16_amp --set cluster.artifact_dir="$ILORA_RUN_LASTFM_CLUSTER/cluster" \
  && E4=$(latest lastfm_cluster_hard) && done_ok "$E4" train && rec ILORA_RUN_LASTFM_E4 "$E4"
# DS14: the variant
python -I rebuild/main.py train --config rebuild/configs/datasets/lastfm_variant_released_lr.yaml --set model.torch_dtype=bfloat16 --set train.precision=bf16_amp \
  && VR=$(latest lastfm_variant_released_lr) && done_ok "$VR" train && rec ILORA_RUN_LASTFM_V_RELEASED_LR "$VR"
```

중단되면 R7의 조건에서 E2는 `R=$(latest lastfm_variant_bf16) && python -I rebuild/main.py train --resume "$R" && done_ok "$R" train && rec ILORA_RUN_LASTFM_E2 "$R" && dsoutputs lastfm_e2_run`, E1·E4는 본문의 재개 명령 그대로다(실행 폴더의 `config.yaml`이 bf16이므로 `--set`이 필요 없다).

Steam:

```bash
# DS6: E2 in bf16 (from scratch; the failed fp16 folder stays as a record)
python -I rebuild/main.py train --config rebuild/configs/datasets/steam_variant_bf16.yaml \
  && E2=$(latest steam_variant_bf16) && done_ok "$E2" train && rec ILORA_RUN_STEAM_E2 "$E2" && dsoutputs steam_e2_run
# DS7: the two re-evaluations with the training config
need ILORA_RUN_STEAM_E2 && python -I rebuild/main.py eval --config rebuild/configs/datasets/steam_variant_bf16.yaml --set eval.source_run="$ILORA_RUN_STEAM_E2" \
  && R=$(latest steam_variant_bf16) && done_ok "$R" eval && rec ILORA_RUN_STEAM_E2_REEVAL "$R"
need ILORA_RUN_STEAM_E2 && python -I rebuild/main.py eval --config rebuild/configs/datasets/steam_variant_bf16.yaml --set eval.source_run="$ILORA_RUN_STEAM_E2" \
  --set eval.batch_invariance_check=true && R=$(latest steam_variant_bf16) && done_ok "$R" eval && rec ILORA_RUN_STEAM_E2_REEVAL_BI "$R" \
  && dsoutputs steam_e2_reeval
# DS8: E1 in bf16
python -I rebuild/main.py train --config rebuild/configs/datasets/steam_llara_baseline.yaml --set model.torch_dtype=bfloat16 --set train.precision=bf16_amp \
  && E1=$(latest steam_llara_baseline) && done_ok "$E1" train && rec ILORA_RUN_STEAM_E1 "$E1" && dsoutputs steam_e1
# DS10: first line (then the two DS10 checks of the main text)
need ILORA_RUN_STEAM_E2 && python -I rebuild/main.py eval --config rebuild/configs/datasets/steam_variant_text_only_eval.yaml \
  --set model.torch_dtype=bfloat16 --set train.precision=bf16_amp --set eval.source_run="$ILORA_RUN_STEAM_E2" \
  && E3=$(latest steam_variant_text_only_eval) && done_ok "$E3" eval && rec ILORA_RUN_STEAM_E3 "$E3"
# DS13: first line, after DS11 and DS12 (then the DS13 export line of the main text)
need ILORA_RUN_STEAM_CLUSTER && python -I rebuild/main.py train --config rebuild/configs/datasets/steam_cluster_hard.yaml \
  --set model.torch_dtype=bfloat16 --set train.precision=bf16_amp --set cluster.artifact_dir="$ILORA_RUN_STEAM_CLUSTER/cluster" \
  && E4=$(latest steam_cluster_hard) && done_ok "$E4" train && rec ILORA_RUN_STEAM_E4 "$E4"
# DS14: the variant
python -I rebuild/main.py train --config rebuild/configs/datasets/steam_variant_released_lr.yaml --set model.torch_dtype=bfloat16 --set train.precision=bf16_amp \
  && VR=$(latest steam_variant_released_lr) && done_ok "$VR" train && rec ILORA_RUN_STEAM_V_RELEASED_LR "$VR"
```

중단되면 R7의 조건에서 E2는 `R=$(latest steam_variant_bf16) && python -I rebuild/main.py train --resume "$R" && done_ok "$R" train && rec ILORA_RUN_STEAM_E2 "$R" && dsoutputs steam_e2_run`, E1·E4는 본문의 재개 명령 그대로다(실행 폴더의 `config.yaml`이 bf16이므로 `--set`이 필요 없다).

## 경로 변수와 실행 폴더 검사 대응

| 변수 (`rec`; `<D>` = `LASTFM`, `STEAM`, `<d>` = `lastfm`, `steam`) | 단계 | 쓰는 시험 (`rebuild/tests/vessl/test_ds_run_outputs.py`) |
|---|---|---|
| `ILORA_RUN_<D>_SMOKE_GPU` | DS4 | `dsoutputs <d>_smoke_gpu` |
| `ILORA_RUN_<D>_OVERFIT` | DS5 | `dsoutputs <d>_overfit` |
| `ILORA_RUN_<D>_E2` | DS6 | `dsoutputs <d>_e2_run`, `dsoutputs <d>_e2_reeval` |
| `ILORA_RUN_<D>_E2_REEVAL`, `ILORA_RUN_<D>_E2_REEVAL_BI` | DS7 | `dsoutputs <d>_e2_reeval` |
| `ILORA_RUN_<D>_E1` | DS8 | `dsoutputs <d>_e1` |
| `ILORA_RUN_<D>_E2_EXPORT` | DS11 | `dsoutputs <d>_e2_export` |
| `ILORA_RUN_<D>_CLUSTER` | DS12 | `dsoutputs <d>_cluster`, `dsoutputs <d>_e4` |
| `ILORA_RUN_<D>_E4`, `ILORA_RUN_<D>_E4_EXPORT` | DS13 | `dsoutputs <d>_e4` |

- 위 표 밖의 변수: `ILORA_RUN_<D>_CHECK_ENV`(DS3, 시험 없음), `ILORA_RUN_<D>_COMPARE`(DS9), `ILORA_RUN_<D>_E3`(DS10), `ILORA_RUN_<D>_E2_BEST`(DS11, 실행 폴더가 아닌 epoch 번호, DS12가 읽음), `ILORA_RUN_<D>_V_RELEASED_LR`, `ILORA_RUN_<D>_COMPARE_ALL`(DS14).
- 단계 하나를 다시 할 때: 같은 명령을 다시 돌리면 새 실행 폴더가 생기고, 그 뒤의 `latest`·`done_ok`·`rec`이 새 경로를 기록한다(`runbook_paths.sh`에서는 나중 줄이 이김). 앞 단계 경로가 바뀌면 그 경로를 쓰는 뒤 단계도 다시 한다(E2 → DS7–DS14, 군집 → DS13, DS14).
