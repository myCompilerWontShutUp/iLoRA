# iLoRA 재구현 설계 문서

상태: **구현 완료, 로컬 시험 통과, VESSL 실행 전** (결정 D1–D26 확정, 모듈 M0–M15 상세 작성·반박 검토·모듈 간 일관성 점검 완료, 코드 구현과 로컬 시험 완료: skip은 `VESSL_REQUIRED`(17.6.10)뿐, 구현 반박 검토 반영)
작성 시작: 2026-11-02 · 최근 갱신: 2026-11-06

이 문서는 iLoRA(NeurIPS 2024)를 논문의 방법론과 공개 코드의 구조를 참고해 처음부터 다시 구현하기 위한 설계서다. 모든 모듈의 설계가 끝난 뒤 제작을 시작하고, 제작도 모듈 단위로 테스트를 통과시키며 진행한다.

---

## 0. 이 문서를 쓰는 규칙

### 0.1 근거 계층과 라벨
설계 항목마다 아래 계층에서 가장 높은 근거를 찾아 라벨을 붙인다. 위 순위가 정한 것은 아래 순위가 바꾸지 않는다.

| 순위 | 라벨 | 뜻 |
|---|---|---|
| 1 | `[논문]` | iLoRA NeurIPS 2024 최종본(proceedings PDF, 2024-10-31 생성). 쪽·절·식·표·그림을 함께 적음. 그림에서 읽은 값이나 식을 결합한 추론은 `(암시)`를 붙임 |
| 1 | `[v1]` | arXiv 2408.10159v1(2024-08-19)에만 있고 이후 판에서 빠진 문장 (warmup 1/100, 최대 5 epoch, batch 128, weight decay 1e-5, seed [0..4]). 최종본과 어긋나면 최종본 우선 |
| 2 | `[LLaRA]` | LLaRA 논문 arXiv 2312.02445 v4(SIGIR 2024). iLoRA p.6 §4 "We request the dataset from LLaRA [9] and maintain exactly the same experimental settings as described in the original paper."로 명세에 포함됨. **§5.1 실험 설정**(데이터·분할·이력·템플릿·epoch·batch·warmup·후보·지표)은 직접 채택 |
| 2 | `[LLaRA 간접]` | LLaRA §4 방법 구성요소(§4.1 projector, §4.2 text-only/hybrid prompt, §4.3 curriculum). iLoRA p.3 "we follow LLaRA [9] and fine-tune the LLM ... on a hybrid prompt"를 통한 해석적 상속. iLoRA 본문과 다르면 iLoRA 우선 |
| 3 | `[코드]` | 두 논문이 모두 침묵할 때만. iLoRA `U`, 2024-08 상태 `U@0a69c15`, LLaRA 코드 `L`. 두 저장소가 같으면 "코드 계보 일치"로 적음. 공개 LLaRA checkpoint(HF `joyliao7777/LLaRA`)의 hparams는 실행 기록으로 보조 근거 |
| 4 | `[결정]` | 어느 근거에도 없거나, 같은 순위 근거끼리 다르거나, 논문이 모호해 우리가 정함. 이유를 반드시 적음 |
| - | `[미정]` | 아직 정하지 않음. 20장에 등록 (현재 없음) |

추가 표기
- **논문-코드 충돌:** 논문과 코드가 다르면 논문을 따르고, 코드 값은 변형 프로파일로만 둔다.
- `[SPEC §n]`: `experiments/EXPERIMENT_SPEC.md`의 연구 제약. 근거가 아니라 지켜야 할 요구다.
- 근거가 아닌 자료: 우리 fork(`F`)와 VESSL baseline 결과는 측정·검증 자료로만 인용한다.
- `VESSL_REQUIRED`: 로컬(Windows, CPU)에서 확인할 수 없어 VESSL에서 확인해야 하는 항목 `[SPEC §0.13]`.

### 0.2 출처 표기
- iLoRA 공개 코드: `U <경로>:<라인>` = AkaliKong/iLoRA `28e47371192121b6ced44401d9e149fedfae13de`
- 2024-08 코드: `U@0a69c15 <경로>`, 2024-08 collater: blob `b69479a07433489ac583b79c40930500be802960`
- LLaRA 코드: `L <경로>:<라인>` = ljy0ustc/LLaRA `9733f732399e95e236f4f2e1d1139c1f0437090e`
- 우리 fork: `F <경로>` = myCompilerWontShutUp/iLoRA `exp/gate-cluster` (baseline `6c7ebac`, 현재 `8a22b13`)
- 재구현 코드: `R <rebuild 아래 경로> <함수·클래스>` = 이 저장소 `rebuild/` (2026-11-06 구현). 모듈 전체에 걸친 사실은 이름 없이 경로만(`R ilora/cluster.py`), `RUNBOOK.md`는 절 이름(`R RUNBOOK.md R7`)
- iLoRA 논문 페이지: NeurIPS 2024 proceedings PDF 기준. `[v1 …]`은 arXiv 2408.10159v1 PDF 기준
- LLaRA 논문: `[LLaRA §절 p.쪽 식/표]` = arXiv 2312.02445 v4 PDF
- 라이브러리 소스 (설치된 wheel 기준, 패키지 폴더 아래 상대 경로)
  - transformers 4.28.0: `T4.28 <경로>:<라인>` (예: `T4.28 models/llama/modeling_llama.py:727`)
  - torch 2.0.0: `torch2.0 <경로>:<라인>` (예: `torch2.0 cuda/amp/grad_scaler.py`)
  - pytorch_lightning 1.8.6: `PL1.8.6 <경로>:<라인>` (예: `PL1.8.6 loops/optimization/optimizer_loop.py:88`). 공개 코드가 고정한 판(`U requirements.txt:149`)의 wheel `pytorch_lightning-1.8.6-py3-none-any.whl`(sha256 `8b6b4126…866d`)을 풀어 읽음. 재구현은 Lightning을 쓰지 않음 (D13)
  - scikit-learn 1.2.2: `sklearn 1.2.2 <경로>:<라인>` (예: `sklearn 1.2.2 cluster/_kmeans.py:1483-1490`)
- 측정 스크립트: `계산:`·`측정:` 뒤의 `design/tmp/…`, `impl/…`, `review/…`, `design_sync/…` 경로는 설계·구현 중 작업 공간에서 돌린 측정 스크립트이고 저장소에 넣지 않음. 결과 값과 조건은 본문에 적음 (저장소만으로는 그대로 다시 돌릴 수 없음)

### 0.3 모듈 작성 틀
각 모듈은 아래 항목을 같은 순서로 채운다.
1. 목적과 범위
2. 입력과 출력 (텐서 모양, dtype, 파일 형식. 3.1 계약과 일치해야 함)
3. 논문 근거
4. 공개 코드에서 가져올 것
5. 버릴 것 (알려진 문제)
6. 설계 (구현 방법)
7. 결정 (20장 번호로 연결)
8. 완료 기준 (테스트)

### 0.4 상태 표시
`뼈대` → `작성 중` → `검토 완료` → `구현 중` → `테스트 통과(로컬)` → `테스트 통과`
- `테스트 통과(로컬)`는 로컬(Windows, CPU) 시험이 모두 통과하고 skip이 `VESSL_REQUIRED`(17.6.10)뿐인 상태다. `테스트 통과`는 `VESSL_REQUIRED` 기준까지 VESSL에서 통과한 상태다

---

## 1. 목표와 범위

### 1.1 목표
- MovieLens에서 iLoRA를 논문 방법론대로 독립 재구현한다 (E2).
- 같은 구현의 K = 1(LoRA, LLaRA와 같음)을 기준선으로 돌려 같은 조건에서 비교한다 (E1).
- 공개 코드에서 확인된 결함(저장 필터, 평가 N, curriculum 카운터, padding, router BatchNorm 등)이 없는 구현을 만든다.
- 이후 연구(gate 분석, SASRec 군집 기반 cluster-hard routing, E4)의 기반으로 쓴다.

### 1.2 범위 밖
- LastFM, Steam 데이터셋 `[SPEC §0.5]`
- 공개 코드의 결함 동작을 그대로 재현하는 것. 그 기준은 기존 fork baseline(`F` 6c7ebac, 결과 보관 파일)이 맡는다 (D26)

### 1.3 원칙
- 저장소의 다른 폴더(`model/`, `data/` 등)를 import하지 않는다. 가져오는 코드는 복사하고 출처를 주석으로 남긴다.
- 기존 fork와 결과 파일은 동결해 비교 기준으로 쓴다.
- dynamic routing(식 11 gate)이 기본 경로다. 실험 변형은 설정으로만 구분한다 `[SPEC §0.1, §0.4]`.
- 평가는 결정적이어야 한다. 같은 seed와 같은 checkpoint면 같은 결과가 나와야 한다.
- checkpoint에는 학습에 쓰인 상태를 빠짐없이 저장하고, 다시 불러왔을 때 출력이 같아야 한다.
- KMeans 등 군집화는 train 표본으로만 fit한다 `[SPEC §0.8]`.
- 측정하지 않은 결과를 적지 않는다. 결과가 없으면 `TBD` 또는 `NOT_RUN` `[SPEC §0.11]`.

---

## 2. 기준 자료

| 자료 | 식별자 | 비고 |
|---|---|---|
| iLoRA 논문 최종본 | NeurIPS 2024, PDF 생성 2024-10-31 | 근거 1순위. Table 1 MovieLens: VR 0.9891, HR@1 0.5275 |
| iLoRA arXiv | 2408.10159 v1(2024-08-19) ~ v4(2025-01-21) | Table 1 수치는 모든 판 동일. v1 App E에만 있는 설정 `[v1]` |
| LLaRA 논문 | arXiv 2312.02445 v4 (SIGIR 2024) | 근거 2순위. Table 2 LLaRA(SASRec) MovieLens VR 0.9684, HR@1 0.4421 = iLoRA Table 1 LLaRA 행 |
| iLoRA 공개 코드 | AkaliKong/iLoRA `28e4737` | 근거 3순위. 2024-11-13(65e2fcd)에 prompt 조건 반전 |
| iLoRA 2024-08 코드 | `0a69c15` (그대로는 실행 불가) | collater blob `b69479a0` |
| LLaRA 코드 | ljy0ustc/LLaRA `9733f73` | 근거 3순위. `SASRecModules_ori.py` 원본 (blob `3ac3c975`) |
| LLaRA 공개 checkpoint | HF `joyliao7777/LLaRA` movielens.ckpt 2개 (2024-03, 2024-07) | 실행 기록(hparams)으로만 보조 |
| 우리 baseline | fork `6c7ebac`, 결과 보관 파일 sha256 `ede386ce…0398` | test 95/86/30, VR 0.9053, HR 0.3488 (정답/유효) |
| 감사·평가 도구 | fork `8a22b13` | `scripts/eval_prompt_semantics.py` 등 |

---

## 3. 모듈 목록과 의존 관계

| 번호 | 모듈 | 파일 | 의존 | 상태 |
|---|---|---|---|---|
| M0 | 설정과 실행 관리 | `config.py`, `sections.py`, `run.py`, `seeding.py`, `precision.py`, `commands.py` | - | 테스트 통과(로컬) |
| M1 | 데이터 | `data.py` | M0 | 테스트 통과(로컬) |
| M2 | 후보 생성 | `candidates.py` | M1 | 테스트 통과(로컬) |
| M3 | SASRec 인코더 | `sasrec.py`, `tools/convert_sasrec.py` | M1 | 테스트 통과(로컬) |
| M4 | prompt와 curriculum | `prompts.py` | M1, M2 | 테스트 통과(로컬) |
| M5 | tokenization과 batch 구성 | `batching.py` | M4 | 테스트 통과(로컬) |
| M6 | projector | `projector.py` | M3 | 테스트 통과(로컬) |
| M7 | gate와 routing | `router.py` | M3 | 테스트 통과(로컬) |
| M8 | MoE-LoRA 층 | `moe_lora.py` | M7 | 테스트 통과(로컬) |
| M9 | LLM 연결 | `model.py` | M5, M6, M8 | 테스트 통과(로컬) |
| M10 | 학습 루프 | `train.py` | M9 | 테스트 통과(로컬) |
| M11 | 생성과 평가 | `evaluate.py` | M9, M2 | 테스트 통과(로컬) |
| M12 | 기록과 gate 분석 | `records.py`, `analysis.py` | M10, M11 | 테스트 통과(로컬) |
| M13 | 테스트 | `tests/`, `pytest.ini`, `smoke.py` | 전 모듈 | 테스트 통과(로컬) |
| M14 | 실험 구성 | `configs/`, `RUNBOOK.md`, `tools/compare_runs.py` | 전 모듈 | 테스트 통과(로컬) |
| M15 | SASRec 군집과 cluster-hard 배정 | `cluster.py` | M3, M7, M12 | 테스트 통과(로컬) |

- 테스트 통과(로컬) = Windows CPU 전체 시험 통과 (2026-11-06, 745 passed, 37 skipped in 1039.78s (0:17:19)). `VESSL_REQUIRED` 기준은 RUNBOOK 실행 후 확인한다.
- `의존`은 그 모듈이 정의를 가져다 쓰는 주된 모듈이다(전부가 아님). 명령 함수는 실행할 때 더 넓게 부르고, 그때 생기는 순환은 함수 안 import로 끊는다 `[결정]`: M10 `train`은 M11 `evaluate_split`·`eval_dir`·`load_eval_result`와 M12 `TrainRecorder`·`ActivationMaxMonitor`를, M11 `evaluate_split`은 M12 `write_eval`·`save_npz`를, M11 `run_eval_command`는 M10 `load_model_state`를, M12 `run_export`는 M10 `read_trainable_state`와 M11 `EvalRecord`(`read_eval_records`)를 부르고, cluster_hard 경로(M10·M11·M12)는 M15를 부른다. M15는 M12의 NPZ 형식과 z 계산(`compute_representations`, 16.6.1)을 모듈 수준에서 씀. 이유: checkpoint 읽기(14.6.6)·평가(15장)·기록 형식(16장)을 한 곳에서만 정의함. 모듈 수준 import에는 순환이 없음 (15.6.1) `R ilora/evaluate.py run_eval_command`, `R ilora/records.py run_export`, `R ilora/cluster.py`

### 3.1 모듈 간 계약
모듈 사이를 오가는 값은 아래 정의만 쓴다. 모듈 문서와 코드는 이 표와 일치해야 하며, 바꾸려면 이 표를 먼저 고친다. 각 장의 '계약 추가 제안' 절은 2026-11-06 병합 때 이 표에 반영했다 (제안 문구는 작성 기록으로 남김).

| 이름 | 정의 | 만드는 곳 | 쓰는 곳 |
|---|---|---|---|
| `Config` | 구역별 frozen dataclass (`run, data, candidates, sasrec, prompt, batching, projector, router, lora, model, train, eval, records, cluster`), 키 87개(`projector`는 키 없음). 스키마 `ilora/sections.py`, 전체 키 표 18.6.2. 프로파일 YAML(`extends` 상속, 4.6.2) + `--set` 덮어쓰기. 읽기는 `config.load_config(path, overrides=()) -> (cfg, applied)`로만 (`applied` = 적용한 덮어쓰기 dict. 실행 폴더의 `config.yaml`도 같은 함수로 읽음), 설정 오류는 `ConfigError`(ValueError), 설정 해시는 `config.config_hash(cfg)`(4.6.3). 구역 간 검사 `Config.validate`: `router.routing_mode == cluster_hard` ⇔ `cluster.artifact_dir` 있음, `routing_mode == none` ⇔ `lora.num_experts == 1`, `model.torch_dtype` ↔ `train.precision` (float16 ↔ fp16_amp, bfloat16 ↔ bf16_amp, float32 ↔ fp32), `model.backbone == tiny_random` ⇔ (`device == cpu`, `torch_dtype == float32`). 구역 검사: `data.train_rows`가 비어 있지 않으면 `data.subset == smoke` | M0 | 전부 |
| `derive_seed(main_seed, *keys) -> int` | `"{main_seed}\|{key1}\|{key2}\|..."`의 SHA-256 앞 8 byte를 63 bit 정수로. key는 str 또는 int. 모든 난수원은 이 함수로만 만든다 (key 목록 4.6.4, 예외는 KMeans `random_state` 하나) | M0 | 전부 |
| 경로와 해시 | 경로 값 키는 절대 경로 또는 저장소 루트 기준 상대 경로이고 `run.resolve_path(value, repo_root)` 하나로만 해석. `run.content_sha256(path, text=False)` (text면 CRLF→LF 정규화), `run.EXPECTED_INPUTS: dict[경로, (sha256, is_text)]`, 그 파일들의 현재 값 `run.input_hashes(repo_root) -> dict[경로, sha256 또는 None]` (`meta.json` `inputs`, M15 `data_sha256`), 설정·환경이 위치를 정하는 파일의 값 `commands.asset_record(cfg, *, repo_root, llm_path, tokenizer_dir)` (`meta.json` `assets`, M10 checkpoint·M11 eval 명령이 대조). `meta.json`은 `run.read_meta(run_dir)`로 읽고 `run.update_meta(run_dir, fields)`로만 블록을 더함 (4.6.3) | M0 | M1, M3 도구, M4, M5, M10–M15 |
| precision 도구 | `precision.autocast_for(precision, device)`: `fp16_amp` → CUDA float16 autocast, `bf16_amp` → bfloat16, `fp32` → 아무것도 안 함. `precision.make_grad_scaler(tc, device)`: `fp16_amp`일 때만 켜고 값은 `train.scaler_*` | M0 | M10, M11 |
| 명령 분기 | `commands.run_command(command, cfg, overrides=(), *, profile, root, repo_root, argv, llm_path, tokenizer_dir, profile_path, resume, stamp) -> Path`: `train`, `eval`, `export`, `cluster`, `check-env`, `smoke`. `overrides`는 `--set` 원문 목록(`meta.json`에는 해석한 값), `profile`은 폴더 이름 접두어, `root`는 `runs/rebuild` 또는 smoke 단계의 `<smoke 실행>/steps`. `repo_root`부터는 기본값이 있는 keyword `[결정]`(smoke와 재개에 필요): `llm_path` 기본 `LLM_PATH`(llama2_7b만), `tokenizer_dir` 기본 LLaMA 폴더(llama2_7b) 또는 `ILORA_TEST_TOKENIZER`(tiny_random) (`commands.llm_path_for`·`tokenizer_dir_for`, M10·M11·M13도 같은 규칙), `profile_path`는 smoke에 필수, `resume`은 train만 `R ilora/commands.py run_command`. 공통 준비(실행 폴더와 `run.lock`, `config.yaml`, `meta.json`, `run.log`, 디스크·환경 점검, `candidates.self_check()`, `tracked_run`) 뒤 명령 함수 호출: M10 `train(cfg, run_dir, *, repo_root, llm_path, resume)`, M11 `run_eval_command(cfg, run_dir, *, repo_root, llm_path)`, M12 `run_export(cfg, run_dir, *, repo_root)`, M15 `run_cluster(cfg, run_dir, *, repo_root)`, M13 `run_smoke(profile_path, overrides, run_dir, *, repo_root, llm_path, tokenizer_dir)`(보고서 `passed`가 거짓이면 `CommandError`), M0 `check_env`. eval·export는 `eval.source_run`·`records.source_run`이 없으면 실행 폴더를 만들기 전에 `ConfigError`. `train --resume`은 failed·미완료(running) 학습 실행만 다시 열고(`commands.load_run_config(run_dir) -> (cfg, profile)`, 설정 해시가 `config.yaml`·`meta.json`과 같아야 함) `meta.json`의 `resumes`에 추가. 다른 살아 있는 프로세스가 `run.lock`을 쥔 폴더는 아무것도 바꾸지 않고 `CommandError` (4.6.3) | M0 | M10–M15 |
| `ItemCatalog` | frozen `ItemCatalog(titles, title_mode, num_items, pad_id)` (5.6.1 순서, 네 필드 모두 필수이고 기본값 없음): `titles: tuple[str, ...]` (길이 1,682, index = item id 0–1681, 정규화 제목), `title_mode: str` (`code`/`fixed`, D20), `num_items = 1682`, `pad_id = 1682` | M1 | M2, M4, M11, M12 |
| `Sample` | frozen. `split: str` (`train`/`val`/`test`), `row: int` (배포 `.df` 파일의 원래 행 번호, 필터 전), `sample_id`(속성) = `f"{split}:{row}"`, `user_id: int` (pseudo, D14), `history: tuple[int, ...]` (padding 없음, 오래된 것부터, 길이 3–10), `target: int` | M1 | M2–M12, M15 |
| `LoadedData` | frozen `LoadedData(catalog, train, val, test, subset, smoke_users)`, `samples(split)`, `title_mode`. `load_dataset(cfg, repo_root)`가 표본을 얻는 유일한 경로 (unpickle 전 해시 확인). smoke는 `select_smoke(data, main_seed, sizes, *, train_rows)`: `train_rows`가 비어 있지 않으면 train을 그 행들의 표본으로 고정하고 `("smoke_users", "train")`을 추첨하지 않음 | M1 | M4, M5, M10–M13, M15 |
| `pad_histories` | `data.pad_histories(histories, pad_id, length) -> (hist_ids [N, length] int64, len_seq [N] int64)`. CPU, 오른쪽 padding 1682 | M1 | M3, M5, M12 |
| 후보 | `tuple[int, ...]` 길이 20: 정답 정확히 1개, 중복 없음, `history`와 겹치지 않음, 순서 섞임. `candidates_for(sample, *, main_seed, epoch, cfg)` (`cfg` = `Config.candidates`, epoch: 학습 int, 평가 None). `CandidateFn` = `functools.partial(candidates_for, main_seed=…, cfg=…)` (spawn worker로 pickle 가능). `candidate_flags(cands, target, catalog)` (`FLAG_NAMES` 5개), `match_norm(title)` (판정 정규화의 유일한 정의), `self_check()` (golden 대조) | M2 | M4, M5, M11, M12 |
| prompt | `str`. hybrid면 이력마다 `"제목 [HistoryEmb]"`, 후보마다 `"제목 [CansEmb]"`, text-only면 둘 다 `"제목 [PH]"`. 템플릿 index와 함께 넘김. API: `load_templates(path)` (해시 확인, `prompt/movie.txt`만 허용), `TemplateSet`, `choose_template(cfg, seed, n, sample, epoch)`, `build_prompt(sample, cands, catalog, templates, template_id, hybrid)`, `eval_is_hybrid(cfg)`, `curriculum_hybrid(mode, seed, tau, total)` | M4 | M5, M11 |
| `BatchPlan` | frozen `BatchPlan(epoch, batches_per_epoch, total, curriculum, entries: tuple[PlanEntry(tau, sample_indices, hybrid), ...])`, `digest()`, `hybrid_count()`. epoch은 0부터, `tau = epoch × NB + b + 1` (1부터), `T = epochs × NB`. `sample_indices`는 `LoadedData.train`(행 오름차순) 안의 위치. main process에서 `make_batch_plan`/`build_batch_plan`으로 만들고 worker는 계획대로 tokenization만 함 | M4 | M5, M10, M12, M13 |
| `TokenIds`, `TokenizerBundle` | `TokenIds(bos 1, eos 2, unk 0, newline 13, pad 32000, ph 32001, history_emb 32002, cans_emb 32003, item_emb 32004, base_vocab 32000, vocab_size 32005, new_token_ids)`. `TokenizerBundle(tokenizer, ids, model_sha256)`. `load_tokenizer(dir, expected_sha256)`가 transformers 4.28.0과 sentencepiece 0.1.97을 확인. 토큰 id는 이것으로만 씀 | M5 | M9, M10, M11, M13 |
| `TrainBatch` | `input_ids [B, L] int64`, `attention_mask [B, L] int64`, `labels [B, L] int64` (응답 토큰 외 -100), 오른쪽 padding. `hist_ids [B, 10] int64` (오른쪽 padding 1682), `len_seq [B] int64`, `cand_ids [B, 20] int64`, `hybrid: bool` (batch 전체 공통), `tau: int`, `sample_ids: list[str]`, `template_ids: list[int]`. `pin_memory()`, `to(device)`. 변형 `label_mode = released_pair`에서만 labels에 두 번째 `[BOS]`도 포함 | M5 | M9, M10 |
| `EvalBatch` | `input_ids`, `attention_mask` (왼쪽 padding), `hist_ids`, `len_seq`, `cand_ids`, `target_ids [B]`, `cand_titles: list[list[str]]`, `target_titles: list[str]`, `hybrid: bool`, `sample_ids`, `template_ids`. `pin_memory()`, `to(device)` | M5 | M9, M11 |
| loader | `make_train_loader(plan, data, cfg, epoch, *, bundle, templates) -> Iterator[TrainBatch]`, `make_eval_loader(data, split, cfg, *, bundle, templates)`, 공개 `collate_train`/`collate_eval`. DataLoader는 `batch_size=None` | M5 | M10, M11, M13 |
| `z` | 표본 표현 `[B, 64] float32`. SASRec eval 모드, `no_grad`, autocast 밖, CUDA에서는 TF32 끔(`strict_fp32`), 정답 미포함. `SASRecEncoder(hist_ids, len_seq)`, `embed_items(ids)` (0–1681, padding 1682 거부), `encode_all` | M3 | M7, M9, M12, M15 |
| 아이템 표현 | `[..., 64] float32` (SASRec 아이템 임베딩 조회) | M3 | M6 |
| 행동 토큰 | `[n, 4096]`. `build_projector(in_dim, out_dim, main_seed)`, autocast 아래 계산 | M6 | M9 |
| `ω` (gate) | `[B, K] float32`, 행 합 1 (허용오차 1e-5), NaN/Inf 없음. `Router(rcfg, *, num_experts, z_dim, main_seed, hard_gate)`가 세 방식의 유일한 공급원이고 `forward(z, sample_ids) -> ω`. dynamic은 `Softmax(W_g z)`, cluster_hard는 one-hot (`HardGate`), none(K = 1)은 모두 1. 기록은 `start_recording`/`stop_recording` → `GateRecord` | M7, M15 | M8, M9, M12 |
| `GateContext` | 모델 하나에 하나 있는 공유 객체. `GateContext(num_experts).use(omega)` 문맥 안에서만 ω가 있고 끝나면 비움. MoE-LoRA 층은 batch 크기·장치를 확인하고 ω가 없으면 오류. `attach_moe_lora(llm, cfg.lora, gate_ctx, main_seed) -> list[str]`, `lora_named_parameters(llm)`. K의 유일한 정의는 `lora.num_experts` | M8 | M9 |
| `ILoRAModel` | `build_model(cfg, bundle, catalog, *, llm_path, hard_gate=None)`. `train_forward(TrainBatch) -> TrainForward(loss, n_label_tokens, omega, z)`, `generate(EvalBatch) -> GenerationOutput(new_token_ids [B, T] int64 CPU, omega, z)`, `param_groups()` (projector, gate(dynamic만), lora), `trainable_state()`, `load_trainable_state(sd, strict=True)`. ω 공급 속성은 `omega_source`, checkpoint key 접두사는 `omega_source.` | M9 | M10, M11, M12, M13 |
| checkpoint | `checkpoints/epoch{e}/{trainable.pt, train_state.pt, STATE_SAVED, val_metrics.json, COMPLETE}`. `trainable.pt` = M9 `trainable_state()` + `generated_frozen_state()`(key 접두사 `generated_frozen.`). 읽기는 `read_trainable_state(ckpt_dir)`와 `load_model_state(ckpt_dir, model) -> str`로만 (표지 파일·sha256 확인. 반환값 = `trainable.pt`의 sha256 = `run.content_sha256(trainable.pt)`, M11 `ckpt_meta`와 M12 manifest의 `trainable_sha256`과 같은 값). `eval/selection.json`: `rule, val, best_val_epoch, last_epoch, test_dirs, primary` | M10 | M11–M14 |
| `StepRecord`, `TrainSummary` | `StepRecord`: optimizer step 한 행. ω 평균은 K개 tuple 필드 `omega_mean` 하나이고, `as_row()`가 `omega_mean_0..omega_mean_{K-1}`로 펼쳐 M12 `step_fields(K)` 열 순서의 dict를 만듦 (frozen dataclass는 K에 따라 필드 수가 바뀔 수 없음) `R ilora/train.py StepRecord.as_row`. `TrainSummary` 필드: `status`, `routing_mode`, `num_experts`, `counts`, `epochs_complete`, `best_val_epoch`, `last_epoch`, `val` (epoch → `SplitMetrics`), `test` (epoch → tags, metrics, aux_protocols, eval_dir), `trainable_params` (그룹별과 `total`), `seconds` (`train`, `validation`, `save`, `test`, `total`), `skipped_steps`, `nonfinite_loss`, `gate_calls`, `cluster_lookups`, `epoch_max_mem_gib`, `unused_experts`, `resumes`. M12 `run_summary.json`의 `train_summary`로 남고 M14가 `trainable_params`·`seconds.total`을 읽음 (18.6.7) `R ilora/train.py TrainSummary` | M10 | M12, M14 |
| `EvalRecord` | 표본별 frozen dataclass: `sample_id, split, user_id, template_id, hybrid, cand_ids, cand_titles, target_id, target_title, raw_text, first_line, matched, valid, correct, gates, flags, gen_token_ids` (이 순서 = `records.EVAL_RECORD_FIELDS`). `gates: tuple[float, ...]`는 ω 한 행(K개)이고 `records.jsonl`에서는 `gate_0..gate_{K-1}` key로 펼침 (frozen dataclass는 K에 따라 필드 수가 바뀔 수 없음, 15.6.1) `R ilora/records.py write_eval`. `raw_text`는 strip 전 decode 결과, `matched`는 match_norm 제목이 첫 줄의 부분 문자열인 후보 id (후보 순서), `flags = candidate_flags`, `gen_token_ids`는 첫 EOS까지의 새 토큰 | M11 | M12, M14 |
| `SplitMetrics`, `EvalResult` | `SplitMetrics(split, n_total, n_valid, n_correct, valid_ratio, hr_at_1, combined_metric)`, `EvalResult(split, ckpt_tag, metrics, aux, records, omega, z, out_dir, seconds)`. `evaluate_split(model, bundle, data, split, cfg, run_dir, *, ckpt_tag, ckpt_meta=None, templates=None)`. 폴더 `eval_dir(run_dir, ckpt_tag, split)`, 완료 폴더 다시 읽기 `load_eval_result(out_dir, *, num_experts)`(M10 재개), eval 명령 `run_eval_command(cfg, run_dir, *, repo_root, llm_path) -> dict`(`eval/summary.json` 내용) | M11 | M0, M10, M12–M14 |
| 기록 | `TrainRecorder`(M10이 `attach_monitor(monitor)`로 `ActivationMaxMonitor`를 넘기고 재개는 `resume`), `ActivationMaxMonitor`, NPZ 표현 형식(`RepresentationTable`, `save_npz`·`load_npz`), z·ω 표 계산 `compute_representations`(M15도 z에 씀), `write_eval`, 열 순서의 유일한 정의 `step_fields(K)`·`EVAL_RECORD_FIELDS`, `run_export(cfg, run_dir, *, repo_root)`. 오류는 `RecordsError`(ValueError) | M12 | M10, M11, M13–M15 |
| 군집 산출물 | `cluster/` 폴더 (manifest에 `data.subset`, `smoke_users`, `train_rows` 포함). `load_cluster_artifacts`, `parse_expert_mapping`, `ClusterHardGate` (`HardGate` 구현), `verify_live_encoder`, `run_cluster(cfg, run_dir, *, repo_root)`. 준비 순서 고정: 읽기 → mapping → `ClusterHardGate` → `build_model` → `verify_live_encoder` (학습 M10·평가 M11. 모델을 만들지 않는 export M12는 `ClusterHardGate` 다음 CPU 인코더로 `verify_live_encoder`). 결과 dict는 학습 `meta.json`, 평가 `eval/summary.json`, export `manifest.json`의 `cluster_live_check` `R ilora/records.py run_export` | M15 | M7, M9–M12, M14 |
| 실행 폴더 | `runs/rebuild/<프로파일>_<UTC>/`: `config.yaml`, `meta.json`, `run.log`, `run.lock`(4.6.3), `checkpoints/epoch{e}/`, `eval/{ckpt_tag}/{split}/` (`records.jsonl`, `metrics.json`(마지막에 씀 = 완료 표시), `representations.npz`, `batch_invariance.json`), `eval/selection.json`, `eval/summary.json`, `records/train/`, `records/export/epoch{e}/`, `records/analysis/epoch{e}/{split}/`, `cluster/`. 명령별 추가: `check_env.json`(check-env, 4.6.5. RUNBOOK R4가 같은 폴더에 `pip_freeze.txt`를 더함, 18.6.6), `smoke_report.json`과 하위 실행 폴더 `steps/<단계>_<이름>_<UTC>/`(smoke, 17.6.6). 비교 도구(M14)는 `compare_<UTC>/`를 만듦(`meta.json`, `run.log`, 비교 파일. `config.yaml`·`run.lock` 없음, 18.6.8). 재개·실패로 남은 미완료 산출물은 지우지 않고 `.incomplete-<UTC>`(이름이 겹치면 `-<n>`)를 붙여 옮김 `[결정]`: `checkpoints/epoch{k}.incomplete-<UTC>/`, `checkpoints/epoch{e}/val_metrics.incomplete-<UTC>.json`(14.6.10, `R ilora/train.py _keep_stale_val_metrics`), `eval/{ckpt_tag}/{split}.incomplete-<UTC>/`(15.6.8), `records/train/<이름>.incomplete-<UTC>.<확장자>`(16.6.2) | M0 | 전부 |
| 시험 기반 | `rebuild/pytest.ini` (marker `slow`, `vessl`, `criterion`), `tests/refs_upstream.py`, `tests/vessl/`, `ilora/smoke.py` (`smoke_plan`, `run_smoke`, `overfit_generation_check`), `tools/compare_runs.py` | M13, M14 | 사용자, RUNBOOK |

### 3.2 전체 흐름
**학습 한 micro-batch (M10)**
1. main process가 epoch의 `BatchPlan`을 만든다: 순서 = `torch.Generator(derive_seed(seed, "train_order", epoch))`의 permutation, drop_last, 각 batch의 `tau`와 curriculum 추첨(D3)
2. worker: 표본마다 후보(D10)·템플릿(D15)을 key로 정하고 prompt(M4)를 만들어 `TrainBatch`로 tokenization(M5)
3. 모델(M9): `z = SASRec(hist_ids, len_seq)`(M3) → `ω = router(z, sample_ids)`(M7) → `GateContext.use(ω)` 문맥 안에서 → hybrid면 `[HistoryEmb]`·`[CansEmb]` 자리에 `projector(SASRec 아이템 임베딩)`(M6) 삽입 → LLaMA forward(MoE-LoRA 층 M8이 `ω` 사용) → 응답 토큰 NLL
4. 창(A = `train.grad_accum` = 16개 micro-batch, epoch 끝은 남은 9개)마다 시작에 lr을 정하고(D6), micro-batch loss를 A로 나눠 backward(epoch 끝 9개 창도 16으로 나눔, D24), 창 끝에 GradScaler step

**평가 (M11)**
1. split 전체를 행 순서대로, 고정 후보·고정 템플릿, hybrid prompt(D4), 왼쪽 padding으로 batch 구성 (drop_last 없음)
2. greedy 생성 64 토큰 → 첫 줄 → 후보 부분 문자열 판정 → ValidRatio, HR@1, CombinedMetric (D17)
3. `eval/{ckpt_tag}/{split}/`에 표본별 `EvalRecord`, 지표, `ω`·`z` 저장 (M11, M12)

---

## 4. M0 설정과 실행 관리
상태: 테스트 통과(로컬)

### 4.1 목적과 범위
- **범위:** 실행 설정(설정 파일과 덮어쓰기), 시드와 난수 흐름, 실행 폴더, 출처 기록(meta), 실행 전 환경 점검, 기본 로깅, 실행 진입점, 코드 폴더 구성
- **범위 밖:** 데이터·모델·학습 로직 자체(M1–M11), 지표 기록 형식(M12), 실험 프로파일의 값 결정(M14)

### 4.2 입력과 출력
**입력**
- 설정 파일(YAML) 1개와 명령행 덮어쓰기 (`--set train.lr_peak=1e-4`). `train --resume <실행 폴더>`는 설정 파일 대신 그 폴더의 `config.yaml`을 씀
- 환경 변수 `LLM_PATH` (LLaMA 경로, `model.backbone = llama2_7b`일 때만 읽음. 없으면 환경 점검 실패), `ILORA_TEST_TOKENIZER` (`tiny_random` 프로파일의 tokenizer 폴더. `llama2_7b`는 `LLM_PATH` 폴더의 tokenizer를 씀) `[결정]`: CPU smoke 실행(17.6.6)에도 tokenizer가 필요하지만 LLaMA 폴더가 없음 `R ilora/commands.py llm_path_for`, `R ilora/commands.py tokenizer_dir_for`. Hugging Face token은 M0이 읽지도 기록하지도 않음
- 저장소 상태 (git commit, 변경 여부)

**출력**
- 확정된 설정 객체 `Config`: 다른 모듈은 이 객체만 받음 (3.1)
- 실행 폴더 `runs/rebuild/<프로파일>_<UTC 시각>/`
  - `config.yaml`: 모든 항목이 채워진 최종 설정
  - `meta.json`: 출처 기록 (4.6.3)
  - `run.log`: 콘솔과 같은 로그
  - `run.lock`: 실행 중인 프로세스가 쥐는 배타 잠금과 쥔 프로세스 기록 (4.6.3)
  - `check_env.json`: check-env 명령의 점검 결과 (4.6.5)
  - `checkpoints/`, `eval/`, `records/`: 내용은 M10–M12에서 정함
- 난수 도구: `derive_seed(main_seed, *keys)`와 이를 쓰는 numpy·torch 난수원 생성 함수 (4.6.4)
- precision 도구: fp16 autocast 범위와 GradScaler 생성 (D9)

### 4.3 논문 근거
- "our iLoRA framework is implemented on a single Nvidia A100." `[논문 App B p.16]` (메모리 크기 미기재). A100 80GB 1장, CUDA:0만 사용 `[SPEC §0.12]`
- "All experiments are carried out using Python 3.8 and PytorchLightning 1.8.6." `[논문 App B p.16]`: 실행 환경 기술로만 참고하고 Python 3.10과 Lightning 미사용으로 확정 (D13)
- seed: 전통 기준선은 [0, 1, 2, 3, 4]의 평균 `[v1 App E p.16]`, 5회 평균 `[LLaRA §5.1.2 p.7]`(전통 추천기 문장). LLM 방법의 seed는 iLoRA·LLaRA 어디에도 없고, LLaRA의 LLM 수치는 모두 k/95라 단일 실행으로 보임 (D2)

### 4.4 공개 코드에서 가져올 것
- 설정 항목의 이름과 의미 (출발점으로만) `U main.py:77-138`
- 기본 seed 1234 `U main.py:84`, `L main.py:84`

공개 코드는 기본값과 실행 스크립트 값이 서로 다르다. 재구현은 각 값을 해당 모듈에서 정하고, 프로파일 파일에 모두 명시한다.

| 항목 | `main.py` 기본값 | `train_movielens.sh` | 논문 | 재구현 값 (결정) |
|---|---|---|---|---|
| micro-batch × accumulation | 4 × 32 (:82, :86) | 8 × 16 | 128 `[v1 App E p.16]` `[LLaRA §5.1.2 p.7]` (App C의 실효 128은 RQ1 분석 설정) | 8 × 16 (M10) |
| 최대 lr | 1e-4 (:85) | 8e-4 | 1e-4 `[논문 App E p.17]` | 1e-4 (D5) |
| warmup 시작 lr | 1e-6 (:91) | 8e-6 | 최대의 1/100 `[v1 App E p.16]` `[LLaRA §5.1.2 p.7]` | 1e-6 (D6) |
| lr 최솟값 | 1e-6 (:90) | 8e-6 | 없음 | 1e-6 (D6, 코드 계보) |
| router | `unshare` (:138) | `share` | gating network 하나 `[논문 p.4 그림 2, p.6 식 11]` | 선형 gate 1개 (D1) |
| epoch | 5 (:120) | 5 | 최대 5 `[v1 App E p.16]` `[LLaRA §5.1.2 p.7]`, 총 5 `[LLaRA §5.4 p.9]` | 5 (D11) |
| seed | 1234 (:84) | 지정 안 함 (1234) | 없음 | 1234 (D2) |
| DataLoader worker | 8 (:83) | 지정 안 함 (8) | 없음 | 설정값, 결과와 무관 (4.6.4) |
| precision | 16 (:79) | 지정 안 함 (16) | half precision `[논문 App E p.17]` | fp16 AMP (D9) |
| 후보 수 | 20 (:122) | 20 | 20 `[논문 App B p.16]` `[LLaRA §5.1.3 p.7]` (정답 포함 여부 모호) | 정답 포함 20 (D10) |
| r, α, dropout, K | 8, 32, 0.1, 4 (:128-131) | 지정 안 함 | K = 4 `[논문 p.9 §4.3.2]` | 8, 32, 0.1, 4 (D7) |
| padding item id | 3581 (:108)과 경로 분기(:142-147)는 읽히지 않는 죽은 인자 | 지정 안 함 | 없음 | 1682 고정 `U data/movielens_data.py:18` (M1) |

### 4.5 버릴 것 (알려진 문제)
- **숨은 기본값:** batch, accumulation, lr 값(1e-4/1e-6/1e-6 vs 8e-4/8e-6/8e-6)을 빠뜨리면 조용히 다른 실험이 되고, router와 dataset을 빠뜨리면 실행이 실패함 (`unshare`는 `U model/model_interface.py:230` AttributeError, steam 기본 경로 없음)
- **죽은 경로 분기:** `data_dir`에 "movielens"가 있으면 `args.padding_item_id`를 1682로 바꾸지만(`U main.py:142-147`) 이 인자는 읽히지 않음. 실제 padding id는 `U data/movielens_data.py:18`의 1682 고정
- **잘못된 인자 타입:**
  - `--devices type=list` (:78)
  - `--if_rand type=bool` (:136): 빈 문자열이 아닌 모든 문자열("False" 포함)이 True. 단 이 인자는 읽히지 않음
  - `--lora_r type=float` (:128): 기본값 8은 int로 남아 r//K = 2로 동작. 명령행으로 넘기면 8.0이 되어 `U model/peft/tuners/moelora.py:616`의 `nn.Linear(in_features, 8.0)`에서 층 생성 실패
- **쓰이지 않는 인자:** `local_rank`(기본 3), `aug_prob`, `load_best`·`load_dir`·`load_ver`·`load_v_num`, `if_rand`, `padding_item_id`, `no_augment`
- **읽히지만 깨졌거나 효과 없는 인자:** `peft_dir`(없는 `self.llm_model` 참조로 AttributeError, `U model/model_interface.py:321-322`), `auto_lr_find`(`U main.py:59-65`가 lr finder를 돌리지만 training_step의 수동 scheduler가 매 micro-batch lr을 덮어써 무의미)
- **router 기본값 `unshare` 경로:** 실행 불가. router를 만들지 않는데(`U model/model_interface.py:30-31`) configure_optimizers(:230)와 generate(:93)가 `self.router`를 씀
- **결과 덮어쓰기:** `output_dir`의 `test.csv`와 `valid.csv`를 매번 덮어씀 `U model/model_interface.py:179, :214`. 실행 식별자 없음
- **출처 기록 없음:** commit, 환경, 입력 파일 해시가 결과에 남지 않음 (fork에서 `metrics.json`에 일부 추가)
- **seed 처리:** `pl.seed_everything` 하나에 의존 `U main.py:40`. 후보 추출(`U data/movielens_data.py:27`), 템플릿 선택(`U data/data_interface.py:32`), curriculum 추첨(`U data/data_interface.py:39`)이 DataLoader worker의 Python `random`에 묶여 worker 수에 따라 결과가 달라짐

### 4.6 설계

#### 4.6.1 코드 폴더 구성 `[결정]`
```
rebuild/
  DESIGN.md
  RUNBOOK.md         M14: VESSL 실행 순서와 정확한 명령
  main.py            진입점: python rebuild/main.py <명령> --config <파일> [--set 키=값 ...]
                     재개: python rebuild/main.py train --resume <실행 폴더> (--config·--set 없음)
                     명령: train | eval | export | cluster | smoke | check-env
                     (스스로 경로를 잡아 실행 위치와 무관)
                     종료 코드: 0 완료, 2 인자·설정 오류 ("config error: ..."), 1 그 밖의 예외
  ilora/             파이썬 패키지, 파일과 모듈 번호가 대응
    config.py        M0 설정 읽기·검사·저장
    sections.py      M0 설정 스키마 (구역별 dataclass, 키 87개)
    run.py           M0 실행 폴더, meta, 환경 점검, 경로·해시, 로깅
    seeding.py       M0 난수
    precision.py     M0 autocast·GradScaler 도구
    commands.py      M0 명령 분기 (main.py는 인자 해석만)
    data.py          M1
    candidates.py    M2
    sasrec.py        M3
    prompts.py       M4
    batching.py      M5
    projector.py     M6
    router.py        M7
    moe_lora.py      M8
    model.py         M9
    train.py         M10
    evaluate.py      M11
    records.py       M12 기록
    analysis.py      M12 gate 분석·그림
    cluster.py       M15
    smoke.py         M13 smoke 계획과 검사
  configs/           M14 프로파일 (YAML 17개)
  tests/             M13 (conftest.py, refs_upstream.py 공개 코드 기준 구현, vessl/ VESSL 전용 시험)
  pytest.ini         M13 marker (slow, vessl, criterion)
  tools/             convert_sasrec.py (SASRec pickle → state_dict), compare_runs.py (실행 비교표)
```
- 저장소의 다른 폴더를 import하지 않음. 패키지 이름 `ilora`는 기존 코드에 없는 이름
- 실행 산출물은 `runs/rebuild/` 아래에만 쓰고, 저장소 안 다른 경로에는 쓰지 않음

#### 4.6.2 설정
- **형식:** YAML + dataclass로 타입 검사 `[결정]` (PyYAML은 공개 requirements에 이미 있음)
  - `extends: <파일>`(그 프로파일 기준 상대 경로)로 다른 프로파일을 이어받음. 구역 안에서는 키 단위로 덮고, 순환은 오류 (18.6.1의 E1·E4·변형) `R ilora/config.py load_raw`
  - `--set 구역.항목=값`의 값은 YAML로 해석하고, 이어받은 결과에 있는 키만 덮음. 실수 자리의 `1e-4`처럼 PyYAML이 문자열로 읽는 숫자 표기는 실수로 받음 `R ilora/config.py load_config`
- **구역:** `run`, `data`, `candidates`, `sasrec`, `prompt`, `batching`, `projector`, `router`, `lora`, `model`, `train`, `eval`, `records`, `cluster`
- **규칙:**
  - 코드 안 기본값을 두지 않음: `extends`를 푼 프로파일에 모든 키(87개)가 있어야 하고, 빠지면 실행 거부 `[결정]`: 의미를 바꾸는 키를 따로 고를 필요가 없고, 5.6.8('M0 규칙상 모든 키 필수')과 18.6.2가 이 규칙을 전제함 `R ilora/config.py _build_section`
  - 알 수 없는 키는 오류 (오타가 조용히 무시되지 않게)
  - 타입 엄격: bool은 true/false만, 정수 자리에 실수는 거부, 선택지가 정해진 항목은 목록 밖 값 거부
  - 값 검사 `[결정]`: 원소가 정해진 목록 키(`lora.target_modules`, `eval.checkpoints`·`eval.splits`, `records.export_checkpoints`·`records.analysis_splits`·`records.figure_formats`)는 비어 있지 않고 중복 없음, 실수는 유한값, sha256 키(`sasrec.source_sha256`, `sasrec.state_digest`, `batching.tokenizer_sha256`)는 소문자 16진 64자(YAML에서 따옴표로 적음), 선택 경로 키는 null 또는 빈칸이 아닌 문자열. 이유: 12.6.7·16.6.9가 '⊆ {…}' 꼴로만 적은 목록 키의 빈 목록·중복 허용 여부를 15.6.12의 eval 목록 규칙과 같게 정함 `R ilora/sections.py`
  - 명령별 필수 키: `eval`은 `eval.source_run`, `export`는 `records.source_run`이 없으면 실행 폴더를 만들기 전에 `ConfigError` `R ilora/commands.py command_requirements`
  - `--set 구역.항목=값`으로 덮어쓴 항목은 `meta.json`에 따로 기록
  - 항목 정의마다 근거 라벨(`[논문]` `[v1]` `[LLaRA]` `[LLaRA 간접]` `[코드]` `[결정]` `[SPEC]`)을 `config.item(label, doc)`의 필드 metadata로 적고(라벨이 없거나 형식이 틀리면 정의할 때 오류), 해당 모듈 문서와 일치시킴 `[결정]`: 주석은 시험(4.8 #2)이 읽을 수 없음 `R ilora/config.py item`
- **프로파일 (M14, 18장):** `paper_spec.yaml`(E2), `llara_baseline.yaml`(E1, K = 1), `cluster_hard.yaml`(E4), `smoke_cpu.yaml`, `smoke_gpu_100.yaml`, `overfit_gpu.yaml`, 변형 11개(`variant_*.yaml`). 모두 17개
- **키 표:** 각 구역의 키는 해당 모듈 장의 설정 절에서 정의하고, 전체 87개 키와 `paper_spec` 값은 18.6.2에 모은다. `run` 구역은 M0 소유:

  | 키 | 타입 | paper_spec | 근거 | 설명 |
  |---|---|---|---|---|
  | `seed` | int | 1234 | `[SPEC §0.6]` D2, 값 계보 `[코드]` `U main.py:84`, `L main.py:84` | 모든 `derive_seed`의 main seed (단일 실행) |

- **경로 값:** 경로 키는 절대 경로 또는 저장소 루트 기준 상대 경로(작업 폴더 기준이 아님). `run.resolve_path` 하나로 해석. 경로 키 8개(`sasrec.source_path`, `sasrec.state_path`, `sasrec.reference_path`, `prompt.template_path`, `eval.source_run`, `records.source_run`, `cluster.artifact_dir`, `cluster.dynamic_export_dir`)를 푼 값은 `meta.json` `paths`에 남김 `R ilora/commands.py resolved_paths`

#### 4.6.3 실행 폴더와 출처 기록
- **위치:** `runs/rebuild/`. 저장소 `.gitignore`의 `runs/`에 이미 포함되어 commit되지 않음
- **이름:** `<프로파일>_<UTC 시각>`(`YYYYMMDDTHHMMSSZ`). 이미 있으면 실행 거부 (덮어쓰기 금지). 프로파일 이름은 명령과 무관하게 `--config` 파일 이름(확장자 제외)이라, 같은 프로파일로 돌린 train·eval·export·cluster·check-env 폴더가 같은 접두어를 가짐 `[결정]`: RUNBOOK의 `latest <프로파일>`이 이 접두어로 가장 최근 폴더를 찾음 (18.6.6). `train --resume`은 새 폴더 없이 원래 폴더를 다시 엶. smoke 단계는 `<smoke 실행>/steps/<단계>_<이름>_<UTC>/`(17.6.6), 비교 도구(M14)는 `compare_<UTC>/` `R ilora/run.py create_run_dir`
- **`meta.json` 항목:**
  - 코드: git commit, 작업 폴더 변경 여부, `rebuild/` 파일 해시
  - 환경: Python, torch, CUDA, cuDNN, transformers, tokenizers, sentencepiece, numpy, pandas, scikit-learn, PyYAML 버전, GPU 이름·개수·메모리, 호스트 이름, torch CPU thread 수 `torch_num_threads` (15.6.10) `R ilora/run.py environment_info`
  - 입력 파일 해시: 데이터 파일 4개, SASRec 가중치(변환본과 원본), prompt 파일, LLaMA `config.json`과 그 안의 `torch_dtype`. 두 블록으로 나눔 `[결정]`: `inputs` = `run.EXPECTED_INPUTS` 6개(데이터 4개, `prompt/movie.txt`, 원본 `rec_model/movielens.pt`)의 현재 sha256(`run.input_hashes`), `assets` = 변환 SASRec state·기준 파일, `tokenizer.model`, LLaMA `config.json`(sha256과 `torch_dtype`), cluster_hard면 군집 `manifest.json` (파일이 없으면 sha256 null). 이유: `inputs`는 기대값이 고정된 저장소 파일(4.6.5 표)이고 `assets`는 위치가 설정·환경 변수로 정해지는 파일임. M10 checkpoint(14.6.6)와 M11 eval 명령(15.6.11)은 두 블록을 함께 대조 `R ilora/commands.py asset_record`
  - 실행: 명령(`command`), 프로파일과 그 파일 경로, 명령행, 확정 설정의 해시, 덮어쓴 항목, 시드, 경로 키를 푼 값(`paths`), `llm_path`·`tokenizer_dir`, eval·export의 원 학습 실행(`source_run`), 공통 준비 결과(`preparation`: 디스크 여유, 환경 점검, `self_check`), 시작·종료 시각(UTC), 경과 시간, 종료 상태와 오류 메시지
    - `config_hash` = 확정 설정을 key 정렬 JSON(UTF-8)으로 쓴 sha256. `overrides` = `--set` 값을 YAML로 해석한 dict `R ilora/config.py config_hash`
    - 경과 시간 `elapsed_s` = 마지막 `ended_utc` − 첫 `started_utc` (벽시계) `[결정]`. `train --resume`은 `started_utc`를 두고 `ended_utc`·`elapsed_s`·`error`를 비운 뒤 `resumes`에 {`utc`, `argv`, `previous_status`, `previous_ended_utc`, `previous_error`, `complete_epochs`, `start_epoch`}를 더함. 그래서 재개한 실행의 `elapsed_s`는 멈춘 시간을 포함함. 이유: 강제 종료된 구간의 끝 시각은 측정되지 않아 구간 합을 만들 수 없고(`[SPEC §0.14]`), 계산 시간은 M10 `TrainSummary.seconds`가 따로 셈 (18.6.7 ElapsedTime·WallClock) `R ilora/run.py finish_meta`, `R ilora/commands.py _reopen_for_resume`
  - 기록 금지: 환경 변수 전체, token·비밀번호, `HF_TOKEN` 등 이름에 TOKEN·KEY·SECRET·PASSWORD가 들어간 값. 명령행은 이름에 TOKEN·KEY·SECRET·PASSWORD·PASSWD·CREDENTIAL이 든 인자의 값과 token 형태 문자열(`hf_`·`ghp_`·`gho_`·`github_pat_`·`sk-`·`AKIA`로 시작하는 긴 값)을 `***`로 가리고, 그런 문자열이 남은 JSON은 쓰지 않음(`SecretLeakError`) `[결정]` `R ilora/run.py scrub_argv`, `R ilora/run.py write_json_atomic`
  - 모듈 기록: 모듈은 `run.update_meta(run_dir, fields)`(최상위 key 단위 병합, 원자적 쓰기, 위 기록 금지 검사)로 블록을 더함: `train`(M10: 프로세스별 `segments`, epoch별 시간·메모리, `step_counts`, `trainable_params`, `parameter_report`, 새 토큰 행의 `generated_frozen_sha256`(13.8 #14), `summary`), `cluster_live_check`(cluster_hard 학습, M15 `verify_live_encoder` 결과), `smoke`(M13), `compare`(M14 비교 도구. 그 입력 목록 `compare_inputs`는 시작 때 `start_meta`가 씀) `R ilora/run.py update_meta`
- **비정상 종료 구분:** 시작할 때 상태를 `running`으로 쓰고 끝날 때 `completed`/`failed`로 갱신, 예외 메시지 기록. 강제 종료(SIGKILL, OOM 등)는 `running`이 남음
  - **`run.lock`** `[결정]`: `run_command`는 실행 폴더를 만든 직후(재개는 `meta.json`을 읽거나 쓰기 전) `<실행 폴더>/run.lock`에 배타적 비차단 OS 잠금(Linux `fcntl.flock(LOCK_EX | LOCK_NB)`, Windows `msvcrt.locking(LK_NBLCK)`)을 잡고 명령이 끝날 때까지 쥠. 잠금은 프로세스가 어떻게 끝나든 커널이 풀어 오래된 잠금이 없고, 잡힌 잠금은 살아 있는 프로세스를 뜻함. 파일은 덧붙이기로만 열고 지우거나 비우지 않으며, 잡은 프로세스마다 한 줄(UTC, 호스트, pid, 명령)을 남김. 다른 프로세스가 쥔 폴더의 `train --resume`은 `CommandError`(`tmux attach -t rebuild` 안내)로 거부하고 `meta.json`·`config.yaml`·`run.log`를 바꾸지 않음. 잠금을 지원하지 않는 파일 시스템이면 `run.log`에 경고를 남기고 계속함. 이유: 셸이 끊겨도 tmux 안의 학습은 살아 있어, 같은 폴더에 두 번째 프로세스가 붙으면 기록 CSV와 checkpoint가 망가짐. pid 확인은 컨테이너의 pid 재사용으로 틀릴 수 있어 쓰지 않음 (18.6.6 R7, 14.6.10) `R ilora/run.py hold_run_lock`, `R ilora/commands.py run_command`

#### 4.6.4 난수
- 모든 난수원은 `derive_seed(main_seed, *keys)`로만 만든다 (3.1). 쓰는 key:
  - `("train_order", epoch)`: DataLoader 순서용 `torch.Generator`
  - `("curriculum", tau)`: micro-batch마다 hybrid 추첨 (D3)
  - `("candidates", split, row)`(평가), `("candidates", "train", row, epoch)`(학습): 후보 (D10)
  - `("template", split, row)`(평가), `("template", "train", row, epoch)`(학습): 템플릿 (D15)
  - `("smoke_users", split)`: smoke 사용자 선택 (D14). `data.train_rows`가 비어 있지 않으면 train은 추첨하지 않음
  - `("init", 이름)`: 모듈 초기화 (`torch_seed_scope`, CPU 난수만 고정·복원). `gate`(dynamic만, M7), `projector`(M6), `lora`(M8, 층 0–31 × q, k, v, o, gate, up, down 순서), `new_tokens`(M9 새 토큰 임베딩), `tiny_llama`(M9 CPU 시험용 backbone)
  - `("torch_global",)`: dropout과 DataLoader 기본 seed 등 전역 torch 난수. `build_model` 뒤 학습 시작 시 1회 (M10)
  - `("test", 이름)`, main seed 0: M13 합성 시험 입력 전용
  - 예외: KMeans `random_state`는 `cluster.random_state`(1234)를 그대로 쓰고 `derive_seed`를 거치지 않음 `[SPEC §7]` (M15)
- epoch은 모든 key에서 0부터 센다 (`train_order`, 학습 후보, 학습 템플릿). `tau`만 1부터 (D3)
- 후보, 템플릿, curriculum 추첨은 worker 수나 worker 상태가 아니라 key로 결정 → worker 수(0, 2, 8)와 무관
- **평가:** 같은 checkpoint와 같은 설정이면 결과가 같아야 함 (같은 GPU·같은 소프트웨어 기준). CPU에서는 torch 스레드 수도 같아야 함: 행렬 곱의 합산 순서가 스레드 수에 따라 달라 ω·z 끝자리가 바뀜 (측정: 1 스레드로 학습·평가한 초소형 실행을 14 스레드로 다시 평가하면 생성 토큰은 같고 gate 값이 달라 `records.jsonl` sha256이 다름). CPU 재현 시험은 1 스레드로 돌림 (15.6.10)
- **학습:** GPU 연산의 비결정성은 허용 (D16)

#### 4.6.5 실행 전 환경 점검 (실패 시 즉시 중단)
- 버전: torch 2.0.0, transformers 4.28.0 (D13), numpy 1.24.2·pandas 1.5.3 (배포 `.df` unpickle과 numpy 난수 재현, M1·M2), sentencepiece 0.1.97 (M5), scikit-learn 1.2.2 (M15)
- transformers 설치 무결성: 설치된 모든 `.py`의 sha256이 wheel RECORD와 같음. fork 환경처럼 `debug/` 사본으로 바뀐 파일이 있으면 실패 (D13)
- GPU 프로파일(`model.device = cuda:0`): CUDA 사용 가능, 보이는 GPU 수가 정확히 1(CUDA:0만 사용) `[SPEC §0.12]`, GPU 0의 전체 메모리 75 GiB 이상 (A100 80GB). CPU smoke 프로파일은 이 항목 생략
  - 점검 항목은 명령이 아니라 프로파일로 정함 `[결정]`: export·cluster처럼 GPU를 쓰지 않는 명령도 `cuda:0`·`llama2_7b` 프로파일이면 GPU와 `LLM_PATH`를 요구함. 이유: 위 항목이 프로파일 단위로 정의되어 있고 VESSL 단계(18.6.6 R10·R11)는 모두 같은 A100 기계에서 돌림. 로컬 시험 M15-11은 `smoke_cpu` + `data.subset=full`로 군집 명령을 돌림 `R ilora/commands.py environment_check`
  - 여유 메모리는 보지 않음 `[결정]`: 기준값을 A100에서 먼저 재야 하고(`smoke_gpu_100`은 한 프로세스에서 단계마다 점검을 되풀이), 같은 실행 폴더의 두 번째 프로세스는 `run.lock`(4.6.3)이 막고, GPU 작업을 한 번에 하나만 돌리는 규칙은 RUNBOOK이 지킴. 단계 사이 여유 메모리 측정은 VESSL 후속 (TBD)
- LLaMA (`llama2_7b` 프로파일만): 경로(`LLM_PATH`가 없으면 실패), shard 파일(`pytorch_model*.bin` 또는 `model*.safetensors`), `tokenizer.model`, `config.json`의 `torch_dtype`이 float16 (D9). 공개 mirror NousResearch/Llama-2-7b-hf의 config.json은 `"torch_dtype": "float16"`. VESSL 파일 자체는 `VESSL_REQUIRED`
- 입력 파일 해시가 기대값과 같음 (데이터, SASRec 가중치). `train_data.df`는 iLoRA판(blob `73b47263`, `cluster` 열 추가)과 LLaRA판(sha256 `a73b119d…`)이 다르므로 기준 파일을 명시하고 `(seq, len_seq, next)` 내용 일치를 따로 확인
- **텍스트 입력은 CRLF→LF 정규화 후 해시** `[결정]`: 로컬 Windows 저장소가 `core.autocrlf=true`라 `u.item`, `movie.txt`의 작업 트리 바이트가 Linux(VESSL)와 다름. 정규화한 값은 git blob 해시와 같음(확인함). 텍스트 파일은 항상 텍스트 모드(universal newline)로 읽어 두 환경의 파싱 결과를 같게 함
- 기대 해시 (sha256, 텍스트는 LF 기준)

  | 파일 | sha256 |
  |---|---|
  | `data/ref/movielens/train_data.df` | `bac4c88b11e6f76d914c2b22aadbb97315a78432bb79f424f7015775053b9546` |
  | `data/ref/movielens/Val_data.df` | `9a950900d0055649e2983ef403d189d5fdc5bd4f84d229a377806d371b32e460` |
  | `data/ref/movielens/Test_data.df` | `d38308a0078d4518dc9eeb619c8d1afb53d830712cc7d36ac51cc47eaafc97d6` |
  | `data/ref/movielens/u.item` | `553841ebc7de3a0fd0d6b62a204ea30c1e651aacfb2814c7a6584ac52f2c5701` |
  | `prompt/movie.txt` | `957fc1f27bb930ca5acbe1b545186ecc04fd93ea2c5307644c706ad44395ecd3` |
  | `rec_model/movielens.pt` | `6928a06778d0e5b64ead0641501f73f13beb2c61ca2d881ee15582630d101df9` |
- 실행 폴더 쓰기 가능(폴더를 만들 때 확인), 디스크 여유 `[결정]`: 실행 폴더가 있는 디스크의 여유 공간 ≥ 1 GiB, `llama2_7b`의 `train`·`smoke`는 `train.epochs` × 424 MiB(14.6.6 checkpoint 크기)를 더함 (paper_spec 학습: 1 + 5 × 0.414 = 3.07 GiB). 설계에 값이 없어 checkpoint를 모두 쓸 공간만 확인함 `R ilora/commands.py disk_check`
- **`check-env` 명령** `[결정]`: 위 점검 뒤 모듈 코드가 필요한 점검 3개를 모두 돌림: 데이터 적재(M1 `load_dataset`), SASRec(원본 pickle sha256 = `sasrec.source_sha256`, M3 `SASRecEncoder.from_config` digest, `verify_against_reference`: CPU 차 0, CUDA 1e-4 이하, 7.8 #5·#12. 예외든 `passed` 거짓이든 실패), tokenizer(M5 `load_tokenizer`). 결과를 `check_env.json`(`passed`, `device`, `checks: [{check, passed, result 또는 error}]`)에 쓴 뒤 실패를 모아 한 번에 `EnvironmentCheckError`로 냄. 이유: VESSL에서 한 번 실행으로 고칠 것을 모두 보고 결과 파일을 남김 (18.6.6 R4) `R ilora/commands.py check_env`

#### 4.6.6 로깅
- 표준 `logging`(logger `ilora`, 모듈은 `ilora.<모듈>`). 콘솔과 `run.log`(UTF-8)에 같은 내용. 첫 줄은 `run_dir=<경로>`, 둘째 줄은 명령·프로파일·재개 여부, 셋째 줄은 `run.lock` 상태(잡았는지, 못 잡았으면 경고와 이유) (17.6.12 제안 3) `R ilora/commands.py run_command`
- 줄 시각은 `logging` 기본 asctime인 실행 기계의 지역 시각(시간대 표시 없음, 초 단위) `[결정]`. 실행 폴더 이름과 `meta.json`·기록 파일의 시각은 UTC이고 판정·비교는 그 값만 씀. 로그와 맞춰 볼 때는 기계의 시간대를 고려함 `R ilora/run.py setup_logging`
- 진행 표시줄 대신 일정 간격의 요약 줄 (baseline 로그 17,924,937 byte 중 99.6%가 진행 표시줄 갱신이었음)
- 지표 기록 형식은 M12에서 정함

### 4.7 결정
- D9, D13 확정, D2 seed, D16 결정성 수준 (20장)

### 4.8 완료 기준 (M13에서 테스트로 구현)
1. 설정을 저장했다가 다시 읽으면 같음. 알 수 없는 키, 잘못된 타입("False" 문자열의 bool, 정수 자리의 8.0), 빠진 항목, 선택지 밖 값은 모두 오류로 거부됨
2. 모든 설정 항목에 근거 라벨이 있음 (라벨 없는 항목이 있으면 테스트 실패)
3. 같은 이름의 실행 폴더는 거부됨. `meta.json`에 필수 키가 모두 있고, token 형태 문자열이나 환경 변수 전체가 없음
4. `derive_seed`가 같은 입력에 같은 값, key가 다르면 다른 값. worker 수(0, 2, 8)가 달라도 학습 batch 순서, 후보, 템플릿, curriculum 추첨이 같음
5. 버전 불일치, GPU 개수 ≠ 1, LLaMA 경로 없음, `torch_dtype` 불일치, 데이터 해시 불일치를 각각 모의로 넣으면 환경 점검이 실패함
6. 예외나 강제 종료로 끝난 실행을 `meta.json` 상태로 구분할 수 있음. 강제 종료로 `running`이 남은 폴더와 살아 있는 실행은 `run.lock`으로 구분함: 살아 있는 프로세스가 쥔 폴더의 `train --resume`은 거부되고 `meta.json`이 그대로이며, 프로세스가 끝난 폴더는 다시 열림

---

<!-- MODULE_SECTIONS_BEGIN -->

## 5. M1 데이터
상태: 테스트 통과(로컬)

### 5.1 목적과 범위
- **범위:** LLaRA가 배포하고 iLoRA가 그대로 받은 MovieLens100K 처리 파일 4개(`train_data.df`, `Val_data.df`, `Test_data.df`, `u.item`)를 읽어 `ItemCatalog`와 `Sample` 목록을 만든다
  - len_seq 필터, padding 제거, 제목 정규화(D20), pseudo 사용자 ID(D14), smoke 부분집합(D14), 입력 내용 검증
- **하지 않는 것:** 분할을 다시 만들지 않음, 원본 `u.data`를 쓰지 않음(저장소에 없음), 후보(M2)·prompt(M4)·tensor 변환(M5) 없음
- **범위 밖:** iLoRA판 `train_data.df`의 `cluster` 열은 읽자마자 버리고 어디에도 쓰지 않음 `[SPEC §0.8, §7]`
- 기대 해시 표의 관리는 M0(4.6.5). M1은 그 표로 unpickle 전에 확인만 함

### 5.2 입력과 출력
**입력** (저장소 기준 고정 경로 `data/ref/movielens/`, M0이 해시를 점검하는 파일과 같은 파일)

| 파일 | 형식 (계산으로 확인) | 내용 |
|---|---|---|
| `train_data.df` | pandas pickle, `DataFrame` 68,388행, `RangeIndex(0, 68388, 1)` | 열 `seq, len_seq, next, cluster` (`cluster`는 int32, 사용 금지) |
| `Val_data.df` | 94행, `RangeIndex(0, 94, 1)` | 열 `seq, len_seq, next` |
| `Test_data.df` | 95행, `RangeIndex(0, 95, 1)` | 열 `seq, len_seq, next` |
| `u.item` | ISO-8859-1 텍스트 1,682줄, 줄마다 `\|` 구분 필드 24개 | 필드 0 = movie_id(1–1682, 줄 순서), 필드 1 = 제목 |

- 설정: `Config.data` (5.6.8), smoke 선택에만 `Config.run.seed`

**출력** (3.1 계약)
- `ItemCatalog`: `titles` 길이 1,682 (index = item id 0–1681), `num_items = 1682`, `pad_id = 1682`
- `Sample` tuple (split마다 `row` 오름차순): full이면 train 66,126, validation 94, test 95. smoke는 5.6.7
- 반환 묶음 `LoadedData` (계약 추가 제안, 5.6.1). pandas 객체는 load 함수 밖으로 나가지 않음
- 파일을 쓰지 않음. 요약은 로그로만 (입력 해시 기록은 M0 `meta.json`)

### 5.3 논문 근거
- 데이터 출처: "We request the dataset from LLaRA [9] and maintain exactly the same experimental settings as described in the original paper." `[논문 p.6 §4]`
- "we choose the MovieLens100K dataset for our experiment" `[LLaRA §5.1.1 p.7]`, 943 sequence·1,682 item·100,000 interaction `[LLaRA Table 1 p.7]`
- 분할: "we arrange sequences chronologically and divide the data into train, validation, and test subsets at a ratio of 8:1:1" `[LLaRA §5.1.1 p.7]`
  - 배포 파일은 sequence(사용자) 단위 분할과 맞음: 754 + 94 + 95 = 943 (5.6.5)
- 이력: "we retain the last 10 interactions as the historical sequence, padding sequences with fewer than 10 interactions." `[LLaRA §5.1.1 p.7]`
- 표본: "a historical interaction sequence represented as i<n = [i1, i2, ..., in−1], where each ij is an item interacted with at the j-th step" `[논문 p.3 §2]` → (이력 i<n, 다음 아이템 i_n), 이력은 오래된 것부터
- 아이템 텍스트 = 제목: "contains user ratings and movie titles" `[LLaRA §5.1.1 p.7]` `[v1 App B p.15]` (최종본은 "containing user ratings and metadata for movies" `[논문 App B p.16]`), "The items in the textual prompts are represented by their respective titles." `[LLaRA §5.3 p.8]`
- 논문에 없는 것: len_seq 필터, padding id·방향, 제목 정규화 방법, 사용자 ID, 평점 사용 여부 → `[코드]` 또는 `[결정]`

### 5.4 공개 코드에서 가져올 것
`U data/movielens_data.py`와 `L data/movielens_data.py`는 같은 blob `2bf0e156` (코드 계보 일치). 라인 번호는 둘 다 같음.
- 배포 파일을 `pd.read_pickle`로 그대로 읽음 `:78`. 파일 이름 `:52-57`, 경로 `data/ref/movielens` `U train_movielens.sh:8`
- 모든 split에 `len_seq >= 3` `:79` (논문 근거 없음)
- padding `(1682, 0)` `:18-19`, padding 제거 `:80-88`, 평점은 버리고 item id만 씀 `:95-101`
- item id = movie_id − 1, u.item은 ISO-8859-1로 줄마다 `strip('\n').split('|')` `:71-74`
- 제목 정규화: 끝 7자 제거 `:74` (`ll[1][:-7]`), 관사 이동 `:61-66` (`get_mv_title`). paper_spec은 이 동작 그대로 (D20)

### 5.5 버릴 것 (알려진 문제)
- **예외 삼킴:** `remove_padding`은 위치 검사 없이 `(1682, 0)`을 최대 10번 지우고 맨 `except: break`로 끝냄 `:83-86` → 오른쪽 정렬을 검사하고 어긋나면 오류
- **숨은 기본값:** `cans_num=10` `:13` (실제 20은 `U main.py:122`가 넘김), 쓰이지 않는 `no_augment`·`aug` `:15, :17`
- **표본 번호:** 공개 코드와 fork는 필터 후 위치를 씀 (`iloc[i]` `:26`, `F data/movielens_data.py:30`의 `sample_idx`) → 필터 전 행 번호 `row`로 바꿈 (3.1). 대응 규칙은 5.6.4
- **Dataset 안 후보 추출:** `__getitem__`마다 Python `random`으로 재추출 `:27, :44-48` → M2로 옮김 (D10)
- **`cluster` 열:** 공개 코드도 읽지 않지만 DataFrame에 그대로 남음 → 읽은 즉시 3개 열로 투영
- **제목 결함 (D20: paper_spec에서는 고치지 않고 기록)** u.item 재계산 (계산: `design/tmp/m1_titles.py`)
  - `", A"`를 `", An"`보다 먼저 검사해 4개 손상: 435 `A American Werewolf in Londonn`, 601 `A American in Parisn`, 965 `A Affair to Remembern`, 1226 `A Awfully Big Adventuren`
  - `s.replace(", The", "")`가 모든 위치를 지움: 176 `Good, The Bad and The Ugly, The` → `The Good Bad and The Ugly` (기존 목록에 없던 결함)
  - 괄호 속 `", The"`를 앞으로 옮김: 1004 → `The Double vie de Véronique, La (Double Life of Veronique)`, 1018 → `The Die xue shuang xiong (Killer)` (기존 목록에 없던 id)
  - 연도 없는 제목: item 266 `unknown` → 빈 문자열
  - 잘림: item 1411 `... Giving (1995) (V)` → `Land Before Time III: The Time of the Great Giving (19`
  - 끝 공백: 1127, 1200, 1634 (원본이 `(연도) `로 끝남)
  - 중복: 정규화 제목 22개를 id 45개가 공유. 원본 제목 그대로의 중복 18개(36 id)에 연도 제거로 4개 제목(Cape Fear, Shall We Dance?, Sabrina, The Scarlet Letter, 8 id)이 생기고 That Darn Cat! 묶음에 1443이 더해짐. 소문자 비교도 같음
  - 학습 영향: item 266이 정답인 train 표본 6개(응답이 줄바꿈뿐, M5), 이력 등장 76회. validation·test에는 266이 이력·정답으로 없음
  - 판정 영향은 M2 flag(6.6.4)로 표본마다 표시

### 5.6 설계

#### 5.6.1 공개 API (`rebuild/ilora/data.py`)
```python
NUM_ITEMS: Final = 1682;  PAD_ID: Final = 1682;  PAD_RATING: Final = 0
HISTORY_LEN: Final = 10;  MIN_LEN_SEQ: Final = 3
SPLITS: Final = ("train", "val", "test")
DATA_DIR: Final = "data/ref/movielens"               # 저장소 기준
SPLIT_FILES: Final = {"train": "train_data.df", "val": "Val_data.df", "test": "Test_data.df"}
ITEM_FILE: Final = "u.item"
USER_OFFSET: Final = {"train": 0, "val": 754, "test": 848}
EXPECTED_ROWS: Final = {"train": 68388, "val": 94, "test": 95}      # 필터 전 행 수 (5.6.3 7단계)
EXPECTED_SAMPLES: Final = {"train": 66126, "val": 94, "test": 95}   # 필터 후 표본 수. M11 표본 수 확인(15.6.2)과 M15 full 판정(19.6.7 1단계)이 import
TRAIN_USERS: Final = 754
EXPECTED_FINGERPRINTS: Final[dict[str, str]]                        # 5.6.9 기대값 (VESSL 시험 5.8 #12도 씀)

class DataError(RuntimeError): ...

@dataclass(frozen=True)
class ItemCatalog:                       # 3.1
    titles: tuple[str, ...]
    title_mode: str                      # "code" | "fixed" (계약 추가 제안)
    num_items: int                       # 항상 1682
    pad_id: int                          # 항상 1682

@dataclass(frozen=True)
class Sample:                            # 3.1
    split: str; row: int; user_id: int
    history: tuple[int, ...]; target: int
    @property
    def sample_id(self) -> str: return f"{self.split}:{self.row}"

@dataclass(frozen=True)
class RawRow:                            # 필터 전 한 행
    row: int; len_seq: int; history: tuple[int, ...]; target: int

@dataclass(frozen=True)
class LoadedData:                        # 계약 추가 제안
    catalog: ItemCatalog
    train: tuple[Sample, ...]; val: tuple[Sample, ...]; test: tuple[Sample, ...]
    subset: str                          # "full" | "smoke"
    smoke_users: tuple[int, int, int] | None   # smoke면 고른 (train, val, test) 사용자 수, full이면 None (M11 15.6.2 표본 수 확인)
    def samples(self, split: str) -> tuple[Sample, ...]: ...   # 없는 split이면 DataError
    @property
    def title_mode(self) -> str: ...     # = catalog.title_mode (M11 15.6.8이 data.title_mode로 읽음)

def read_raw_titles(path: Path) -> tuple[str, ...]
def code_title(raw: str) -> str          # U :61-66, :74 그대로
def fixed_title(raw: str) -> str         # variant_fixed_titles (5.6.6)
def build_catalog(raw_titles: Sequence[str], title_mode: str) -> ItemCatalog
def title_report(raw_titles: Sequence[str]) -> dict[str, tuple[int, ...]]   # 결함 범주별 id
def project_frame(obj: object, split: str) -> pd.DataFrame                  # 형식·열 검사 후 3개 열
def frame_fingerprint(df: pd.DataFrame) -> str
def parse_rows(df: pd.DataFrame, split: str) -> tuple[RawRow, ...]
def assign_users(rows: Sequence[RawRow], split: str) -> tuple[int, ...]
def to_samples(rows: Sequence[RawRow], users: Sequence[int], split: str) -> tuple[Sample, ...]
def load_dataset(cfg: Config, repo_root: Path) -> LoadedData
def select_smoke(data: LoadedData, main_seed: int, sizes: tuple[int, int, int], *,
                 train_rows: tuple[int, ...]) -> LoadedData          # train_rows = () 이면 5.6.7 기본 규칙
def pad_histories(histories: Sequence[Sequence[int]], pad_id: int, length: int
                  ) -> tuple[torch.Tensor, torch.Tensor]          # (hist_ids [N, length], len_seq [N]) CPU int64, 오른쪽 padding (5.6.13)
```
- `Sample.__post_init__`: split ∈ SPLITS, row ≥ 0, user_id ≥ 0, `history`는 tuple, 3 ≤ len(history) ≤ 10, `row`·`user_id`·모든 id가 Python int(bool·numpy 정수 제외)이고 id는 0–1681, 이력 중복 없음, target ∉ history. 어기면 `DataError`
- `ItemCatalog.__post_init__`: len(titles) == num_items == 1682, pad_id == 1682, 모든 원소 str, title_mode ∈ (`code`, `fixed`)
- `LoadedData.__post_init__`: split마다 그 split의 `Sample`만 `row` 엄격한 오름차순으로 담음 (M4 `sample_indices`와 M11 행 순서가 이것을 전제로 함). `subset == "full"`이면 `smoke_users`가 None, `"smoke"`면 int 3개. 어기면 `DataError` `R ilora/data.py LoadedData`
- 합성 데이터 테스트는 `project_frame`·`parse_rows`·`assign_users`·`to_samples`를 직접 부름. `load_dataset`은 실제 배포 파일 전용이라 기대 수를 항상 검사함

#### 5.6.2 파일 형식 (계산으로 확인, `design/tmp/m1_inspect.py`, `m1_verify.py`)
- `seq`: 길이 10 list, 원소는 `(item, rating)` 2-tuple. 실제 항목은 `numpy.int64`, padding `(1682, 0)`은 Python int → 모두 `int()`로 바꾸고 bool은 거부
- padding은 항상 오른쪽: 첫 padding 뒤는 모두 padding, 앞은 padding 없음 (세 파일 전 행)
- `len_seq = max(1, k)` (k = padding 아닌 항목 수). train은 (len_seq, k) = (1, 0) 754행, (1, 1) 754행, (k, k) k = 2–10 (k = 2–5 각 754행, 6·7 각 753행, 8·9 각 752행, 10은 60,854행: 짧은 블록은 긴 window가 없음). validation (9, 9) 2행·(10, 10) 92행, test (10, 10) 95행
- `next`: `(item, rating)` 2-tuple (`numpy.int64`). 정답 item 0–1681, 1682는 정답으로 없음
- 평점 값은 3·4·5뿐 (이력·정답, 세 파일 모두). 1·2점이 없으므로 배포 파일은 원본 100,000 평점 전부가 아님 (원본 `u.data`가 없어 처리 규칙은 확인 불가). 평점은 쓰지 않음
- 이력 안 중복 0건, 정답이 이력에 있는 행 0건 (세 파일 전 행)
- 이력은 오래된 것부터: train에서 다음 행의 이력 = (이전 이력 + 이전 정답)의 마지막 10개 (5.6.5)
- 필터 후: train 66,126 (2,262 = 754 × 3행 제거), validation 94, test 95 (제거 없음). 필터 후 항상 len(history) == len_seq, 길이 3–10 (validation 9–10, test 10)

#### 5.6.3 읽기 순서 (`load_dataset`)
1. 4개 파일 경로 = `repo_root / DATA_DIR / 파일`. 없으면 `DataError`
2. **unpickle 전에** M0 해시 함수 `run.content_sha256(path, text)`로 sha256 계산(`text`는 `run.EXPECTED_INPUTS`의 표시 그대로: 4개 중 `u.item`만 True, CRLF→LF 정규화)하고 같은 표의 기대값(4.6.5)과 비교. 다르면 `DataError(파일, 기대값, 실제값)` (pickle은 임의 코드를 실행할 수 있으므로 확인된 파일만 엶). 두 이름은 M0 코드 `rebuild/ilora/run.py`에 이미 있음
3. `u.item`: 텍스트 모드(universal newline), `encoding="ISO-8859-1"`로 전체를 읽고 `"\n"`으로 나눔. 마지막 빈 원소 1개만 버림
   - `str.splitlines()`는 쓰지 않음 (ISO-8859-1의 `\x85` 등도 줄 끝으로 봄. 현재 파일엔 없음)
   - 줄 수 1,682, 줄마다 필드 24개, `int(필드0) == 줄 번호 + 1`. 아니면 `DataError`
   - `build_catalog(raw_titles, cfg.data.title_mode)`
4. split마다 `obj = pd.read_pickle(path)` → `project_frame(obj, split)`
   - `DataFrame`이 아니면, index가 `RangeIndex(0, len, 1)`이 아니면 오류
   - 열 집합 허용: train `{seq, len_seq, next, cluster}`(iLoRA판) 또는 `{seq, len_seq, next}`(LLaRA판), validation·test `{seq, len_seq, next}`. 그 밖은 오류
   - LLaRA판은 M0 기대 표에 없어 `load_dataset`에서는 2단계에서 거부됨. `project_frame`이 LLaRA판을 받는 것은 fingerprint 비교 도구·테스트용
   - `df.loc[:, ["seq", "len_seq", "next"]].copy()`만 반환. `cluster` 값은 읽지 않음
5. `frame_fingerprint(df)`가 기대값(5.6.9)과 다르면 오류 (pandas 판에 따라 unpickle 결과가 달라지는 경우도 잡음)
6. `parse_rows` → `assign_users`(필터 전 행 전부) → `to_samples`(필터 `len_seq >= MIN_LEN_SEQ`를 여기서 하고 남은 행만 `Sample`로 만듦) `R ilora/data.py to_samples`
7. 기대 수 검사: 행 68,388/94/95, 필터 후 66,126/94/95, train 사용자 754
8. `cfg.data.subset == "smoke"`면 `select_smoke(data, cfg.run.seed, cfg.data.smoke_users, train_rows=cfg.data.train_rows)`
9. INFO 로그: split별 행·표본·사용자 수, title_mode, `title_report` 범주별 개수, smoke 선택 결과

#### 5.6.4 행 번호와 표본 ID
- `row` = 필터 전 DataFrame의 0부터 시작하는 위치 `p`. index가 `RangeIndex(0, n, 1)`임을 확인하므로 index label과 같음
- 필터는 행을 지우기만 하고 `row`는 바꾸지 않음. split tuple은 `row` 오름차순이고, M10 `BatchPlan`의 `sample_indices`는 이 tuple의 위치
- `sample_id = f"{split}:{row}"` (예: `train:3`, `val:0`)
- fork 기록과의 대응 (계산: `design/tmp/m1_rowmap.py`): train은 `row = F sample_idx + 3·(user_id + 1)` (66,126개 모두 성립), validation·test는 `row = F sample_idx`

#### 5.6.5 pseudo 사용자 ID (D14)
- **train 블록 규칙** (필터 전 행으로 계산): `p = 0`이거나 `history[p] != (history[p−1] + (target[p−1],))[-10:]`이면 새 블록. `user_id` = 블록 번호(0부터, 파일 순서)
- 검증 (계산: `design/tmp/m1_verify.py`, `m12_extra.py`)
  - 블록 754개. (item, rating) tuple로 비교해도 같은 경계
  - 블록 시작 = 빈 이력 행 754개와 정확히 일치
  - 블록마다 window 길이가 0, 1, 2, …, 10, 10, … (블록 크기 6–509, 합 68,388)
  - user 0은 0행, user 1은 72행, user 753은 68,368행에서 시작
- **필터와의 관계:** 필터는 블록마다 정확히 첫 3행(len_seq 1, 1, 2)만 지움. 남는 window는 블록당 3–506개, 빈 블록 없음 → train 사용자 754명 모두 표본이 있음. 필터 후 행에 같은 규칙을 적용해도 같은 754개 분할
- **validation·test:** 1행 = 1사용자. `user_id = 754 + row`(754–847), `848 + row`(848–942)
  - validation·test 이력은 split 안에서 모두 다르고 서로도 겹치지 않음. train의 어떤 window·이력과도 겹치지 않음 (계산)
- 검사: `assign_users`는 블록이 빈 이력으로 시작하지 않거나 window 길이 순서가 어긋나면 `DataError`
- user_id는 분석·smoke·D25 변형용이고 모델 입력이 아님

#### 5.6.6 제목 정규화 (D20)
- **`code` (paper_spec):** `code_title(raw) = get_mv_title(raw[:-7])`, `get_mv_title`은 `U :61-66`을 그대로 옮김 (`[", The", ", A", ", An"]` 순서로 `in` 검사, 처음 걸린 것으로 `sub[2:] + " " + s.replace(sub, "")`)
- **`fixed` (`variant_fixed_titles`):** 아래 순서 `[결정]`
  1. `s = raw.strip()`
  2. `re.fullmatch(r"(?P<body>.+) \((?P<year>\d{4})\)(?: \(V\))?", s)`가 맞으면 `s = body` (맞지 않는 것은 item 266 `unknown`뿐 → 그대로 `unknown`)
  3. `re.fullmatch(r"(?P<main>.+?), (?P<art>The|A|An)(?P<alias> \(.+\))?", s)`가 맞으면 `s = art + " " + main + (alias or "")`. 괄호 속 관사와 외국어 관사(`, La` 등)는 옮기지 않음
- `fixed`는 `code`와 정확히 12개 id만 다름: 176, 266, 435, 601, 965, 1004, 1018, 1127, 1200, 1226, 1411, 1634 (계산: `design/tmp/m2_flags.py`)
  - 예: 435 `An American Werewolf in London`, 176 `The Good, The Bad and The Ugly`, 1004 `Double vie de Véronique, La (Double Life of Veronique, The)`, 1411 `Land Before Time III: The Time of the Great Giving`
  - `fixed`에는 빈 제목이 없고, `unknown`은 다른 제목의 부분 문자열이 아님. 중복 22개 제목·45 id는 그대로 (D20 변형 범위 밖)
- `title_report` 범주: `article_an`, `article_double_the`, `article_in_parentheses`, `empty`, `truncated`, `trailing_space`, `duplicate_title` (위 5.5 목록과 같은 id). 판정 규칙 `[결정]`: 5.5는 예시 id만 주므로 그 id를 정확히 재현하는 규칙을 정함 `R ilora/data.py title_report`. `cut = raw[:-7]`, 걸린 관사 = `[", The", ", A", ", An"]` 순서로 `cut`에 처음 든 것
  - `article_an`: 걸린 관사가 `", A"`이고 `cut`에 `", An"`이 있음
  - `article_double_the`: `cut`에 `", The"`가 2번 이상
  - `article_in_parentheses`: 걸린 관사가 나온 자리 중 하나가 괄호 안
  - `empty`: `code` 제목이 빈 문자열. `trailing_space`: `code` 제목이 공백으로 끝남
  - `truncated`: `cut`이 비어 있지 않고 `cut.rstrip()`이 `fixed` 2단계의 연도를 뺀 본문(`raw.strip()` 기준, 연도가 없으면 그대로)의 `rstrip()`과 다름
  - `duplicate_title`: `code` 제목을 다른 id와 함께 쓰는 id
- 제목 문자열은 정규화 외에 바꾸지 않음 (비 ASCII 9개 제목은 ISO-8859-1 decode 그대로, 예: 1632 `Á köldum klaka (Cold Fever)`)

#### 5.6.7 smoke 부분집합 (D14)
- full로 읽은 뒤 고름 (필터 후 표본에서). user_id·row는 다시 매기지 않음. 반환 `LoadedData`는 `subset = "smoke"`, `smoke_users = sizes` (full은 `"full"`, None). `select_smoke`는 `subset == "full"`인 `LoadedData`만 받음 (아니면 `DataError`)
- 순서 (split마다 독립된 key, 호출 1회)
  1. train: `users = sorted({s.user_id for s in data.train})` (754명), `pick = np_rng(seed, "smoke_users", "train").choice(len(users), size=n_train, replace=False)` → 고른 사용자의 **모든** window를 row 순서로
  2. validation: `pick = np_rng(seed, "smoke_users", "val").choice(len(data.val), size=n_val, replace=False)` → `data.val`의 그 위치 표본, row 순서로
  3. test: 같은 방법, key `("smoke_users", "test")`
- `np_rng(seed, *keys) = np.random.Generator(np.random.PCG64(derive_seed(seed, *keys)))` (M0 4.6.4)
- 크기 범위: 1 ≤ n_train ≤ 754, 1 ≤ n_val ≤ 94, 1 ≤ n_test ≤ 95. 밖이면 `DataError`
- seed 1234 결과 (계산: `design/tmp/m12_extra.py`, numpy 1.24.2)
  - train 사용자 80명, window 7,230개 (처음 5명 43, 51, 60, 66, 69)
  - validation 행 1, 7, 15, 20, 26, 44, 59, 61, 77, 90 (user 755, 761, 769, 774, 780, 798, 813, 815, 831, 844)
  - test 행 4, 9, 14, 20, 50, 66, 71, 72, 82, 92 (user 852, 857, 862, 868, 898, 914, 919, 920, 930, 940)
- 후보·템플릿 key가 `row` 기반이라 smoke 표본의 후보·템플릿은 full 실행의 같은 행과 같음 (M2, M4)
- **`train_rows`가 비어 있지 않으면** (`overfit_gpu` 전용, M13 17.6.8) `[결정]`: train만 위 1단계 대신 `row ∈ train_rows`인 표본을 row 순서로 고름. `("smoke_users", "train")` key는 쓰지 않음. validation·test는 2·3단계 그대로
  - 행마다 필터 후 train 표본이어야 함 (len_seq < 3이라 지워진 행, train 밖 번호 → `DataError`)
  - 고른 표본의 서로 다른 `user_id` 수 == `sizes[0]` (아니면 `DataError`). `LoadedData.smoke_users = sizes`
  - 목록 형식(1–64개, 엄격한 오름차순, 0 이상)과 `subset == "smoke"` 조합은 설정 단계에서 확인 (5.6.8 `validate()`, `ConfigError`)
  - 이유: smoke 사용자 선택은 seed 1234에서 1명이 사용자 117(132표본)이라 과적합 점검용 8–16표본을 만들 수 없음 (M13 17.6.8). 결과를 보지 않는 고정 행 목록을 프로파일에 적음

#### 5.6.8 설정 (`data` 구역)
| 키 | 타입 | paper_spec 값 | 라벨 | 설명 |
|---|---|---|---|---|
| `title_mode` | str (`code`, `fixed`) | `code` | `[코드] U data/movielens_data.py:61-66, :74` (D20) | 제목 정규화. `fixed`는 `variant_fixed_titles` |
| `subset` | str (`full`, `smoke`) | `full` | `[결정]` D14 | `smoke`면 5.6.7로 사용자 선택 |
| `smoke_users` | tuple[int, ...] (길이 3) | `[80, 10, 10]` | `[결정]` D14 | (train, val, test) 사용자 수. `full`이면 범위 검사만 함 (M0 규칙상 모든 키 필수) |
| `train_rows` | tuple[int, ...] | `[]` | `[결정]` | 비어 있으면 쓰지 않음. 비어 있지 않으면 smoke train을 이 행들의 표본으로 고정 (5.6.7). `overfit_gpu`만 씀 (M13 17.6.8) |

- `validate()`: `len(smoke_users) == 3`이고 각 값 ≥ 1, `train_rows`가 비어 있지 않으면 `subset == "smoke"`이고 1–64개·엄격한 오름차순·0 이상 (어기면 `ConfigError`). 행이 실제 train 표본인지는 데이터를 읽은 뒤 `select_smoke`가 확인 (`DataError`). `smoke_users`의 상한(754, 94, 95)은 데이터를 읽은 뒤 `load_dataset`이 subset과 무관하게 확인 (`DataError`) `[결정]`: 표의 'full이면 범위 검사만 함'의 범위를 5.6.7의 범위(상한 포함)로 읽음. full 프로파일 값도 `data.subset=smoke`로 바꾸면 그대로 쓰이므로 같은 검사를 함 `R ilora/data.py load_dataset`
- 설정이 아닌 고정 상수 (변형 없음): `MIN_LEN_SEQ = 3` `[코드] :79`, `HISTORY_LEN = 10` `[LLaRA §5.1.1 p.7]`, `PAD_ID = 1682` `[코드] :18`, `NUM_ITEMS = 1682` `[LLaRA Table 1 p.7]`, 파일 경로 `[코드] :52-57`, `U train_movielens.sh:8`, 인코딩 ISO-8859-1 `[코드] :71`
- 데이터 경로를 설정으로 두지 않는 이유 `[결정]`: M0이 해시를 점검한 파일과 M1이 읽는 파일이 항상 같게

#### 5.6.9 내용 fingerprint (계산: `design/tmp/m1_fingerprint.py`)
- 정의: 필터 전 행 `p`마다 `json.dumps([p, [[item, rating], …10개], len_seq, [next_item, next_rating]], separators=(",", ":"))`, 줄을 `"\n"`으로 이어 UTF-8 sha256. `cluster` 등 다른 열은 넣지 않음
- 기대값
  - train `85bafab832f1f78e5095b1fd880a418097054d3ea7dbb7d4fe9e1528f019aa99`: iLoRA판(sha256 `bac4c88b…9546`)과 LLaRA판(sha256 `a73b119d…8aad`)이 같은 값 → 두 배포 파일의 `(seq, len_seq, next)`가 행 단위로 같음 (4.6.5의 "내용 일치" 확인)
  - validation `de7ac66b6703dc6935e1339183c00e08af2ad40f6d0877e7556b483a9aa65523`
  - test `01636285c16faf95cc7136bc937b9aa315c66de5e23fed5efec0e2b2e84dff09`
- 파일 해시 (sha256, 텍스트는 LF 기준): 4.6.5 표와 같음을 다시 계산함 (train `bac4c88b…`, validation `9a950900…`, test `d38308a0…`, `u.item` `553841eb…`, `prompt/movie.txt` `957fc1f2…`, `rec_model/movielens.pt` `6928a067…`)
  - Windows 작업 트리 원시 바이트는 다름: `u.item` 238,026 byte(CRLF 1,682개, `36e14e4a…`), `movie.txt` 583 byte(CRLF 2개, `d0a3194e…`). 정규화하면 git blob과 같음

#### 5.6.10 dtype·device·오류 처리
- `pad_histories`(5.6.13) 밖에서는 torch를 쓰지 않음. 모든 id는 Python int, 제목은 str. Sample·ItemCatalog만 DataLoader worker로 넘어감 (pickle 가능한 frozen dataclass)
- 오류는 모두 `DataError`(설정 값 오류는 M0 `ConfigError`)로 즉시 중단. 메시지에 파일·split·row·기대값·실제값
  - 파일 없음·해시 불일치 (unpickle 전), DataFrame·RangeIndex·열 구성 불일치, fingerprint 불일치
  - `seq` 형식(길이 10 list, 정수 2-tuple), 가운데·왼쪽 padding, item 범위 밖, rating ∉ 1–5, `len_seq != max(1, k)`, `next` 형식
  - 필터 후 이력 중복, 정답 ∈ 이력, 블록 규칙 위반, 기대 수 불일치, u.item 줄·필드·id 순서 불일치, smoke 크기 범위 밖
- `try/except`로 오류를 삼키지 않음. 대체 경로 없음 (예: LLaRA판 파일로 자동 전환하지 않음)

#### 5.6.11 다른 모듈과의 연결
- M0: 해시 함수·기대 표 사용, `np_rng`로 smoke 선택. `load_dataset(cfg, repo_root)`는 진입점이 환경 점검 뒤에 부름
- M2: `Sample.history`, `Sample.target`, `ItemCatalog.titles`(flag)
- M3·M5·M12: `history`를 `pad_histories(histories, PAD_ID, HISTORY_LEN)`(5.6.13)으로 오른쪽 padding해 `hist_ids [B, 10]`, `len_seq = len(history)` (M5 collate, M3 `encode_all`, M12 export가 같은 함수)
- M4: `catalog.titles[i]`로 이력·후보·정답 제목
- M10: `LoadedData.train` 순서가 `BatchPlan` 위치의 기준. M11: `val`·`test` 행 순서 그대로 평가. M12·M15: `sample_id`, `user_id`, `split`

#### 5.6.12 계약 추가 제안
- `ItemCatalog.title_mode: str` (`"code"`/`"fixed"`) 필드 추가: M11·M12가 어떤 제목으로 판정했는지 기록
- `LoadedData`와 `load_dataset(cfg, repo_root) -> LoadedData`를 M1 공개 API로: M10·M11·M15가 표본을 얻는 유일한 경로. 필드 `smoke_users`(smoke일 때만, full은 None)와 `title_mode` 속성은 M11 15.6.2·15.6.8이 읽는 이름에 맞춤
- `BatchPlan`의 `sample_indices`는 `LoadedData.train`(row 오름차순 tuple)의 위치로 정의 (3.1에 기준이 적혀 있지 않음)
- M0의 해시 함수와 기대 해시 표를 3.1·4.6.5에 공개 API로 명시: 코드에는 이미 `run.content_sha256(path: Path, text: bool = False) -> str`와 `run.EXPECTED_INPUTS: dict[str, tuple[str, bool]]`(값 = (sha256, 텍스트 여부))로 있음. M1은 이 둘만 씀 (unpickle 전 확인)
- M0 환경 점검(4.6.5)에 numpy 1.24.2 `U requirements.txt:68`, pandas 1.5.3 `:82` 판 **확인(불일치 시 실패)** 추가: unpickle 결과(M1)와 Generator 출력(M1 smoke, M2 후보)이 판에 묶임. 판 **기록**은 M0 `run.environment_info()`가 이미 `meta.json`에 함 (numpy, pandas, sentencepiece 등)
- 공유 함수 `pad_histories`(5.6.13)를 M1 공개 API로: M3 7.6.8·M5 9.6.6·M12 16.6.11이 두 곳(M1, M5)을 제안했던 것을 최종 검토에서 M1로 정함 (세 사용처가 모두 M1에 의존하고 `PAD_ID`·`HISTORY_LEN`이 여기 있음)
- `data.train_rows` 키와 `select_smoke(..., train_rows=)` (M13 17.6.11 제안 채택)

#### 5.6.13 `pad_histories` (공유 padding 함수)
- `pad_histories(histories, pad_id, length)`: `hist_ids[i] = list(h) + [pad_id] * (length − len(h))`, `len_seq[i] = len(h)`. 반환 `(torch.int64 [N, length], torch.int64 [N])`, CPU, contiguous
- 검사 (`DataError`, 위치 포함): N ≥ 1, 이력마다 문자열이 아닌 길이 있는 sequence(tuple, list, numpy 배열), `1 ≤ len(h) ≤ length`, 원소는 Python·numpy 정수(bool 제외)이고 `0 ≤ id < pad_id`, `pad_id`·`length`는 1 이상 Python int `R ilora/data.py pad_histories`. 호출하는 쪽은 `pad_id = PAD_ID`(1682), `length = HISTORY_LEN`(10)을 넘김
- 사용처: M5 `collate_train`·`collate_eval`(9.6.6), M3 `encode_all`(7.6.4, 기준 파일의 `hist_ids`·`len_seq`와 같아야 함 M3 7.8 #10), M12 `compute_representations`(16.6.5). 같은 규칙이 세 곳에 따로 있지 않게 함

### 5.7 결정
- D14 (pseudo 사용자 ID, smoke 80/10/10), D20 (`code` 제목이 기본, `fixed`는 변형)
- 이 장의 새 `[결정]`
  - unpickle 전 해시 확인, 읽을 때마다 내용 fingerprint 확인: 확인된 내용만 쓰기 위함
  - 데이터 경로·`MIN_LEN_SEQ`·`HISTORY_LEN`은 설정이 아닌 상수: 변형이 없고 M0 점검 대상과 일치시키기 위함
  - `variant_fixed_titles` 규칙(5.6.6): D20 변형 범위(관사 순서, 빈 제목, 잘림, 끝 공백)를 최소로 고침
  - smoke 선택 방식(정렬된 사용자 목록에서 `choice`, 결과는 row 순서, 번호 유지): 같은 행이 full과 같은 후보·템플릿을 갖게
  - `train_rows`(과적합 점검 전용 고정 행 목록, 비어 있으면 미사용): seed 선택으로는 8–16표본을 만들 수 없어 결과를 보지 않는 행 목록을 프로파일에 적음 (5.6.7)
  - 공유 `pad_histories`는 M1 `data.py`: M3·M5·M12가 같은 padding 규칙을 쓰고 모두 M1에 의존 (5.6.13)

### 5.8 완료 기준
1. (CPU) 4개 파일 sha256이 M0 표와 같을 때만 읽음. 1 byte 바꾼 사본은 `pd.read_pickle` 호출 전에 `DataError` (호출 여부를 mock으로 확인)
2. (CPU) 행 수 68,388/94/95, 필터 후 66,126/94/95. 필터 후 `len(history) == len_seq`, 길이 범위 3–10 (validation 9–10, test 10)
3. (CPU) train 블록 754개, 블록마다 빈 이력으로 시작하고 window 길이 0, 1, 2, …, 10, 10, …. 필터가 블록마다 정확히 3행 제거, 빈 블록 없음. validation user 754–847, test 848–942
4. (CPU) fingerprint가 train `85bafab8…`, validation `de7ac66b…`, test `01636285…`. 투영 후 열이 정확히 (`seq`, `len_seq`, `next`)이고 `Sample`·`LoadedData`에 `cluster` 정보 없음
5. (CPU) `code` 제목이 5.5 결함 목록을 그대로 재현: 435/601/965/1226 손상 문자열, 176 `The Good Bad and The Ugly`, 1004/1018 괄호 관사 이동, 266 == `""`, 1411이 `(19`로 끝남, 1127/1200/1634 끝 공백 1개, 중복 22개 제목·45 id
6. (CPU) `fixed` 제목이 `code`와 정확히 위 12개 id에서만 다르고 빈 제목 0개, 중복 22/45 유지
7. (CPU) CRLF 사본과 LF 사본의 `u.item`에서 같은 catalog, 같은 정규화 해시
8. (CPU) train `row == F sample_idx + 3·(user_id + 1)` (66,126개 모두), validation·test `row == F sample_idx`
9. (CPU) smoke (seed 1234): train 80명·window 7,230개, validation 행 [1, 7, 15, 20, 26, 44, 59, 61, 77, 90], test 행 [4, 9, 14, 20, 50, 66, 71, 72, 82, 92]. 두 번 불러도 같음
10. (CPU) 합성 DataFrame으로 오류 사례마다 `DataError`: 왼쪽·가운데 padding, `len_seq != max(1, k)`, 정답 1682, 범위 밖 id, rating 0, 이력 중복, 정답 ∈ 이력, 다른 열 구성, RangeIndex 아님, 빈 이력으로 시작하지 않는 블록
11. (CPU) `Sample`이 길이 2·11 이력, bool id, 범위 밖 id를 거부. `sample_id` 형식 `"{split}:{row}"`
12. (VESSL_REQUIRED) VESSL 작업 트리에서 4개 파일 해시와 pandas 1.5.3·numpy 1.24.2 판, 같은 fingerprint와 표본 수
13. (CPU) `train_rows`: `overfit_gpu` 설정(행 71, 196, …, 2874)으로 train 16표본·사용자 16명(0–15), validation·test는 9번과 같은 행. 필터로 지워진 행(예: 0), train 밖 번호(68,388), 사용자 수가 `smoke_users[0]`과 다름 → `DataError`. 중복, 내림차순, 65개, `subset = full`과의 조합 → `ConfigError`
14. (CPU) `pad_histories`: 길이 3·10 이력이 1682로 오른쪽 padding된 `[N, 10]` int64와 `len_seq`, 길이 0·11 이력, 1682 이상 id, bool id → `DataError`

---

## 6. M2 후보 생성
상태: 테스트 통과(로컬)

### 6.1 목적과 범위
- **범위:** 표본마다 후보 20개(정답 1 + 음성 19)를 정함 (D10, D21)
  - 평가(validation·test) 후보는 `(seed, split, row)`로 고정, 학습 후보는 `(seed, "train", row, epoch)`로 epoch마다 결정적 재추출
  - 후보 집합의 제목 모호성 flag 계산: M11 판정 규칙 때문에 정답을 출력해도 무효가 되는 표본을 미리 표시 (D20)
- **범위 밖:** 제목 문자열 만들기(M1), prompt 조립(M4), tensor 변환(M5), 생성·판정·지표(M11)

### 6.2 입력과 출력
**입력**
- `Sample` (M1): `split`, `row`, `history`, `target`
- `main_seed` = `Config.run.seed` (M0), `epoch` (학습만, M10이 넘기는 음이 아닌 int, `train_order`·`template` key와 같은 번호)
- `Config.candidates` (6.6.7), flag 계산에만 `ItemCatalog` (M1)

**출력**
- 후보 `tuple[int, ...]` 길이 20 (Python int, 3.1 계약): 정답 정확히 1개, 중복 없음, `history`와 겹치지 않음, 순서 섞임
  - M4가 `catalog.titles[c]`로 제목을 붙이고, M5가 `cand_ids [B, 20] int64`로 바꿈
- flag `dict[str, bool]`: 키는 `FLAG_NAMES` 5개(순서 고정, 항상 모두 있음) → M11 `EvalRecord.flags`
- 파일·로그를 쓰지 않음. 평가 후보는 M11이 `EvalRecord.cand_ids`로 기록

### 6.3 논문 근거
- "we construct candidate sets for each sequence by randomly selecting 20 non-interacted items while ensuring the inclusion of the correct next item." `[논문 App B p.16]`
- 같은 내용: "For each sequence, we randomly select 20 non-interacted items to construct the candidate set, ensuring the inclusion of the correct subsequent item." `[LLaRA §5.1.3 p.7]`
  - 모호한 점: 20개에 정답이 포함되는지(20 vs 21), "non-interacted"의 기준, 후보를 고정하는지 → D10(코드 계보로 정답 포함 20), D21(보이는 window 기준), 고정 방식 `[결정]`
- 학습 prompt에도 후보 집합: "(3) Candidate Set: the set of candidate items, from which the LLM is to generate responses to the given task" `[LLaRA 간접 §4.2 p.5]`
- 유효 응답의 정의가 후보에 묶임: "the proportion of valid responses (i.e., items within the candidate set) across all sequences" `[논문 App B p.16]` → 후보 제목의 모호성이 ValidRatio·HR@1에 직접 영향

### 6.4 공개 코드에서 가져올 것
`U data/movielens_data.py`는 `L`과 같은 blob `2bf0e156` (코드 계보 일치).
- 음성 풀 `[i for i in list(self.item_id2name.keys()) if i not in seq_unpad and i!=next_item]` `:45`: 보이는 이력 window(최대 10)와 정답을 뺀 아이템, id 오름차순(u.item 줄 순서 = dict 삽입 순서) (D21)
- 순서: 음성 `cans_num − 1`개를 균등 비복원으로 뽑고(`random.sample`), 정답을 끝에 붙인 뒤 섞음(`random.shuffle`) `:46-47` → 같은 순서를 numpy Generator로 재현
- 후보 수 20: `U main.py:122`, `U train_movielens.sh:9`, template 문구 "following 20 movie titles" `U prompt/movie.txt:1-2`
- 판정 규칙 (flag 정의의 기준, 구현은 M11) `U model/model_interface.py:468-492`
  - 생성문은 첫 줄만 (`strip().split("\n")[0]` `:165`, `:200`)
  - 생성문·정답·후보 제목을 모두 `strip().lower().strip()` `:476-478`
  - 후보 제목이 생성문의 **부분 문자열**이면 목록에 넣음 `:480-482`, 목록이 **정확히 1개**일 때만 유효 `:483-484`, 그 1개가 정답 제목과 **문자열로 같으면** 정답 `:485-486`

### 6.5 버릴 것 (알려진 문제)
- **worker 난수:** 후보를 Python 전역 `random`으로 뽑음 `:27, :46-47`. 이 상태는 worker마다 `base_seed + worker_id`로 다시 seed되고(`torch2.0 utils/data/_utils/worker.py:223-224`) 어느 worker가 어느 표본을 맡는지에 따라 소비 순서가 달라짐 → worker 수·순서에 따라 후보가 달라짐 (4.5). key마다 독립 Generator로 바꿈
- **평가 후보가 매번 다름:** `__getitem__`마다 재추출 `:27` → validation 후보가 epoch마다 바뀌어 checkpoint 선택(D11)이 흔들리고, 실행 간 비교도 짝이 안 맞음. 평가 후보 고정 (D10)
- **풀 구성 비용:** list에 대한 `in` 검사 `:45`(표본당 약 1.7만 번 비교) → boolean mask (결과 집합·순서 같음)
- **판정 규칙의 약점 (고치지 않음, D17·D20 코드 계보):** flag로 표시만 함
  - 빈 제목(item 266, `code` 제목)은 모든 생성문에 일치
  - 짧은 제목이 무관한 제목 안에서 일치: `code` 제목 79개가 다른 제목의 부분 문자열(순서쌍 774개). 655 `M`은 541개, 1286 `Ed`는 80개, 646 `Ran`은 25개 제목에 들어감 (계산: `design/tmp/m12_extra.py`)
  - 같은 제목(22개 제목, 45 id)은 문자열로만 비교되어 구별 불가

### 6.6 설계

#### 6.6.1 공개 API (`rebuild/ilora/candidates.py`)
```python
NUM_CANDIDATES: Final = 20                          # 3.1 계약
FLAG_NAMES: Final = ("empty_title_candidate", "duplicate_title_candidates",
                     "substring_title_candidates", "target_title_duplicated",
                     "target_unscorable")

class CandidateError(ValueError): ...

def candidate_key(sample: Sample, epoch: int | None) -> tuple[str | int, ...]
def negative_pool(history: tuple[int, ...], target: int, num_items: int) -> np.ndarray  # int64, 오름차순
def draw_candidates(history: tuple[int, ...], target: int, rng: np.random.Generator,
                    *, num_items: int, num_candidates: int) -> tuple[int, ...]
def candidates_for(sample: Sample, *, main_seed: int, epoch: int | None,
                   cfg: CandidatesConfig) -> tuple[int, ...]
def check_candidates(cands: tuple[int, ...], sample: Sample, *, num_items: int,
                     num_candidates: int) -> None
def match_norm(title: str) -> str                   # 판정 정규화, M11과 공유
def candidate_flags(cands: tuple[int, ...], target: int, catalog: ItemCatalog) -> dict[str, bool]
def self_check() -> None                            # golden 비교 (6.6.5)
```
- `num_items`는 M1 상수 `NUM_ITEMS = 1682`를 씀 (설정 아님)
- 상태 없음, 캐시 없음: 같은 인자면 어느 process에서 불러도 같은 값

#### 6.6.2 추출 알고리즘 (난수 호출 순서 고정)
1. key (`candidate_key`)
   - `sample.split == "train"`: `epoch`는 int ≥ 0 필수, key = `("candidates", "train", sample.row, epoch)`
   - `"val"`·`"test"`: `epoch`는 `None`이어야 함, key = `("candidates", sample.split, sample.row)`
   - 어긋나면 `CandidateError` (예: 학습 표본을 평가 key로 부르는 것 금지)
2. `rng = np_rng(main_seed, *key)` = `np.random.Generator(np.random.PCG64(derive_seed(main_seed, *key)))` (M0 4.6.4). 표본마다 새로 만들고 다른 용도와 공유하지 않음
3. 풀: `mask = np.ones(1682, dtype=bool)`, `mask[list(history)] = False`, `mask[target] = False`, `pool = np.flatnonzero(mask).astype(np.int64)`
   - 오름차순, 길이 = 1,682 − len(history) − 1 = 1,671–1,678 (history는 중복 없음, target ∉ history)
4. **난수 호출 1:** `neg = rng.choice(pool, size=num_candidates - 1, replace=False)` (`p` 없음, `shuffle` 기본값 True 그대로)
5. `cands = np.concatenate([neg, np.array([target], dtype=np.int64)])` → 정답이 index 19
6. **난수 호출 2:** `rng.shuffle(cands)` (제자리)
7. `out = tuple(int(x) for x in cands)` → `check_candidates(out, sample, ...)` → 반환
- rng는 이 두 호출에만 씀. 같은 key면 worker 수, 호출 순서, smoke 여부와 무관하게 같은 후보
- **numpy 판 고정:** numpy는 BitGenerator 비트열은 고정하지만 `Generator.choice`·`shuffle` 알고리즘은 판에 따라 바뀔 수 있다고 밝힘 (NEP 19) → numpy 1.24.2 고정 (`U requirements.txt:68`, fork VESSL 설치가 이 파일을 씀 `F setup_vessl.sh:43-44`), golden 값으로 확인 (6.6.5)

#### 6.6.3 계약 검사 (`check_candidates`)
- `cands`는 tuple이고 길이 == `num_candidates`(20), 원소가 Python int(bool·numpy 정수 제외)이고 0 ≤ id < 1682, 중복 없음, `target`이 정확히 1개, `history`와 교집합 없음. M4 `build_prompt`도 이 함수로 후보 계약을 다시 확인 (8.6.4)
- 입력 표본이 `target ∈ history`면 풀을 만들기 전에 `CandidateError` (배포 파일에는 0건이고 M1 `Sample`도 거부하지만 이중 확인)
- 실패 메시지에 `sample_id`, key, 위반 내용. `check_candidates`는 key를 받지 않으므로(6.6.1 서명) `candidates_for`가 그 `CandidateError`에 `(key …)`를 덧붙여 다시 올림 `[결정]` `R ilora/candidates.py candidates_for`. 다시 뽑거나 고쳐서 넘기지 않음

#### 6.6.4 모호성 flag (`candidate_flags`)
- `match_norm(t) = t.strip().lower().strip()`: 판정 규칙 정규화 `U model/model_interface.py:476-478`와 같음. M11 판정도 이 함수를 씀 (계약 추가 제안)
- n(c) = `match_norm(catalog.titles[c])`, 음성 = `cands`에서 `target`을 뺀 19개, `x in y`는 Python 부분 문자열 검사

| flag | 정의 | 판정에 미치는 영향 |
|---|---|---|
| `empty_title_candidate` | 어떤 후보의 n(c) == `""` | 빈 제목은 모든 생성문에 일치: 정답 출력도 무효, 다른 후보와 안 맞는 아무 출력이 "유효·오답" |
| `duplicate_title_candidates` | 서로 다른 후보 c ≠ c'의 n(c) == n(c') | 그 제목을 출력하면 무효 |
| `substring_title_candidates` | n(c) ≠ `""`, n(c) ≠ n(c'), n(c) in n(c')인 후보 쌍이 있음 | 긴 제목을 출력하면 무효 |
| `target_title_duplicated` | 어떤 음성 c의 n(c) == n(target) | 정답 제목을 출력하면 무효 |
| `target_unscorable` | 어떤 음성 c의 n(c) in n(target) (빈 제목·같은 제목·부분 문자열 포함) | 정답 제목을 그대로 출력해도 정답으로 세지 않음 |

- `target_unscorable`은 "판정 규칙에 첫 줄 = 정답 제목을 넣었을 때 정답이 아님"과 동치 (정답 제목을 포함한 출력에는 그 음성 제목도 들어 있으므로 목록이 2개 이상). val·test 189표본, 두 제목 모드에서 판정 규칙 모의 실행과 일치 확인 (계산: `design/tmp/m2_flags.py`)
- flag 사이 관계 (정의에서 나옴): `target_title_duplicated`이면 `duplicate_title_candidates`와 `target_unscorable`도 참. `target_unscorable`을 만든 음성 제목은 빈 제목·같은 제목·부분 문자열 중 하나라서 `empty_title_candidate`·`duplicate_title_candidates`·`substring_title_candidates` 중 하나도 참. 그래서 혼자 켜질 수 있는 flag는 앞의 세 개뿐이고, 빈 제목인 음성은 늘 `target_unscorable`도 켬 (val row 66, test row 5)
- 평가 표본(val·test)에만 계산. 학습에는 판정이 없어 계산하지 않음
- 반환은 `{name: bool for name in FLAG_NAMES}`. 직렬화 형식은 M11·M12가 정함

#### 6.6.5 golden 값과 `self_check`
- 조건: seed 1234, numpy 1.24.2, M0 `derive_seed`(SHA-256 앞 8 byte big-endian, 63 bit mask) (계산: `design/tmp/m2_flags.py`)
  - `derive_seed(1234, "candidates", "val", 0) = 4835709385070275973`
  - val row 0 (history `(880, 269, 361, 1024, 881, 890, 1242, 1104, 989, 333)`, target 321) → `(980, 156, 321, 1240, 892, 1299, 448, 1094, 1502, 728, 1598, 208, 1042, 1386, 902, 1282, 1029, 378, 515, 554)`
  - test row 0 (target 3) → `(1625, 1239, 800, 944, 769, 316, 840, 381, 1450, 228, 1367, 1393, 634, 3, 1497, 365, 101, 832, 1160, 638)`
  - split 전체 직렬화 `json.dumps([[row, list(cands)], …], separators=(",", ":"))` (row 순서)의 sha256: val `9a07320651cadd0d049efc8644cb9a6103a78bdc2b45f2bd793ae204a545f01a`, test `1ffdb6499d3a316fb8cf94c322f09e5af509520f8cdcc6574634b99a7a30bef3`
- `self_check()`: 위 val row 0 표본을 상수로 만들어(`Sample("val", 0, 754, history, 321)`) `candidates_for(..., main_seed=1234, epoch=None)`를 계산하고 tuple 비교. 다르면 `CandidateError("numpy Generator 출력이 설계 golden과 다름: numpy 판 확인")`
- `self_check()`는 tuple 비교 전에 `derive_seed(1234, "candidates", "val", 0) == 4835709385070275973`인지 먼저 확인하고, 다르면 M0 `derive_seed`를 보라는 `CandidateError`를 냄 `[결정]`: numpy 판 문제와 원인을 가름. golden 불일치 메시지 뒤에는 numpy 판과 얻은 tuple을 붙임 `R ilora/candidates.py self_check`
- M0 `run_command`가 모든 명령의 공통 준비에서 환경 점검 직후 1회 부르고 `meta.json`의 `preparation.candidates_self_check`에 기록 (3.1 명령 분기) `R ilora/commands.py run_command`. 데이터 파일 없이 동작

#### 6.6.6 계산한 flag 빈도 (seed 1234, `code` 제목, 6.6.5의 고정 후보)
| split | N | empty | duplicate | substring | target_dup | unscorable | 하나 이상 |
|---|---|---|---|---|---|---|---|
| val | 94 | 1 (row 66) | 0 | 1 (row 36) | 0 | 1 (row 66) | 2 |
| test | 95 | 1 (row 5) | 0 | 4 (rows 11, 38, 39, 43) | 0 | 2 (rows 5, 11) | 5 |

- 사례
  - val row 66, test row 5: 후보에 item 266(빈 제목) → 정답 "Monty Python and the Holy Grail", "Dirty Dancing"을 정확히 출력해도 무효
  - test row 11: 후보 655 `M`이 정답 "Mr. Holland's Opus"를 포함한 다른 후보 5개 제목의 부분 문자열 → 정답 출력도 무효
  - val row 36 `manhattan` ⊂ `sudden manhattan`, test row 38 `ran` ⊂ `carmen miranda: bananas is my business`, row 39 `die hard` ⊂ `die hard 2`, row 43 `glory` ⊂ `paths of glory`
- 뜻: 이 후보로는 정답 출력으로도 맞출 수 없는 표본이 val 1, test 2 → CombinedMetric 상한 93/94, 93/95 (계산값, 실험 결과 아님)
- 참고 `variant_fixed_titles`: val은 substring 1 (하나 이상 1), test는 substring 4·unscorable 1(row 11) (하나 이상 4)
- 모든 실험 프로파일이 seed 1234(D2)를 쓰므로 E1·E2·E4의 평가 후보와 flag가 같음 → 표본 단위 짝 비교 가능

#### 6.6.7 설정 (`candidates` 구역)
| 키 | 타입 | paper_spec 값 | 라벨 | 설명 |
|---|---|---|---|---|
| `num_candidates` | int (선택지 20) | 20 | `[논문 App B p.16]` `[LLaRA §5.1.3 p.7]` `[코드] U main.py:122` (D10) | 정답 포함 후보 수. 3.1 계약(`cand_ids [B, 20]`)과 template 문구 "20 movie titles"에 묶여 다른 값 거부 |

- 설정이 아닌 고정 규칙 (변형 없음): 음성 풀(D21), 평가 고정·학습 epoch별 재추출(D10), 난수 호출 순서(6.6.2), flag 정의(6.6.4)

#### 6.6.8 dtype·device·성능·오류 처리
- numpy CPU만, torch 없음. 출력은 Python int tuple
- 학습: worker 안에서 표본마다 계산 (epoch당 계획에 든 66,120회 = 8,265 × 8, drop_last로 빠진 6개는 계산하지 않음. 표본당 Generator 1개 + 호출 2회). 값은 main process의 `BatchPlan`과 무관 (key에 batch 정보 없음)
- 평가: M11이 표본마다 계산 (val 94, test 95). 같은 값을 다시 계산할 수 있으므로 캐시 파일 없음
- 오류: 잘못된 split·epoch 조합, epoch < 0 또는 int 아님, `target ∈ history`, 계약 위반, `num_candidates != 20`, golden 불일치 → 모두 `CandidateError`로 즉시 중단. 다시 뽑기·건너뛰기 없음

#### 6.6.9 다른 모듈과의 연결
- M4·M5 (학습 worker): `candidates_for(sample, main_seed=cfg.run.seed, epoch=e, cfg=cfg.candidates)` → `catalog.titles[c]`로 prompt 후보 목록
  - M5에는 `functools.partial(candidates_for, main_seed=cfg.run.seed, cfg=cfg.candidates)`로 넘김 (module 수준 함수 + frozen 설정이라 Windows spawn에서 pickle 가능). `epoch`는 키워드 전용이므로 M5는 반드시 `fn(sample, epoch=e)` / `fn(sample, epoch=None)`으로 부름 (위치 인자로 부르면 `TypeError`)
- M11 (평가): `candidates_for(sample, main_seed=cfg.run.seed, epoch=None, cfg=cfg.candidates)`, `candidate_flags(cands, sample.target, catalog)` → `EvalRecord.cand_ids`, `cand_titles`, `flags`. 판정 정규화는 `match_norm`
- M12·M14: flag별 표본 수와 flag 없는 표본만의 보조 지표를 보고에 씀 (값 정의는 M11·M12)
- M13: 아래 완료 기준을 테스트로 구현

#### 6.6.10 계약 추가 제안
- `candidates_for`, `candidate_flags`, `FLAG_NAMES`, `match_norm`을 M2 공개 API로. `EvalRecord.flags`의 내용 = `candidate_flags` 결과
- M11 판정 규칙의 정규화는 `candidates.match_norm` 하나만 씀 (flag와 판정이 같은 정규화를 쓰게)
- 실행 진입점이 환경 점검 직후 `candidates.self_check()`를 부름. M0 환경 점검에 numpy 1.24.2 판 확인(불일치 시 실패) 추가 (M1 5.6.12와 같은 제안. 판 기록은 `run.environment_info()`가 이미 함)
- M5의 후보 함수 형식 (`CandidateFn`, 9.6.1): `functools.partial(candidates_for, main_seed=..., cfg=...)`, 호출은 `fn(sample, epoch=e | None)` (6.6.9)

### 6.7 결정
- D10 (정답 포함 20, 평가 고정, 학습 epoch별 재추출), D21 (보이는 window와 정답 제외), D20 (`code` 제목 유지, flag로 표시), D17 (판정 규칙은 M11)
- 이 장의 새 `[결정]`
  - 난수 호출 순서 `choice` → 정답 붙이기 → `shuffle`: 공개 코드 순서(`:46-47`)를 그대로 따름
  - numpy 1.24.2 고정과 golden `self_check`: Generator 메서드 출력이 판에 묶이므로
  - flag 5개 정의와 `target_unscorable` 추가: 판정 규칙이 정답 출력도 무효로 만드는 표본을 표본 단위로 분리하기 위함
  - `num_candidates`를 선택지 20 하나인 키로 둠: 프로파일에 명시하되 계약과 template 문구에서 벗어나지 않게

### 6.8 완료 기준
1. (CPU) 계약: val·test 전 표본과 train 처음 2,000 표본(epoch 0, 1)에서 길이 20, 정답 1개, 중복 없음, history와 교집합 없음, id 0–1681
2. (CPU) 결정성: 같은 key 두 번 → 같은 tuple. train 같은 row의 epoch 0과 1 → 다름 (예: train row 3). DataLoader worker 0·2·8에서 표본별 후보가 같음
3. (CPU) golden: seed 1234에서 val row 0, test row 0 tuple이 6.6.5와 같고 직렬화 sha256이 val `9a073206…`, test `1ffdb649…`. `self_check()` 통과
4. (CPU) smoke 부분집합(M1 5.6.7)의 val·test 표본 후보가 full 데이터 같은 행의 후보와 같음
5. (CPU) flag: 손으로 만든 작은 catalog로 flag마다 발동 사례와 아깝게 빗나간 미발동 사례를 만들고 켜지는 flag 집합 전체를 비교. `target_title_duplicated`·`target_unscorable`은 정의상 혼자 켜지지 않으므로(6.6.4) 함께 켜지는 flag까지 기대값에 넣음 `[결정]`. seed 1234 고정 후보에서 표 6.6.6의 수 (val 1/0/1/0/1, 하나 이상 2. test 1/0/4/0/2, 하나 이상 5)
6. (CPU) val·test 전 표본에서 `target_unscorable == not M11판정(first_line=정답 제목).correct` (M11 판정 함수로 교차 확인)
7. (CPU) 잘못된 호출 거부: train 표본에 `epoch=None`, val 표본에 `epoch=0`, `epoch=-1`, `epoch=True`, `target ∈ history`, `num_candidates=21` → 각각 오류
8. (CPU) 균등성: train row 10(이력 길이 10인 첫 표본, |pool| = 1,671)을 epoch 0–9,999로 뽑아 아이템별 음성 빈도(기대값 10,000 × 19 / 1,671 ≈ 113.7)와 정답 위치 0–19 빈도에 카이제곱 적합도 검정, 둘 다 p > 0.001. 설계 시점 계산값 p = 0.380, 0.398 (고정 seed라 결정적, 계산: `design/tmp/m2_uniform.py`)
9. (VESSL_REQUIRED) VESSL 환경 numpy가 1.24.2이고 `self_check()` 통과 (fork 설치 절차상 예상되나 VESSL에서 확인)

---

## 7. M3 SASRec 인코더
상태: 테스트 통과(로컬)

### 7.1 목적과 범위
- **목적:** 사전학습된 SASRec(공개 `rec_model/movielens.pt`)을 고정한 채, 이력 표현 `z`(gate 입력, M7·M12·M15)와 아이템 표현(projector 입력, M6)을 만든다.
- **범위:**
  - 일회성 변환 도구 `tools/convert_sasrec.py`: 통째 pickle → 일반 state 파일 + 원본 코드로 계산한 기준 출력 + `manifest.json`
  - 재구현 클래스 `ilora/sasrec.py`의 `SASRecEncoder`: 원본 `cacul_h`(`U recommender/A_SASRec_final_bce_llm.py:202-217`)와 `cacu_x`(`:219-222`)가 쓰는 계산만, 같은 의미로 다시 구현
  - 입력 검사, 고정(eval, `no_grad`, `requires_grad = False`), fp32 계산 (D12, D23)
  - 동등성 확인 함수 (M0 check-env, M13)
- **범위 밖:** SASRec 학습·미세조정, 다음 아이템 점수(`s_fc`), Caser·GRU 인코더, KMeans(M15), 기록 파일 쓰기(M12), 이력 window 구성(M1)

### 7.2 입력과 출력
**입력**
- 원본 pickle `rec_model/movielens.pt`: 973,163 byte, git blob `41c057dc`, SHA-256 `6928a06778d0e5b64ead0641501f73f13beb2c61ca2d881ee15582630d101df9`. 변환 도구만 읽음
- 변환 state 파일(`sasrec.state_path`): `torch.load(..., weights_only=True)`로 읽는 dict (7.6.2)
- `hist_ids [B, 10] int64`(오른쪽 padding 1682), `len_seq [B] int64`: 3.1 `TrainBatch`·`EvalBatch` (M5)
- 아이템 id `[...] int64`, 값 0–1681: M9가 prompt에 넣을 이력·후보 아이템

**출력**
- `z [B, 64] float32`: 3.1 `z` (eval, `no_grad`, autocast 밖, 정답 미포함). 입력과 같은 장치
- 아이템 표현 `[..., 64] float32`: 3.1 아이템 표현
- `SASRecEncoder.state_digest: str`: meta.json, checkpoint의 출처 기록 (M15 산출물은 변환 state 파일 sha256으로 연결, 7.6.8)
- 도구 산출물(`runs/rebuild/assets/sasrec/`): `sasrec_movielens_state.pt`, `sasrec_movielens_reference.pt`, `manifest.json`

### 7.3 논문 근거
- 이력 표현: "we utilize a sequential recommender (e.g., SASRec [24]) to extract its representation as follows: z = SR-EMBθ(i<n). (10)" `[논문 p.5 §3.2.2 식 10]`, "Here z ∈ R^d" `[논문 p.6]`. 입력이 i<n뿐이라 정답 i_n은 들어가지 않음
- 아이템 표현: "item i's ID embedding z, pretrained using the encoder SR-EMB(·) of sequential recommender (e.g., SASRec [24], GRU4Rec [28]), is represented as: z = SR-EMBθ(i). (5)" `[논문 p.4 §2 식 5]`, "e_s^i = SR-EMB(i; Θe)" `[LLaRA 간접 §4.1 p.5 식 4]`
- 두 용도에 같은 모델: 식 5와 식 10이 같은 SR-EMB_θ `[논문 p.4-5 (암시)]`
- 고정: 그림 2의 SR-EMB 두 블록 모두 눈송이 `[논문 p.4 그림 2 (암시)]`, "The sequential recommender is well-trained and frozen" `[LLaRA 그림 2 caption p.4]`, 목적식이 Θ, Θp만 최적화(Θe 없음) `[LLaRA 간접 §4.3 p.6 식 11]`
- d = 64: "SASRec, with a dimension of 64" `[논문 App C p.17]`, 전통 추천기 "an embedding dimension of 64" `[논문 App E p.17]`, "an embedding dimension d of 64" `[LLaRA §5.1.2 p.7]`. SR-EMB가 이 SASRec이라는 것은 `(암시)`. 값은 가중치에서 다시 확인 `[SPEC §4]` (7.6.2)
- 이력 길이: "we retain the last 10 interactions as the historical sequence, padding sequences with fewer than 10 interactions." `[LLaRA §5.1.1 p.7]`
- 논문에 없는 것: pooling 위치, 층·head 수, dropout 모드, 계산 정밀도 → `[코드]`, D12, D23

### 7.4 공개 코드에서 가져올 것
- **가중치:** `U rec_model/movielens.pt` = `U rec_model/SASRec_movielens.pt` = `L rec_model/movielens.pt` (blob `41c057dc`, 코드 계보 일치). zip 안 기록 이름 `SASRec_val_0.4206_hr1_0.4222_…_lr0.001_l2decay_0.0001_seed_1234` (`[논문 App E p.17]`의 lr 0.001과 맞지만 근거로 쓰지 않음)
- **구조** (pickle 검사로 확인, 7.6.1): item 임베딩 1683×64(행 1682 = padding), 위치 임베딩 10×64, LayerNorm 4개(`ln_1`, `ln_2`, `ln_3`, `feed_forward.layer_norm`, eps 1e-5), attention 1층 1 head(`linear_q/k/v` 64→64, bias 있음), FFN `Conv1d(64, 64, 1)` 2개 + ReLU, dropout p = 0.1 네 곳. `U recommender/A_SASRec_final_bce_llm.py:147-170`, `SASRecModules_ori.py:7-94` (blob `3ac3c975`, `U` = `L`)
- **`cacul_h` 계산 순서** `U recommender/A_SASRec_final_bce_llm.py:202-217` (`L :202-215`. 실행되는 차이는 `U :203-204` 장치 이동 2줄뿐이고, 그 밖에는 `U :216` 주석 1줄):
  - 아이템 + 위치 임베딩(0–9) → (emb dropout) → `item ≠ 1682` mask 곱 → `ln_1` → attention(query = `ln_1` 출력, key·value = mask된 입력) → `ln_2` → FFN → mask 곱 → `ln_3` → `len_seq − 1` 위치 (`extract_axis_1`, `:13-18`)
  - attention `SASRecModules_ori.py:40-94`: 점수 ÷ √hidden_size(head 크기가 아님, `:59`), key mask = key 합 ≠ 0(`:62-65`), 그다음 causal mask(`:68-72`), 가림 값 −2^32+1, softmax, query mask = query 합 ≠ 0(`:78-80`), (dropout `:83`), query를 더하는 residual(`:92`)
  - FFN `SASRecModules_ori.py:15-22`: `w_2(relu(w_1(x)))`, (dropout), `layer_norm(출력 + x)`
- **`cacu_x`** = item 임베딩 조회 `U :219-222` (`L :217-220`)
- **padding id** = `item_num` = 1682 `U data/movielens_data.py:18`, mask `U recommender/A_SASRec_final_bce_llm.py:208`
- **로드 의도:** `eval()`과 `requires_grad = False` `U model/model_interface.py:406-412` (`L :271-277`). 의도는 유지하고 구조로 강제함 (D12)

### 7.5 버릴 것 (알려진 문제)
- **통째 pickle:** `__main__.SASRec` 클래스가 있어야 읽히고 `torch.load`가 임의 코드를 실행할 수 있음. 저장 당시 상태가 그대로 들어 있음: `training = True`, `device = cuda:0`, `state_size`·임베딩 개수가 numpy int64 (pickle 검사) → 런타임에는 쓰지 않고 한 번 변환 (7.6.1)
- **학습 중 dropout 켜짐:** 로드할 때 eval이지만 Lightning이 학습 시작과 validation 뒤에 `model.train()`을 불러 SASRec dropout이 켜짐 (PL 1.8.6 `trainer/trainer.py:1194`, 감사 확인). CPU 측정: 같은 1,304개 표본에서 train 모드 z와 eval z의 최대 차이 9.63 (`design/tmp/proto_convert_sasrec.py`) → D12
- **fp16 계산:** Lightning precision 16(`U main.py:79-80`)의 autocast 안에서 `cacul_h`가 돌아 bmm·linear·conv가 fp16. 가림 값 −2^32+1은 fp16에서 −inf가 됨 (확인) → D23
- **숨은 장치 이동:** `cacul_h`가 입력을 pickle의 `self.device`(cuda:0)로 옮김 `U :203-204` → 재구현은 장치가 다르면 오류
- **`s_fc`(1682×64 + 1682)와 `self.dropout`:** z와 아이템 표현에 쓰이지 않음 → 변환본에서 뺌 (manifest에 기록)
- **표본별 Python 반복:** `extract_axis_1` `U :13-18` → `gather` 한 번 (값 동일, 7.8)
- **가운데 1 차원:** `cacul_h` 출력 `[B, 1, 64]`(`:17`의 unsqueeze) → 3.1 `z`는 `[B, 64]`
- **fork 수정 파일:** `F SASRecModules_ori.py`는 bf16 tril 우회로 바뀜 (`F 761dea8`, blob `e1e33498`). fp32 값은 같지만 원본 blob이 아니므로 변환 도구가 쓰지 않음. D23(fp32)이라 재구현에는 우회가 필요 없음

### 7.6 설계

#### 7.6.1 변환 도구 `tools/convert_sasrec.py` `[결정]`
```python
UPSTREAM_COMMIT = "28e47371192121b6ced44401d9e149fedfae13de"
ORIGINAL_BLOBS = {"SASRecModules_ori.py": "3ac3c975a58d43ee866b637ee5d4f99534974a7b",
                  "recommender/A_SASRec_final_bce_llm.py": "dc7c94999a4db5f1b97971a2913baae44687c124"}
PICKLE_BLOB = "41c057dc96df2cc6d432d7f839d98b57ac341b26"
PICKLE_SHA256 = "6928a06778d0e5b64ead0641501f73f13beb2c61ca2d881ee15582630d101df9"
EXPECTED_GLOBALS: frozenset[tuple[str, str]]     # 아래 18개
REF_CHUNK = 4096
REF_THREADS = 1

def main(argv: Sequence[str] | None = None) -> int
def materialize_original_sources(repo: Path, dest: Path) -> dict[str, str]          # 경로 -> blob id
def scan_pickle_globals(pt_path: Path) -> frozenset[tuple[str, str]]               # 실행하지 않는 정적 검사
def load_original_model(pt_path: Path, original_module: ModuleType) -> nn.Module
def describe_original_model(model: nn.Module) -> dict                              # manifest "observed"
def read_reference_samples(data_dir: Path) -> dict[str, tuple[list[str], torch.Tensor, torch.Tensor]]
def compute_reference(model: nn.Module, hist_ids: torch.Tensor, len_seq: torch.Tensor) -> torch.Tensor
def write_outputs(out_dir: Path, state: dict, reference: dict, manifest: dict) -> None
class ConversionError(RuntimeError): ...                                         # 거부 사유 (7.8 #1)
def convert(repo: Path, out_dir: Path) -> dict                                    # 1–12단계, manifest 반환
def build_parser() -> argparse.ArgumentParser                                    # --repo, --out. M14 RUNBOOK 시험이 명령 검사에 씀
```
- 실행: `python -I rebuild/tools/convert_sasrec.py --repo <저장소 루트> --out runs/rebuild/assets/sasrec`. CPU만, 난수 없음. 출력 폴더가 있고 비어 있지 않으면 거부 (덮어쓰기 금지, 4.6.3)
  - `--repo` 기본값은 도구가 든 저장소이고, 상대 경로 `--out`은 `--repo` 기준 `[결정]`: M0 경로 규칙(4.6.2)과 같음. 거부는 모두 `ConversionError`이고 `main`이 `convert_sasrec: refused: <사유>`를 출력한 뒤 1을 돌려줌 `R tools/convert_sasrec.py main`
- `state_digest`와 파일 형식 상수는 `ilora.sasrec`에서 import (런타임 검사와 같은 함수). 기준 z는 원본 코드로만 계산
- 순서:
  1. `git -C <repo> cat-file -e <UPSTREAM_COMMIT>^{commit}`. 실패하면 중단 ("전체 git 이력 필요: git fetch --unshallow")
  2. 원본 파일마다 `git rev-parse <commit>:<경로>`가 고정 blob과 같은지 확인하고 `git cat-file blob <id>`로 임시 폴더에 씀. 임시 폴더는 출력 폴더 옆(`<out>/../convert_sasrec_src_*`)에 만들고 끝나면 지움 `[결정]`: 쓰기가 모두 `runs/rebuild/`(시험은 tmp) 아래에 머물게 함. 바이트로 git blob 해시를 다시 계산해 확인. 작업 트리 파일은 쓰지 않음 (7.5 fork 수정본)
  3. pickle 파일의 git blob 해시와 SHA-256 확인
  4. 정적 검사: zip 안 `data.pkl`을 `pickletools.genops`로 훑어(실행하지 않음) global 집합이 아래 18개와 정확히 같은지 확인
     - `__main__.SASRec`, `SASRecModules_ori.{MultiHeadAttention, PositionwiseFeedForward}`, `torch.nn.modules.{sparse.Embedding, normalization.LayerNorm, linear.Linear, conv.Conv1d, dropout.Dropout, activation.Softmax}`, `torch._utils.{_rebuild_tensor_v2, _rebuild_parameter}`, `torch.{FloatStorage, device}`, `collections.OrderedDict`, `__builtin__.set`, `numpy.dtype`, `numpy.core.multiarray.scalar`, `_codecs.encode`
  5. 임시 폴더를 이 도구 프로세스의 `sys.path` 맨 앞에만 넣고 `A_SASRec_final_bce_llm`을 import. `SASRecModules_ori.__file__`이 임시 폴더 안인지 확인 (저장소 루트의 수정본을 잡지 않게). 비교는 `Path(...).resolve()`끼리 (Windows에서 `__file__`은 `/`와 `\`가 섞여 문자열 비교가 틀림, 확인 `design/tmp/rv_enc/rv_sasrec.py`)
  6. `torch.load(path, map_location="cpu", pickle_module=<대응 모듈>, weights_only=False)`. 대응 모듈의 `Unpickler.find_class`는 허용 목록 밖 global을 거부하고 `__main__.SASRec`만 원본 클래스로 바꿈 (`FloatStorage`는 torch 2.0.0의 `UnpicklerWrapper`가 먼저 처리하므로 허용 목록은 17개)
  7. 구조 확인과 기록: 객체 타입, 속성(`device`, `hidden_size`, `item_num`, `state_size`, `training`), 하위 모듈 이름, dropout p, LayerNorm eps, head 수, Conv1d 설정, state 텐서 22개의 이름·모양·dtype. 기대와 하나라도 다르면 중단
  8. `model.device = torch.device("cpu")` (pickle의 cuda:0을 덮어씀. `cacul_h`가 입력을 이 장치로 옮기므로 필요), `model.eval()`, 모든 하위 모듈 `training == False` 확인
  9. 기준 표본: pandas로 배포 파일 3개를 직접 읽음 (M1과 독립된 기준). `.df`는 pickle이라 읽기 전에 M0 `content_sha256`으로 `EXPECTED_INPUTS` 값과 대조. split마다 `len_seq ≥ 3`인 행 전부를 원래 행 순서로 (train 66,126 / val 94 / test 95 = 66,315), `sample_id = f"{split}:{row}"`, `hist_ids`는 `seq` 튜플의 첫 원소. 오른쪽 padding 배치 확인
  10. `torch.set_num_threads(REF_THREADS)`, `no_grad`에서 split마다 따로 `REF_CHUNK`개씩 원본 `cacul_h` → `z_ref [66315, 64]`. `cacu_x(arange(1682))` → `item_emb_ref [1682, 64]`
  11. 원본 자체 점검 (manifest에 기록): 같은 입력 반복 시 bitwise 동일, padding 칸 내용을 바꿔도 pooled z bitwise 동일
  12. 쓰기 (임시 이름으로 쓴 뒤 rename): state 파일(7.6.2), 기준 파일(`format` = `"ilora-rebuild-sasrec-ref-v1"`(`sasrec.SASREC_REFERENCE_FORMAT`), `sample_ids`, `hist_ids`, `len_seq`, `z`, `item_ids`, `item_emb`, `chunk`, `num_threads`, `state_digest`), `manifest.json`(저장소 HEAD와 변경 여부, 고정 commit·blob, pickle 해시·크기·zip 기록 이름, 18개 global, 7단계 관찰값, digest, 뺀 key의 모양 `state.dropped`, 11단계 결과 `self_check`, 데이터 파일 SHA-256, Python·torch·numpy·pandas 버전, UTC 시각, 두 `.pt` 파일의 sha256 `outputs`). 쓴 뒤 두 파일을 `weights_only=True`로 다시 읽고 state 파일은 `load_sasrec_state`로 확인 `R tools/convert_sasrec.py _reread`
      - 두 `.pt` 파일에는 텐서·str·int·float·list·dict만 넣음. pickle의 `state_size`는 numpy int64라 `int()`로 바꿔 씀: torch 2.0.0 `weights_only=True`는 numpy 스칼라를 `UnpicklingError`로 거부 (확인 `design/tmp/rv_enc`). manifest의 관찰값은 타입 이름(`"int64"`)과 Python int 값을 따로 기록 (`json`은 numpy int64를 쓰지 못함)
- prototype(`design/tmp/proto_convert_sasrec.py`, venv_rb CPU)에서 4–12단계 동작 확인: `find_class` 대응으로 원본 클래스 로드 성공, `weights_only=True` 다시 읽기 성공, 11단계 차이 둘 다 0

#### 7.6.2 state 파일과 digest
- 파일 내용 (`weights_only=True`로 읽히는 dict):
  - `format: "ilora-rebuild-sasrec-v1"`
  - `arch: {"hidden_size": 64, "item_num": 1682, "state_size": 10, "num_heads": 1, "layernorm_eps": 1e-5}` (pickle에서 읽은 값을 Python int·float로 바꿔 씀, 7.6.1 12단계)
  - `state_dict`: 텐서 20개, 모두 float32, 값 129,664개. `item_embeddings.weight [1683, 64]`, `positional_embeddings.weight [10, 64]`, `ln_{1,2,3}.{weight,bias} [64]`, `mh_attn.linear_{q,k,v}.weight [64, 64]`·`.bias [64]`, `feed_forward.w_{1,2}.weight [64, 64, 1]`·`.bias [64]`, `feed_forward.layer_norm.{weight,bias} [64]`
  - `dropped_keys: ["s_fc.bias", "s_fc.weight"]`, `source: {path, git_blob, sha256}`, `digest`
- `state_digest(state)`: key를 정렬해 key마다 `f"{key}|{dtype}|{모양을 쉼표로}\n"`(dtype은 `float32`처럼 `torch.` 없이)와 텐서 원시 byte(CPU, contiguous, little-endian)를 SHA-256에 차례로 넣음. 파일 byte가 아닌 내용 해시라 파일 이름·저장 형식과 무관
- prototype 계산값 `6a78697b92ce8994250842dc780eac723285e5f51741264c09706e0057c51dae` (`design/tmp/state_digest.py`). 도구 출력과 같아야 하며 프로파일 `sasrec.state_digest`에 적음. 구현한 도구의 로컬 변환도 같은 값을 냄 (`runs/rebuild/assets/sasrec/manifest.json`의 `state.digest`, `created_utc` 2026-11-05T23:06Z, torch 2.0.0+cpu). VESSL에서 만든 state 파일은 check-env의 `load_sasrec_state`가 이 값과 대조
- 관찰 (설계 영향 없음): padding 행 1682의 최대 절댓값 6.1e-40. 아이템 160개의 임베딩 norm < 1e-30이고 그중 157개는 `train_data.df`에 나오지 않음 (M6 10.6.5)

#### 7.6.3 공개 API (`ilora/sasrec.py`)
```python
SASREC_STATE_FORMAT = "ilora-rebuild-sasrec-v1"
EXPORT_CHUNK = 4096      # 기준 계산과 같은 묶음 크기
EXPORT_THREADS = 1       # CPU export의 torch 스레드 수
SASREC_REFERENCE_FORMAT = "ilora-rebuild-sasrec-ref-v1"   # 기준 파일 format (7.6.1 12단계)
CUDA_REFERENCE_ATOL = 1e-4   # CUDA z와 CPU 기준의 허용 차이 (7.8 #12). CPU는 bitwise. M13 smoke 환경 점검이 import
STATE_FILE_KEYS: tuple[str, ...]        # state 파일 key 6개 (7.6.2)
ARCH_KEYS: tuple[str, ...]              # arch key 5개 (7.6.2)
REFERENCE_FILE_KEYS: tuple[str, ...]    # 기준 파일 key 10개 (7.6.1 12단계, 변환 도구도 import)

class SASRecError(RuntimeError): ...    # state·기준 파일, 생성 시 대조, 기준 비교 실패 (7.6.6)

@dataclass(frozen=True)
class SASRecArch:
    hidden_size: int; item_num: int; state_size: int; num_heads: int; layernorm_eps: float
    def as_dict(self) -> dict[str, Any]: ...    # ARCH_KEYS 순서 (state 파일 arch)

def expected_state_shapes(arch: SASRecArch) -> dict[str, tuple[int, ...]]   # 7.6.2 텐서 20개 이름·모양 (변환 도구도 씀)
def state_digest(state: Mapping[str, torch.Tensor]) -> str
def load_sasrec_state(path: Path, expected_digest: str) -> tuple[SASRecArch, dict[str, torch.Tensor], str]
def strict_fp32(device: torch.device) -> ContextManager[None]       # autocast 끔 + CUDA TF32 끔, 끝나면 복원
# pad_histories는 M1 data.py (5.6.13)에서 import: encode_all이 data.pad_histories(histories, PAD_ID, HISTORY_LEN)을 씀

class SASRecEncoder(nn.Module):
    hidden_size: int; item_num: int; state_size: int; state_digest: str
    def __init__(self, arch: SASRecArch, digest: str) -> None
    @classmethod
    def from_config(cls, cfg: Config, catalog: ItemCatalog, history_len: int) -> "SASRecEncoder"   # CPU에 생성
    def train(self, mode: bool = True) -> "SASRecEncoder"
    def forward(self, hist_ids: torch.Tensor, len_seq: torch.Tensor) -> torch.Tensor     # z [B, 64]
    def embed_items(self, item_ids: torch.Tensor) -> torch.Tensor                       # [..., 64]
    def encode_all(self, histories: Sequence[Sequence[int]], *, chunk: int = EXPORT_CHUNK,
                   num_threads: int = EXPORT_THREADS) -> torch.Tensor                  # [N, 64] CPU, 한 split씩

def verify_against_reference(encoder: SASRecEncoder, reference_path: Path, device: torch.device) -> dict
```
- 다른 모듈이 부르는 이름은 이것뿐: `encoder(hist_ids, len_seq)`(= `forward`), `encoder.embed_items(ids)`, `encoder.hidden_size`, `encoder.state_digest` (M9 13.6.1의 `SeqEncoder` 프로토콜이 같은 이름을 씀)
- 경로 설정(`sasrec.state_path` 등)은 저장소 루트(M0 `REPO_ROOT`) 기준 상대 경로
- 하위 모듈 이름은 원본 state key와 같음(`item_embeddings`, `positional_embeddings`, `ln_1`–`ln_3`, `mh_attn.linear_q/k/v`, `feed_forward.w_1/w_2/layer_norm`) → `load_state_dict(strict=True)`
- Dropout 모듈은 만들지 않음 `[결정]`: D12로 항상 eval이라 항등이고, 없으면 출력이 학습 모드 표시에 의존할 길이 없음. 원본 위치 3곳(임베딩 합 뒤, attention 확률 뒤, FFN 출력 뒤)은 코드 주석으로 남김
- head 분할·연결, ÷√hidden_size, 가림 값, mask 순서, `Conv1d(k=1)`을 원본 식 그대로 옮김 (Linear로 바꾸지 않음: 같은 연산이어야 CPU bitwise 일치)
- `from_config`: `load_sasrec_state(cfg.sasrec.state_path, cfg.sasrec.state_digest)` → 생성 → 대조(7.6.6) → 고정. 장치 이동은 M9
- `verify_against_reference(encoder, reference_path, device)` `[결정]` (설계가 반환과 실패 방식을 정하지 않았음): encoder는 이미 `device`에 있어야 함 (아니면 `ValueError`). 기준 파일의 `chunk`·`num_threads`로 split마다 `encode_all`, 기준 아이템 id로 `embed_items`를 계산하고, 기준 `hist_ids`·`len_seq`가 M1 `pad_histories` 결과와 같은지도 확인. CPU는 z와 아이템 표현이 bitwise, CUDA는 z 최대 차이 ≤ `CUDA_REFERENCE_ATOL`이고 아이템 표현이 bitwise면 통과 (허용 오차는 M3가 적용)
  - 반환 dict: `passed`, `device`, `reference_path`, `state_digest`, `chunk`, `num_threads`, `tolerance`, `n`, `splits`(split마다 `n`, `max_abs_diff`, `bitwise`, `padding_matches`), `item_emb`(`n`, `max_abs_diff`, `bitwise`), `max_abs_diff`, `bitwise`. M13 smoke의 환경 점검 검사가 `max_abs_diff`·`bitwise`를 읽음
  - 실패하면 같은 dict를 메시지와 `.report`에 담은 `SASRecError` (7.6.6: 검사 실패는 값과 함께 즉시 올림). M0 check-env는 예외와 `passed` false를 똑같이 실패로 봄 `R ilora/sasrec.py verify_against_reference`

#### 7.6.4 `forward(hist_ids, len_seq)` 단계
1. 입력 검사 (`ValueError`, 메시지에 값 포함):
   - `hist_ids` int64 `[B, state_size]`, `len_seq` int64 `[B]`, B ≥ 1, 두 텐서와 parameter가 같은 장치
   - `1 ≤ len_seq ≤ state_size`
   - 오른쪽 padding: 위치 `< len_seq`는 0–1681, 위치 `≥ len_seq`는 1682. 왼쪽 padding·중간 1682는 거부
2. 상태 검사 (`RuntimeError`): parameter가 모두 float32이고 `requires_grad == False`, `self.training == False`
3. `torch.no_grad()`와 `strict_fp32(device)` 안에서:
   1. `e = item_embeddings(hist_ids) + positional_embeddings(arange(state_size))`
   2. `mask = (hist_ids != item_num).float().unsqueeze(-1)`, `seq = e * mask`
   3. `q = ln_1(seq)` → attention(q, seq) (7.4 순서 그대로) → `a`
   4. `f = feed_forward(ln_2(a))` → `f = f * mask` → `f = ln_3(f)`
   5. `z = f.gather(1, (len_seq − 1)[:, None, None].expand(−1, 1, hidden_size)).squeeze(1)`
4. 반환 전 검사: `[B, hidden_size]`, float32, 유한값. z는 일반 텐서(`inference_mode` 아님)라 gate(M7)의 autograd 입력이 될 수 있음
- `embed_items(item_ids)`: int64, 모든 값 0–1681 (1682와 범위 밖은 `ValueError`), 같은 장치, `no_grad` 조회 → float32
- `train(mode)`: 요청과 무관하게 `super().train(False)` (부모 모델의 `.train()`이 와도 eval 유지, D12)
- `encode_all(histories, chunk, num_threads)`: M1 `pad_histories`(5.6.13) → `chunk`개씩 `forward`. CPU면 호출 동안 `torch.set_num_threads(num_threads)` 후 복원. 기준 파일의 `chunk`·`num_threads`로 한 split씩 행 순서대로 부르면 CPU에서 기준과 bitwise 같음 (`verify_against_reference`가 이 값을 기준 파일에서 읽어 씀). 이력 형식 오류는 `pad_histories`의 `DataError`(`forward`의 텐서 검사는 `ValueError`), `chunk`·`num_threads`가 1 이상 int가 아니면 `ValueError` `R ilora/sasrec.py SASRecEncoder.encode_all`

#### 7.6.5 dtype과 장치
- parameter: float32, `requires_grad = False`, optimizer·GradScaler 밖. CPU에서 만들고 M9가 모델 장치로 옮김
- 계산: autocast 밖 fp32 (D23). torch 2.0.0 기본값은 `torch.backends.cudnn.allow_tf32 = True`, `torch.backends.cuda.matmul.allow_tf32 = False` (확인) → A100에서 FFN `Conv1d`가 TF32로 계산됨. `strict_fp32`가 호출 동안 두 flag를 끄고 이전 값으로 복원 `[결정]`. `torch.backends.cudnn.flags()`는 인자를 모두 덮어써(`enabled` 기본 False) 쓰지 않음
- 수치 (CPU 측정, `design/tmp/proto_chunk_check.py`, `proto_split_chunk_check*.py`, `proto_threads_check.py`):
  - 같은 묶음·같은 스레드 수면 원본과 bitwise 같음 (split별 4,096 단위, 스레드 1과 14 모두)
  - batch 8로 나누면 기준과 최대 7.63e-6, 스레드 1과 14 차이 최대 2.9e-6. 원본 코드 자체도 묶음 크기에 따라 3.8e-6까지 다름
  - 스레드 1에서 1,024개씩(M12 `records.export_batch_size`)과 4,096개씩은 train 66,126개 모두 bitwise 같음 (로컬 CPU 측정, `design/tmp/rv_enc/rv_sasrec.py`). 다른 CPU에서도 같다는 보장은 없어 기계 사이 비교는 허용 오차로 함
  - 같은 z 함수이고 차이는 fp32 반올림뿐. D12·D23의 "gate 입력 = KMeans 입력"은 이 오차 안에서 성립
- `.half()` 등으로 parameter dtype이 바뀌면 다음 호출에서 오류 (조용히 fp16으로 계산하지 않음)

#### 7.6.6 오류 처리
- 로드: 파일 없음, `format` 불일치, key 집합·모양·dtype 불일치, 유한하지 않은 값, 다시 계산한 digest가 파일 안 값이나 `sasrec.state_digest`와 다름 → `SASRecError`로 중단
- 생성 시 대조: `catalog.pad_id`·`catalog.num_items`(M1) ≠ `item_num`, 이력 길이(M1·M5) ≠ `state_size` → `SASRecError`로 중단
- 기준 비교 실패 → `SASRecError` (측정값은 `.report`, 7.6.3). 입력 검사는 `ValueError`, 상태 검사는 `RuntimeError` (7.6.4)
- `SASRecError`는 `RuntimeError`의 하위 클래스 `[결정]`: 설계에 이름이 없어 M1 `DataError`, M5 `BatchingError`처럼 모듈 고유 예외를 둠 `R ilora/sasrec.py SASRecError`
- 예외를 잡아 기본값으로 바꾸지 않음. 검사 실패는 모두 해당 값과 함께 즉시 올림

#### 7.6.7 설정 (`sasrec` 구역)
| 키 | 타입 | paper_spec 값 | 라벨 | 설명 |
|---|---|---|---|---|
| `source_path` | str | `rec_model/movielens.pt` | `[코드]` U·L 같은 blob `41c057dc` | 원본 pickle. 변환 도구 입력, meta.json 해시 기록 |
| `source_sha256` | str | `6928a06778d0e5b64ead0641501f73f13beb2c61ca2d881ee15582630d101df9` | `[코드]` (계산값) | check-env가 원본 파일을 대조. M0 `EXPECTED_INPUTS["rec_model/movielens.pt"]`와 같아야 함 (다르면 설정 오류) |
| `state_path` | str | `runs/rebuild/assets/sasrec/sasrec_movielens_state.pt` | `[결정]` | 변환 state 파일 |
| `state_digest` | str | `6a78697b92ce8994250842dc780eac723285e5f51741264c09706e0057c51dae` | `[결정]` (prototype 계산값) | 7.6.2 내용 digest |
| `reference_path` | str | `runs/rebuild/assets/sasrec/sasrec_movielens_reference.pt` | `[결정]` | 원본 코드 기준 출력 (check-env, M13) |
- 차원(64), item 수(1682), 길이(10), head 수는 설정 항목이 아니라 state 파일에서 읽고 M1 값과 대조 `[SPEC §4]`
- dropout 모드와 계산 정밀도는 변형이 없어 설정 항목을 두지 않음 (D12, D23)

#### 7.6.8 다른 모듈과의 연결
- M1: `ItemCatalog.pad_id`·`num_items`, 이력 길이 10과 생성 시 대조. `Sample.history`는 `pad_histories`로 `hist_ids`가 됨
- M5: `TrainBatch`·`EvalBatch`의 `hist_ids`, `len_seq`를 그대로 받음
- M9: 매 micro-batch·평가 batch마다 `z = encoder(hist_ids, len_seq)`. hybrid batch면 prompt에 들어갈 이력(행마다 `len_seq`개)·후보(20개) id만 모아 `embed_items` → M6
- M7: `z`가 gate 입력, `strict_fp32` 공유
- M12·M15: z 표(`RepresentationTable`)는 M12 `compute_representations`가 `records.export_batch_size`(1024)개씩 `encoder(hist_ids, len_seq)`를 불러 만들고 (`records.export_num_threads` 1), M15도 같은 함수로 군집 입력을 만듦 (KMeans fit은 train만, D25). `encode_all`은 기준 대조(check-env, M13) 전용
- M15: 배정 산출물 출처는 변환 state 파일의 sha256(`load_cluster_artifacts(sasrec_sha256=...)`)과 live z 대조(`verify_live_encoder`)로 확인. 내용 digest는 meta.json·checkpoint에 기록
- M0: check-env가 `source_sha256`, `state_digest`, `verify_against_reference`(CUDA 허용 오차 7.8)를 실행. meta.json에 두 해시
- M10: SASRec parameter를 optimizer에 넣지 않음. SASRec 가중치는 checkpoint에 넣지 않고 `train_state.pt`의 `sasrec_state_digest`로 기록하며, epoch 끝마다 `state_digest(model.sasrec.state_dict())`를 다시 계산해 시작 값과 비교 (14.6.4, 14.6.6, 7.8 #13) `R ilora/train.py _Trainer.run_epoch`
- M14: RUNBOOK이 clone 뒤 변환 도구를 한 번 실행 (전체 git 이력 필요)
- `[SPEC §10]` 판정: A (사전학습 + 고정). 학습 중 z가 바뀌지 않아 고정 군집 배정을 쓸 수 있음
- **계약 추가 제안:**
  - `pad_histories(histories, pad_id, length) -> (hist_ids [N, length] int64, len_seq [N] int64)`: padding 규칙 한 곳. 최종 검토에서 M1 `data.py`(상수 `PAD_ID`, `HISTORY_LEN`과 같은 곳, M1 5.6.13)로 정함 — M3에 두면 M5(의존 M4)가 M3를 import하게 됨. M3 `encode_all`, M5 collate, M12 export가 함께 씀
  - 3.1 아이템 표현: 입력 id는 0–1681 (padding id 1682는 오류)
  - 3.1 `z`: "CUDA에서는 cuBLAS·cuDNN TF32 끔"을 정의에 추가

### 7.7 결정
- D12 (항상 eval), D23 (fp32, autocast 밖): 20장
- 이 장의 새 `[결정]`:
  - 원본 코드는 고정 blob을 git에서 꺼내 씀 (작업 트리의 fork 수정본 대신): 기준 출력이 원본 코드의 출력이어야 함
  - 변환본은 일반 dict + `weights_only=True` 로드 + 내용 digest: 런타임에 임의 코드 실행이 없고 저장 형식과 무관하게 검증
  - Dropout 모듈 없음 + `train()` 덮어쓰기: D12를 구조로 보장
  - CUDA TF32 끔: D23의 fp32를 A100에서도 지킴
  - 기준 표본은 `len_seq ≥ 3` 전부(66,315), split별 4,096 단위, CPU 스레드 1: 재구현을 원본과 bitwise로 대조 가능. M12·M15 export(1,024개씩, 스레드 1)도 로컬 CPU에서는 기준과 bitwise 같았으나(7.6.5) 기계가 다르면 허용 오차로 대조
  - 오른쪽 padding 엄격 검사, `embed_items`는 padding id 거부: 이력 조립 오류를 조용히 넘기지 않음

### 7.8 완료 기준
1. (CPU) 도구가 다음 각각에서 중단: 고정 blob 불일치, pickle 해시 불일치, 허용 목록 밖 global이 든 pickle(시험용으로 만든 파일), 비어 있지 않은 출력 폴더, upstream commit 없음
2. (CPU) state 파일: 텐서 20개(7.6.2 모양), 값 129,664개, `dropped_keys` = `["s_fc.bias", "s_fc.weight"]`(7.6.2의 이름 목록. 모양 `s_fc.weight [1682, 64]`, `s_fc.bias [1682]`는 `manifest.json`의 `state.dropped`), digest `6a78697b…1dae`(재검토 때 독립 재계산으로 같은 값 확인), `weights_only=True`로 읽힘, `arch` 값이 모두 Python int·float
3. (CPU) manifest 관찰값: `training` true, `device` cuda:0, dropout 0.1 ×4, LayerNorm eps 1e-5 ×4, head 1, `state_size` 10(numpy int64), `item_num` 1682, `hidden_size` 64, global 18개
4. (CPU) 기준 파일: 66,315행(66,126 / 94 / 95), `sample_id` = `"{split}:{row}"`, `z [66315, 64]`, `item_emb [1682, 64]`. 원본 자체 점검(반복, padding 내용) 차이 0
5. (CPU) 동등성: split별 4,096 단위·스레드 1이면 `torch.equal(z, z_ref)` 참 (prototype: 세 split 모두 참). batch 8이면 최대 차이 ≤ 1e-5 (prototype 7.63e-6). `embed_items` 결과 bitwise 같음
6. (CPU) D12: 부모 모듈 `.train()` 뒤와 `encoder.train(True)` 뒤에도 encoder와 모든 하위 모듈 `training == False`. encoder 안에 `nn.Dropout` 없음
7. (CPU) 고정: parameter 전부 `requires_grad == False`, z `requires_grad == False`. z를 입력으로 한 gate 손실의 backward가 오류 없이 W_g에만 gradient를 줌. parameter를 `requires_grad_(True)`로 바꾸면 다음 forward가 `RuntimeError`
8. (CPU) D23: CPU bf16 autocast 안에서 부른 z가 float32이고 밖에서 부른 값과 bitwise 같음. parameter를 fp16으로 바꾸면 `RuntimeError`
9. (CPU) 입력 검사: float `hist_ids`, 모양 `[B, 9]`, `len_seq` 0·11, 왼쪽 padding, 실제 구간 안의 1682, 1683 이상 id, `embed_items`의 1682 → 모두 `ValueError`
10. (CPU) `encode_all`을 기준 파일의 `chunk`·`num_threads`로 split마다 부른 결과가 기준 z와 bitwise 같고, 호출 뒤 torch 스레드 수가 원래 값. `pad_histories` 결과가 기준 파일의 `hist_ids`·`len_seq`와 같음. M12 `compute_representations`(1,024개씩, 스레드 1)의 z와 기준 z의 최대 차이 ≤ 1e-5 (측정값 기록, 로컬 측정은 0)
11. (CPU) check-env 모의: digest 불일치, 원본 SHA-256 불일치, catalog `pad_id` 불일치를 각각 넣으면 실패
12. (VESSL_REQUIRED) CUDA z와 CPU 기준의 최대 차이 ≤ 1e-4 (측정값 기록. TF32가 켜지면 넘을 것으로 보고 정한 값이라, `strict_fp32` 없이 TF32를 켠 차이도 한 번 측정해 함께 기록). 같은 입력 두 번 bitwise 같음. 호출 뒤 TF32 flag 두 개가 호출 전 값. CUDA `embed_items`는 CPU와 bitwise 같음
13. (VESSL_REQUIRED) 학습 각 epoch 끝에 다시 계산한 SASRec state digest가 시작 값과 같음

---

## 8. M4 prompt와 curriculum
상태: 테스트 통과(로컬)

### 8.1 목적과 범위
- **범위:** 템플릿 파일 읽기와 검증, 표본별 템플릿 선택(D15), prompt 문자열 구성(hybrid, text-only), 학습 curriculum 추첨(D3), epoch별 `BatchPlan`(순서, drop_last, τ 번호), 평가 prompt 형식(D4)
- **범위 밖:** 제목 정규화(M1, D20), 후보 추출(M2, D10), tokenization·padding·label(M5), 행동 토큰 삽입(M9), 학습 루프(M10), 생성과 판정(M11)
- 파일 `rebuild/ilora/prompts.py`, 설정 구역 `prompt` (4.6.1, 4.6.2)

### 8.2 입력과 출력
**입력**
- `Config.prompt`(8.6.8), `Config.run.seed`, `Config.train.epochs`·`Config.train.micro_batch_size`(M10 구역, BatchPlan에만 씀)
- 템플릿 파일 `prompt/movie.txt` (저장소 루트 기준). 내용은 M0 환경 점검(4.6.5)이 CRLF→LF 정규화 해시로 확인 (sha256 `957fc1f2…ecd3` = git blob `d25b3c01` 내용, 측정)
- `ItemCatalog.titles`(M1), `Sample`(M1), 후보 `tuple[int, ...]` 길이 20(M2)

**출력**
- `TemplateSet`: 템플릿 문자열 3개
- prompt `str`와 템플릿 index `int` (3.1 "prompt", M5로)
- `BatchPlan` (3.1): epoch마다 main process에서 만든 `[(tau, sample_indices, hybrid)]`. `sample_indices`는 M1 train 표본 목록(행 순서)의 위치, 길이 = micro-batch 크기
- 모두 Python 객체 (텐서 없음). permutation만 CPU `torch.randperm` 결과를 Python int로 바꿔 씀

### 8.3 논문 근거
- 이력 순서: "i<n = [i1, i2, ..., in−1], where each ij is an item interacted with at the j-th step" `[논문 p.3 §2]` → 오래된 것부터
- hybrid 항목: "e = Concat(s, Projτ1(z))." `[논문 p.4 식 6]`, "By converting each item into a hybrid token representation (cf. Equation (6)), we can rewrite the input prompt x describing the sequence i<n" `[논문 p.4]`
- prompt 구성: "(1) Task Definition ... (2) Interaction Sequence ... (3) Candidate Set" `[LLaRA 간접 §4.2 p.5]`
- text-only 항목: "Within the input prompt, each item is represented using the textual tokens followed by a placeholder token." `[LLaRA 간접 §4.2 p.5]`, "<PH> indicates a special placeholder token, reserved for substitution by the behavioral token" `[LLaRA 간접 그림 3 caption p.5]` (§4.2 방법 그림이라 `간접`, 0.1)
- curriculum: "p(τ) = τ/T (0 ≤ τ ≤ T), (9) with the total training time accumulated to T.", I(τ) = 1 w.p. p(τ) (식 10), 목적 식 11 `[LLaRA 간접 §4.3 p.6 식 9-11]`. "LLaRA framework adopts a single-stage curriculum learning approach." `[LLaRA §5.4 p.9]`
- iLoRA는 curriculum을 언급하지 않고 식 6-8은 hybrid prompt만 기술 `[논문 p.4-5]`. 상속은 "Hereafter, we follow LLaRA [9] and fine-tune the LLM (i.e., Llama-2 [21]) on a hybrid prompt" `[논문 p.3 §1]` 경유 해석 (D3)
- 문자적 읽기(처음부터 전부 hybrid)는 변형 `all_hybrid`. LLaRA의 비교군 "the direct training method confounds the model with the hard task throughout the entire process" `[LLaRA §5.4 p.9]`와 같은 형태
- 평가 hybrid: "Ultimately, our LLM-based recommender will be fully integrated with the hybrid item representation." `[LLaRA 간접 §4.3 p.6]` (D4 표기는 `[LLaRA §4.3 p.6]`, §4 방법 문장이라 0.1 기준 `간접`), 식 7-8의 입력 x가 hybrid `[논문 p.4-5 (암시)]` (D4)
- 템플릿: "the instruction format for training and testing is randomly sampled from several prompts." `[LLaRA §5.1.2 p.7]` (D15). 템플릿 문구와 추출 단위는 두 논문 모두 없음

### 8.4 공개 코드에서 가져올 것
- 템플릿 3개 문구 `U prompt/movie.txt:1-3` (blob `d25b3c01`, "20 movie titles". `L prompt/movie.txt`는 "10 movie titles"로 후보 20과 어긋남)
- 읽기 `U data/data_interface.py:182-186`: `splitlines()` 후 줄마다 `strip()` → 줄 끝 공백이 빠져 "Answer:", "is:", "Recommendation:"으로 끝남 (측정)
- 항목 형식 `U data/data_interface.py:43-48`(hybrid), `:56-63`(text-only): 제목 + `" [HistoryEmb]"`·`" [CansEmb]"`, text-only는 둘 다 `" [PH]"`, 항목 구분 `", "`. `L data/data_interface.py:43-59`와 코드 계보 일치
- 이력 제목 순서 = 배포 파일 seq 순서(padding 제거) `U data/movielens_data.py:89-91` → `Sample.history` 순서 그대로
- curriculum 단위: collate 호출(micro-batch)마다 1회 추첨, batch 전체 같은 형식, 카운터 1부터 `U data/data_interface.py:28, :38-40, :67` (`L data/data_interface.py:28, :38-40, :62`)
- 평가는 학습 추첨과 무관하게 hybrid `U data/data_interface.py:40` (`or not self.train`), `L data/data_interface.py:40`
- 학습 순서는 epoch마다 섞고 마지막 불완전 batch는 버림 `U data/data_interface.py:138-143` (shuffle, drop_last). torch `RandomSampler`도 `torch.randperm(n, generator)`로 순서를 만듦 (`torch2.0 utils/data/sampler.py:132`)

### 8.5 버릴 것 (알려진 문제)
- **worker별 curriculum 카운터:** collater 복사본마다 `cur_step`이 1에서 시작하고 epoch마다 새로 만들어짐 `U data/data_interface.py:28, :67`. `max_steps = 5·(66,126//8)//8 = 5,165` `U data/data_interface.py:135`(`L data/data_interface.py:125`)라 hybrid 확률이 epoch마다 0에서 약 0.2까지만 오름 (감사 확인). 전체 학습의 τ/T(D3)가 아님
- **2024-08 방향 반전:** `if p > thresh_hold:` → 학습 hybrid 확률 1 − τ/T, 평가 collater는 `max_step=1`이라 항상 text-only `U@0a69c15 data/data_interface.py:22, :40` (blob `b69479a0`) → 학습은 변형 `reverse`, 평가는 `eval_prompt: text_only`(E3)로만
- **작업자 난수 의존:** 템플릿 `random.choice`(`U data/data_interface.py:32`)와 curriculum `random.random()`(`:39`)이 DataLoader 작업자의 Python `random`에 묶여 작업자 수에 따라 결과가 바뀜 (M0 4.5). 예시 출력의 `random.choice`도 전역 난수를 소비 `U data/data_interface.py:188`
- **batch 단위 템플릿:** batch 전체가 같은 템플릿 `U data/data_interface.py:32-33` → 표본 단위 key 추첨 (D15)
- **`[TargetHere]`→`[ItemEmb]` 경로:** 정답 제목을 prompt에 넣는 경로 `U data/data_interface.py:49-51`. MovieLens 템플릿에는 없고 쓰면 정답 누설 → 구현하지 않고 템플릿 검증에서 오류 (8.6.2)
- **템플릿 파일이 없으면 빈 목록:** `U data/data_interface.py:189-190`이 조용히 `[]`로 두고 첫 batch의 `random.choice([])`에서야 실패 → 시작 시 오류
- **순차 `str.replace`:** `[HistoryHere]`를 바꾼 뒤 `[CansHere]`를 바꿔 이력 제목 안의 자리 표시 문자열까지 바뀔 수 있음 `U data/data_interface.py:45, :48`. 현재 제목에는 `[`·`]`가 없어 결과는 같음 (측정) → 한 번에 조립

### 8.6 설계

#### 8.6.1 공개 API (`rebuild/ilora/prompts.py`)
```python
class PromptError(ValueError): ...

HISTORY_SLOT, CANDS_SLOT = "[HistoryHere]", "[CansHere]"
HISTORY_TAG, CANS_TAG, PH_TAG = " [HistoryEmb]", " [CansEmb]", " [PH]"
ITEM_SEP = ", "
NUM_CANDIDATES = 20                                   # 3.1 후보, D10
CURRICULA = ("linear", "all_hybrid", "reverse")       # D3와 변형

@dataclass(frozen=True)
class TemplateSet:
    templates: tuple[str, ...]                        # strip된 줄, 파일 순서
    parts: tuple[tuple[str, str, str, str, str], ...] # (앞, 첫 자리 이름, 가운데, 둘째 자리 이름, 뒤)

TEMPLATE_FILE = "prompt/movie.txt"                    # M0 run.EXPECTED_INPUTS의 키

def load_templates(path: Path) -> TemplateSet          # 8.6.2 (내용 해시 확인 후 읽음). 호출: repo_root / cfg.prompt.template_path (M11 15.6.1과 같은 꼴)
def parse_templates(data: bytes, *, source: str) -> TemplateSet          # 8.6.2 2–6단계 (해시 확인 없음, 시험용으로도 씀)
def check_catalog_titles(catalog: ItemCatalog) -> None
def choose_template(cfg: PromptConfig, seed: int, n_templates: int,
                    sample: Sample, epoch: int | None) -> int
def render_prompt(templates: TemplateSet, template_id: int, history_titles: Sequence[str],
                  cand_titles: Sequence[str], hybrid: bool) -> str
def build_prompt(sample: Sample, cands: tuple[int, ...], catalog: ItemCatalog,
                 templates: TemplateSet, template_id: int, hybrid: bool) -> str
def eval_is_hybrid(cfg: PromptConfig) -> bool

def curriculum_hybrid(mode: str, seed: int, tau: int, total: int) -> bool

class PlanEntry(NamedTuple):                          # 3.1의 (tau, sample_indices, hybrid)
    tau: int
    sample_indices: tuple[int, ...]
    hybrid: bool

@dataclass(frozen=True)
class BatchPlan:
    epoch: int                                        # 0부터
    batches_per_epoch: int                            # NB
    total: int                                        # T
    curriculum: str
    entries: tuple[PlanEntry, ...]
    def __len__(self) -> int: ...                     # len(entries) = NB
    def __iter__(self) -> Iterator[PlanEntry]: ...    # 3.1의 [(tau, sample_indices, hybrid)] 목록으로 쓰임
    def __getitem__(self, i: int | slice) -> PlanEntry | tuple[PlanEntry, ...]: ...   # entries[i] 그대로 (M10 창 자르기)
    def digest(self) -> str: ...
    def hybrid_count(self) -> int: ...

def batches_per_epoch(num_samples: int, micro_batch_size: int) -> int
def make_batch_plan(*, seed: int, epoch: int, epochs: int, num_samples: int,
                    micro_batch_size: int, curriculum: str) -> BatchPlan          # 순수 함수 (8.6.6)
def build_batch_plan(samples: Sequence[Sample], epoch: int, cfg: Config) -> BatchPlan   # M10이 부르는 형태 (14.6.4)
```
- `PromptConfig`는 M0 설정의 `prompt` 구역 dataclass, `ItemCatalog`·`Sample`은 M1 (3.1)
- `build_batch_plan(samples, epoch, cfg)` = `make_batch_plan(seed=cfg.run.seed, epoch=epoch, epochs=cfg.train.epochs, num_samples=len(samples), micro_batch_size=cfg.train.micro_batch_size, curriculum=cfg.prompt.train_curriculum)`. 그 전에 `samples`가 모두 `split == "train"`이고 `row`가 엄격히 오름차순(M1 `LoadedData.train`, full 또는 smoke)인지 확인, 아니면 `PromptError`

#### 8.6.2 템플릿 읽기와 검증 (`load_templates`: 0–1단계 후 `parse_templates`: 2–6단계)
0. 설정 `prompt.template_path`의 선택지는 `TEMPLATE_FILE` 하나 (8.6.8): M0 환경 점검이 해시를 확인하는 파일과 읽는 파일이 항상 같게 함 (M1 5.6.8의 데이터 경로 `[결정]`과 같은 이유)
1. 파일이 없으면 `PromptError` (공개 코드처럼 빈 목록으로 두지 않음)
   - 읽기 전에 `run.content_sha256(path, text=True)`가 `run.EXPECTED_INPUTS[TEMPLATE_FILE][0]`(`957fc1f2…ecd3`)와 같은지 확인, 다르면 `PromptError(경로, 기대값, 실제값)`. 환경 점검을 거치지 않는 경로(CPU 시험, `smoke`)에서도 같은 내용만 쓰게 함
2. bytes를 UTF-8 strict로 decode → `str.splitlines()` → 줄마다 `strip()`. `U data/data_interface.py:185-186`과 같은 결과이고 `\r\n`·`\n` 파일이 같은 결과
3. 줄이 하나도 없거나(빈 파일) 빈 줄이 하나라도 있으면 오류
4. 템플릿마다 `re.findall(r"\[[^\[\]]*\]", t)`가 정확히 `[HistoryHere]` 1개와 `[CansHere]` 1개
   - 그 밖의 대괄호 묶음(`[TargetHere]`, `[PH]`, `[HistoryEmb]` 등), 두 자리 표시를 뺀 뒤 남는 `[`·`]`, `<s>`·`</s>`·`<unk>` 문자열이 있으면 오류 (메시지에 줄 번호와 문자열)
5. 템플릿을 두 자리 표시 기준으로 5조각으로 나눠 `parts`에 저장 (자리 순서는 템플릿마다 그대로 기록)
6. paper_spec 결과 (측정, 정확히 이 3개):
```
0: This user has watched [HistoryHere] in the previous. Please predict the next movie this user will watch. Choose the answer from the following 20 movie titles: [CansHere]. Answer:
1: This user has watched [HistoryHere] in the previous. Given the following 20 movie titles: [CansHere], recommend one movie for this user to watch next. The movie title you recommend is:
2: The visit history of this user is: [HistoryHere]. Recommend a next movie for this user to watch from the following movie title set: [CansHere]. The recommendation should contain one movie title only. Recommendation:
```
- `check_catalog_titles`: 제목에 `[`, `]`, `<s>`, `</s>`, `<unk>`가 있으면 오류 (자리 표시나 특수 토큰을 만들거나 깨뜨림). 현재 1,682개 중 0개 (측정). 부르는 곳 `[결정]` (설계에 호출자가 없었음): M5 dataset 생성자(`PlannedTrainBatches`, `EvalBatches`)가 dataset마다 한 번 부름 `R ilora/batching.py PlannedTrainBatches`. `build_prompt`를 직접 부르는 M13 과적합 점검은 따로 부름 `R ilora/smoke.py overfit_generation_check`. 이유: 표본마다 1,682개 제목을 다시 훑지 않고 batch를 만들기 전에 실패하게 함
  - 빈 제목(item 266), 끝 공백 제목(1127, 1200, 1634), 비ASCII 제목 9개는 D20대로 그대로 허용 (측정)

#### 8.6.3 템플릿 선택 (`choose_template`, D15)
- `template_choice = random`
  - 학습: `k = int(np_rng(seed, "template", "train", sample.row, epoch).integers(0, n))`
  - 평가: `k = int(np_rng(seed, "template", sample.split, sample.row).integers(0, n))`
  - `np_rng`는 M0 `seeding.np_rng` (= `np.random.Generator(PCG64(derive_seed(...)))`). 생성기 하나에서 `integers` 1회만 뽑음
- `template_choice = fixed`: `k = fixed_template_index`, 추첨 없음
- 학습 표본(`split == "train"`)에 `epoch is None`, 평가 표본에 정수 epoch, `epoch`가 bool이거나 음수, `k`가 0–(n−1) 밖 → `PromptError`. 이 검사는 `fixed`에서도 똑같이 함 (변형 프로파일에서만 잘못된 호출이 통과하지 않게)
- epoch는 0부터 (8.7). 같은 행도 epoch마다 템플릿이 바뀔 수 있음 (epoch 0과 1이 같은 비율 0.3351, 측정)
- 측정(seed 1234): train epoch 0 템플릿 0/1/2 = 22,033 / 22,031 / 22,062 (66,126개), validation 30 / 33 / 31, test 22 / 36 / 37

#### 8.6.4 prompt 구성 (`render_prompt`, `build_prompt`)
- `tag_h, tag_c = (HISTORY_TAG, CANS_TAG)` if hybrid else `(PH_TAG, PH_TAG)`
- `H = ITEM_SEP.join(t + tag_h for t in history_titles)`, `C = ITEM_SEP.join(t + tag_c for t in cand_titles)`
- `prompt = 앞 + (첫 자리의 값) + 가운데 + (둘째 자리의 값) + 뒤`. 한 번에 조립하므로 제목 안 문자열이 다시 치환되지 않음
- 이력 제목 = `catalog.titles[i]` (`Sample.history` 순서, 오래된 것부터), 후보 제목 = M2가 준 순서
- `build_prompt`의 경계 검사 (`PromptError`):
  - `3 ≤ len(sample.history) ≤ 10` (3.1 `Sample`)
  - 후보: 길이 20, 중복 없음, `sample.target` 정확히 1개, `history`와 겹침 없음, 모든 id 0–1681. M2 `check_candidates`를 그대로 불러 `CandidateError`를 `PromptError`로 바꿈 (계약 검사 정의는 한 곳, 후보는 Python int tuple) `R ilora/prompts.py build_prompt`
- 응답(정답 제목 + `"\n"`)은 M5가 만듦 (D18)
- 문자열 수준 결과: hybrid는 `"[HistoryEmb]"` len(history)개·`"[CansEmb]"` 20개·`"[PH]"` 0개, text-only는 `"[PH]"` len(history) + 20개. 토큰 수준 검사는 M5 (9.6.5)

#### 8.6.5 curriculum 추첨 (`curriculum_hybrid`, D3)
- 기호: `N` = M1 train 표본 수, `B` = `train.micro_batch_size`, `NB = N // B` (drop_last), `E = train.epochs`, `T = E × NB`
  - paper_spec: N = 66,126, B = 8 → NB = 8,265 (6개 버림), T = 5 × 8,265 = 41,325 (측정, D3와 같음)
- τ: epoch `e`(0부터)의 `b`번째(0부터) batch는 `τ = e × NB + b + 1` (1부터, 전체 학습 통틀어 연속)
- 공통 검사: `T ≥ 1`, `1 ≤ τ ≤ T`, `mode ∈ CURRICULA`. 아니면 `PromptError`
- `linear` (D3): `u = seeding.uniform(seed, "curriculum", τ)` (PCG64 `random()` 1회, [0, 1) float64), `hybrid = u < τ / T` (Python float 나눗셈)
  - τ = T이면 τ/T = 1.0 > u라 항상 hybrid (D3 "마지막 micro-batch는 p = 1")
- `all_hybrid`: 항상 True, 추첨 없음
- `reverse`: 같은 u로 `hybrid = u < 1 − τ/T` (τ = T이면 항상 text-only)
- key가 τ라 추첨 순서·작업자·중간 재개와 무관. 같은 τ는 언제 다시 계산해도 같음
- 측정(seed 1234, paper_spec):
  - `linear` epoch별 hybrid micro-batch 수 846 / 2,550 / 4,076 / 5,766 / 7,436 (합 20,674 / 41,325). 기댓값 Σ τ/T = 826.6 / 2,479.6 / 4,132.6 / 5,785.6 / 7,438.6 (합 20,663)
  - 첫 hybrid τ = 432. τ = 1의 u = 0.260466 (text-only), τ = T의 u = 0.615392 (hybrid)
  - `reverse` 7,442 / 5,764 / 4,100 / 2,522 / 823 (합 20,651)

#### 8.6.6 BatchPlan 생성 (`make_batch_plan`, main process. M10은 감싼 함수 `build_batch_plan(samples, epoch, cfg)`를 부름, 8.6.1)
1. 검사: `micro_batch_size ≥ 1`, `num_samples ≥ micro_batch_size` (아니면 "train 표본 N개 < micro-batch B" 오류), `0 ≤ epoch < epochs`
2. `g = seeding.torch_generator(seed, "train_order", epoch)` (CPU `torch.Generator`)
3. `perm = torch.randperm(num_samples, generator=g).tolist()`: `g`에서 뽑는 유일한 난수
4. `b = 0 … NB−1`마다 `idx = tuple(perm[b·B:(b+1)·B])`, `τ = epoch·NB + b + 1`, `hybrid = curriculum_hybrid(curriculum, seed, τ, epochs·NB)`
5. 남는 `perm[NB·B:]` (paper_spec 6개)는 그 epoch에 쓰지 않음 (drop_last, `U data/data_interface.py:142`)
6. `BatchPlan(epoch, NB, T, curriculum, entries)` 반환
- `digest()` = `hashlib.sha256("\n".join(f"{e.tau}:{int(e.hybrid)}:{','.join(map(str, e.sample_indices))}" for e in self.entries).encode("ascii")).hexdigest()`. M10이 epoch 시작에 `run.log`에 기록
- `BatchPlan`은 `__len__`·`__iter__`·`__getitem__`을 `entries`에 위임: M10이 `len(plan)`, plan 순회, 창 단위 자르기를 3.1 목록처럼 씀 (14.6.4)
- `BatchPlan`은 만들 때 한 epoch 전체인지 확인하고 어기면 `PromptError` `[결정]`: entries 수 = NB, b번째 entry의 `tau = epoch·NB + b + 1`, micro-batch 크기가 모두 같음, `hybrid`는 bool, epoch 안에서 index 중복 없음, `total`은 NB의 배수이고 그 epoch을 포함. 이유: 손으로 만든 잘못된 계획이 학습까지 가지 않게 함 (M10 `check_plan`은 학습 위치와의 τ 연속을 다시 봄, 14.6.4). 그래서 `plan[i:j]`는 `PlanEntry` tuple이고 더 작은 `BatchPlan`이 아님 `R ilora/prompts.py BatchPlan`
- 측정(seed 1234, torch 2.0.0 CPU, `linear`): epoch 0 첫 batch `(57248, 62110, 9506, 49288, 37323, 59238, 25930, 18480)`, 서로 다른 index 66,120개. digest (테스트 상수):
```
epoch 0  206691cdaee8a823141bb6e99c5429643148e4f43edd4a8e0c136231a385a589
epoch 1  8ba8ff1c0cb65cadbe43d95c846328443eafd3e60fe20044acc0f6469b7e5d5e
epoch 2  b470b7450f1c8f87074b4b883932b0f7eb474211daad820fd073e84096a644dc
epoch 3  c5fb0b531aa98d4572b07587ea83c080ec61eb17ce2260fa2c9f7047ed6a2ff8
epoch 4  945a6f7b8500cef6b852b7b85e879456bf14b9ee526f4c4101e87e0af1d9ec5f
```
- 계획은 (seed, epoch, E, N, B, curriculum)의 순수 함수 → 중간 재개 때 다시 만들어 다음 τ부터 이어 감 (M10)
- 작업자는 계획을 바꾸지 않음. DataLoader의 sampler는 쓰지 않음 (M5, `batch_size=None`)

#### 8.6.7 dtype과 device
- GPU, fp16 autocast, GradScaler와 무관: 문자열, int, Python float만 다룸 (D9의 fp16 base·fp32 학습 parameter는 M9·M10)
- permutation은 CPU 생성기로 만들어 CPU 시험과 VESSL에서 같은 값 (torch 2.0.0 고정, D13)
- `u`와 `τ/T`는 float64 Python float. 텐서나 float32로 바꾸지 않음 (경계 근처 판정이 달라질 수 있음)

#### 8.6.8 설정 (`prompt` 구역)
| 키 | 타입 | paper_spec 값 | 라벨 | 설명 |
|---|---|---|---|---|
| `template_path` | str (선택지 `prompt/movie.txt` 하나) | `prompt/movie.txt` | `[코드]` | 저장소 루트 기준. `U prompt/movie.txt:1-3`, `U train_movielens.sh:10`. 다른 값은 M0 해시 검사와 어긋나므로 거부, 읽을 때 해시 재확인 (8.6.2) |
| `template_choice` | str (`random`, `fixed`) | `random` | `[LLaRA §5.1.2 p.7]` | D15. `fixed`는 `variant_fixed_template` |
| `fixed_template_index` | int 또는 null | null | `[결정]` | `fixed`면 0–(n−1) 정수, `random`이면 null이어야 함. 변형 값 0 |
| `train_curriculum` | str (`linear`, `all_hybrid`, `reverse`) | `linear` | `[LLaRA 간접 §4.3 p.6 식 9-11]` | D3. 나머지 둘은 변형 |
| `eval_prompt` | str (`hybrid`, `text_only`) | `hybrid` | `[논문 p.4 식 6-7 (암시)]` | D4. `text_only`는 E3 |
- `template_choice`와 `fixed_template_index`의 조합과 하한(`fixed`면 0 이상 정수)은 구역 dataclass의 `validate()`에서 확인 (M0 설정 규칙, `ConfigError`). 상한 n−1은 템플릿 파일을 읽어야 알 수 있어 `choose_template`이 확인하고 조합도 다시 확인함 (`PromptError`, 8.6.3) `[결정]` `R ilora/sections.py PromptConfig.validate`, `R ilora/prompts.py choose_template`
- 코드 상수(설정 아님): 자리 표시 2개, 태그 3개, 구분자 `", "` `[코드]` `U data/data_interface.py:43-63`, 후보 수 20 (3.1)
- `run.seed`, `train.epochs`, `train.micro_batch_size`는 다른 구역 소유

#### 8.6.9 오류 처리
- 모든 오류는 `PromptError`로 즉시 중단. 메시지에 파일·줄·`sample_id`·τ 등 위치를 넣음
- 대체값·재시도 없음 (템플릿 경로 선택지 위반은 M0 `ConfigError`): 템플릿 파일 해시 불일치, 템플릿 파일 없음, 빈 줄, 잘못된 자리 표시, 템플릿 index 범위 밖, `template_choice`와 `fixed_template_index` 불일치, 알 수 없는 curriculum, τ 범위 밖, N < B, epoch 범위 밖, 후보·이력 계약 위반, 제목의 금지 문자열

#### 8.6.10 다른 모듈과의 연결
- M1 → M4: `ItemCatalog.titles`, train 표본 목록 (그 위치가 `sample_indices`)
- M2 → M4: 후보 (`build_prompt`가 3.1 조건을 다시 확인)
- M4 → M5: `BatchPlan`, `TemplateSet`, `choose_template`, `build_prompt`, `eval_is_hybrid`. 작업자가 표본마다 불러 `TrainBatch.hybrid`·`tau`·`template_ids`를 채움
- M4 → M10: epoch 시작마다 main process에서 `build_batch_plan(data.train, e, cfg)` (M10 14.6.4 1단계의 호출 형태 그대로), `digest()`·`hybrid_count()` 기록, 받은 batch의 `tau`가 계획 순서인지 확인
- M11·M12: 평가 템플릿 key `("template", split, row)`와 `eval_prompt`, `EvalRecord.template_id`·`hybrid`
- **계약 추가 제안**
  - `BatchPlan`을 frozen dataclass로: `entries`(원소 `PlanEntry(tau, sample_indices, hybrid)` NamedTuple = 3.1의 3-tuple 그대로) + `epoch`, `batches_per_epoch`, `total`, `curriculum`, `digest()`, `hybrid_count()`, 목록처럼 쓰는 `__len__`·`__iter__`·`__getitem__`
  - 만드는 함수 두 개: 순수 `make_batch_plan(*, seed, epoch, epochs, num_samples, micro_batch_size, curriculum)`(시험·재현용)과 M10용 `build_batch_plan(samples, epoch, cfg)` (M10 14.6.10 제안 2와 같은 형태)
  - epoch 번호는 모든 key에서 0부터: `("train_order", epoch)`, `("candidates", "train", row, epoch)`, `("template", "train", row, epoch)` (paper_spec 0–4)
  - M10 구역에 `train.epochs`, `train.micro_batch_size` (이름은 M10 장 기준)
  - 4.6.3 `meta.json` 문서 항목에 numpy·sentencepiece 버전을 명시 (`np_rng` 난수열과 tokenization이 구현에 따름). M0 코드 `run.environment_info()`는 이미 numpy·pandas·sentencepiece 버전을 기록하므로 문서만 맞추면 됨
  - `TEMPLATE_FILE`(`prompt/movie.txt`)은 M0 `run.EXPECTED_INPUTS`의 키를 그대로 씀. `prompt.template_path`는 그 값 하나만 허용

### 8.7 결정
- D3, D4, D15를 구현. 새 `[결정]`:
  - epoch 번호 0부터: Lightning `current_epoch`, D24의 "epoch 4"(마지막 epoch) 표기와 같음
  - 템플릿 검증 규칙(자리 표시 2개 각 1회, 그 밖의 대괄호·특수 토큰 문자열 금지)과 한 번에 조립: 정답 누설 경로와 재치환을 구조적으로 막음
  - 추첨 함수는 M0 도구 그대로: curriculum `seeding.uniform`, 템플릿 `np_rng(...).integers(0, n)`, 순서 `torch_generator` + `randperm`. `reverse`는 `u < 1 − τ/T`
  - `BatchPlan.digest()` 형식: 로컬과 VESSL의 순서·추첨 일치를 값 하나로 비교
  - `build_prompt`에서 후보 계약 재확인: 잘못된 후보가 tokenization까지 가지 않게 경계에서 실패
  - 템플릿 경로는 `prompt/movie.txt` 하나만 허용하고 읽기 전에 M0 기대 해시로 확인: M0이 점검한 파일과 M4가 읽는 파일이 달라지는 경로를 막음 (M1의 데이터 경로 상수화와 같은 이유)

### 8.8 완료 기준
- M13에서 테스트로 구현. (CPU)는 로컬 Windows에서, (VESSL_REQUIRED)는 VESSL에서 확인
1. (CPU) `load_templates(repo_root / "prompt/movie.txt")`가 8.6.2의 세 문자열과 정확히 같음. 같은 내용의 CRLF 파일과 LF 파일이 같은 결과 (둘 다 정규화 해시 `957fc1f2…`)
2. (CPU) 검증 오류: `[TargetHere]` 추가, `[CansHere]` 없음, `[HistoryHere]` 2회, `[PH]` 포함, 빈 줄, 짝 없는 `[`, `<s>` 포함 → 모두 `parse_templates`에서 `PromptError`. 파일 없음과 1 byte 바꾼 사본(해시 불일치)은 `load_templates`에서 `PromptError`, 설정 `prompt.template_path = "prompt/game.txt"`(선택지 밖)는 M0 설정 적재에서 `ConfigError`
3. (CPU) `render_prompt`가 공개 코드의 `replace` 연쇄(`U data/data_interface.py:43-48, :56-63`)와 같은 문자열: validation 94 + test 95 표본 × 템플릿 3개 × 형식 2개(고정 후보). hybrid는 `"[HistoryEmb]"` len_seq개·`"[CansEmb]"` 20개, text-only는 `"[PH]"` len_seq + 20개
4. (CPU) `build_prompt`가 후보 19개·21개, 중복, 정답 없음·2개, 이력과 겹침, id 1682를 모두 거부
5. (CPU) `choose_template`: 같은 입력에 같은 결과. seed 1234에서 train epoch 0 개수 22,033 / 22,031 / 22,062, validation 30 / 33 / 31, test 22 / 36 / 37. `fixed`면 모두 0. 학습에 epoch 없음·평가에 epoch 있음 → 오류
6. (CPU) curriculum (seed 1234, T = 41,325): `linear` epoch별 hybrid 수 846 / 2,550 / 4,076 / 5,766 / 7,436, `reverse` 7,442 / 5,764 / 4,100 / 2,522 / 823, `all_hybrid` 8,265씩. τ = 1은 text-only(u = 0.260466), τ = T는 `linear` hybrid·`reverse` text-only. τ = 0, T + 1, 알 수 없는 mode → 오류
7. (CPU) `make_batch_plan` (seed 1234, N 66,126, B 8, E 5): epoch마다 8,265개, τ 연속(epoch 0은 1–8,265, epoch 4는 33,061–41,325), epoch 0 첫 batch `(57248, 62110, 9506, 49288, 37323, 59238, 25930, 18480)`, 서로 다른 index 66,120개, digest = 8.6.6 값, 두 번 만들면 같음. N < B, epoch = 5 → 오류. `build_batch_plan(data.train, e, cfg)`(paper_spec 설정)가 같은 digest를 내고, `len(plan)`·순회·`plan[i]`가 `entries`와 같음. val 표본이 섞인 목록·row 순서가 어긋난 목록 → `PromptError`
8. (CPU) 작업자 수 무관: M5 loader로 smoke 계획을 `num_workers` 0과 2로 읽어 `(tau, hybrid, template_ids, sample_ids)` 순서가 같음 (4.8의 4번)
9. (VESSL_REQUIRED) VESSL(Linux, torch 2.0.0+cu117)에서 epoch 0–4 digest와 hybrid 수가 7번·6번 값과 같음 `[SPEC §0.13]`

---

## 9. M5 tokenization과 batch 구성
상태: 테스트 통과(로컬)

### 9.1 목적과 범위
- **범위:** tokenizer 적재와 특수 토큰 추가, prompt·응답 tokenization, 학습 label(D18)과 변형, 학습 오른쪽·평가 왼쪽 padding(D22), 자리 표시 토큰 개수 검사, `TrainBatch`·`EvalBatch` 구성, DataLoader(작업자는 계획대로 tokenization만)
- **범위 밖:** prompt 문자열과 `BatchPlan`(M4), 후보(M2), 임베딩 resize·새 행 초기화·행동 토큰 삽입(M9), 손실 계산과 정규화(M9, M10), 생성과 decode(M11)
- 파일 `rebuild/ilora/batching.py`, 설정 구역 `batching` (4.6.1, 4.6.2)
- 출처 표기: `T4.28 <파일>:<라인>` = transformers 4.28.0 패키지 소스(D13 고정 버전), `torch2.0 <파일>:<라인>` = torch 2.0.0 `torch/` 아래 소스

### 9.2 입력과 출력
**입력**
- tokenizer 폴더: 실행은 `LLM_PATH`(M0), CPU 시험은 `ILORA_TEST_TOKENIZER`(M13 `tokenizer_dir` fixture)
- `Config.batching`(9.6.9), `Config.eval.batch_size`(M11 구역), `Config.prompt`(M4), `Config.run.seed`
- M4: `BatchPlan`, `TemplateSet`, `choose_template`, `build_prompt`, `eval_is_hybrid`. M1: `Sample` 목록, `ItemCatalog`. M2: 후보 함수

**출력**
- `TokenizerBundle` (tokenizer + `TokenIds`, 계약 추가 제안)
- `TrainBatch` (3.1): `input_ids`·`attention_mask`·`labels` `[B, L] int64` 오른쪽 padding, `hist_ids [B, 10]`(1682로 오른쪽 padding), `len_seq [B]`, `cand_ids [B, 20]`, `hybrid`, `tau`, `sample_ids`, `template_ids`
- `EvalBatch` (3.1): `input_ids`·`attention_mask` `[B, L] int64` 왼쪽 padding, `hist_ids`, `len_seq`, `cand_ids`, `target_ids [B]`, `cand_titles`, `target_titles`, `hybrid`, `sample_ids`, `template_ids`
- 텐서는 모두 CPU `int64`. label은 input_ids와 같은 위치에 정렬하고 shift는 손실 쪽에서 함 (`T4.28 models/llama/modeling_llama.py:705-706`)

### 9.3 논문 근거
- 학습 목표는 응답 토큰의 로그우도: "max ϕ Σ(x,y) Σ_{t=1}^{|y|} log Pϕ(yt|x, y<t), (2)" `[논문 p.3 식 2]`, 식 7-8 `[논문 p.4-5]`, L_easy·L_hard `[LLaRA 간접 §4.3 p.6 식 7-8]` → prompt와 padding은 손실에서 제외
- 응답: "the output y comprises the textual tokens corresponding to the next item" `[LLaRA 간접 §4.2 p.5]`
- 텍스트 토큰은 LLM tokenizer와 word embedding: "s = LLM-TKZϕ(txt). (4)" `[논문 p.4 식 4]`
- placeholder는 특수 토큰: "<PH> indicates a special placeholder token" `[LLaRA 간접 그림 3 caption p.5]`
- micro-batch 8 × accumulation 16 = 128은 M10 `[v1 App E p.16]` `[LLaRA §5.1.2 p.7]`
- tokenizer 종류, 특수 토큰 목록, 응답 종결 문자, padding 방향, label 경계는 두 논문 모두 없음 → `[코드]`·`[결정]` (D13, D18, D22)

### 9.4 공개 코드에서 가져올 것
- tokenizer: `LlamaTokenizer.from_pretrained(llm_path, use_fast=False)` `U model/model_interface.py:313` (`L model/model_interface.py:201`, 코드 계보 일치)
- 특수 토큰 추가 순서: `pad_token` `[PAD]` → additional `[PH]`, `[HistoryEmb]`, `[CansEmb]`, `[ItemEmb]` `U model/model_interface.py:315, :317` (`L model/model_interface.py:203, :205`). 결과 id 32000–32004, 길이 32,005 (측정)
  - `U model/model_interface.py:314`의 `pad_token = eos_token`은 다음 줄이 덮어써 효과 없음 (측정: 최종 pad id 32000)
- 응답 = 정답 제목 + `"\n"` `U data/data_interface.py:21, :69-72`
- prompt와 응답을 따로 tokenization, truncation 없음, batch 최장 길이로 padding `U data/data_interface.py:73-82`
- 평가 입력 = `[BOS]` + prompt `U data/data_interface.py:94-100`
- 행동 토큰 자리 = `[HistoryEmb]`·`[CansEmb]` id 위치, 이력·후보 순서대로 대응 `U model/model_interface.py:441-461` (M9가 사용)
- 학습 오른쪽 padding `U model/model_interface.py:316`

### 9.5 버릴 것 (알려진 문제)
- **token_type_ids label mask** `U model/model_interface.py:38-42`: 4.28.0 쌍 인코딩의 token_type_ids가 input_ids보다 1 길어 `[:, 1:]`로 맞춰지고, 남는 label은 두 번째 `[BOS]` + 응답. 4.29.0부터 길이가 같아져 shape 오류 (D13, D18) → label을 직접 만들고, 코드 동작은 `variant_released_label`
- **생성용 오른쪽 padding** `U model/model_interface.py:316`: 짧은 prompt의 첫 생성 토큰을 `[PAD]` 위치에서 예측 (D22) → 평가는 왼쪽 padding
- **zip 절단** `U model/model_interface.py:456, :460`: `[HistoryEmb]` 수가 len_seq(또는 `[CansEmb]` 수가 20)와 달라도 짧은 쪽에 맞춰 조용히 넘어감 → 개수 불일치는 오류 (9.6.5)
- **`[ItemEmb]` 입력 경로** `U data/data_interface.py:49-51`, `U model/model_interface.py:443, :448, :462-464`: MovieLens 템플릿에 없고 쓰면 정답 누설 → 입력 경로는 구현하지 않음 (vocabulary 항목은 유지, 9.6.2)
- **collate 안의 추첨:** 템플릿 `U data/data_interface.py:32`, curriculum `:39`가 작업자 난수 → M4 계획과 key 추첨
- **test drop_last** `U data/data_interface.py:158` → 평가는 drop_last 없음 (D17)
- **매 batch special id 조회** `U model/model_interface.py:441-443` (tokenizer 호출 + `.item()`) → 적재 때 한 번 정하고 검사

### 9.6 설계

#### 9.6.1 공개 API (`rebuild/ilora/batching.py`)
```python
class BatchingError(ValueError): ...

PAD_TOKEN = "[PAD]"
EXTRA_TOKENS = ("[PH]", "[HistoryEmb]", "[CansEmb]", "[ItemEmb]")   # 순서 고정
TERMINATOR = "\n"
HIST_LEN, NUM_CANDS, PAD_ITEM = 10, 20, 1682   # 값은 data.HISTORY_LEN, candidates.NUM_CANDIDATES, data.PAD_ID에서 가져옴
LABEL_MODES = ("answer_only", "released_pair")
IGNORE_INDEX = -100                             # 응답 밖 label. M9가 import
BASE_VOCAB, VOCAB_SIZE = 32000, 32005           # M9 build 1단계와 M13 smoke가 import
FINGERPRINTS: tuple[tuple[str, tuple[int, ...]], ...]   # 9.6.2 6단계 지문 2개 (VESSL 시험 9.8 #11도 씀)

@dataclass(frozen=True)
class TokenIds:
    bos: int; eos: int; unk: int; newline: int                         # 1, 2, 0, 13
    pad: int; ph: int; history_emb: int; cans_emb: int; item_emb: int  # 32000–32004
    base_vocab: int                                                    # 32000
    vocab_size: int                                                    # 32005
    new_token_ids: tuple[int, ...]                                     # (32000, …, 32004)

@dataclass(frozen=True)
class TokenizerBundle:
    tokenizer: LlamaTokenizer
    ids: TokenIds
    model_sha256: str

def load_tokenizer(tokenizer_dir: Path, expected_sha256: str) -> TokenizerBundle
def encode_text(bundle: TokenizerBundle, text: str) -> list[int]
def encode_train(bundle: TokenizerBundle, prompt: str, answer: str,
                 label_mode: str) -> tuple[list[int], list[int]]          # input_ids, labels
def encode_eval(bundle: TokenizerBundle, prompt: str) -> list[int]
def check_token_counts(bundle: TokenizerBundle, input_ids: Sequence[int], *, hybrid: bool,
                       len_seq: int, prompt_len: int, n_bos: int, max_seq_len: int,
                       where: str) -> None                 # where = sample_id, τ 등 오류 위치

@dataclass(frozen=True)
class TrainBatch:                         # 필드는 3.1 그대로 (모든 텐서 CPU int64)
    input_ids: Tensor                     # [B, L] 오른쪽 padding (pad 32000)
    attention_mask: Tensor                # [B, L] 1…1 0…0
    labels: Tensor                        # [B, L] 응답 토큰 외 -100 (released_pair는 두 번째 BOS 포함)
    hist_ids: Tensor                      # [B, 10] 오른쪽 padding 1682
    len_seq: Tensor                       # [B]
    cand_ids: Tensor                      # [B, 20] M2 순서
    hybrid: bool; tau: int
    sample_ids: list[str]; template_ids: list[int]
    def pin_memory(self) -> "TrainBatch": ...
    def to(self, device: torch.device) -> "TrainBatch": ...           # non_blocking=True
@dataclass(frozen=True)
class EvalBatch:                          # 필드는 3.1 그대로, 같은 두 메서드
    input_ids: Tensor; attention_mask: Tensor                         # [B, L] 왼쪽 padding
    hist_ids: Tensor; len_seq: Tensor; cand_ids: Tensor               # [B, 10], [B], [B, 20]
    target_ids: Tensor                                                # [B]
    cand_titles: list[list[str]]; target_titles: list[str]
    hybrid: bool; sample_ids: list[str]; template_ids: list[int]

def collate_train(samples: Sequence[Sample], cands: Sequence[tuple[int, ...]], template_ids: Sequence[int],
                  encoded: Sequence[tuple[list[int], list[int]]], bundle: TokenizerBundle, *,
                  hybrid: bool, tau: int) -> TrainBatch            # 9.6.6, (input_ids, labels) 표본별 → 오른쪽 padding
def collate_eval(samples: Sequence[Sample], cands: Sequence[tuple[int, ...]], template_ids: Sequence[int],
                 encoded: Sequence[list[int]], catalog: ItemCatalog, bundle: TokenizerBundle, *,
                 hybrid: bool) -> EvalBatch                         # 9.6.6, 왼쪽 padding. M13 과적합 점검(train 표본)도 씀

class CandidateFn(Protocol):                                         # M2 6.6.9: partial(candidates_for, main_seed=, cfg=)
    def __call__(self, sample: Sample, *, epoch: Optional[int]) -> tuple[int, ...]: ...   # epoch None = 평가 key

class PlannedTrainBatches(torch.utils.data.Dataset):                 # 원소 1개 = TrainBatch 1개
    def __init__(self, plan: BatchPlan, samples: Sequence[Sample], catalog: ItemCatalog,
                 templates: TemplateSet, candidates: CandidateFn, bundle: TokenizerBundle,
                 prompt_cfg: PromptConfig, batching_cfg: BatchingConfig, seed: int): ...
    def __len__(self) -> int: ...                                    # len(plan) = NB
class EvalBatches(torch.utils.data.Dataset):                         # 원소 1개 = EvalBatch 1개
    def __init__(self, samples: Sequence[Sample], batch_size: int, catalog: ItemCatalog,
                 templates: TemplateSet, candidates: CandidateFn, bundle: TokenizerBundle,
                 prompt_cfg: PromptConfig, batching_cfg: BatchingConfig, seed: int): ...
    def __len__(self) -> int: ...                                    # ceil(len(samples) / batch_size)

def make_loader(dataset: PlannedTrainBatches | EvalBatches, num_workers: int,
                pin_memory: bool) -> torch.utils.data.DataLoader
def make_train_loader(plan: BatchPlan, data: LoadedData, cfg: Config, epoch: int, *,
                      bundle: TokenizerBundle, templates: TemplateSet) -> Iterator[TrainBatch]   # M10 14.6.4 3단계
def make_eval_loader(data: LoadedData, split: str, cfg: Config, *,
                     bundle: TokenizerBundle, templates: TemplateSet) -> Iterator[EvalBatch]    # M11
```
- `make_train_loader`: `plan.epoch == epoch`, `len(plan.entries[0].sample_indices) == cfg.train.micro_batch_size`, 모든 index < `len(data.train)`이 아니면 `BatchingError`. 후보 함수 = `functools.partial(candidates.candidates_for, main_seed=cfg.run.seed, cfg=cfg.candidates)`, `seed = cfg.run.seed`, `pin_memory = (cfg.model.device == "cuda:0")` (M9 `model` 구역). 반환은 `iter(make_loader(PlannedTrainBatches(...), cfg.batching.num_workers, pin_memory))`
- `make_eval_loader`: `split ∈ {"val", "test"}`, 표본 = `data.samples(split)`, `batch_size = cfg.eval.batch_size`, 나머지는 같음

#### 9.6.2 tokenizer 적재 (`load_tokenizer`)
1. `transformers.__version__ == "4.28.0"` (M0와 겹치지만 tokenization 결과가 버전에 묶여 있어 여기서도 확인, D13). `sentencepiece.__version__ == "0.1.97"` `[결정]`: slow tokenizer의 실제 분할은 `sp_model.encode`(sentencepiece C++)가 하므로 같은 이유로 고정 (`U requirements.txt:116`, fork VESSL 설치도 같은 판 `F requirements.txt:117`, 로컬 시험 환경 0.1.97 확인)
2. `tokenizer.model` 내용의 sha256 = `batching.tokenizer_sha256` (다르면 기대값과 실제값을 담아 오류)
3. `tok = LlamaTokenizer.from_pretrained(str(dir), use_fast=False)` (`U model/model_interface.py:313`). 확인: `type(tok) is LlamaTokenizer`, `not tok.is_fast`, `tok.vocab_size == 32000`, bos/eos/unk = 1/2/0, `add_bos_token` True·`add_eos_token` False, `tok.sp_model_kwargs == {}` (subword sampling 금지)
4. `tok.add_special_tokens({"pad_token": "[PAD]"}) == 1`, 이어서 `tok.add_special_tokens({"additional_special_tokens": list(EXTRA_TOKENS)}) == 4` (`U model/model_interface.py:315, :317` 순서). `T4.28 tokenization_utils.py:428`이 `len(self) + i`로 차례대로 id를 줌
5. `TokenIds` 구성과 확인: pad 32000, ph 32001, history_emb 32002, cans_emb 32003, item_emb 32004, `len(tok) == 32005`, 특수 토큰 문자열마다 id 하나로 encoding, `newline` = piece `<0x0A>` = 13
6. 지문 검사 (4.28 동작 확인, 하나라도 다르면 오류. 값은 test tokenizer로 측정):
   - `encode_text("Star Wars\n") == [7828, 9149, 13]` (`▁Star ▁Wars <0x0A>`)
   - `encode_text("watched Toy Story [HistoryEmb], Heat [HistoryEmb] in the previous") == [20654, 29411, 13740, 32002, 1919, 940, 271, 32002, 297, 278, 3517]`: 특수 토큰 뒤 조각이 `▁,`(1919)로 시작. 4.31 이상에서 `legacy: false`면 특수 토큰 뒤 `▁`를 떼므로 이 지문이 달라짐 (D13의 감사 내용)
- tokenizer의 padding 기능(`padding_side`, `pad`)과 쌍 인코딩은 쓰지 않음. padding은 9.6.6에서 직접
- **`[ItemEmb]` 유지 `[결정]`:** vocabulary 32,005 (`U model/model_interface.py:317`, `L model/model_interface.py:205` 계보 일치)
  - 4.28.0 resize는 새 임베딩 행과 새 lm_head 행을 모두 N(0, 0.02)로 채움 (`T4.28 modeling_utils.py:1467-1471, :1539-1543`, `models/llama/modeling_llama.py:347-356`). Llama는 `tie_word_embeddings` 기본 false라 lm_head도 따로 늘어남 (`T4.28 modeling_utils.py:1416-1420`, `models/llama/configuration_llama.py:98`)
  - base가 고정이라 새 lm_head 행 5개는 학습 내내 고정된 무작위 logit으로 softmax 분모에 들어감. 빼면(32,004) 출력 vocabulary가 코드와 달라지므로 근거 없이 바꾸지 않음
  - 입력에는 나오지 않음: M4 템플릿 검증과 9.6.5 개수 검사(0개)가 막음. 다른 id는 `[ItemEmb]` 유무와 무관 (마지막 id, 측정)
  - 새 행의 값은 M9가 `("init", "new_tokens")`로 정함. 생성에 나오면 M11이 decode 규칙대로 처리

#### 9.6.3 학습 encoding (`encode_train`, D18)
- `encode_text(s) = tok(s, add_special_tokens=False)["input_ids"]` (= `convert_tokens_to_ids(tokenize(s))`: `_encode_plus`의 `get_input_ids`, `T4.28 tokenization_utils.py:614-617, :649`)
- `p = encode_text(prompt)`, `a = encode_text(answer + TERMINATOR)`, `answer = catalog.titles[target]`
- `answer_only` (paper_spec, D18): `input_ids = [BOS] + p + a`, `labels = [-100] × (1 + len(p)) + a`
- `released_pair` (`variant_released_label`): `input_ids = [BOS] + p + [BOS] + a`, `labels = [-100] × (1 + len(p)) + [BOS] + a`
- **SentencePiece dummy prefix와 segment tokenization (4.28 slow)**
  - `tokenize`는 문자열을 특수 토큰 경계로 자르고 (`T4.28 tokenization_utils.py:516-517`), str로 등록된 특수 토큰 왼쪽 조각의 끝 공백과 오른쪽 조각의 앞 공백을 지운 뒤 (`:532-537`), 조각마다 `sp_model.encode`를 따로 부름 (`:539-547`, `models/llama/tokenization_llama.py:119-121`)
  - Llama SentencePiece는 조각마다 앞에 dummy prefix `▁`를 붙임: 응답 `"Star Wars\n"` → `▁Star ▁Wars <0x0A>`, 특수 토큰 뒤 `", Heat"` → `▁, ▁He at` (측정). 따로 tokenization한 응답도 공백 뒤 단어처럼 시작
  - 공개 코드의 쌍 인코딩도 두 segment를 각각 `tokenize`하고 (`T4.28 tokenization_utils.py:698-701, :725-735`) `build_inputs_with_special_tokens`로 `[BOS] p [BOS] a`를 만듦 (`models/llama/tokenization_llama.py:178-187`, `add_eos_token` False). 그래서 `p`, `a`는 코드와 같고 차이는 두 번째 `[BOS]`뿐
  - 코드 label: `create_token_type_ids_from_sequences`(`models/llama/tokenization_llama.py:249-254`)는 cls/sep id가 None인 1원소 목록 길이까지 세어 0을 len(p) + 2개, 1을 len(a) + 1개 만듦 (input_ids보다 1 김). `_pad`는 input_ids 길이 차이만큼만 늘려 차이를 유지 (`T4.28 tokenization_utils_base.py:3367-3393`). `U model/model_interface.py:42`의 `[:, 1:]` 뒤 1로 남는 자리 = 두 번째 `[BOS]` + `a` (측정: validation 처음 8표본 batch에서 input_ids (8, 257), token_type_ids (8, 258). validation 12 batch 모두 열 차이 1)
  - 측정 (test tokenizer, M2 실제 후보·M4 템플릿 key 추첨, seed 1234, train은 처음 3,000 표본의 epoch 0 key): validation 94 + test 95 + train 3,000 표본 × 형식 2 = 6,378건 모두 `answer_only` = 코드에서 두 번째 `[BOS]`만 뺀 input_ids·label, `released_pair` = 코드와 같음, 개수 검사(9.6.5) 위반 0. `prompt + " " + 응답`을 한 문자열로 tokenization해도 6,378건 같았지만 정의는 segment 방식
- 의미: `answer_only`는 prompt 마지막 토큰(`:`) 다음에 바로 응답 첫 토큰(`▁Star`)을 예측하도록 학습 → 평가 입력 `[BOS] + p`에서 생성이 시작하는 자리와 같음. `released_pair`는 먼저 `[BOS]`를 예측하도록 학습 (M11은 특수 토큰을 빼고 decode)
- 측정: 1,682개 제목의 `a`는 모두 unk·특수 id 없이 13으로 끝나고 길이 2–23 (최장 23은 item 1241, 1331 두 개). 빈 제목(item 266)은 `[29871, 13]` (`▁ <0x0A>`)
- 측정 (`answer_only` 학습 입력 길이, M2 실제 후보·M4 템플릿, seed 1234): 위 6,378건 163–314 토큰 (중앙값 246), train 66,126 표본 전체의 epoch 0 key 153–334 (중앙값 245). hybrid와 text-only 길이는 같음 (태그가 모두 토큰 1개). 계산 `design/tmp/adv/adv_m5.py`, `adv_m5_len.py`

#### 9.6.4 평가 encoding (`encode_eval`)
- `input_ids = [BOS] + encode_text(prompt)`. `U data/data_interface.py:94-100`의 `add_special_tokens=True` 결과와 같음 (측정, 6,378건)
- 응답·label 없음. 정답은 `target_ids`·`target_titles`로만 전달
- 측정 (hybrid, M2 고정 후보·M4 템플릿, seed 1234): 평가 입력 길이 validation 211–290 토큰 (중앙값 241.5), test 204–279 (중앙값 240). M11의 KV cache 계산은 이 값(최장 290)을 씀

#### 9.6.5 개수 검사 (`check_token_counts`, padding 전 표본마다)
| 형식 | `[HistoryEmb]` | `[CansEmb]` | `[PH]` |
|---|---|---|---|
| hybrid | len_seq | 20 | 0 |
| text-only | 0 | 0 | len_seq + 20 |
- 항상: `[ItemEmb]`·`[PAD]`·unk·eos 0개. `[BOS]`는 `answer_only`·평가 1개(위치 0), `released_pair` 2개(위치 0과 1 + `prompt_len`, `prompt_len = len(p)`)
- 학습: label 위치 수 = len(a) (`released_pair`는 + 1), `a`의 마지막 id = `newline`
  - `check_token_counts`는 label을 받지 않음(9.6.1 서명) `[결정]`: worker가 `prompt_len` = (앞쪽 -100 label 수) − 1로 구해 넘김. `check_token_counts`가 개수, `[BOS]` 위치, 길이, 응답 부분 `input_ids[n_bos + prompt_len:]`의 끝 id = `newline`을 확인하고, 비공개 도우미 `_check_train_labels`가 label(응답 자리는 그 input_ids와 같고 나머지는 -100, 끝은 `newline`)을 확인 `R ilora/batching.py PlannedTrainBatches.__getitem__`
  - 평가는 `prompt_len = len(input_ids) − 1` (응답 부분 없음)
- 길이: `len(input_ids) ≤ batching.max_seq_len`. 넘으면 오류, 자르지 않음 (`U data/data_interface.py:79` truncation 없음)
- 하나라도 어긋나면 `BatchingError` (sample_id, τ, 형식, 실제 개수 포함)
- M9는 이 검사를 전제로 k번째 `[HistoryEmb]` = `hist_ids[b, k]`, k번째 `[CansEmb]` = `cand_ids[b, k]`로 대응 (row-major 순서)

#### 9.6.6 padding과 텐서 (D22)
- 학습 (`collate_train`): `L = max(n)`, `input_ids = seq + [pad]·(L − n)`, `attention_mask = [1]·n + [0]·(L − n)`, `labels = lab + [-100]·(L − n)`. 실제 토큰 position이 0부터 연속이라 position_ids 불필요
- 평가 (`collate_eval`): `input_ids = [pad]·(L − n) + seq`, `attention_mask = [0]·(L − n) + [1]·n`. HF `padding_side="left"` 결과와 같음 (측정). position_ids는 생성 쪽이 mask로 만듦 (`T4.28 models/llama/modeling_llama.py:736-737`, M11)
- `hist_ids[b] = history + [1682]·(10 − len)`, `len_seq[b] = len(history)`: M1 공유 함수 `data.pad_histories(histories, PAD_ID, HISTORY_LEN) -> (hist_ids [B, 10] int64, len_seq [B] int64)`(M1 5.6.13)로 만들고 M5 안에 따로 padding 코드를 두지 않음 (M3 `encode_all`, M12 export와 같은 함수). `cand_ids[b] = 후보`(M2 순서), `target_ids[b] = target`
- `collate_train`·`collate_eval`은 공개 함수(9.6.1): dataset이 표본별 encoding 뒤 부르고, M13 과적합 점검(17.6.8)이 train 표본의 평가 형식 batch를 같은 함수로 만듦 (padding 규칙을 한 곳에)
- 모두 `torch.int64` CPU 텐서. `B`는 학습 = micro-batch(8), 평가 = 그룹 크기
- `pin_memory()`는 텐서마다 `.pin_memory()`한 사본, `to(device)`는 `non_blocking=True` 사본
- M5는 float 텐서를 만들지 않아 fp16 autocast·GradScaler와 무관. fp16 base와 fp32 학습 parameter(D9)는 M9·M10

#### 9.6.7 학습 DataLoader (작업자는 tokenization만)
- `PlannedTrainBatches.__getitem__(i)`:
  1. `tau, idx, hybrid = plan.entries[i]`
  2. `idx` 순서대로 표본마다: `s = samples[j]` → `cands = candidates(s, epoch=plan.epoch)` (M2, key `("candidates", "train", row, epoch)`. `epoch`는 키워드 전용, M2 6.6.9) → `k = choose_template(prompt_cfg, seed, n, s, plan.epoch)` → `prompt = build_prompt(s, cands, catalog, templates, k, hybrid)` → `encode_train` → `check_token_counts` → label 검사 (9.6.5)
  3. `collate_train` → `TrainBatch(hybrid=hybrid, tau=tau, sample_ids=[s.sample_id …], template_ids=[k …])`
- `__len__` = `len(plan)` (SequentialSampler가 `range(len(dataset))`를 씀), `EvalBatches.__len__` = 묶음 수
- `make_loader`: `DataLoader(ds, batch_size=None, shuffle=False, sampler=None, num_workers=num_workers, collate_fn=_identity, pin_memory=pin_memory, drop_last=False)`. `_identity`는 module 수준 함수(spawn pickle). `pin_memory`는 실행 device가 CUDA일 때 True. generator는 넘기지 않음: iterator를 만들 때마다 main process의 전역 CPU 생성기에서 base seed 1개를 뽑음 (`torch2.0 utils/data/dataloader.py:605`). 학습은 M10 14.6.5가 이것을 전역 난수 사용으로 셈하고, 평가는 M11이 `fork_rng`로 감쌈 (15.6.3) `R ilora/batching.py make_loader`
  - `batch_size=None`이면 자동 batch 묶기를 끄고 원소 하나를 그대로 넘김 (`torch2.0 utils/data/dataloader.py:339-343, :365-369`, `utils/data/_utils/fetch.py:52-54`)
  - 순서는 SequentialSampler = 계획 순서. 작업자가 여럿이어도 index 순서로 돌려받음: 먼저 도착한 batch는 보관했다가 `_rcvd_idx` 순서로 냄 (`torch2.0 utils/data/dataloader.py:1299-1346`)
  - 사용자 정의 batch의 pin은 그 객체의 `pin_memory()` 호출 (`torch2.0 utils/data/_utils/pin_memory.py:74-75`)
  - 작업자 안에서 전역 난수(torch, numpy, `random`)를 쓰지 않음 → 작업자 수·seed와 결과 무관 (M0 4.6.4). 작업자는 시작할 때 `base_seed + worker_id`로 전역 난수를 다시 seed하지만(`torch2.0 utils/data/_utils/worker.py:223-229`) 어떤 결과에도 쓰이지 않음
  - Windows(spawn)는 dataset을 pickle: `LlamaTokenizer`는 `sp_model`을 빼고 보낸 뒤 다시 읽음 (`T4.28 models/llama/tokenization_llama.py:98-106`). 후보 함수도 pickle 가능한 객체여야 함 (`functools.partial` + module 수준 함수 + frozen 설정)
  - 설계 확인 (로컬 Windows, `design/tmp/adv/dlproto/`): 같은 구조(frozen dataclass batch, `batch_size=None`, `_identity`, 키워드 `partial`)로 30 batch를 `num_workers` 0과 2로 읽은 결과가 같고, pickle 왕복한 tokenizer의 encoding이 같음
  - epoch마다 새 dataset과 loader (계획이 epoch마다 다름). M10은 `batch.tau`가 1씩 늘어나는지 확인
- 측정: slow tokenizer로 8개 hybrid 표본 tokenization에 6.1 ms/micro-batch (로컬 CPU, 단일 process). VESSL 속도는 VESSL_REQUIRED

#### 9.6.8 평가 DataLoader
- 표본 = M1 split 목록 (행 순서). 그룹 = `[range(i, min(i + bs, n)) for i in range(0, n, bs)]`, `bs = eval.batch_size`, drop_last 없음 (D17: test 95, validation 94 전부)
- 표본마다 `candidates(s, epoch=None)` (`("candidates", split, row)`), `choose_template(..., epoch=None)` (`("template", split, row)`), `hybrid = eval_is_hybrid(prompt_cfg)` (D4), `encode_eval`, 개수 검사
- `cand_titles[b] = [catalog.titles[c] for c in cands]` (prompt와 같은 문자열, M11 판정용), `target_titles[b] = catalog.titles[target]`
- 같은 `make_loader` 사용. 결과는 `num_workers`와 무관

#### 9.6.9 설정 (`batching` 구역)
| 키 | 타입 | paper_spec 값 | 라벨 | 설명 |
|---|---|---|---|---|
| `label_mode` | str (`answer_only`, `released_pair`) | `answer_only` | `[결정]` D18 | `released_pair` = `variant_released_label` |
| `num_workers` | int (0 이상) | 8 | `[코드]` `U main.py:83` | 결과와 무관 (M0 4.6.4) |
| `max_seq_len` | int ≥ 1 | 1024 | `[결정]` | 표본 토큰 수 상한, 넘으면 오류 (자르지 않음). 구성한 최악 639 + 생성 64 = 703 (`released_pair`는 640 + 64 = 704) < 1024 |
| `tokenizer_sha256` | str | `9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347` | `[결정]` | `tokenizer.model` 내용 해시. 로컬 test tokenizer 값, VESSL 파일 일치는 VESSL_REQUIRED |
- 코드 상수(설정 아님): 특수 토큰 문자열과 추가 순서, 종결 `"\n"`, `HIST_LEN` 10, `NUM_CANDS` 20, `PAD_ITEM` 1682 `[코드]`, slow tokenizer (D13)
- `eval.batch_size`는 M11 구역, `prompt.*`는 M4 구역
- 최악 길이 639는 `", 제목"` 조각 토큰 수가 가장 큰 제목 30개(이력 10 + 후보 20, 그중 응답이 가장 긴 item 1241 또는 1331을 정답 후보로)와 가장 긴 템플릿 2로 만든 표본 (측정 `design/tmp/adv/adv_m5_len.py`: 템플릿 0/1/2 = 631/632/639, `released_pair`는 각 +1, hybrid·text-only 같음). 특수 토큰 경계에서 조각이 따로 tokenization되므로 순서와 무관하게 이 선택이 상한
- 실제 표본 최장: train epoch 0 key 334, 평가 290 (9.6.3, 9.6.4)

#### 9.6.10 오류 처리
- `BatchingError`로 즉시 중단: transformers·sentencepiece 버전, tokenizer 파일 해시, fast tokenizer, `sp_model_kwargs`, base vocabulary·특수 id·지문 불일치, 개수 검사, 길이 초과, `history` 길이 10 초과, 빈 batch, 알 수 없는 `label_mode`, 평가 표본 목록에 split이 섞이거나 val·test가 아닌 표본이 있음(`EvalBatches`는 평가 key만 씀. train 표본의 평가 형식 batch는 `collate_eval`로 만듦, 17.6.8), `eval.batch_size < 1`
- M4 `PromptError`와 M2 오류는 그대로 올림. 작업자 예외도 잡지 않음 (DataLoader가 main process에서 다시 던짐)
- 대체 동작 없음: 자르기, 개수 맞추기, 다른 tokenizer 사용을 하지 않음

#### 9.6.11 다른 모듈과의 연결
- M4 → M5: `BatchPlan`(작업자가 계획대로), `TemplateSet`(`load_templates`), `build_prompt`, `choose_template`, `eval_is_hybrid`
- M5 → M9: `TrainBatch`·`EvalBatch`, `TokenizerBundle`. M9는 `build_model(cfg, bundle, catalog, *, llm_path, hard_gate)`(13.6.1)로 bundle을 받아 `bundle.ids`(`TokenIds`)만 쓰고 (M9에 별도 id 클래스 없음, 13.6.13), build 1단계에서 `len(bundle.tokenizer) == ids.vocab_size == 32005`, `new_token_ids`, bos/eos/pad를 다시 대조함. resize 길이는 `len(bundle.tokenizer)` = `TokenIds.vocab_size` = 32,005
- M5 → M10: `make_train_loader(plan, data, cfg, e, bundle=, templates=)` → `TrainBatch` 반복자 (`tau` 연속 확인). `hybrid`에 따른 projector 사용은 M9
- M5 → M11: `EvalBatches` + `make_loader` (M11 15.6.2 4단계) 또는 `make_eval_loader`, `TokenIds.pad`·`eos` (생성 `pad_token_id`와 종료), `cand_titles`·`target_titles`
- M5 → M12: `sample_ids`, `template_ids`, `hybrid`. `hist_ids`·`len_seq`는 공유 `pad_histories`와 같은 규칙 (9.6.6)
- **계약 추가 제안**
  - `TokenIds`, `TokenizerBundle` (M5 → M9, M11): 9.6.1의 필드. M9·M11은 이 `TokenIds`만 씀 (M9 13.6.13에서 별도 id 클래스를 없앰, 최종 검토 반영)
  - `collate_train`, `collate_eval` 공개 (M5 → M13 과적합 점검)
  - `CandidateFn` (M2 → M5, M11): `functools.partial(candidates_for, main_seed=cfg.run.seed, cfg=cfg.candidates)`, 호출 `fn(sample, epoch=e)`(학습) / `fn(sample, epoch=None)`(평가). pickle 가능
  - `make_train_loader(plan, data, cfg, epoch, *, bundle, templates) -> Iterator[TrainBatch]`: M10 14.6.4 3단계의 `make_train_loader(plan, data, cfg, e)`에 `bundle=`·`templates=` 두 키워드가 더 필요 (tokenizer와 템플릿 없이는 batch를 만들 수 없음). M10은 학습 시작 때 `load_tokenizer`·`load_templates`를 한 번 부르고 같은 객체를 M9·M11에도 넘김
  - `TrainBatch`·`EvalBatch`에 메서드 `pin_memory()`, `to(device)` (필드는 3.1 그대로)
  - `variant_released_label`에서는 `TrainBatch.labels`에 두 번째 `[BOS]`도 들어감 (3.1 "응답 토큰 외 -100"의 변형 예외)
  - M0 환경 점검에 sentencepiece 0.1.97 확인 추가 (9.6.2. M5도 tokenizer 적재 때 확인)

### 9.7 결정
- D13(4.28.0 slow tokenizer), D18, D22, D17(평가 drop_last 없음)을 구현. 새 `[결정]`:
  - `[ItemEmb]`를 vocabulary에 유지 (32,005): 9.6.2
  - tokenizer 파일 해시 고정과 지문 검사: tokenization이 다르면 모든 비교가 무의미하므로 시작 때 실패
  - sentencepiece 0.1.97 고정: slow tokenizer의 분할을 실제로 하는 라이브러리라 transformers 판 고정과 같은 이유 (9.6.2)
  - 후보 함수는 키워드 `epoch`로 부름: M2 `candidates_for`의 키워드 전용 인자와 맞추고, 학습·평가 key 혼동을 호출 꼴에서 드러냄
  - tokenizer의 쌍 인코딩·padding 대신 직접 구성: token_type_ids 의존을 없애고 두 label 방식이 같은 코드 경로
  - `max_seq_len` 1024 상한, 자르지 않음
  - 계획된 batch의 dataset + `batch_size=None`: 작업자 수 무관 (M0 4.6.4)

### 9.8 완료 기준
- M13에서 테스트로 구현. (CPU)는 로컬 Windows에서, (VESSL_REQUIRED)는 VESSL에서 확인
- tokenizer가 필요한 시험(1–10번)은 `tokenizer_dir` fixture(`ILORA_TEST_TOKENIZER`)를 씀. 로컬은 hf-internal-testing/llama-tokenizer 사본(`tokenizer.model` sha256 `9e556afd…d347`), 없으면 "test tokenizer not available" 메시지로 skip. VESSL은 `ILORA_TEST_TOKENIZER=$LLM_PATH`로 같은 시험
1. (CPU) `load_tokenizer`: 길이 32,005, `[PAD]` 32000, `[PH]` 32001, `[HistoryEmb]` 32002, `[CansEmb]` 32003, `[ItemEmb]` 32004, bos/eos/unk/newline 1/2/0/13, 지문 2개 통과. 해시 불일치, fast tokenizer, transformers·sentencepiece 버전 불일치(모의), `sp_model_kwargs` 비어 있지 않음 → `BatchingError`. 이 `bundle`로 M9 `build_model` 1단계의 `bundle.ids` 재대조(길이 32,005, `new_token_ids`, bos/eos/pad)가 통과함
2. (CPU) 코드 대조: validation 94 + test 95 + train 처음 3,000 표본(M2 후보·M4 템플릿 key 추첨, seed 1234, train은 epoch 0 key) × hybrid·text-only에서 `answer_only` = 코드 쌍 인코딩(`tokenizer([[p, a]])`, batch 8, `padding="longest"` + `U model/model_interface.py:38-42` mask)에서 두 번째 `[BOS]`를 뺀 input_ids·label, `released_pair` = 코드와 같음. 기대 6,378 / 6,378 (설계 확인 측정 6,378 / 6,378, `design/tmp/adv/adv_m5.py`)
3. (CPU) 평가 encoding = 코드 `tokenizer(prompts, add_special_tokens=True)`의 표본별 input_ids (같은 6,378건)
4. (CPU) 개수 검사: hybrid는 `[HistoryEmb]` = len_seq·`[CansEmb]` = 20, text-only는 `[PH]` = len_seq + 20. 제목에 `" [HistoryEmb]"`를 덧붙인 catalog, 후보 19개, `[ItemEmb]`가 든 prompt → 오류
5. (CPU) 1,682개 응답: unk·특수 id 없음, 끝 id 13, 길이 2–23 (23은 item 1241, 1331), item 266 → `[29871, 13]`
6. (CPU) padding: 학습 batch의 mask `1…1 0…0`, label은 pad·prompt 자리 −100이고 응답 자리 = `a`. 평가 batch = HF `padding_side="left"` 결과. 모든 텐서 int64, `hist_ids [B, 10]`(1682 padding), `cand_ids [B, 20]`
7. (CPU) 길이: 9.6.9의 구성한 최악 표본이 템플릿 0/1/2에서 631/632/639 토큰(`released_pair` +1)이고 모두 ≤ 1024. train epoch 0 key 전체 최장 334, 평가 최장 290. 1,025 토큰 표본 → 오류
8. (CPU) 작업자 수 무관: smoke 계획 30 batch를 `num_workers` 0과 2로 읽어 모든 텐서·목록이 같음. pickle 왕복한 dataset의 `encode_text` 결과가 같음. 후보 함수를 위치 인자 epoch로 부르는 코드가 없음 (`fn(s, 0)`은 `TypeError`)
9. (CPU) 평가 loader: test 95명, `eval.batch_size` 8 → 12 batch(마지막 7), 행 순서, 누락·중복 없음
10. (CPU) `make_train_loader`: `plan.epoch ≠ epoch`, micro-batch 크기 불일치, `len(data.train)` 밖 index → `BatchingError`. 반환 반복자가 `len(plan)`개를 계획 순서로 내고 끝남
11. (VESSL_REQUIRED) `$LLM_PATH/tokenizer.model`의 sha256 = `tokenizer_sha256`, 1–10번을 실제 tokenizer로 다시 통과 `[SPEC §0.13]`
12. (VESSL_REQUIRED) Linux에서 `num_workers` 8로 읽은 epoch 0 처음 50 batch가 `num_workers` 0과 같음

---

## 10. M6 projector
상태: 테스트 통과(로컬)

### 10.1 목적과 범위
- **목적:** SASRec 아이템 표현(64차원)을 LLM 입력 임베딩 공간(4096차원)의 행동 토큰으로 바꾼다 (Proj_τ1). hybrid prompt의 `[HistoryEmb]`·`[CansEmb]` 자리에 들어간다.
- **범위:** projector 구조, 초기화(난수 순서), dtype, 입력 검사, parameter 이름
- **범위 밖:** 특수 토큰 위치 찾기와 삽입(M9), hybrid 추첨(M4, D3), optimizer·scheduler 생성과 그 설정값(lr 배수·weight decay는 M10 `train` 구역 14.6.9), 아이템 표현 계산(M3)

### 10.2 입력과 출력
**입력**
- 아이템 표현 `[n, 64] float32` (3.1, M3 `embed_items`). n = batch 안 `[HistoryEmb]`·`[CansEmb]` 토큰 수 (행마다 `len_seq` + 20)
- `in_dim` = `SASRecEncoder.hidden_size` (state 파일에서 읽은 64), `out_dim` = LLaMA `config.hidden_size` (Llama-2-7B 4096, M9가 넘김)
- main seed (M0 `run` 구역): `derive_seed(seed, "init", "projector")`

**출력**
- 행동 토큰 `[n, 4096]` (3.1). CUDA fp16 autocast 안에서는 float16, autocast 밖(CPU smoke)에서는 float32
- 학습 parameter 4개 (float32, 17,047,552개): M10의 projector group

### 10.3 논문 근거
- "the ID embedding is transferred into a behavior token representation through a trainable projector Proj(·) parameterized by τ1" `[논문 p.4 §2]`, e = Concat(s, Projτ1(z)) `[논문 p.4 식 6]` → 출력 차원 = LLM 토큰 임베딩 차원 `(암시)`. 아이템 하나에 행동 토큰 하나
- 그림 2의 "Adapter"(학습됨, 불꽃)가 SR-EMB 뒤에 있음 `[논문 p.4 그림 2 (암시)]`
- 구조: "we project the ID-based item representation e_s^i into the LLM space with a trainable projector Proj (i.e., two-layer perceptions)" `[LLaRA 간접 §4.1 p.5 식 5]`
- hybrid 표본에서만 학습: L_easy(text-only, 식 7)에는 Θp가 없고 L_hard(식 8)에만 있으며, 목적식은 Θ, Θp 최소화 `[LLaRA 간접 §4.3 p.6 식 7, 8, 11]`
- lr: "the projector parameters are updated using the same learning rate curve integrated with iLoRA during training" `[논문 App E p.17]`, "using the same learning rate curve with LoRA" `[v1 App E p.16]`
- weight decay: "we apply a weight decay factor of 1e-5 to prevent overfitting" `[v1 App E p.16]` (적용 범위 미기재 → D19)
- 논문에 없는 것: 은닉 폭, 활성 함수, 초기화, dropout → `[코드]`, `[결정]`

### 10.4 공개 코드에서 가져올 것
- **구조:** `Linear(64, 4096) → GELU → Linear(4096, 4096)`, bias 있음, dropout 없음 `U model/mlp_projector.py:6-10` (`L` 같은 blob `07f66b36`, 코드 계보 일치). GELU는 `nn.GELU()` 기본값(erf, `approximate="none"`)
- **폭:** 입력 64 = `rec_size` `U main.py:107`, 출력·은닉 폭 = LLaMA `hidden_size` `U model/model_interface.py:393`
- **초기화:** 명시 초기화 없음 → `nn.Linear` 기본 (`U model/mlp_projector.py`)
- **parameter 이름과 dtype:** `mlp_proj.0.*`, `mlp_proj.2.*`. 공개 LLaRA checkpoint(HF `joyliao7777/LLaRA` movielens.ckpt 앞부분)에서 `projector.mlp_proj.0.weight`가 `FloatStorage`(fp32)
- **optimizer:** projector group lr = 기본 lr, weight decay 적용 `U model/model_interface.py:228` (`L :154`), `weight_decay` 기본 1e-5 `U main.py:102`

### 10.5 버릴 것 (알려진 문제)
- **requires_grad 토글:** batch 형식에 따라 projector의 `requires_grad`를 껐다 켬 `U model/model_interface.py:137-142` (`L :65-70`). 저장 필터(`U :296-307`)가 `requires_grad == False`인 parameter를 빼서, 마지막 batch가 text-only였던 checkpoint에는 projector가 없음 (`F` 측정: epoch 0·2·4 checkpoint에 projector key 0개, 1·3에 4개) → 토글하지 않고, text-only batch에서는 projector를 부르지 않음
- **버리는 계산:** 사용자 표현에 projector를 적용하고 결과를 버림 `U model/model_interface.py:428`. 이력 padding 칸과 정답 아이템(`:446-448`의 `batch["item_id"]`)에도 projector를 계산하지만 영화 템플릿에는 `[ItemEmb]`가 없음 (`U prompt/movie.txt:1-3`) → prompt에 들어가는 아이템만 계산
- **동적 생성:** `importlib` + `inspect.getargspec`으로 클래스를 찾아 만듦 `U model/model_interface.py:384-404` (`getargspec`은 Python 3.11에서 제거됨) → 직접 생성

### 10.6 설계

#### 10.6.1 공개 API (`ilora/projector.py`)
```python
class Projector(nn.Module):
    in_dim: int
    out_dim: int
    def __init__(self, in_dim: int, out_dim: int) -> None          # 직접 부르지 않고 build_projector 사용
    def forward(self, item_repr: torch.Tensor) -> torch.Tensor      # [n, in_dim] float32 -> [n, out_dim]

def build_projector(in_dim: int, out_dim: int, main_seed: int) -> Projector
def projector_param_count(in_dim: int, out_dim: int) -> int        # in·out + out + out·out + out
```
- 내부: `self.mlp_proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))`
- `in_dim`, `out_dim`은 설정 항목이 아니라 M3 state와 LLaMA config에서 받음 (숨은 기본값 없음). paper_spec에서 4096이 아니면 M9의 LLaMA config 확인(13.6.2 2단계)이 생성 시 중단. 64는 `sasrec.state_digest`로 고정한 state 파일의 값이라 다른 state는 M3 로드(7.6.6)에서 멈추고, M9가 64를 따로 확인하지는 않음
- projector는 설정을 읽지 않음 (weight decay·lr 배수는 optimizer 설정이라 M10이 `train` 구역에서 읽음)

#### 10.6.2 초기화와 난수 순서
- `build_projector`는 M0 `seeding.torch_seed_scope(main_seed, "init", "projector")` 안에서 `Projector`를 만든다 (4.6.4의 `("init", 이름)` key, 새 key 없음)
  - `torch_seed_scope`: `torch.random.fork_rng(devices=[])` 안에서 CPU 기본 generator만 `derive_seed(main_seed, "init", "projector")`로 seed하고, 끝나면 이전 CPU 상태로 복원. CUDA generator는 건드리지 않음 (`rebuild/ilora/seeding.py`)
  - 두 Linear를 `device="cpu", dtype=torch.float32`로 명시해 생성 (기본 장치 설정과 무관하게 CPU generator 사용). torch 2.0의 `nn.init` 함수는 generator 인자가 없어(확인) 이 방식을 씀
- 추첨 순서 (`nn.Linear.reset_parameters`, 같은 CPU generator, 행 우선):
  1. `mlp_proj.0.weight [4096, 64]`: kaiming_uniform(a=√5) = U(−1/8, 1/8), 262,144개
  2. `mlp_proj.0.bias [4096]`: U(−1/8, 1/8)
  3. `mlp_proj.2.weight [4096, 4096]`: U(−1/64, 1/64), 16,777,216개
  4. `mlp_proj.2.bias [4096]`: U(−1/64, 1/64)
- 이후 M9가 모델 전체를 장치로 옮김. CPU에서 만들므로 CPU smoke와 GPU 실행의 초기값이 같음
- 다른 모듈 초기화와 독립: gate를 만들지 않는 모드(cluster_hard, none)에서도 projector 초기값이 같고, 전역 torch 난수(`("torch_global",)`)를 쓰지도 바꾸지도 않음 (`fork_rng` 복원 확인, `design/tmp/proto_gate_projector.py`)

#### 10.6.3 forward 단계
1. 입력 검사 (`ValueError`, 값 포함): float32, 2차원 이상, 마지막 차원 = `in_dim`, 원소 1개 이상
2. 상태 검사 (`RuntimeError`): parameter 4개가 모두 float32 (D9)
3. `return self.mlp_proj(item_repr)`. dtype 변환은 하지 않음 (autocast가 정함)
- M9가 fp16 autocast 안에서 부름: Linear 두 개는 fp16으로 계산, GELU는 autocast 목록에 없어 입력 dtype(fp16)을 따름, 출력 fp16. CPU smoke(autocast 없음)는 fp32
- 출력 유한값 검사는 M9가 삽입 직전에 다른 검사와 묶어 한 번에 함

#### 10.6.4 학습에서의 동작
- hybrid batch에서만 호출 (D3): M9가 `TrainBatch.hybrid`가 참일 때만 부름. text-only batch의 그래프에는 projector가 없어 gradient가 생기지 않음 `[LLaRA 간접 §4.3 p.6 식 7]`
- `requires_grad`는 항상 True. 토글하지 않음
- 한 optimizer step의 micro-batch가 모두 text-only면 projector gradient가 None → Adam이 이 parameter를 건너뜀 (moment·weight decay 갱신 없음, 확인). M10이 `zero_grad(set_to_none=True)`(torch 2.0.0 기본값, 확인)를 써야 성립: 0 텐서로 지우면 weight decay와 moment 때문에 값이 움직임 (확인, `design/tmp/proto_gate_projector.py`)
- 섞인 step: hybrid micro-batch의 gradient만 쌓이고, 손실은 D24대로 micro-batch마다 `grad_accum`(16)으로 나눈 값 (epoch 끝 9개 창도 16)
- lr: LoRA와 같은 곡선 ×1 (warmup 시작·최대·최솟값 모두 같음) `[논문 App E p.17]`. weight decay 1e-5, Adam의 coupled L2 (D19). 두 값 모두 M10 `train` 구역의 projector 그룹 항목 (14.6.9)
- 평가(D4, hybrid): 모든 평가 batch에서 `no_grad` + autocast 안에서 호출

#### 10.6.5 dtype, 장치, 메모리
- parameter float32 (D9): 17,047,552개 = 65.0 MiB. gradient와 Adam 상태 2개를 더해 약 260 MiB
- GradScaler: scale된 gradient를 step 전에 unscale. projector gradient의 inf/NaN도 step 건너뛰기 판정에 들어감
- 관찰: SASRec 아이템 160개의 임베딩이 거의 0(norm < 1e-30, 7.6.2)이라 이 아이템들의 행동 토큰은 거의 같은 벡터 `W2·GELU(b1) + b2`가 된다. 고정 SASRec의 성질이므로 바꾸지 않음

#### 10.6.6 오류 처리
- 생성: `in_dim`·`out_dim`이 양의 정수가 아니면 `ValueError`
- forward: 10.6.3의 검사. 예외를 잡아 0이나 다른 값으로 바꾸지 않음
- checkpoint 로드(M10 `load_model_state`·`load_checkpoint` → M9 `load_trainable_state(strict=True)`, 13.6.8): projector key 4개 중 하나라도 없으면 중단 (공개 코드처럼 조용히 빠진 채 평가하지 않음)

#### 10.6.7 설정 (`projector` 구역)
| 키 | 타입 | paper_spec 값 | 라벨 | 설명 |
|---|---|---|---|---|
| (없음) | | | | 프로파일에는 빈 구역 `projector: {}` (3.1 구역 목록 유지) |
- 구조·폭·활성 함수·초기화는 변형이 없어 설정 항목을 두지 않음 (`[LLaRA 간접]`, `[코드]`)
- projector 그룹의 weight decay 1e-5(`[v1 App E p.16]`, 적용 범위 `[코드]` `U model/model_interface.py:228`, D19)와 lr 배수 1.0(`[논문 App E p.17]`)은 optimizer 설정이라 M10 `train` 구역(14.6.9)에 한 번만 둠. 같은 값을 이 구역에 또 두지 않음 (두 곳에 두면 어긋날 수 있음)

#### 10.6.8 다른 모듈과의 연결
- M3: `embed_items(ids)` → 아이템 표현 `[n, 64] float32`
- M9: hybrid batch에서 `[HistoryEmb]`(행마다 이력 순서로 `len_seq`개)와 `[CansEmb]`(후보 순서로 20개) 자리의 아이템 id만 모아 projector를 한 번 부름 → 행동 토큰 `[n, 4096]`을 `inputs_embeds` dtype으로 맞춰 자리마다 넣음. 토큰 수 ≠ n이면 오류. text-only batch에서는 부르지 않음
- M10: M9 `param_groups()["projector"]`로 projector 그룹을 만들고 lr 배수·weight decay는 M10 `train` 구역 값(1.0, 1e-5), `zero_grad(set_to_none=True)`(M10 14.6.3과 일치), checkpoint에 항상 저장 (저장 필터 없음)
- M11: 평가 batch마다 호출 (D4)
- **계약 추가 제안:**
  - 초기화 난수는 M0에 이미 있는 `seeding.torch_seed_scope(main_seed, "init", "projector")`를 씀 (M7 `("init", "gate")`, M8 `("init", "lora")`, M9 `("init", "new_tokens")`와 같은 함수). 새 M0 함수는 필요 없음
  - `projector` 구역은 키가 없는 빈 구역으로 둠 (lr·weight decay는 M10 `train` 구역 한 곳)

### 10.7 결정
- D3 (hybrid batch만 projector 사용), D4 (평가 hybrid), D9 (fp32 parameter), D19 (lr ×1, weight decay 1e-5): 20장
- 이 장의 새 `[결정]`:
  - CPU·fp32에서 seed를 정해 만든 뒤 옮김: 장치·기본 장치 설정과 무관하게 같은 초기값
  - parameter 이름을 원본과 같게(`mlp_proj.0`, `mlp_proj.2`): 공개 checkpoint와 key를 대조할 수 있음
  - 폭·출력 차원은 설정 항목이 아니라 M3 state와 LLaMA config에서 받음: 숨은 기본값 없음
  - 유한값 검사는 M9의 삽입 지점에서 한 번
  - optimizer 값(weight decay, lr 배수)은 projector 구역에 두지 않고 M10 `train` 구역 한 곳: 같은 값이 두 구역에 있으면 어긋나도 알 수 없음

### 10.8 완료 기준
1. (CPU) `build_projector(64, 4096, main_seed=1234)`: parameter 17,047,552개 (`projector_param_count`와 같음, prototype 확인). key·모양 `mlp_proj.0.weight [4096, 64]`, `mlp_proj.0.bias [4096]`, `mlp_proj.2.weight [4096, 4096]`, `mlp_proj.2.bias [4096]`, 모두 float32·`requires_grad` True
2. (CPU) 같은 seed면 bitwise 같은 초기값, 다른 seed면 다름. 생성 전후 전역 CPU 난수 상태가 같음. gate를 함께 만들든 아니든 projector 값이 같음
3. (CPU) 초기값 범위: `mlp_proj.0.*`는 |w| ≤ 1/8, `mlp_proj.2.*`는 |w| ≤ 1/64
4. (CPU) forward: `[n, 64]` → `[n, 4096]` float32. 입력 fp16·int64·마지막 차원 63·빈 텐서 → `ValueError`. parameter를 fp16으로 바꾸면 `RuntimeError`
5. (CPU, tiny LLaMA) text-only micro-batch: projector forward hook 호출 0회, backward 뒤 projector grad None. hybrid micro-batch: grad 존재
6. (CPU, tiny LLaMA) 모든 micro-batch가 text-only인 optimizer step 뒤 projector 값과 그 Adam state(`step`, moment)가 step 전과 bitwise 같음 (학습 첫 step이면 Adam state가 비어 있음)
7. (CPU) M10 optimizer: projector group weight decay 1e-5, 모든 step에서 lr이 LoRA group lr과 같음
8. (CPU) checkpoint 저장·로드 뒤 같은 입력에 bitwise 같은 출력. text-only batch 직후 저장한 checkpoint에도 projector key 4개
9. (VESSL_REQUIRED) fp16 autocast 안 출력 dtype float16, 유한값. 학습 로그에 GradScaler의 step 건너뛰기 횟수 기록

---

## 11. M7 gate와 routing
상태: 테스트 통과(로컬)

### 11.1 목적과 범위
- **목적:** 표본마다 expert 가중치 `ω [B, K]`를 한 곳(`Router`)에서 만들고 검사해 MoE-LoRA 층(M8)에 넘긴다. M9는 세 모드 모두 이 `Router` 하나만 부르고, 그 호출 지점이 gate 값의 유일한 수집 지점이다. 평가 때는 같은 지점에서 ω와 z를 기록한다 `[SPEC §4]`.
- **모드 (`router.routing_mode`, SPEC CLI `--routing_mode`):**
  - `dynamic` (기본, E2): ω = Softmax(W_g z) (식 11, D1)
  - `cluster_hard` (E4): M15 `ClusterHardGate`가 sample_id로 미리 계산한 군집을 찾아 mapping을 적용한 one-hot ω. gate를 만들지도 부르지도 않고 optimizer에도 없음 `[SPEC §9, §11]`
  - `none` (E1, K = 1): ω = 1, gate 없음 (LLaRA와 같은 단일 LoRA)
- **범위 밖:** ω를 층에 전달하는 `GateContext`(M8, M9), KMeans fit·predict, 배정 산출물 로드·검사, cluster→expert mapping 파싱과 one-hot 생성(M15), 기록 파일 쓰기와 gate 분석(M12), optimizer 생성과 그 설정값(M10)

### 11.2 입력과 출력
**입력**
- `z [B, 64] float32` (3.1, M3), `sample_ids: list[str]` (`TrainBatch`·`EvalBatch`, z와 같은 행 순서)
- K: M8 `lora.num_experts` (D7: 4, E1: 1, M9가 인자로 넘김). main seed (M0 `run` 구역)
- cluster_hard만: M15 `ClusterHardGate` 객체 (M10·M11이 M15 `load_cluster_artifacts` 뒤 만들어 M9 `build_model`에 넘김, 19.6.7. M12 export는 LLM 없이 `Router`에 직접 넘김, 16.6.5)

**출력**
- `ω [B, K] float32` (3.1): 행 합 1, 0 이상, NaN/Inf 없음, z와 같은 장치
- `GateRecord` (평가만): `split`, `sample_ids`, `omega [N, K]`, `z [N, 64]` (CPU float32), 기록 중 `gate_calls` 증가분 → M11이 split 끝에 받아 batch마다 받은 `GenerationOutput.omega`·`z`와 대조 (15.6.3)
- 학습 parameter: dynamic만 `W_g [K, 64]` (K = 4면 256개)
- `gate_calls: int` (gate forward 호출 수), `provenance() -> dict` (checkpoint 기록)

### 11.3 논문 근거
- 식 11: "we get the contribution scores via a linear transformation with a softmax function: ω = Softmax(Projτ2(z)), (11) where Projτ2(z) = Wgz with the trainable transformation matrix Wg ∈ R^{K×d}" `[논문 p.6 §3.2.2 식 11]` → bias·은닉층·정규화 없음 (D1)
- 입력: z = SR-EMBθ(i<n) `[논문 p.5 식 10]`, "By using the sequence representation as the guidance signal" `[논문 p.6]`
- expert 쌍마다 가중치 하나: ΔW(i<n) = Σ_k ω_k B_k A_k `[논문 p.6 §3.2.3 식 12]`, "we use a gating network to obtain attention scores across the arrays of up- and down-projection experts" `[논문 p.5 §3.2 (2)]`
- gate 하나를 모든 층이 공유, sequence마다 ω 하나: 그림 2의 Gating Network 1개, 그림 3c의 sequence당 4차원 벡터 1개 `[논문 p.4 그림 2, p.7 그림 3c (암시)]`
- K = 4: "Consequently, we adapt 4 experts as the default setting." `[논문 p.9 §4.3.2]`
- dense soft routing이고 손실은 식 8의 NLL뿐 (보조·load-balancing 손실 없음) `[논문 p.5 식 8, p.6 식 11-12]`
- hard routing: "we do not extensively investigate the effects of using hard routing for recommendations with a large number of experts" `[논문 p.9 §6]` → cluster_hard는 논문 변형이 아니라 연구 확장 `[SPEC §9]`
- 학습 설정: gate의 lr·초기화는 논문에 없음. projector만 "the same learning rate curve integrated with iLoRA" `[논문 App E p.17]` → gate ×1 `(암시)` (D19). weight decay 1e-5 `[v1 App E p.16]` (범위 미기재 → D19)
- 추가 parameter: "with less than a 1% relative increase in trainable parameters" `[논문 p.1 Abstract]` → gate 256개와 맞음

### 11.4 공개 코드에서 가져올 것
- 공유 router 1개가 모든 MoE-LoRA 층에 같은 gate를 줌: `U train_movielens.sh:3`(`--router share`), `U model/model_interface.py:30-31, :47-58`, `U model/peft/tuners/moelora.py:759, :780`
- 같은 ω를 학습 forward와 generate 전 단계에 씀 `U model/model_interface.py:47-58, :72-88`
- 식 11과 같은 모양의 gate: `Dense` = `nn.Linear(64, num_moe, bias=False)` + Softmax `U model/peft/tuners/gating.py:5-16` (만들어지지만 쓰이지 않음, `moelora.py:779` 주석) → 초기화는 `nn.Linear` 기본(kaiming_uniform a=√5)
- router group의 weight decay `U model/model_interface.py:230` (값 `U main.py:102`의 1e-5)

### 11.5 버릴 것 (알려진 문제)
- **CNN router:** 실제 router는 BatchNorm1d 19개가 든 1D CNN + bias 있는 `Linear(64, 4)` (90,772 parameter, 출력 `[B, 1, 4]`, U 소스로 생성해 확인) `U model/router/nlpr.py:39-84`. 식 11과 다르고(D1), 학습 중 BatchNorm이 batch 통계를 써서 한 표본의 ω가 같은 batch의 다른 표본에 따라 달라짐. BatchNorm buffer는 저장 필터(`U model/model_interface.py:300-305`)에 빠져 다른 프로세스 평가는 초기 통계로 계산됨
- **router lr ×0.3:** warmup 시작·최대·최솟값 모두 ×0.3 `U model/model_interface.py:230, :253-267`. CNN router용 값 → `variant_released_gate_lr`로만 (D19)
- **층마다 쓰이지 않는 Dense gate:** 224개 × 256 = 57,344 parameter가 trainable로 집계되지만 optimizer에서 빠짐 (`U model/model_interface.py:232`의 `"gating" not in n`) → 만들지 않음
- **`--router unshare` 기본값:** router를 만들지 않는데 `:93`, `:230`이 `self.router`를 써서 실행 불가 (4.5)
- **gate 오류 숨김:** MoE-LoRA forward가 모든 예외를 잡아 0 텐서를 반환 `U model/peft/tuners/moelora.py:797-799` → 잘못된 ω가 조용히 층 출력을 0으로 만듦. 재구현은 ω를 넘기기 전에 검사하고 예외를 잡지 않음
- **암묵적 모양 `[B, 1, K]`:** `out.unsqueeze(1)` `U model/router/nlpr.py:80`과 broadcasting `moelora.py:780` → `[B, K]`로 명시
- **kwargs 전달:** patched transformers의 `gate_weights` kwargs와 PEFT 모델 안 list 저장 `moelora.py:195-204` → `GateContext` (M8, M9)

### 11.6 설계

#### 11.6.1 공개 API (`ilora/router.py`)
```python
ROW_SUM_ATOL = 1e-5            # 행 합 허용 오차 (11.7)
ROUTING_MODES = ("dynamic", "cluster_hard", "none")

class RouterError(ValueError): ...

class HardGate(Protocol):                      # M15 ClusterHardGate가 구조적으로 만족 (M7은 cluster.py를 import하지 않음)
    lookups: int                               # 누적 조회 표본 수
    def __call__(self, sample_ids: Sequence[str], device: torch.device) -> torch.Tensor: ...   # [B, K] float32 one-hot

@dataclass(frozen=True)
class GateRecord:                              # M7 -> M11 (계약 추가 제안)
    split: str
    routing_mode: str
    num_experts: int
    sample_ids: tuple[str, ...]
    omega: torch.Tensor                        # [N, K] float32 CPU
    z: torch.Tensor                            # [N, d] float32 CPU
    gate_calls: int                            # 기록 동안 늘어난 gate 호출 수 (cluster_hard·none은 0)

class LinearGate(nn.Module):                   # 식 11
    def __init__(self, in_dim: int, num_experts: int) -> None    # self.proj = nn.Linear(in_dim, K, bias=False)
    def forward(self, z: torch.Tensor) -> torch.Tensor           # [B, K] float32

class Router(nn.Module):                       # M9 OmegaSource의 유일한 구현 (세 모드 공통)
    routing_mode: str
    num_experts: int
    z_dim: int
    gate: LinearGate | None
    hard_gate: HardGate | None                 # nn.Module이 아님 (state_dict에 들어가지 않음)
    gate_calls: int
    def __init__(self, rcfg: RouterConfig, *, num_experts: int, z_dim: int, main_seed: int,
                 hard_gate: HardGate | None) -> None
    def forward(self, z: torch.Tensor, sample_ids: Sequence[str]) -> torch.Tensor    # ω [B, K]
    def trainable_parameters(self) -> list[nn.Parameter]                             # dynamic만 [W_g]
    def provenance(self) -> dict
    def start_recording(self, split: str) -> None
    def stop_recording(self, expected_sample_ids: Sequence[str]) -> GateRecord
    def discard_recording(self) -> None
```
- `rcfg`는 `router` 구역(`routing_mode` 하나). K는 `num_experts` 인자로(`lora.num_experts`), `z_dim`은 `SASRecEncoder.hidden_size`(64)로 받음. Router는 `router` 구역 밖의 설정을 직접 읽지 않음 (M9가 연결)
- M12 export(16.6.5)의 `gate_fn(z, sample_ids)`도 세 모드 모두 이 `Router`를 씀. M15 `ClusterHardGate`는 인자가 `(sample_ids, device)`라 `gate_fn` 자리에 직접 넣지 않음

#### 11.6.2 생성과 모드별 검사
- 모든 모드: `rcfg.routing_mode` ∈ `ROUTING_MODES`. 어기면 `RouterError`
- `dynamic`: K ≥ 2, `hard_gate is None` (dynamic 경로는 군집 산출물을 만들지도 참조하지도 않음 `[SPEC §22]`)
  - M0 `seeding.torch_seed_scope(main_seed, "init", "gate")`(4.6.4 key, CPU generator만 seed·복원) 안에서 `LinearGate`를 `device="cpu", dtype=torch.float32`로 생성. 장치 이동은 M9
  - 난수: `proj.weight [K, 64]` 하나, kaiming_uniform(a=√5) = U(−1/8, 1/8), 행 우선 K × 64개 (`derive_seed(seed, "init", "gate")`). 다른 모듈 초기화·전역 torch 난수와 독립
- `cluster_hard`: K ≥ 2, `self.gate = None` (만들지 않음), `hard_gate is not None`. 산출물 해시·fit split·SASRec sha256·표본 id 범위 검사는 `hard_gate`를 만들기 전에 M15 `load_cluster_artifacts`가 끝냄. 학습 장치의 z 대조(`verify_live_encoder`)는 `build_model` 뒤에 함 (3.1 준비 순서, 19.6.7)
- `none`: K == 1, `hard_gate is None`, gate 없음
- K == 1인데 `dynamic`·`cluster_hard`, K ≥ 2인데 `none` → `RouterError` `[결정]`
- 위 검사 실패는 모두 `RouterError` (어긴 값 포함). 난수 없음: cluster_hard, none

#### 11.6.3 `forward(z, sample_ids)` 단계
1. 입력 검사 (`RouterError`): z가 float32 2차원 `[B, z_dim]` (`[B, 1, 64]` 같은 모양은 squeeze하지 않고 거부), B ≥ 1, `len(sample_ids) == B`, z 유한값
2. 모드별 ω:
   - `dynamic`: `strict_fp32(z.device)`(M3) 안에서, 즉 autocast 끔·CUDA TF32 끔 상태에서 `logits = F.linear(z, W_g)` → `softmax(logits, dim=−1)`. `gate_calls += 1`
   - `cluster_hard`: `ω = hard_gate(sample_ids, z.device)` (M15: dict 조회 → mapping → one-hot, 모르는 id는 M15의 `KeyError`가 그대로 올라감). z는 검사만 하고 계산에 쓰지 않음. gate 호출 없음
   - `none`: `ω = torch.ones(B, 1, dtype=torch.float32, device=z.device)`
3. 출력 검사 (조건을 묶어 동기화 한 번): 모양 `[B, K]`, float32, z와 같은 장치, 유한값, 0 이상, `|행 합 − 1| ≤ ROW_SUM_ATOL`. cluster_hard는 값이 0 또는 1이고 행마다 1이 정확히 하나 (M15 검사와 겹치지만 M7 출력 계약으로 다시 확인). 어기면 `RouterError` (첫 위반 행 번호와 sample_id 포함)
4. 기록 중이면 (11.6.5) `sample_ids`, `ω`, `z`를 `detach().float().cpu()`로 추가
5. ω 반환. dynamic이면 autograd가 W_g까지 이어짐. Router는 기록 중이 아닐 때 표본별 값을 쌓아 두지 않고, 어느 때도 파일을 쓰지 않음 (11.6.5)

#### 11.6.4 cluster→expert mapping
- 형식·파싱·거부 규칙은 M15 `parse_expert_mapping`(19.6.7)과 설정 `cluster.expert_mapping` 한 곳 `[SPEC §9]`. M7에는 파서를 두지 않음 (같은 규칙이 두 곳에 있으면 어긋남)
- M7이 보장하는 것은 출력 계약뿐: one-hot, 행 합 정확히 1, expert index 0..K−1 (11.6.3 3단계) `[SPEC §11]`
- 주 실험은 항등 `"0:0,1:1,2:2,3:3"`, 군집 번호에는 의미가 없음 `[SPEC §9]`

#### 11.6.5 gate 값 수집 경로와 기록기 `[SPEC §4]`
- 수집 지점은 M9가 batch마다 한 번 부르는 `Router.forward` 하나. MoE-LoRA 층(M8) 안에서는 수집하지 않음
- 학습: M9 `train_forward`가 ω를 `TrainForward.omega`(detach)로 돌려주고 M10이 step 기록(ω 평균·entropy)을 만듦 (14.6.8). Router는 학습 ω를 쌓지 않음 (평가 ω가 학습 로그에 섞이지 않음)
- 평가 (M11 15.6.3): M9 `generate`가 ω·z를 `GenerationOutput`(CPU float32)으로 돌려주고 M11이 `EvalRecord.gate_0..K−1`을 채움 → M12가 split 끝에 한 번 씀 (16.6.4). 같은 평가 동안 기록기가 같은 호출의 ω·z를 모아 `GateRecord`로 넘기고, M11이 batch별 출력과 bitwise 대조 (한 지점 수집의 확인)
  - `start_recording(split)` → 평가 loop → `stop_recording(기대 sample_id 순서)` → `GateRecord`. 예외로 평가가 끊기면 M11이 `discard_recording()` 뒤 예외를 다시 올림. `stop_recording`은 id 대조가 실패해도(`RouterError`) 기록을 끝내고, `discard_recording`은 기록 중이 아니면 아무것도 하지 않음 `[결정]`: 실패한 stop 뒤에 M11이 부르는 discard도 안전함 `R ilora/router.py Router.stop_recording`. `start_recording`의 `split`은 비어 있지 않고 `:`가 없는 str. `router.py`는 파일을 쓰지 않음
  - 기록 중인 forward는 `torch.is_grad_enabled()`가 거짓이어야 함 (학습 step에서 기록하면 `RouterError`)
  - 검사 (`RouterError`): 기록 중 `start_recording` 다시 호출, `"{split}:"`로 시작하지 않는 id, 중복 id, `stop_recording`의 기대 id와 개수·순서 불일치, 기록 중이 아닌데 `stop_recording`
  - 메모리: train 전체를 기록해도 66,126 × (64 + 4) × 4 byte ≈ 18 MB
- export: M12가 checkpoint의 router state와 M3 z로 LLM 없이 `Router`를 다시 불러 train·val·test 전체 ω를 만들고, `EvalRecord`의 ω와 대조 (16.6.5). ω는 z와 W_g만의 함수라 가능. 학습 전 gate도 seed로 같은 초기값을 다시 만들 수 있음
- cluster_hard의 표본 id 범위(세 split 전부 배정됨)는 `hard_gate`를 만들기 전 M15 `load_cluster_artifacts(expected_ids=...)`가 확인 (19.6.7). M7에는 따로 범위 검사 함수를 두지 않음 (`hard_gate`로 미리 조회하면 `lookups`가 늘어 E4 조회 수 기록이 틀어짐)

#### 11.6.6 dtype, 장치, gradient
- W_g float32 (D9). CPU에서 seed로 만든 뒤 M9가 장치로 옮김
- ω는 autocast 밖, TF32 끔 상태의 fp32. M8이 autocast 안 expert 출력에 ω를 곱함 (dtype 변환은 M8이 정함)
- gradient 경로: GradScaler로 scale된 loss → M8의 ω 곱 → softmax → W_g. z는 `no_grad` 결과라 SASRec으로 흐르지 않음. GradScaler가 inf로 step을 건너뛰면 W_g도 갱신되지 않음
- 첫 optimizer step: LoRA B가 0이라 ω에 대한 gradient가 0 (M8 12.6.3, 12.8 항목 6) → 이 step의 W_g 변화는 weight decay 항에서만 옴. 설계 변경 없음
- 측정 (`design/tmp/proto_gate_projector.py`, 임의 torch seed 0–4, derive_seed 값 아님):
  - z norm은 21.14–34.10 (66,315개)이라 기본 초기화의 logit 표준편차가 1.67–2.18 (seed별 train 전체 logit의 표준편차)
  - 학습 전 ω도 균등하지 않음: 행 최대값 평균 0.67–0.78, 정규화 entropy 평균 0.43–0.59, 한 expert의 argmax 비율 최대 0.87
  - D1의 초기화는 그대로 둔다. M12는 학습 전 ω(같은 seed로 재생성)를 collapse 진단의 비교 기준으로 함께 보고하는 것이 좋음
- fp32 softmax(K = 4)의 행 합 오차는 측정 최대 2.4e-7 (logit 최대 약 46)

#### 11.6.7 optimizer와 checkpoint
- `trainable_parameters()`: dynamic이면 `[W_g]`, cluster_hard·none이면 빈 목록. M9 `param_groups()["gate"]`가 이 목록을 이름과 함께 냄 (cluster_hard·none이면 `gate` 키 없음)
- M10 group: lr 배수(paper_spec 1.0, `variant_released_gate_lr` 0.3)와 weight decay(1e-5)는 M10 `train` 구역의 gate 항목(14.6.9). 빈 목록이면 group을 만들지 않음 (cluster_hard의 gate는 optimizer에 없음 `[SPEC §11]`)
- state_dict: dynamic = `{"gate.proj.weight"}`, cluster_hard·none = 빈 dict (`hard_gate`는 module이 아님). 모델 안 이름은 `omega_source.gate.proj.weight` (M9 13.6.7–13.6.8)
- `provenance()`: 키 `routing_mode`, `num_experts`, `z_dim`, `gate_shape` (JSON 값. `gate_shape`는 dynamic이면 `[K, d]`, 나머지는 None). M10이 checkpoint(`train_state.pt`의 `router_provenance`)에 넣고 다시 불러올 때 같지 않으면 중단 (14.6.6). cluster_hard·none은 state_dict가 둘 다 비어 strict 로드로는 구별되지 않으므로, 그 둘과 K가 다른 cluster_hard는 이 비교로만 가려짐. cluster_hard의 산출물 출처(manifest sha256, mapping, `fit_split`, SASRec 변환본 sha256)는 M15 `ClusterArtifacts.manifest`와 `cluster` 구역에서 M10이 기록 (19.6.7)
- `[SPEC §11]` 확인 항목 대응: forward 호출 = `gate_calls` (cluster_hard는 0), gradient·갱신 = optimizer에 gate parameter 없음, TrainableParams 차이 = 256 (dynamic − cluster_hard, K = 4, d = 64), 조회 수 = M15 `hard_gate.lookups`

#### 11.6.8 설정 (`router` 구역)
| 키 | 타입 | paper_spec 값 | 라벨 | 설명 |
|---|---|---|---|---|
| `routing_mode` | str (`dynamic`·`cluster_hard`·`none`) | `dynamic` | `[논문 p.6 식 11]` (D1), `[SPEC §9]` | ω 공급 방식. E1 프로파일 `none`, E4 `cluster_hard` |
- K는 이 구역에 두지 않음 (M8 `lora.num_experts` 하나만 씀)
- gate lr 배수·weight decay는 M10 `train` 구역 (14.6.9), 군집 산출물 경로·mapping은 M15 `cluster` 구역(`cluster.artifact_dir`, `cluster.expert_mapping`, 19.6.10). 같은 값을 이 구역에 두지 않음
- 구역 사이 조합 검사 (M0 `Config.validate`): `routing_mode == cluster_hard` ⇔ `cluster.artifact_dir`가 null이 아님, `routing_mode == none` ⇔ `lora.num_experts == 1`
- `ROW_SUM_ATOL`은 실험 의미가 없는 검사 상수라 코드 상수

#### 11.6.9 다른 모듈과의 연결
- M3: z 입력, `strict_fp32`, `hidden_size`
- M9: `build_model`이 `Router`를 만들어 `ILoRAModel.omega_source`로 둠. 매 forward·generate 직전에 `ω = omega_source(z, batch.sample_ids)` → `GateContext`에 넣음 → 끝나면 비움. generate는 모든 단계에 같은 ω (D1)
- M8: `GateContext`에서 ω `[B, K] float32`를 읽음. `x.shape[0] == ω.shape[0]` 확인은 M8
- M10: optimizer group(11.6.7), `gate_calls`와 `hard_gate.lookups`를 step 기록에, checkpoint에 `provenance()`. cluster_hard면 학습 loop 전에 3.1의 준비 순서대로 M15 `load_cluster_artifacts`(표본 id 전체 대조 포함) → `parse_expert_mapping` → `ClusterHardGate` → `build_model(hard_gate=...)` → `verify_live_encoder`(모델의 SASRec, 모델 장치) (14.6.4 0단계 2–3, 19.6.7)
- M11: `GenerationOutput.omega`로 `EvalRecord.gate_0..K−1`을 채우고, 평가 시작·끝에 기록 시작·종료 (`GateRecord`와 batch 출력 대조, `gate_calls` 증가분 확인). cluster_hard 표본 범위는 M15 로더가 이미 확인함 (M7에는 미리 조회하는 범위 검사 함수가 없음, 11.6.5)
- M12: export의 `gate_fn` = `Router` (dynamic은 checkpoint의 `omega_source.gate.proj.weight`를 접두어를 떼고 strict 로드)
- M15: `ClusterHardGate`(`HardGate` 프로토콜)를 만들어 넘김. dynamic·none 실행에서는 M15 로더를 부르지 않음 `[SPEC §22]`
- **계약 추가 제안:**
  - 3.1 `ω`: 만드는 곳을 "M7 `Router` (cluster_hard 값은 M15 `ClusterHardGate`)"로 정리 (수집 지점 하나 `[SPEC §4]`)
  - 3.2 학습 3단계의 `ω = gate(z)`를 `ω = router(z, sample_ids)`로 (cluster_hard 조회에 sample_id 필요)
  - `HardGate` 프로토콜 (`__call__(sample_ids, device) -> [B, K] float32`, `lookups`): M15 `ClusterHardGate`가 만족. 타입은 `router.py`에 두어 의존 방향 M15 → M7 유지
  - 3.1에 `GateRecord` 행 추가: 만드는 곳 M7, 쓰는 곳 M11 (평가 대조)
  - 설정 키 이름 `router.routing_mode` (SPEC `--routing_mode`, M15 19.6.7 표기와 같게)
  - 초기화 난수는 M0에 이미 있는 `seeding.torch_seed_scope` (새 함수 없음)

### 11.7 결정
- D1 (선형 gate, 공유), D7 (K = 4), D9 (fp32), D19 (lr ×1, weight decay 1e-5): 20장
- 이 장의 새 `[결정]`:
  - `none` ⇔ K = 1: E1에 gate parameter가 없어야 LLaRA와 같은 구조이고 TrainableParams 비교가 맞음
  - 세 모드를 `Router` 하나로: M9·M12가 모드별 분기 없이 같은 호출 `router(z, sample_ids)`를 씀, gate 값 수집 지점이 하나 `[SPEC §4]`
  - ω는 autocast 밖·TF32 끔 상태의 fp32: gate 값이 장치 설정에 흔들리지 않음
  - 행 합 허용 오차 1e-5: fp32 softmax 측정 최대 오차 2.4e-7의 약 40배 여유 (M8 `GateContext`, M12 `records.gate_rowsum_atol`도 같은 값)
  - 기록은 `no_grad`인 평가에서만, 파일 쓰기 없음: 학습 step의 기록·I/O를 구조로 막음 `[SPEC §4]`. 학습 ω 누적(이전 초안의 `pop_running_mean`)은 두지 않음 — 평가 ω가 다음 학습 로그에 섞였고(M11 15.6.14 계약 추가 제안 4 지적), M10이 `TrainForward.omega`로 직접 계산함
  - mapping 파싱·산출물 검사는 M15 한 곳, M7은 출력 계약만 확인

### 11.8 완료 기준
1. (CPU) `LinearGate(64, 4)`: weight `[4, 64]` float32, bias 없음, parameter 256개, |w| ≤ 1/8. 같은 seed면 bitwise 같고, 생성 전후 전역 CPU 난수 상태가 같음
2. (CPU) dynamic ω: `[B, 4]` float32, 행 합 오차 ≤ 1e-5, 0 이상, fp64로 따로 계산한 softmax와 차이 ≤ 1e-6
3. (CPU) CPU bf16 autocast 안에서 불러도 ω가 float32이고 밖에서 부른 값과 bitwise 같음
4. (CPU) batch 독립: 한 표본의 ω가 혼자일 때와 다른 표본과 함께일 때 차이 ≤ 1e-6 (공개 코드 BatchNorm과 대비)
5. (CPU) gradient: ω로 만든 손실의 backward 뒤 W_g grad 존재, SASRec parameter grad 없음
6. (CPU) cluster_hard ω (가짜 `HardGate`와 실제 M15 `ClusterHardGate` 둘 다): 값 0/1, 행 합 정확히 1, expert = mapping[군집]. 가짜 `HardGate`가 행 합 2인 행, 값 0.5, `[B, 1, K]`, float64, 다른 장치를 내면 각각 `RouterError`. 모르는 sample_id는 M15의 `KeyError`가 그대로 올라옴 `[SPEC §21]`
7. (CPU) cluster_hard: `router.gate is None`, `trainable_parameters() == []`, state_dict 빈 dict, 합성 1 epoch 뒤 `gate_calls == 0`이고 `hard_gate.lookups` = 조회 표본 수, M10 optimizer에 gate group 없음, TrainableParams 차이 256
8. (CPU) dynamic이나 none에 `hard_gate`를 주면, cluster_hard에 `hard_gate`를 주지 않으면 각각 `RouterError`. dynamic 실행에서 M15 로더 호출 0회 `[SPEC §22]`
9. (CPU) none: K = 1에서 ω = `ones [B, 1]`, parameter 없음. K = 4 + none, K = 1 + dynamic, K = 1 + cluster_hard → 오류
10. (CPU) 실행 중 검사: z에 NaN, z 모양 `[B, 1, 64]`, `len(sample_ids)` ≠ B, z fp16 → 각각 `RouterError` (squeeze 없음). 출력 검사가 NaN ω를 잡는지 W_g에 NaN을 넣어 확인
11. (CPU) 기록기: 기본 꺼짐. grad가 켜진 상태의 기록, 이중 start, 다른 split의 id, 중복 id, `stop_recording` 기대 목록 불일치, 기록 없이 stop → 각각 오류. 정상 경로는 N행·호출 순서·CPU float32, `discard_recording` 뒤 다시 start 가능. `router.py`에 파일 쓰기 코드 없음 (정적 검사: `open(`, `np.save`, `torch.save`, `.to_csv` 없음)
12. (CPU, tiny LLaMA) 평가 중 기록한 `GateRecord`의 ω·z가 batch마다 받은 `GenerationOutput`과 bitwise 같고, checkpoint의 router state와 M3 z로 LLM 없이 다시 계산한 `Router` 출력과도 같음 (같은 장치·같은 batch 구성에서 bitwise)
13. (CPU) checkpoint: dynamic state_dict key는 `gate.proj.weight` 하나, cluster_hard·none은 빈 dict. `provenance()`의 mode·K·z_dim이 다른 설정으로 다시 불러오면 오류 (gate가 있는 쪽은 strict 로드가 실패하고, 빈 state_dict끼리는 provenance가 달라 M10 비교가 멈춤, 11.6.7)
14. (VESSL_REQUIRED) paper_spec 학습의 `gate_calls` = 학습 micro-batch 수 41,325 + 평가 batch 수. 평가 `EvalRecord`의 gate가 validation 94행, test 95행, 행 합 오차 ≤ 1e-5, NaN/Inf 없음
15. (VESSL_REQUIRED) cluster_hard 학습 전체에서 `gate_calls == 0`이고 기록된 ω가 모두 one-hot

---

## 12. M8 MoE-LoRA 층
상태: 테스트 통과(로컬)

### 12.1 목적과 범위
- **목적:** LLaMA의 대상 Linear 층 하나를 감싸 h' = Wh + Σ_k ω_k · s · B_k A_k h를 계산하는 층(`MoELoRALinear`)과, 모델 하나가 공유하는 ω 보관 객체(`GateContext`)를 만든다. s는 D8의 scaling
- **범위:** 층 구조와 초기화, expert 분할(r* = r/K), scaling 규칙(D8), dropout 위치, autocast 아래 dtype, ω 확인, 대상 모듈 교체 함수(`attach_moe_lora`), LoRA parameter 수 확인
- **범위 밖:** ω를 만드는 일(M7 gate, M15 cluster-hard), LLaMA 로드와 ω를 넣는 시점(M9), optimizer·weight decay(M10, D19)

### 12.2 입력과 출력
**입력**
- 감쌀 층: `nn.Linear(d_in, d_out, bias=False)`, 가중치 fp16 고정 (D9). Llama-2-7B에서 q/k/v/o_proj 4096→4096, gate/up_proj 4096→11008, down_proj 11008→4096
- `x`: `[B, L, d_in]`(학습, 생성 첫 단계) 또는 `[B, 1, d_in]`(KV cache 생성 단계). GPU에서는 autocast 아래 fp16
- `ω`: `GateContext`에서 읽음. `[B, K] float32`, 행 합 1 (3.1)
- 설정 `lora` 구역 (12.6.7)

**출력**
- `[B, L, d_out]` (또는 `[B, 1, d_out]`), dtype = base 출력 dtype (GPU: fp16)
- 학습 parameter: 층마다 `lora_A.weight [r, d_in]`, `lora_B.weight [d_out, r]`, fp32. Llama-2-7B 224층 합계 448 텐서, 19,988,480개 (12.8 항목 1)

### 12.3 논문 근거
- 표준 LoRA: ΔW = BA, B ∈ R^{d_out×r}(up), A ∈ R^{r×d_in}(down), h' = Wh + (α/r)BAh, "where W remains frozen" `[논문 p.3 §2 식 3]`
- expert 분할: B = [B_1, …, B_K], A = [A_1, …, A_K], B_k ∈ R^{d_out×r*}, A_k ∈ R^{r*×d_in}, "r∗ = r/K is the partial rank determined by the total rank r of LoRA" `[논문 p.5 §3.2.1 식 9]` → r은 K로 나누어져야 함
  - §3.2.1 첫 문장만 B를 "down", A를 "up"이라 부름. 모양과 §2, 식 9 뒤 문장(B_k up, A_k down)으로 뜻은 분명함 `[논문 p.3, p.5]`
- 혼합: ΔW(i<n) = Σ_k ω_k B_k A_k `[논문 p.6 §3.2.3 식 12]`
- expert는 모든 sequence가 공유하고 ω만 sequence마다 다름: "Instead of establishing various LoRA modules, we implement the mixture-of-experts (MoE) concept" `[논문 p.5 §3.2]`
- parameter 수 보존: "such a segmentation scheme preserves the overall number of parameters equivalent to that of the standard LoRA" `[논문 p.5 §3.2.1]`, 같은 뜻 `[논문 p.6 §3.2.3]`
- 적용 위치: "We apply such instance-wise LoRA updates on the transformer layers of the base LLM" `[논문 p.6 §3.2.3]`. 모듈 목록은 없음, gate_proj는 App C의 basic LoRA 분석에서 암시 `[논문 App C p.17]`
- K = 4: "we adapt 4 experts as the default setting" `[논문 p.9 §4.3.2]`
- "less than a 1% relative increase in trainable parameters" `[논문 p.1 Abstract]`: 새 parameter는 gate W_g 256개뿐 → 이 구현에서 0.000691% (계산, 12.8 항목 1)
- 논문에 없음: r, α, dropout 값, 대상 모듈 목록, 초기화, 결합 순전파의 scaling(식 3의 r이 전체 rank인지 r*인지) → D7, D8

### 12.4 공개 코드에서 가져올 것
- 층 구조: `lora_A = nn.Linear(d_in, r, bias=False)`, `lora_B = nn.Linear(r, d_out, bias=False)` `U model/peft/tuners/moelora.py:616-617`
- expert 분할: A 출력 r개 채널을 (K, r*)로 나눔 `:774-776`, B 가중치를 (K, r*, d_out)으로 나눔 `:750-753` → expert k = A의 행 [k·r*, (k+1)·r*), B의 열 [k·r*, (k+1)·r*)
- dropout: 입력 x에 한 번 적용, 모든 expert가 같은 mask `:774-776`. p = 0이면 Identity `:608-611`. 모듈마다 별도 dropout (q, k, v가 같은 입력이어도 mask가 다름, pip peft와 같음)
- 초기화: A `kaiming_uniform_(a=√5)`, B 0 `:670-674` (pip peft와 같음). fan_in = d_in이라 A ~ U(−1/√d_in, 1/√d_in)
- scaling α/(r//K) = 32/2 = 16 `:618` → D8 주 설정
- r 8, α 32, dropout 0.1 `U main.py:128-130`, 대상 7개 `U model/model_interface.py:364-371`, `L model/model_interface.py:215-220` → D7. LLaRA 공개 checkpoint 2개도 같은 값, LoRA 텐서 448개 (코드 계보 일치)
- 모든 층이 같은 ω: 모델 공유 리스트 `gate_weights`를 층마다 넘기고 `:191-192, :275-277` forward·generate 때 채워 `:200-202, :212-215` 224층이 읽음 `:759` → `GateContext`로 바꿈
- 메모리 절약 계산 Σ_k ω_k B_k(A_k x) = B((A x) ⊙ ω 확장) `F model/peft/tuners/moelora.py:775-781` (fork commit `8049a98`). 원본 식과의 일치는 `F scripts/test_moelora_equivalence.py`로 확인됨: fp64 출력 ≤1.3e-16, gradient ≤5.9e-16, fp32 ≤2.9e-7 (VESSL 측정)
- 2024-08 판과 같음: `moelora.py`는 `0a69c15`와 `28e4737` 사이 변화 없음 (git diff)

### 12.5 버릴 것 (알려진 문제)
- **[B, L, K, d_out] 텐서:** `calculate_B`의 einsum `:750-753, :777`과 `(B_out * Gate).sum(-2)` `:783`이 expert별 출력을 모든 대상 층에서 backward용으로 저장 → A100 80GB OOM
  - fork 측정(CPU에서 GPU dtype 흐름 재현): LoRA 가지 저장량 803 KiB/token/layer(원본) vs 139 KiB/token/layer(재작성), B = 8·L = 296·32층에서 약 25 GiB 차이. 보정 모형 최대 86.2 GiB(원본) vs 60.7 GiB(재작성, L = 347), 5 epoch 41,325 batch 중 원본은 2,221개가 한도 초과 `F experiments/IMPLEMENTATION_NOTES.md §11g`
- **예외 삼킴:** 모든 예외를 출력만 하고 `torch.zeros_like(x)` 반환 `:797-799`. q/k/v/o_proj에서는 base 출력까지 0이 되어 조용히 틀린 forward, MLP 층에서는 모양이 달라 뒤에서 엉뚱한 오류
- **층마다 쓰지 않는 gate:** Dense gate(Linear 64→4) 224개 생성 `:604-606`, 호출은 주석 `:779`, 이름의 'gating' 때문에 학습 parameter로 남음 `:556` → 만들지 않음 (gate는 M7 하나, D1)
- **오래된 ω 재사용 위험:** 공유 리스트를 다음 호출 때에만 비움 `:195-217` → `GateContext`가 끝날 때 finally에서 비우고, 비어 있으면 층이 오류
- **merge/unmerge** `:718-748`: 고정 BA를 W에 더하므로 sequence마다 다른 ω와 맞지 않음 → 구현하지 않음
- **dtype 처리:** x를 LoRA dtype으로 바꾸고 `result +=`로 더한 뒤 되돌림 `:772, :783, :794` → autocast에 맡기고 출력 dtype을 base 출력 dtype으로 명시
- **쓰지 않는 계산** `:786-788` (gate 벡터 분해) → 버림
- **`--lora_r type=float`**: 명령행으로 넘기면 8.0이 되어 `:616`에서 층 생성 실패 (4.5) → 설정은 int만 받음

### 12.6 설계

#### 12.6.1 공개 API (`rebuild/ilora/moe_lora.py`)
```python
class GateContextError(RuntimeError): ...

@dataclass
class GateUse:                       # use() 블록 하나의 기록
    batch_size: int
    calls: int = 0                   # 블록 안에서 ω를 읽은 층 호출 수

class GateContext:
    def __init__(self, num_experts: int) -> None
    num_experts: int                                     # 읽기 전용
    def use(self, omega: Tensor) -> ContextManager[GateUse]   # 확인 → 보관 → yield → finally 비움
    def get(self) -> Tensor                              # 층이 호출, 비었으면 GateContextError
    is_set: bool                                         # property

class MoELoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, r: int, num_experts: int, alpha: int,
                 dropout: float, scaling_rule: str, gate_ctx: GateContext, name: str) -> None
    base: nn.Linear            # 원래 층, requires_grad False
    lora_A: nn.Linear          # weight [r, d_in] fp32, bias 없음
    lora_B: nn.Linear          # weight [d_out, r] fp32, bias 없음
    lora_dropout: nn.Module    # nn.Dropout(p), p = 0이면 nn.Identity
    r: int; num_experts: int; r_star: int; scaling: float; name: str
    def reset_lora_parameters(self) -> None    # kaiming_uniform_(A, a=√5), zeros_(B); 전역 CPU 난수
    def expert_slices(self, k: int) -> tuple[Tensor, Tensor]   # (A_k [r*, d_in], B_k [d_out, r*]) view
    def forward(self, x: Tensor) -> Tensor

def lora_scaling(alpha: int, r: int, num_experts: int, rule: str) -> float
CANONICAL_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
def attach_moe_lora(llm: LlamaForCausalLM, cfg: LoraConfig, gate_ctx: GateContext,
                    main_seed: int) -> list[str]           # 교체한 모듈 이름 (llm 기준 "model.layers.{i}.{self_attn|mlp}.{대상}"), 초기화 순서대로
def lora_named_parameters(llm: nn.Module) -> list[tuple[str, nn.Parameter]]   # 같은 순서, 모듈마다 lora_A 다음 lora_B. 이름은 llm 기준이라 M9가 "llm."을 붙여 정렬
```

#### 12.6.2 순전파 (`MoELoRALinear.forward`)
1. `omega = gate_ctx.get()` (비었으면 `GateContextError`), 현재 `GateUse.calls` += 1
2. 확인: `x.dim() >= 2`, `x.shape[0] == omega.shape[0]`, `omega.device == x.device`. 어기면 `GateContextError` (층 이름, x 모양, ω 모양)
3. dtype 확인: `x.dtype != lora_A.weight.dtype`인데 x 장치의 autocast가 꺼져 있으면 `RuntimeError` ("fp16 입력은 M0 precision 도구의 autocast 안에서만"). `[결정]` `GateContextError`는 ω와 GateContext의 오류라 여기에 쓰지 않음 `R ilora/moe_lora.py MoELoRALinear.forward`. 판정은 torch 2.0.0 함수로: CUDA는 `torch.is_autocast_enabled()`, CPU는 `torch.is_autocast_cpu_enabled()`
4. `base_out = base(x)`: 고정 fp16, autocast 아래 fp16
5. `a = lora_A(lora_dropout(x))` `[B, …, r]`: dropout mask 하나를 모든 expert가 공유, autocast가 A의 fp32 master를 fp16으로 바꿔 계산
6. `w = omega.to(lora dtype)[:, :, None].expand(-1, K, r*).reshape(B, r)` `[B, r]` (열 j의 값 = ω_{j // r*}, `repeat_interleave(r*, dim=1)`과 같은 값), `[B, 1, …, 1, r]`로 view. `[결정]` `repeat_interleave`는 쓰지 않음: torch 2.0.0에서 `output_size` 없이 부르면 반복 수를 host로 읽어 층 호출마다 `aten::item` 2회, 즉 224층 × forward·생성 단계마다 CUDA 동기화가 생김 (12.7). expand·reshape는 host 읽기 0회이고, 출력과 x·ω·A·B gradient가 이전 계산과 bitwise 같음 (K = 1·2·4·8, dropout 0·0.1, fp32와 CPU bf16 autocast, `review/fix/probes/item_equal.py`) `R ilora/moe_lora.py MoELoRALinear.forward`
7. `g = a.to(lora dtype) * w`: gate 곱은 fp32 (ω gradient의 토큰 방향 합을 fp32로)
8. `lora_out = lora_B(g)` `[B, …, d_out]`: autocast 아래 fp16
9. 반환 `base_out + (lora_out * scaling).to(base_out.dtype)`
- 수식: lora_B(g) = Σ_j B[:, j] a_j ω_{j//r*} = Σ_k ω_k B_k A_k x → 출력 = Wx + Σ_k ω_k · s · B_k A_k x (식 12 + D8)
- 중간 텐서는 `[B, L, d_in]`(dropout 출력, mask)과 `[B, L, r]`(a, g)뿐. `[B, L, K, d_out]` 없음
- K = 1: ω = 1이라 표준 LoRA `Wx + (α/r) B A x` (두 scaling 규칙 모두 α/r = 4, LLaRA와 같음)
- 균등 ω(1/K): expert_rank 규칙에서 Σ_k (1/K)(α/r*) B_k A_k = (α/r) BA, 같은 A, B의 표준 LoRA와 정확히 같음 (20.1의 근거, 12.8 항목 4)
- one-hot ω(cluster-hard): 활성 expert 하나의 rank-2 갱신 s·B_k A_k만 남음. 활성 용량 차이는 `[SPEC §13]` 한계로 M14가 기록

#### 12.6.3 초기화와 난수 순서
- `lora_A`, `lora_B`는 `torch.nn.utils.skip_init(nn.Linear, …, dtype=float32, device="cpu")`로 만들어 생성 단계의 난수 소비를 없앰 (확인: 난수 상태 변화 없음). 생성자는 두 weight를 0으로 채움 `[결정]`: `skip_init`만으로는 `reset_lora_parameters` 전까지 초기화되지 않은 메모리가 남는데, 생성자만 쓰는 경우가 있음 (12.8 항목 1). 0 채우기는 난수를 쓰지 않음 `R ilora/moe_lora.py MoELoRALinear.__init__`
- `attach_moe_lora`가 `seeding.torch_seed_scope(main_seed, "init", "lora")` 안에서 (4.6.4, CPU 난수 fork 후 복원) 아래 순서로 `reset_lora_parameters()` 호출:
  - 층 i = 0 … 31, 층 안에서 `CANONICAL_TARGETS` 순서(q, k, v, o, gate, up, down) 중 `lora.target_modules`에 있는 것
  - 모듈마다 A `[r, d_in]`을 행 우선으로 U(−1/√d_in, 1/√d_in)에서 r·d_in개 (`kaiming_uniform_(a=√5)`와 같은 값, 확인), B는 0 (난수 없음)
  - HF 등록 순서(`LlamaMLP`는 gate, down, up `modeling_llama.py:151-153`)나 `named_modules()` 순서를 쓰지 않음
- LoRA parameter는 CPU fp32로 만들고 M9가 모델 전체를 장치로 옮김 → CPU 테스트와 GPU 실행의 초기값 생성 방식이 같음
- dropout mask: 학습 중 전역 torch 난수(M10이 `("torch_global",)`로 1회 seed)에서 forward 실행 순서대로 뽑음. 층마다 q, k, v (`modeling_llama.py:196-198`), o (`:243`), gate, up, down (`:157`). eval 모드에서는 난수를 쓰지 않음
- B = 0으로 시작하므로 첫 step에서 출력은 base와 같고, ω·A의 gradient는 정확히 0, B만 gradient를 받음 (12.8 항목 6). gate는 B가 0에서 벗어난 다음 step부터 학습됨

#### 12.6.4 GateContext 규칙
- 모델 하나에 하나 (3.1). M9가 만들고 224층이 같은 객체를 참조
- `use(omega)` 확인 (어기면 `GateContextError`, 보관하지 않음):
  - 이미 ω가 있음 (중첩 금지)
  - `omega.dim() == 2`, `omega.shape[1] == num_experts`, `omega.dtype == float32`
  - NaN/Inf 없음, 음수 없음, 행마다 |Σ − 1| ≤ 1e-5 (fp32 softmax 반올림보다 충분히 큰 검사 상수, 실험 의미와 무관)
- 블록이 끝나면 예외가 나도 finally에서 비움. `GateUse.calls`로 블록 안의 층 호출 수를 M9에 돌려줌 (M9가 "모든 대상 층이 단계마다 한 번씩 ω를 읽음"을 확인)
- ω의 계산 그래프는 그대로 보관: dynamic gate는 224층의 gradient가 ω에 모여 W_g로 감. 생성은 no_grad
- 비용: 행 합·유한성 확인이 `use()`마다 GPU 동기화 1회 (층 호출마다가 아님)

#### 12.6.5 dtype과 장치 (D9)
- base 가중치 fp16 고정 (requires_grad False), LoRA A·B fp32 master (requires_grad True)
- GPU: M10·M11이 M0 precision 도구로 연 fp16 autocast 안에서 호출. A·B matmul은 fp16, gradient는 fp32 master에 쌓임
- 출력 dtype = base 출력 dtype. CPU 테스트(fp32 base, autocast 없음)에서는 모두 fp32
- autocast 밖에서 fp16 입력이면 3단계에서 즉시 오류 (조용한 fp32 승격 없음)
- ω는 x와 같은 장치여야 함 (M9가 맞춤)

#### 12.6.6 오류 처리
- forward는 예외를 잡지 않음. 메시지에 층 이름, x 모양, ω 모양
- 생성자 `ValueError`: `r <= 0`, `r % num_experts != 0`(식 9), `alpha <= 0`, dropout ∉ [0, 1), base가 정확히 `nn.Linear`가 아님(fan_in_fan_out 층 미지원), scaling_rule이 선택지 밖, base가 이미 `MoELoRALinear`
- `attach_moe_lora` `ValueError`: 대상 이름이 층에 없음, 교체 수 ≠ 층 수 × 대상 수, LLM이 CPU에 있지 않음

#### 12.6.7 설정 (`lora` 구역)
| 키 | 타입 | paper_spec 값 | 라벨 | 설명 |
|---|---|---|---|---|
| `r` | int | 8 | `[코드]` D7 | 전체 rank, `U main.py:128`, `L` 같음 |
| `alpha` | int | 32 | `[코드]` D7 | `U main.py:129` |
| `dropout` | float | 0.1 | `[코드]` D7 | A 앞 입력 dropout, `U main.py:130` |
| `num_experts` | int | 4 | `[논문 p.9 §4.3.2]` | K. `r % K == 0` (식 9). M7 gate 출력 차원도 이 값을 읽음. E1 프로파일은 1 |
| `scaling_rule` | str ∈ {`expert_rank`, `total_rank`} | `expert_rank` | `[결정]` D8 | expert_rank: α/r* (= 16), total_rank: α/r (= 4, `variant_total_rank_scaling`) |
| `target_modules` | tuple[str, …] ⊆ `CANONICAL_TARGETS` | `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` | `[코드]` D7 | `U model/model_interface.py:369`, `L model/model_interface.py:220`. 빈 목록, 중복, 목록 밖 이름 거부 |
- 숨은 기본값 없음. 초기화 분포(A kaiming a=√5, B 0)는 설정이 아닌 코드 상수 `[코드]` `U moelora.py:670-674`

#### 12.6.8 다른 모듈과의 연결
- M9: `GateContext` 생성, `attach_moe_lora` 호출, forward·generate 동안 `with gate_ctx.use(ω) as use`, 끝난 뒤 `use.calls` 확인
- M7, M15: `ω` 공급 (3.1). M8은 ω를 만들지 않고 모양·값만 확인
- M10: M9 `param_groups()["lora"]`(M9가 `lora_named_parameters`를 이름 사전순으로 정렬해 냄)로 optimizer 그룹 `lora` (weight decay 0, D19)
- M12: `expert_slices`로 expert별 가중치 분석 (선택)

#### 12.6.9 계약 추가 제안
- `GateContext.use(omega)`가 `GateUse`(층 호출 수)를 돌려줌, 오류형 `GateContextError`
- K는 `lora.num_experts` 한 곳에서만 정의 (M7 `Router`·`GateContext`는 M9가 이 값을 인자로 넘김. `router` 구역에는 K가 없음)
- `attach_moe_lora(llm, cfg.lora, gate_ctx, main_seed) -> list[str]`, `lora_named_parameters(llm)`

### 12.7 결정
- D7 (r 8, α 32, dropout 0.1, 224 모듈), D8 (expert_rank 16, 변형 total_rank 4), D9 (fp16 base, fp32 LoRA)
- `[결정]` 계산 방식 B((A x) ⊙ ω 확장): 식 12와 대수적으로 같고 [B, L, K, d_out]를 만들지 않음 (fork 측정으로 OOM 원인 확인)
- `[결정]` gate 곱은 LoRA dtype(fp32): ω gradient가 L 방향 합을 fp16으로 하지 않게. matmul은 autocast fp16 그대로
- `[결정]` 초기화 순서 고정: seed scope `("init", "lora")` 하나 안에서 canonical 순서, `skip_init`. HF 등록 순서나 모듈 생성 방식과 무관하게 초기값 재현
- `[결정]` ω 값 확인은 `GateContext.use`에서 한 번(행 합 1e-5, 유한성, dtype), 층은 batch 크기·장치만 확인: 224층 × 생성 단계마다 동기화하지 않음. 층의 forward·backward도 host로 값을 읽지 않음 (ω 확장은 expand·reshape, 12.6.2 6단계. 측정: 층 forward의 `aten::item`이 `repeat_interleave`일 때 2회, 지금 0회) `R tests/test_m8_moe_lora.py test_layer_call_reads_nothing_back`
- `[결정]` autocast 밖 fp16 입력은 오류: 공개 코드처럼 조용히 fp32로 계산하지 않음
- `[결정]` merge 미구현: sequence별 ω와 맞지 않음

### 12.8 완료 기준
1. (CPU) Llama-2-7B 모양 골격(meta 장치, 4.28.0 `LlamaConfig`)의 7개 대상을 `MoELoRALinear` 생성자로 직접 감싸면 (`attach_moe_lora`는 CPU 모델만 받으므로 이 시험에는 쓰지 않음, LoRA 텐서는 생성자가 CPU에 만듦) 224 모듈, LoRA 텐서 448개, parameter 19,988,480 (q/k/v/o 각 2,097,152, gate/up/down 각 3,866,624). projector 17,047,552 + gate 256과 합치면 37,036,288, gate 비율 256/37,036,032 = 0.000691% (계산: `w4_counts.py`, 재검토 때 손계산으로 같은 값 확인)
2. (CPU) fp64에서 순전파와 gradient가 expert별 naive 구현(Σ_k ω_k s B_k(A_k x))과 공개 코드 식(einsum + gate 합, `U :750-783`) 둘 다와 일치: 출력·x·A·B gradient 상대오차 < 1e-12, ω(fp32) gradient < 1e-6. dropout 0과 0.1(같은 mask), `[3, 7, d]`와 `[2, 1, d]` (설계 확인 측정: 출력 ≤ 2.2e-16, ω ≤ 9.6e-8)
3. (CPU) K = 1: scaling 4.0, 같은 dropout mask에서 `base(x) + 4·B(A(dropout(x)))`와 bitwise 같음 (두 규칙 모두)
4. (CPU) K = 4: scaling expert_rank 16.0, total_rank 4.0. 균등 ω(0.25)에서 expert_rank 출력 = 같은 A, B의 표준 LoRA(α/r = 4), total_rank 출력 = 그 1/4 (fp64 상대오차 < 1e-12)
5. (CPU) backward용 저장 텐서에 `[B, L, K, d_out]` 모양이 없음 (`saved_tensors_hooks`)
6. (CPU) B = 0 초기 상태: 출력 == base(x) bitwise, ω와 A의 gradient 정확히 0, B gradient는 0 아님
7. (CPU) 초기화: A ∈ [−1/√d_in, 1/√d_in], B = 0. 같은 seed로 두 번 감싸면 모든 LoRA 텐서가 bitwise 같고 seed가 다르면 다름. `attach_moe_lora` 전후 전역 CPU 난수 상태가 같음
8. (CPU) GPU fp16의 대리로 CPU bf16 autocast: base bf16, 출력 bf16, A·B·ω gradient fp32, base gradient 없음. autocast 밖 bf16 입력은 명확한 오류
9. (CPU) 오류: GateContext 비었음, batch 크기 불일치, 중첩 `use`, NaN ω, 행 합 ≠ 1, float64 ω, 장치 불일치, `r % K != 0`, 목록 밖 대상 이름 → 모두 예외. 예외 뒤 GateContext는 비어 있음
10. (CPU) 작은 LLaMA(2층, 13.6.9)에서 학습 forward 한 번의 `GateUse.calls` = 14, 생성 T 단계면 14 × T. 감싼 층 하나의 forward·backward는 host 읽기(`aten::item`) 0회 (K = 4, 1), 학습 forward의 host 읽기 수는 2층(14 모듈)과 3층(21 모듈)에서 같음 (12.7)
11. (VESSL_REQUIRED) A100에서 fp16 base + fp16 autocast로 감싼 층 하나: 출력 fp16, A·B gradient fp32, 2번 항목과 같은 입력에서 fp32 기준과의 차이 기록
12. (VESSL_REQUIRED) 감싼 Llama-2-7B의 가장 긴 학습 batch(B = 8, 5 epoch 계획에서 L이 최대인 batch를 CPU에서 M4·M5로 계산해 씀. 참고: fork prompt 분포의 최대 L = 347, M5 9.6.3 측정 train epoch 0 key 표본 최장 334·구성한 최악 639) forward+backward 최대 메모리 기록. 기대: 76 GiB 미만 (fork 재작성 보정 모형 60.7 GiB, bf16 base 기준 `F experiments/IMPLEMENTATION_NOTES.md §11g`)

---

## 13. M9 LLM 연결
상태: 테스트 통과(로컬)

### 13.1 목적과 범위
- **목적:** Llama-2-7B에 MoE-LoRA(M8)를 붙이고 SASRec(M3), projector(M6), ω 공급원(M7 `Router`, cluster_hard는 그 안에서 M15 `ClusterHardGate`)과 연결해, `TrainBatch`에서 학습 loss를, `EvalBatch`에서 생성 토큰을 내는 모델 객체 `ILoRAModel`을 만든다
- **범위:** LLaMA 로드(fp16), 특수 토큰 embedding 확장과 그 초기화, 고정 범위, 224층 교체, `inputs_embeds` 조립(행동 토큰 삽입), z·ω 계산과 `GateContext` 설정, 학습 loss, stock `generate` 호출, train/eval 모드, 모델 쪽 checkpoint 상태, parameter 보고, CPU 테스트용 작은 LLaMA
- **범위 밖:** prompt 문자열과 tokenization·label(M4, M5), optimizer·lr·GradScaler·accumulation·autocast 범위(M10, M0), 생성 문자열 해석과 지표(M11), 기록 형식(M12), 군집 산출물 로드와 검사(M15)

### 13.2 입력과 출력
**입력**
- `TrainBatch` (3.1): `input_ids`, `attention_mask`, `labels` `[B, L] int64` 오른쪽 padding, `hist_ids [B, 10]`, `len_seq [B]`, `cand_ids [B, 20]`, `hybrid`, `sample_ids`
- `EvalBatch` (3.1): 왼쪽 padding `input_ids`, `attention_mask`, `hist_ids`, `len_seq`, `cand_ids`, `hybrid`, `sample_ids`
- tokenizer (M5 `TokenizerBundle`): `LlamaTokenizer`(slow) + `TokenIds` (bos 1, eos 2, pad 32000, ph 32001, history_emb 32002, cans_emb 32003, item_emb 32004, `base_vocab` 32000, `vocab_size` 32005, `new_token_ids`)
- LLaMA 경로: 환경 변수 `LLM_PATH` (M0이 읽어 넘김, 4.2)
- 설정 `model`, `lora`, `router`, `sasrec` 구역과 `run.seed`
- cluster_hard만: M15 `ClusterHardGate` (M10·M11이 `load_cluster_artifacts` 뒤 만들어 넘김, 19.6.7. M12 export는 M9를 거치지 않고 `Router`에 넘김)

**출력**
- `TrainForward`: `loss` (스칼라 float32, 그래프 연결), `n_label_tokens: int`, `omega [B, K] float32`, `z [B, 64] float32` (둘 다 detach, 모델 장치)
- `GenerationOutput`: `new_token_ids [B, T] int64` CPU (1 ≤ T ≤ 64, 4.28.0이 붙이는 BOS 자리표 제거, EOS 뒤는 `[PAD]`), `omega [B, K]`, `z [B, 64]` CPU float32
- `param_groups()`, `trainable_state()`, `generated_frozen_state()`, `parameter_report()` (M10, M12)

### 13.3 논문 근거
- backbone: "we fine-tune the Llama2-7B [21] model" `[논문 App B p.16]`. base/chat 구분 없음 → base `[코드]` `U README.md:13` (meta-llama/Llama-2-7b-hf)
- LLaRA 위에서 LoRA만 바꿈: "we follow LLaRA [9] and fine-tune the LLM (i.e., Llama-2 [21]) on a hybrid prompt, but instead with the personally-activated LoRA" `[논문 p.3 §1]`
- 고정: "W remains frozen" `[논문 p.3 식 3]`, 그림 2의 LLM·tokenizer 고정 표시 `[논문 p.4 그림 2 (암시)]`, LoRA는 "keeps the LLM weights frozen" `[LLaRA 간접 §3 p.4]`
- 아이템 표현: 제목 토큰 = LLM tokenizer + word embedding `[논문 p.4 식 4]`, 행동 토큰 = Proj_τ1(SR-EMB(i)) `[논문 p.4 식 5-6]`, 둘을 이어 붙임 `[논문 p.4 식 6]`
- 자리 교체: "<PH> indicates a special placeholder token, reserved for substitution by the behavioral token" `[LLaRA 간접 그림 3 caption p.5]`, hybrid 항목 = 제목 토큰 뒤 행동 토큰 `[LLaRA 간접 §4.2 p.5-6]`
- text-only 손실에는 projector가 없음: 식 7은 Φ0+ΔΦ(Θ), 식 8에만 Θp `[LLaRA 간접 §4.3 p.6 식 7-8]` → text-only batch는 projector를 부르지 않음
- 목적식: 응답 토큰 y_t의 log P(y_t | x, y<t) 합 `[논문 p.4 식 7, p.5 식 8]`
- gate: z = SR-EMB_θ(i<n) `[논문 p.5 식 10]`, ω = Softmax(W_g z) `[논문 p.6 식 11]`. ΔW(i<n)가 instance당 하나 → prompt·응답의 모든 토큰과 모든 생성 단계에 같은 ω `[논문 p.6 식 12 (암시)]`
- 계산 정밀도: "half-precision computation (mixed-precision training)" `[논문 App E p.17]`
- 논문에 없음: 로드 dtype, 새 토큰 embedding 초기화, loss 평균 방식, padding 방향, 생성 설정 → `[코드]`, `[결정]`

### 13.4 공개 코드에서 가져올 것
- 특수 토큰과 embedding 확장: `[PAD]`, `[PH]`, `[HistoryEmb]`, `[CansEmb]`, `[ItemEmb]` 추가 후 `resize_token_embeddings(len(tokenizer))` `U model/model_interface.py:313-319`, `L model/model_interface.py:201-207` (코드 계보 일치). 새 행은 transformers 4.28.0이 N(0, 0.02)로 채우고 (`modeling_utils.py:1467-1471, :1539-1543`, `modeling_llama.py:347-356`) 이후 고정
- 자리 교체 규칙: 행 i의 k번째 `[HistoryEmb]` ← 이력의 k번째 아이템(앞 `len_seq`개), k번째 `[CansEmb]` ← k번째 후보 `U :436-466`, `L :291-313`. 행동 토큰 = projector(SASRec 아이템 임베딩) `U :414-420`
- 학습 호출: `inputs_embeds`, `attention_mask`, `use_cache=False` `U :49-57`, `L :34-40`. loss = HF 내부 loss(label 토큰 평균 CE) `U :289-292`, 4.28.0 `modeling_llama.py:703-713`
- 생성 설정: `do_sample=False`, `num_beams=1`, `max_new_tokens=64`, `min_new_tokens=1`, `repetition_penalty=1.0`, `pad_token_id=[PAD]` `U :70-88`, `L :43-57`. `temperature=0.8`, `length_penalty=1.0`은 greedy에서 쓰이지 않음 (4.28.0 greedy는 warper 없음 `generation/utils.py:1429-1448`)
- SASRec 고정·eval `U :406-412` (D12)
- `model_interface.py`는 2024-08 판(`0a69c15`)과 test 지표(ctr) 줄만 다름 (git diff) → 위 내용은 두 판 공통

### 13.5 버릴 것 (알려진 문제)
- **bf16 로드 + fp16 autocast** `U model/model_interface.py:318`, `L model/model_interface.py:206` → fp16 그대로 로드 (D9)
- **transformers 내부 파일 교체** `U README.md:41, :43` (`debug/utils.py:1106` kwargs 허용, `debug/modeling_llama.py:653-654` `**kwargs`)와 kwargs로 gate 전달 → `GateContext` (D13). stock 4.28.0 `generate`에는 `inputs_embeds`, `attention_mask`만 넘김 (13.6.6)
- **생성 오른쪽 padding** `U :316` → 왼쪽 padding (D22, M5)
- **조용한 잘림:** `zip`으로 자리와 아이템을 짝지어 개수가 달라도 오류가 없음 `U :456, :460` → 행마다 개수 일치 확인
- **표본별 Python 반복과 제자리 쓰기** `U :453-464` → 한 번의 out-of-place `index_put`
- **쓰이지 않는 계산:** padding 포함 이력 10개와 `item_id`의 projector `U :446-448` (movie 템플릿에 `[ItemEmb]` 자리 없음 `U prompt/movie.txt:1-3`), 사용자 표현의 projector를 계산하고 버림 `U :428`
- **projector `requires_grad` 전환:** text-only batch에서 끄고 켬 `U :137-142` → 저장 필터와 맞물려 projector 누락 (M10). 대신 text-only batch에서는 projector를 부르지 않음
- **seed 없는 resize:** 새 토큰 행이 그 순간의 전역 난수 상태로 정해짐 → seed scope 안에서 resize
- **학습 parameter 확인 없음** (`print_trainable_parameters`만 `U :378`) → 이름·개수·dtype 확인
- **SASRec dropout 켜짐:** Lightning의 `model.train()` (D12) → `train()` 재정의

### 13.6 설계

#### 13.6.1 공개 API (`rebuild/ilora/model.py`)
```python
@dataclass(frozen=True)
class TrainForward:                    # M10 14.6.10의 이름
    loss: Tensor; n_label_tokens: int; omega: Tensor; z: Tensor

@dataclass(frozen=True)
class GenerationOutput:
    new_token_ids: Tensor; omega: Tensor; z: Tensor

class SeqEncoder(Protocol):            # M3 SASRecEncoder가 만족 (이름은 M3 7.6.3 그대로)
    hidden_size: int; state_digest: str
    def __call__(self, hist_ids: Tensor, len_seq: Tensor) -> Tensor     # z [B, 64] fp32
    def embed_items(self, item_ids: Tensor) -> Tensor                    # [n, 64] fp32
    def train(self, mode: bool = True) -> nn.Module                      # 항상 eval 유지 (D12)

class OmegaSource(Protocol):           # M7 Router가 만족 (dynamic·cluster_hard·none 공통, 11.6.1)
    num_experts: int
    def __call__(self, z: Tensor, sample_ids: Sequence[str]) -> Tensor: ...  # [B, K] fp32

class ModelError(RuntimeError): ...   # 이 모듈의 모든 확인 실패 (13.6.10)

PARAM_GROUPS = ("projector", "gate", "lora")                  # param_groups 키 순서 (D19)
OMEGA_SOURCE_PREFIX = "omega_source."                         # checkpoint의 gate key 접두사 (13.6.8, M12)
GENERATED_FROZEN_PREFIX = "generated_frozen."                 # 새 토큰 행 key 접두사 (13.6.8, M10)
GENERATED_FROZEN_KEYS = ("generated_frozen.token_ids", "generated_frozen.embed_rows", "generated_frozen.lm_head_rows")
LLAMA2_7B_SHAPE = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "intermediate_size": 11008}
TINY_LLAMA_SHAPE = {"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "intermediate_size": 128}

def check_token_ids(bundle: TokenizerBundle) -> TokenIds                          # 1단계
def load_base_llm(path: Path, torch_dtype: torch.dtype) -> LlamaForCausalLM      # 2단계: 로드 정보·dtype·CPU 확인
def check_llama_config(config: LlamaConfig, *, shape: Mapping[str, int], base_vocab: int) -> None   # 2단계 config 확인 (load_base_llm에 backbone 인자가 없어 따로 둠)
def resize_new_tokens(llm: LlamaForCausalLM, vocab_size: int, main_seed: int) -> None   # 3단계
def generation_config_for(mcfg: ModelConfig, ids: TokenIds) -> GenerationConfig        # 13.6.6
def parameter_counts(llm, projector, omega_source, *, base_vocab: int, sasrec=None) -> dict[str, int]   # parameter_report가 씀, meta 골격에서도 셈 (13.8 항목 13)
def build_tiny_llama(vocab_size: int, main_seed: int) -> LlamaForCausalLM      # CPU 테스트용
def build_model(cfg: Config, bundle: TokenizerBundle, catalog: ItemCatalog, *, llm_path: Optional[Path],
                hard_gate: Optional[HardGate] = None) -> "ILoRAModel"      # HardGate: M7 11.6.1

class ILoRAModel(nn.Module):
    llm: LlamaForCausalLM; sasrec: SeqEncoder; projector: nn.Module
    omega_source: OmegaSource; gate_ctx: GateContext; ids: TokenIds
    n_wrapped: int; num_experts: int; gen_config: GenerationConfig   # num_experts = lora.num_experts (M11 대조), gen_config는 읽기 전용 사본 (M11 검사·기록)
    device: torch.device                               # property, 읽기 전용 (M11)
    def train_forward(self, batch: TrainBatch) -> TrainForward
    @torch.no_grad()
    def generate(self, batch: EvalBatch) -> GenerationOutput
    def train(self, mode: bool = True) -> "ILoRAModel"
    def sasrec_is_eval(self) -> bool
    def param_groups(self) -> dict[str, list[tuple[str, nn.Parameter]]]   # "projector", "gate"(dynamic만), "lora"; 그룹 안 이름 정렬
    def trainable_state(self) -> dict[str, Tensor]                        # param_groups의 이름 전부, CPU contiguous 사본
    def load_trainable_state(self, state: Mapping[str, Tensor], strict: bool = True) -> None
    def generated_frozen_state(self) -> dict[str, Tensor]                 # 13.6.8
    def parameter_report(self) -> dict[str, int]
```
- `llm_path`는 `backbone = llama2_7b`일 때만 필수이고 `tiny_random`이면 None이어야 함 (아니면 `ModelError`)
- `build_model`은 eval 모드 모델을 돌려줌 `[결정]`: 만든 직후 `generate`를 부를 수 있고(13.6.6 1단계), 학습은 M10이 `model.train()` 뒤 `sasrec_is_eval()`을 확인함 (14.6.4)
- `parameter_report()` 키: `base_before_resize, new_token_rows, base, lora, projector, gate, trainable, sasrec, frozen, total, lora_tensors, wrapped_modules` (`trainable = lora + projector + gate`, `frozen = base + sasrec`). M10이 `meta.json` `train.parameter_report`에 기록 `R ilora/model.py parameter_counts`
- 메서드 이름은 M10(14.6.10 계약 추가 제안)이 부르는 이름과 같음

#### 13.6.2 모델 만들기 (`build_model`, 순서 고정)
1. tokenizer 확인 (`bundle.ids`, M5 9.6.2가 이미 검사한 값을 다시 대조): `len(bundle.tokenizer) == ids.vocab_size == 32005`, `ids.new_token_ids == (32000, …, 32004)`이고 모두 ≥ `ids.base_vocab`, `bos = 1`, `eos = 2`, `pad ≠ eos`. `[ItemEmb]`는 vocabulary에만 있고 입력에 나오면 오류 (13.6.3)
2. LLaMA 로드 (CPU, `device_map` 없음):
   - `llama2_7b`: `from_pretrained(LLM_PATH, torch_dtype=float16, output_loading_info=True)`. `unexpected_keys`, `mismatched_keys`, `error_msgs`는 비어야 하고 `missing_keys`는 `model.layers.*.self_attn.rotary_emb.inv_freq`(init 때 계산되는 buffer)의 부분집합만 허용
   - config 확인: hidden 4096, 32층, 32 head, intermediate 11008, `vocab_size == ids.base_vocab`(32000), `tie_word_embeddings` False, `rope_scaling` 없음(None), `num_key_value_heads`가 있으면 32 (4.28.0 `LlamaConfig`는 두 키를 읽지 않으므로 다른 값이면 조용히 다른 모델이 됨)
   - 모든 parameter dtype = 설정 `torch_dtype`
   - CPU에서 fp16 7B를 만들고 읽으므로 호스트 메모리가 약 13 GiB + 가장 큰 shard 크기만큼 필요하고, 4.28.0은 모듈 생성 때 torch 기본 초기화를 돌려 시간이 걸림 (실측 `VESSL_REQUIRED`, 13.8 항목 14)
   - `tiny_random`: `build_tiny_llama` (13.6.9)
3. 새 토큰 행 (resize): LLaMA가 CPU에 있는지 확인 후 `seeding.torch_seed_scope(seed, "init", "new_tokens")` 안에서 `resize_token_embeddings(len(bundle.tokenizer))` 한 번. 4.28.0의 난수 소비 순서 (모두 CPU 생성기):
   1. `nn.Embedding(32005, 4096)` 생성 시 N(0, 1) 채움, fp32, 버려짐 `modeling_utils.py:1467`
   2. fp16으로 바꾼 뒤 `_init_weights`가 N(0, 0.02)로 전체를 채움 `:1468-1471`, `modeling_llama.py:353-354`. 앞 32,000행은 원래 값으로 덮음 `:1484` → 32000–32004행이 남음
   3. `nn.Linear(4096, 32005)` 생성 시 kaiming 채움, fp32, 버려짐 `:1539`
   4. fp16에서 N(0, 0.02) `:1540-1543`, 앞 32,000행 복사 `:1570` → lm_head 32000–32004행이 남음
   - 결과: 새 embedding은 `padding_idx` 없음, `config.vocab_size = 32005` `:1403`. CPU fp16 `normal_`은 torch 2.0.0에서 동작하고 같은 seed면 bitwise 같음 (확인)
   - 의미: `[PH]` 행은 text-only prompt마다 입력에 들어감. `[PAD]` 행은 mask로 가려짐, `[HistoryEmb]`·`[CansEmb]` 행은 hybrid에서 교체됨. lm_head 새 행은 생성 때 그 토큰들의 logit을 정함 (모두 고정)
4. 고정: LLaMA의 모든 parameter `requires_grad_(False)` (embed_tokens, lm_head, norm 포함). resize가 새로 만든 parameter는 requires_grad True라 반드시 resize 뒤에
5. MoE-LoRA: `gate_ctx = GateContext(lora.num_experts)`, `attach_moe_lora(llm, cfg.lora, gate_ctx, seed)` (M8, scope `("init", "lora")`). 교체 수 = 층 수 × 대상 수 (224)
6. 나머지:
   - SASRec: `SASRecEncoder.from_config(cfg, catalog, history_len=HIST_LEN)` (M3, 고정·eval·fp32. `catalog`은 M1, `HIST_LEN` = 10은 M5 상수)
   - projector: `build_projector(encoder.hidden_size, llm.config.hidden_size, seed)` (M6, scope `("init", "projector")`, fp32)
   - ω 공급원: `Router(cfg.router, num_experts=cfg.lora.num_experts, z_dim=encoder.hidden_size, main_seed=seed, hard_gate=hard_gate)` (M7). dynamic은 scope `("init", "gate")`로 W_g를 만들고, none은 ω = 1, cluster_hard는 `hard_gate`(M15)를 조회. `hard_gate`가 모드와 맞지 않으면 M7이 `RouterError`
   - 각자 자기 scope 안에서 만들므로 만드는 순서가 값에 영향 없음
7. 장치 이동: 모델 전체 `.to(cfg.model.device)` (dtype 유지). `cuda:0`이면 CUDA 사용 가능 확인 (이 확인은 2단계 전에 함: CUDA가 없으면 7B를 읽기 전에 `ModelError`). 이후 `model.device`(property)는 이 장치를 돌려줌
8. 사후 확인 (실패 시 오류): 학습 parameter 목록 (13.6.7), base dtype, `llm.is_gradient_checkpointing == False` (`modeling_utils.py:1648-1655`), SASRec eval, `gen_config.use_cache is True` (13.6.6)
- `from_pretrained`도 4.28.0에서는 모듈 생성 때 전역 CPU 난수를 씀 (`no_init_weights`는 HF 초기화만 끔 `modeling_utils.py:118-131`). 모든 초기값은 자기 scope 안에서 정해지므로 무관
- M0 `torch_seed_scope`는 `fork_rng(devices=[])` 안에서 CPU 기본 generator만 seed·복원하고 CUDA generator는 건드리지 않음 (`rebuild/ilora/seeding.py`; `torch.manual_seed`는 CUDA까지 다시 seed함 `torch/random.py:26-46`). M10의 `("torch_global",)` seed(`torch.manual_seed`, CPU·CUDA)는 `build_model` 다음에 1회 (14.6.5): 그래야 dropout 난수열이 `from_pretrained`가 소비한 전역 난수 양과 무관

#### 13.6.3 inputs_embeds 조립 (`_embed_inputs`)
1. 확인 (어기면 오류):
   - `hist_ids`: 앞 `len_seq[b]`개 ∈ [0, 1681], 나머지 = 1682 (`ItemCatalog.pad_id`), `3 ≤ len_seq ≤ 10`. `cand_ids` ∈ [0, 1681]
   - 행마다 개수: `[ItemEmb]` 0. hybrid면 `[HistoryEmb]` = `len_seq`, `[CansEmb]` = `cand_ids.shape[1]`(20), `[PH]` 0. text-only면 `[HistoryEmb]` = `[CansEmb]` = 0, `[PH]` = `len_seq + 20`
   - 특수 토큰은 `attention_mask = 1`인 곳에만
2. `emb = llm.get_input_embeddings()(input_ids)` `[B, L, H]`, base dtype, gradient 없음
3. text-only: `emb` 그대로 반환 (projector 호출 없음)
4. hybrid:
   - 아이템 id: `ids_h = hist_ids[arange(10)[None] < len_seq[:, None]]` (행 우선, 오래된 것부터), `ids_c = cand_ids.reshape(-1)` (행 우선, 후보 순서)
   - 자리: `pos_h = nonzero(input_ids == ids.history_emb)`, `pos_c = nonzero(input_ids == ids.cans_emb)` (행 우선 정렬)
   - 행마다 개수가 같으므로 행 안 k번째 자리 ↔ k번째 아이템 (`U :456, :460`과 같은 규칙, 잘림 없음)
   - `rep = sasrec.embed_items(cat(ids_h, ids_c))` `[n, 64] fp32`: autocast 끄고 no_grad (3.1 아이템 표현)
   - `tok = projector(rep)` `[n, H]`: 호출자의 autocast 아래 (GPU fp16) (3.1 행동 토큰)
   - `emb = emb.index_put((pos[:, 0], pos[:, 1]), tok.to(emb.dtype))`, `pos = cat(pos_h, pos_c)`: out-of-place, 자리 중복 없음, gradient는 `tok`을 거쳐 projector로. 삽입 직전 `tok` 유한값 확인 (M6 10.6.3)
5. 반환 `emb [B, L, H]`
- projector는 필요한 아이템만 계산 (`n = Σ len_seq + 20B`). 행 단위 연산이라 공개 코드처럼 padding 포함 전체를 계산한 뒤 고른 값과 반올림 범위 안에서 같음. bitwise로는 같지 않음: CPU fp32 GEMM의 행 값이 한 번에 넣는 행 수 M과 thread 수에 따라 달라짐 (측정: projector 64→64와 64→4096에서 M = 1–200의 최대 |차이| 3.6e-7, 4.8e-7, 1 thread와 14 thread도 다름, `impl/check_m9_rows.py`, `design_sync/g3/probe_rows.py`). 그래서 13.8 항목 5는 같은 projector 호출을 넣은 참조와 bitwise, 공개 코드 배치와는 ≤ 1e-5로 확인

#### 13.6.4 z와 ω
1. `with torch.autocast(device_type, enabled=False)` 안에서:
   - `z = sasrec(hist_ids, len_seq)` `[B, 64] fp32`, no_grad, SASRec eval (D12, D23)
   - `omega = omega_source(z, sample_ids)` `[B, K] fp32` (M7 `Router`, dynamic이면 gradient 있음)
2. 확인: `z.dtype == float32`, `z.requires_grad == False`, `omega.shape == (B, lora.num_experts)`. 나머지 ω 값 확인은 M7 출력 검사와 `GateContext.use`
- gate 값은 이 한 곳에서만 수집해 출력으로 돌려줌 (MoE 층 안에서 모으지 않음) `[SPEC §4]`

#### 13.6.5 학습 forward (`train_forward`)
1. batch 텐서를 모델 장치로 옮김 (M5 `TrainBatch.to(device)`, `non_blocking=True`). 2단계 확인과 13.6.3 1단계 확인은 이 이동 전에 M5가 준 CPU batch에서 한 번에 함 `[결정]`: GPU 동기화가 생기지 않고, 실패하면 첫 위반 행의 번호·`sample_id`·형식(hybrid·text-only)과 그 행이 어긴 규칙을 모두 적을 수 있음. 이동 뒤에 확인한 결과와 같음 `R ilora/model.py ILoRAModel._check_batch`
2. 확인: 오른쪽 padding (행마다 mask가 1…1 0…0), `input_ids[:, 0] == bos`, `labels[:, 0] == -100`, label ≠ -100인 곳은 mask = 1이고 `labels == input_ids`, 행마다 label 토큰 ≥ 1
3. `z, ω` (13.6.4), `emb` (13.6.3)
4. `with gate_ctx.use(ω) as use:` `h = llm.model(inputs_embeds=emb, attention_mask=attention_mask, use_cache=False, return_dict=True).last_hidden_state` (마지막 RMSNorm 뒤 `modeling_llama.py:594`)
   - position_ids는 넘기지 않음: 오른쪽 padding이라 4.28.0 기본 `arange` (`:520-525`)가 실제 토큰 위치와 같음 (공개 코드와 같음)
5. 확인: `use.calls == n_wrapped` (224, 작은 모델 14)
6. loss `[결정]`: `sel = labels[:, 1:] != -100`, `logits = llm.lm_head(h[:, :-1][sel])` `[N, V]` (autocast 아래 fp16), `loss = F.cross_entropy(logits.float(), labels[:, 1:][sel], reduction="mean")`
   - HF 내부 loss(`modeling_llama.py:703-713`: shift 후 label 토큰 평균)와 같은 식. fp32 CPU에서 같은 값 확인 (13.8 항목 4)
   - micro-batch 안 label 토큰 평균 (코드 계보와 같음). accumulation 나눗셈은 M10 (D24)
7. 반환 `TrainForward(loss, N, ω.detach(), z)`
- 학습 모드면 LoRA dropout 224개가 전역 난수를 씀 (12.6.3). eval 모드 호출도 허용 (dropout 없음, overfit 점검용)
- gradient: LoRA A·B, projector(hybrid만), gate(dynamic만)에만 생김. base·embedding·lm_head·SASRec은 requires_grad False
- gradient checkpointing은 쓰지 않음: backward의 재계산 때 `GateContext`가 비어 있어 오류가 나므로 build에서 꺼져 있음을 확인

#### 13.6.6 생성 (`generate`)
1. `self.training`이면 오류 (평가는 eval 모드, dropout 없음, 1.3 결정성)
2. 확인: 왼쪽 padding (행마다 mask가 0…0 1…1, 마지막 열 모두 1), 행의 첫 실제 토큰 = BOS. 13.6.5 1단계처럼 13.6.3 1단계 확인과 함께 장치 이동 전 CPU batch에서 하고, 그 뒤 batch를 모델 장치로 옮김
3. `z, ω`, `emb` (13.6.3, 13.6.4)
4. `with gate_ctx.use(ω) as use:` `seq = llm.generate(inputs_embeds=emb, attention_mask=attention_mask, generation_config=self.gen_config)`
   - `gen_config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens, min_new_tokens, repetition_penalty, pad_token_id=ids.pad, bos_token_id=ids.bos, eos_token_id=ids.eos, use_cache=True)`, 값은 `model` 구역에서. `generation_config` 인자를 주면 모델 폴더의 `generation_config.json`은 쓰이지 않음 (`generation/utils.py:1212-1229`)
   - `use_cache=True`는 필수 (설정 항목 아님): 4.28.0은 `past_key_values`가 없으면 매 단계 처음의 `inputs_embeds`를 다시 넣음 (`modeling_llama.py:742-745`). cache 없이 부르면 둘째 단계에서 `RuntimeError: shape '[-1, 7]' is invalid for input of size 24` (작은 모델로 확인, `design/tmp/rv_enc/rv_gen.py`). build에서 확인
5. 확인 후 정리: `seq.shape[0] == B`, `seq[:, 0] == bos` 전부, `1 ≤ T = seq.shape[1] − 1 ≤ max_new_tokens`, `use.calls == n_wrapped × T`. `new_token_ids = seq[:, 1:].cpu()`
6. 반환 `GenerationOutput(new_token_ids, ω.cpu(), z.cpu())`. 문자열 변환은 M11 (`batch_decode(skip_special_tokens=True, clean_up_tokenization_spaces=False)` 후 strip `U :89-90`)

**transformers 4.28.0 소스로 확인한 동작** (CPU 작은 모델로 실행 확인, 13.8 항목 9)
- `inputs_embeds` 허용: `prepare_inputs_for_generation`에 `inputs_embeds` 인자가 있어 통과하고, `input_ids`는 BOS 하나 `[B, 1]`로 만들어 model_kwargs에 둠 `generation/utils.py:537-556, :568-594`
- 반환값 = 그 `input_ids` 뒤에 새 토큰을 붙인 것 `:1304, :2291, :2333` → `[BOS] + 새 토큰`, prompt는 반환되지 않음
- 첫 단계만 `inputs_embeds`, 이후는 마지막 토큰 id `modeling_llama.py:730-731, :742-745`
- position_ids = `attention_mask.cumsum(-1) − 1`, padding 자리는 1, cache 단계는 마지막 값 `modeling_llama.py:733-739` → 왼쪽 padding에서 실제 토큰 위치가 0부터 (D22). mask는 단계마다 1열 늘어남 `generation/utils.py:735-739`
- 최대 길이 = `max_new_tokens + 1`(BOS 자리표) `:1310-1320` → 새 토큰 최대 64 (작은 모델로 폭 65 확인)
- `min_new_tokens=1`: 첫 단계 EOS logit −inf (`prompt_length_to_skip = 1`) `:857-866`, `logits_process.py:157-163`
- `repetition_penalty=1.0`이면 processor를 만들지 않음 `:829-830`. greedy = argmax `:2282`, 끝난 행은 `pad_token_id` `:2285-2288`
- 단계마다 forward 1회 → 감싼 층마다 호출 T회 (작은 모델: 새 토큰 5개에 `q_proj` 호출 5회, 첫 호출 `[B, L, d]`, 이후 `[B, 1, d]`)
- kwargs 검사: `prepare_inputs_for_generation`의 `**kwargs` 때문에 forward 인자가 허용됨 `:1099-1103` (패치 불필요)
- `generate`는 `@torch.no_grad()` `:1114`
- GPU fp16에서 실제로 도는지는 `VESSL_REQUIRED` (13.8 항목 16)

#### 13.6.7 모드, dtype, 장치, 메모리
- `train(mode)`: `super().train(mode)` 뒤 항상 `sasrec.eval()` (D12, M3 `train()` 재정의와 이중 보장). `sasrec_is_eval()`은 SASRec과 모든 하위 모듈의 `training`이 False인지 돌려줌 (M10 14.6.4가 `model.train()` 뒤 확인). 4.28.0 LLaMA에는 dropout 층이 없어 (`modeling_llama.py`에 Dropout 없음) 모드는 LoRA dropout만 바꿈
- `use_cache`: 학습 False (`U :54`), 생성 True (13.6.6)
- dtype (D9): base fp16, LoRA·projector·gate fp32, SASRec fp32. autocast는 M10·M11이 M0 도구로 연다. M9가 autocast를 끄는 곳은 z·ω 계산뿐
- 학습 parameter (`param_groups`): `lora` 448 텐서 (Llama-2-7B 19,988,480), `projector` 4 텐서 (M6, 17,047,552), `gate` 1 텐서 (dynamic일 때만, 256). 그룹 안은 parameter 이름 사전순, 이름은 `ILoRAModel` 기준 (`llm.model.layers.0.self_attn.q_proj.lora_A.weight`, `projector.mlp_proj.0.weight`, `omega_source.gate.proj.weight`). 이 밖에 requires_grad True가 있으면 build 오류
- 장치: `model.device`. batch는 M9가 옮김. ω는 z와 같은 장치
- 메모리: fp16 base에서는 autocast가 고정 가중치의 fp16 복사본을 만들지 않음. fork 측정(bf16 base + fp16 autocast)에서 이 복사본이 층마다 0.378 GiB `F experiments/IMPLEMENTATION_NOTES.md §11g` → 32층 약 12.1 GiB 감소 예상 (계산, 13.8 항목 17에서 측정)

#### 13.6.8 checkpoint 상태 (M10과 연결)
- `trainable_state()`: build 때 고정한 `param_groups()`의 이름 전부 → 텐서 (CPU, contiguous, fp32). 저장 때 `requires_grad`로 거르지 않음 (M10 14.6.6)
- `generated_frozen_state()`: `generated_frozen.token_ids [5] int64` (= `ids.new_token_ids`), `generated_frozen.embed_rows [5, H]`, `generated_frozen.lm_head_rows [5, H]` (base dtype, CPU). 새 토큰 행은 학습되지 않지만 seed로 만든 값이라 함께 저장 (1.3: 다시 불렀을 때 출력이 같아야 함)
- `load_trainable_state(state, strict=True)`: 키 집합이 `param_groups`의 이름과 정확히 같아야 함(누락·여분 오류), 모양·dtype 일치, 모든 확인이 통과한 뒤에만 값 복사 (일부만 바뀐 모델이 남지 않음). `strict=False`는 있는 키만 같은 확인 뒤 복사. `generated_frozen.*`는 덮어쓰지 않음: `load_trainable_state`가 이 키도 여분으로 거부하므로, M10 `_apply_trainable`(`load_model_state`·`load_checkpoint`)이 먼저 새로 만든 모델의 `generated_frozen_state()`와 bit 단위로 비교하고(다르면 `TrainingError`, 14.6.6) 떼어 낸 나머지를 넘김 `R ilora/train.py _apply_trainable`. 같은 seed·같은 torch면 resize가 같은 행을 만들므로 다르면 실행 환경이 달라진 것
- base 가중치는 저장하지 않음 (`LLM_PATH`, M0가 config 해시 기록)
- 새 토큰 행의 기록 (13.8 항목 14): M10이 학습 준비 때 `meta.json` `train.generated_frozen_sha256`에 `generated_frozen_state()` 세 텐서의 내용 digest를 씀. digest는 M3 `state_digest`(7.6.2)와 같은 방식(key 정렬 순서로 `key|dtype|모양` 줄과 텐서 원시 byte를 SHA-256에 넣음)이고 byte를 torch로 꺼내 bf16도 됨 `[결정]`: 13.8 항목 14가 meta 기록을 요구하지만 쓰는 곳을 정하지 않았고, M9에는 실행 폴더가 없음 `R ilora/train.py tensor_digest`
- export(M12)는 `trainable.pt`에서 `omega_source.` 접두어 키만 읽어 접두어를 떼고 LLM 없이 `Router`에 strict 로드 (`gate.proj.weight`)

#### 13.6.9 CPU 테스트 경로
- `tiny_random`: `LlamaConfig(vocab_size=bundle.ids.base_vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=1024, rms_norm_eps=1e-5, initializer_range=0.02, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False)`
  - `torch_seed_scope(seed, "init", "tiny_llama")` 안에서 fp32로 만들고, 임시 폴더에 `save_pretrained` 후 `load_base_llm`으로 다시 읽어 실제 경로(로드 확인 → resize → 고정 → 교체)를 그대로 지남
  - `device = cpu`, `torch_dtype = float32`만 허용: torch 2.0.0 CPU에는 fp16 matmul이 없음 (`"addmm_impl_cpu_" not implemented for 'Half'`, 확인)
- tokenizer: M13 테스트 tokenizer (`ILORA_TEST_TOKENIZER`, hf-internal-testing/llama-tokenizer, vocab 32000) + M5 특수 토큰 → 32005, id `[PAD]` 32000, `[PH]` 32001, `[HistoryEmb]` 32002, `[CansEmb]` 32003, `[ItemEmb]` 32004 (확인)
- SASRec: M3 실제 가중치 (변환 state 파일). batch: M1–M5로 실제 표본에서 만듦
- GPU fp16의 대리: CPU bf16 autocast (base bf16) → logits bf16, loss fp32 (확인)
- 이 대리(13.8 항목 12)와 13.8 항목 2의 fp16 CPU 모델은 `build_model`로 만들 수 없음 (`tiny_random`은 cpu·float32만, 13.6.11) `[결정]`: bf16은 만든 작은 모델의 base parameter(LoRA 밖)를 bf16으로 바꿔 CPU bf16 autocast 안에서 `train_forward`를 부르고, fp16은 `build_tiny_llama` 결과의 사본을 fp16으로 바꿔 `resize_new_tokens`를 부름 `R tests/test_m9_model.py test_bf16_autocast_proxy`

#### 13.6.10 오류 처리
- 모든 확인 실패는 `ModelError`(`RuntimeError` 하위) `[결정]`: 이 장이 오류형을 정하지 않았음. 다른 모듈의 오류(M3 `SASRecError`, M7 `RouterError`, M8 `GateContextError` 등)는 그대로 올라감
- 예외를 잡지 않고 원인을 메시지에 적음: tokenizer 특수 토큰, 로드 정보(missing/unexpected), config 모양·`rope_scaling`, base dtype, 교체 수, 학습 parameter 목록, gradient checkpointing, `use_cache`, 장치
- batch 확인 실패 (13.6.3, 13.6.5, 13.6.6): 첫 위반 행의 번호, `sample_id`, 형식(hybrid·text-only)과 그 행이 어긴 규칙 전부를 메시지에 포함
- label 토큰 0개: HF loss는 조용히 NaN을 내므로 그 전에 오류
- `generate`의 학습 모드 호출, BOS 자리표 불일치, `use.calls` 불일치
- `GateContext`는 예외가 나도 비워짐 (M8)

#### 13.6.11 설정 (`model` 구역)
| 키 | 타입 | paper_spec 값 | 라벨 | 설명 |
|---|---|---|---|---|
| `backbone` | str ∈ {`llama2_7b`, `tiny_random`} | `llama2_7b` | `[논문 App B p.16]`, base는 `[코드]` `U README.md:13` | `tiny_random`은 CPU smoke·test 전용 |
| `torch_dtype` | str ∈ {`float16`, `bfloat16`, `float32`} | `float16` | `[결정]` D9 | base 로드 dtype. `bfloat16`은 D9 대안 프로파일만, `float32`는 `tiny_random`만 |
| `device` | str ∈ {`cuda:0`, `cpu`} | `cuda:0` | `[SPEC §0.12]` | `cpu`는 `tiny_random`만 |
| `gen_strategy` | str ∈ {`greedy`} | `greedy` | `[코드]` `U model/model_interface.py:70`, `L model/model_interface.py:43` | do_sample False, num_beams 1 |
| `gen_max_new_tokens` | int | 64 | `[코드]` `U …:70, :80`, `L …:43, :51` | |
| `gen_min_new_tokens` | int | 1 | `[코드]` `U …:70, :81` | |
| `gen_repetition_penalty` | float | 1.0 | `[코드]` `U …:70, :83` | 1.0이면 processor 없음 |
- 조합 확인 (`validate`): `tiny_random` ⇔ (`cpu`, `float32`), `llama2_7b` ⇒ `cuda:0`이고 `float16` 또는 `bfloat16`, `gen_min_new_tokens ≤ gen_max_new_tokens`
- 구역 사이 조합 (M0 `Config.validate`): M10 `train.precision`과 `fp16_amp` ⇔ `float16`, `bf16_amp` ⇔ `bfloat16`, `fp32` ⇔ `float32` (D9: autocast dtype과 base dtype이 같아야 고정 가중치 복사본이 생기지 않음)
- LLaMA 경로는 설정이 아니라 `LLM_PATH` (기계마다 다름, M0). 작은 모델 모양은 테스트 상수 (실험 의미 없음). 생성의 `use_cache=True`는 설정이 아닌 고정값 (13.6.6)

#### 13.6.12 다른 모듈과의 연결
- M5 → M9: `TokenizerBundle`(`TokenIds`), `TrainBatch`, `EvalBatch`. M1 → M9: `ItemCatalog` (M3 생성 시 대조)
- M3 → M9: `z`, 아이템 표현 (3.1). M6 → M9: 행동 토큰 (3.1)
- M7 → M9: `Router` (`OmegaSource`, 세 모드). M15 → M9: cluster_hard의 `ClusterHardGate`를 `build_model(hard_gate=...)`로 받아 `Router`에 넘김. M8 ↔ M9: `GateContext`, `attach_moe_lora`
- M9 → M10: `train_forward` → `TrainForward`, `param_groups` (D19 그룹), `trainable_state`/`load_trainable_state`, `generated_frozen_state`, `sasrec_is_eval`, `train()`/`eval()`
- M9 → M11: `generate` → `GenerationOutput`, 읽기 전용 `gen_config`·`device`, `omega_source`(M7 `Router`의 기록기·`gate_calls`), `sasrec.hidden_size`·`state_digest`. eval 명령의 checkpoint 적용은 M10 `load_model_state`(14.6.6)를 거쳐 `load_trainable_state` (M11 15.6.11)
- M9 → M12: ω, z, `parameter_report` (TrainableParams `[SPEC §12]`)

#### 13.6.13 계약 추가 제안
- `TrainForward`, `GenerationOutput` (13.6.1)을 3.1에 추가 (M10이 부르는 이름 `TrainForward`로 통일)
- `SeqEncoder`는 M3 이름 그대로 (`__call__`, `embed_items`, `hidden_size`, `state_digest`), `OmegaSource`는 M7 `Router` 하나 (cluster_hard도 `Router`가 M15 `ClusterHardGate`를 부름)
- `build_model(cfg, bundle, catalog, *, llm_path, hard_gate=None)`: tokenizer는 M5 `TokenizerBundle`로 받고 특수 id는 M5 `TokenIds`만 씀 (M9에 별도 id 클래스 없음), `catalog`(M1)은 M3 생성 시 대조용
- `ILoRAModel`의 M10용 메서드: `train_forward`, `param_groups`, `trainable_state`, `load_trainable_state`, `generated_frozen_state`, `sasrec_is_eval` (M10 14.6.10과 같은 이름). M11도 checkpoint 적용에 `load_trainable_state`를 씀
- `ILoRAModel.device`·`gen_config`는 읽기 전용 공개 속성 (M11 15.6.14 계약 추가 제안 6)
- derive_seed key `("init", "tiny_llama")` (CPU 테스트 backbone)
- 생성 설정은 `model.gen_*` 한 곳 (M11은 같은 키를 두지 않음)
- M10은 `build_model` 다음에 `("torch_global",)` seed
- `Config.validate`의 구역 사이 검사: `model.torch_dtype` ↔ `train.precision` (13.6.11)

### 13.7 결정
- D9 (fp16 로드, 학습 parameter fp32), D12 (SASRec eval), D13 (패치 없는 4.28.0, GateContext), D22 (생성 왼쪽 padding, mask 기반 position_ids), D23 (z fp32 autocast 밖), D4 (평가 batch의 `hybrid` 값을 그대로 따름)
- `[결정]` loss 직접 계산 (label 위치만 lm_head, fp32 CE 평균): label 0개일 때 HF loss의 조용한 NaN을 오류로 바꾸고, CE dtype을 명시하고, `[B, L, V]` logits를 만들지 않음. HF 내부 loss와 같은 값 (확인)
- `[결정]` resize는 CPU에서 scope `("init", "new_tokens")` 안: 새 토큰 행(`[PH]` 포함)이 seed로 정해짐
- `[결정]` 생성은 설정으로 만든 `GenerationConfig`를 넘김: 모델 폴더 `generation_config.json`의 내용과 무관하게 greedy. `use_cache=True` 고정 (cache 없이는 4.28.0 `inputs_embeds` 생성이 둘째 단계에서 실패)
- `[결정]` 로드 확인: missing은 `rotary_emb.inv_freq`만 허용, config의 `rope_scaling`·`num_key_value_heads` 확인
- `[결정]` checkpoint에 새 토큰 행 포함, 로드 때는 덮어쓰지 않고 다시 만든 값과 bit 비교 (M10)
- `[결정]` ω 공급원은 M7 `Router` 하나로 받음: dynamic, K = 1, cluster_hard가 같은 경로 (`[SPEC §9]` "gate input만 one-hot으로")
- `[결정]` 생성은 eval 모드에서만, text-only batch는 projector를 부르지 않음 (`requires_grad` 전환 없음)
- `[결정]` 학습은 position_ids를 넘기지 않고 오른쪽 padding을 확인, 생성은 4.28.0의 mask 기반 position_ids
- `[결정]` 작은 LLaMA 모양(64/128/2층/4 head)은 테스트 상수

### 13.8 완료 기준
1. (CPU) `tiny_random` build: 감싼 모듈 14개, 학습 parameter = LoRA 28 텐서(17,408) + projector(64→64, 8,320) + gate(256), base·embedding·lm_head 고정, `config.vocab_size = 32005`. `param_groups()` 키가 dynamic `{projector, gate, lora}`, cluster_hard·none `{projector, lora}`이고 그룹 안 이름이 사전순
2. (CPU) 새 토큰 행: 같은 seed로 두 번 만들면 embedding·lm_head 새 행이 bitwise 같고, seed가 다르면 다름, 앞 32,000행은 resize 전과 같음, 값은 N(0, 0.02) 근처 (설계 확인 측정: 표준편차 0.0198, 0.0191). fp16 CPU 모델에서도 같은 결과
3. (CPU) 초기 상태(B = 0): 감싼 모델의 logits == 감싸지 않은 같은 모델의 logits (bitwise)
4. (CPU) 직접 계산한 loss == HF 내부 loss(`labels=` 전달), 같은 dropout seed, 차이 < 1e-6 (설계 확인 측정: 10.4062376 두 값 같음)
5. (CPU) hybrid 조립 결과가 공개 코드 `wrap_emb`의 행별 반복(`U :453-461`)을 그대로 옮긴 참조 구현에 같은 projector 출력(같은 아이템 행을 같은 순서로 한 번 부른 값)을 넣은 결과와 bitwise 같음. 공개 코드 배치(padding 포함 `[B, 10]` 이력과 `[B, 20]` 후보에 projector, `U :446-448`)와는 최대 차이 ≤ 1e-5 (CPU GEMM 반올림, 13.6.3). text-only는 projector 호출 0회이고 `emb == embed_tokens(input_ids)`
6. (CPU) 오류: `[HistoryEmb]` 하나 빠짐, `[CansEmb]` 하나 더함, hybrid에 `[PH]`, text-only에 `[CansEmb]`, `[ItemEmb]` 등장, hist padding id ≠ 1682, 왼쪽 padding `TrainBatch`, 오른쪽 padding `EvalBatch`, `labels ≠ input_ids`, label 0개 → 모두 예외
7. (CPU) gradient 흐름 (B를 무작위로 둔 상태): LoRA A·B, projector(hybrid), gate가 0이 아닌 gradient. base·SASRec은 gradient 없음. text-only batch에서는 projector gradient가 None
8. (CPU) `model.train()` 뒤에도 SASRec eval이고 `sasrec_is_eval()`이 True, 두 번 호출한 z가 bitwise 같음
9. (CPU) 생성: `seq[:, 0]`가 BOS 자리표이고 제거됨, T ≤ 64, 왼쪽 padding batch 생성 == 한 표본씩 생성 (토큰 일치, fp32), `use.calls == 14 × T`, 학습 모드 호출은 오류, 생성 뒤(예외 포함) `GateContext` 비어 있음, 두 번 생성 결과 같음 (설계 확인 측정: 3행 batch가 `[3, 65]`, 한 표본씩과 일치). `gen_config.use_cache`를 False로 바꾼 모델은 build 확인에서 오류
10. (CPU) position_ids: 왼쪽 padding mask에서 실제 토큰 위치가 0, 1, 2, … (4.28.0 `prepare_inputs_for_generation`, 첫 단계는 `inputs_embeds`만)
11. (CPU) checkpoint 왕복: 학습 parameter를 바꾼 모델의 `trainable_state()`를 저장 → 새로 만든 모델에 `load_trainable_state`로 불러오면 logits와 생성 결과가 같음, `generated_frozen_state()`가 두 모델에서 bit 단위로 같음. 키 누락·여분, 모양·dtype 불일치는 오류. `omega_source.` 키만으로 만든 LLM 없는 `Router`의 ω가 모델 안 `omega_source`와 같음
12. (CPU) bf16 autocast 대리: logits bf16, loss fp32, 학습 parameter gradient fp32
13. (CPU) meta 장치 Llama-2-7B 골격(4.28.0 `LlamaConfig`, 가중치 없음)으로 센 parameter 수: base 6,738,415,616 → resize 뒤 6,738,456,576 (+2 × 5 × 4,096), LoRA 19,988,480, E2 학습 parameter 37,036,288, gate 없는 경우 37,036,032 (계산: `w4_counts.py`, 재검토 때 손계산으로 같은 값 확인). `parameter_report()`는 build된 모델에서 같은 항목을 같은 식으로 셈
14. (VESSL_REQUIRED) `LLM_PATH`의 Llama-2-7b-hf를 fp16으로 로드: 로드 정보가 허용 범위, 모든 parameter fp16, `rope_scaling` None, resize 뒤 32005. 새 토큰 행의 sha256이 같은 seed로 두 번 로드해도 같음 (시험 값은 `measurements.json`. 학습 실행에서는 M10이 `meta.json` `train.generated_frozen_sha256`에 씀, 13.6.8). CPU 로드 시간과 최대 호스트 메모리 기록
15. (VESSL_REQUIRED) fp16 autocast 학습 micro-batch 하나: logits fp16, loss fp32 유한, 직접 계산 loss와 HF 내부 loss의 차이 기록, 첫 step의 ω·lora_A gradient 0, lora_B gradient ≠ 0 (GradScaler로 scale하지 않은 backward로 확인: B = 0에서 scale된 fp16 gradient가 overflow하면 0 × inf = NaN이 되어 초기 상태를 확인할 수 없음. loss 차이는 dropout 없는 eval 모드에서 잼)
16. (VESSL_REQUIRED) 패치 없는 stock `generate(inputs_embeds, attention_mask)`가 A100 fp16에서 동작, 반환 첫 열 BOS, 같은 batch 두 번 생성 결과 같음, batch 생성과 한 표본씩 생성의 일치율 기록 (fp16이라 100%를 요구하지 않음)
17. (VESSL_REQUIRED) 가장 긴 학습 batch의 최대 메모리 기록. 12.1 GiB 감소 예상(13.6.7)과 비교

---

## 14. M10 학습 루프
상태: 테스트 통과(로컬)

### 14.1 목적과 범위
- **범위:** `train` 명령의 학습 전체
  - epoch마다 `BatchPlan`(M4·M5) 순서로 micro-batch 손실(M9)을 역전파하고, accumulation 창이 차면 GradScaler step과 lr 설정(D6)
  - epoch 끝: 상태 저장, validation(M11, 94명), 학습 기록 전달(M12)
  - 학습 뒤: checkpoint 선택(D11), test 평가 호출(M11). 중단된 실행의 재개
- **하지 않는 것:** 후보·템플릿·curriculum 추첨·tokenization(M2, M4, M5), 순전파와 손실 계산(M9), 생성과 판정(M11), 기록 파일 형식(M12), 프로파일 값(M14)
- **원칙:** Lightning이 숨기던 동작을 여기서 모두 명시한다 (D13): `model.train()` 뒤 SASRec 모드(D12), max_steps 절단 없음과 epoch 끝 부분 accumulation(D24), 저장 필터 없음

### 14.2 입력과 출력
**입력**
- `Config`: `train` 구역(14.6.9), `run.seed`, `batching`(M5), `eval`(M11), `records`(M12), `cluster`(cluster_hard만, M15)
- `LoadedData`(M1 5.6.1 계약 추가 제안): train 66,126, validation 94, test 95 표본 (smoke는 M1 5.6.7)
- 학습 시작 때 한 번 만드는 `TokenizerBundle`(M5 `load_tokenizer`)과 `TemplateSet`(M4 `load_templates`): 같은 객체를 M5 loader·M9·M11에 넘김 (M5 9.6.11 계약 추가 제안)
- 모델(M9): 학습 parameter 그룹, `TrainBatch` 손실, `EvalBatch` 생성 (계약 추가 제안 14.6.10)
- cluster_hard만: M15 `ClusterArtifacts`·`ClusterHardGate` (14.6.4 0단계, M15 19.6.7)
- epoch마다 `BatchPlan`(3.1): 8,265개 `(tau, sample_indices, hybrid)`, `tau`는 전체 학습 기준 1부터
- `TrainBatch`(3.1): `input_ids`·`attention_mask`·`labels [B, L] int64`(오른쪽 padding, 응답 외 −100), `hist_ids [B, 10]`, `len_seq [B]`, `cand_ids [B, 20]`, `hybrid`, `tau`, `sample_ids`, `template_ids`

**출력** (실행 폴더 3.1)
- `checkpoints/epoch{e}/` (e = 0–4): `trainable.pt`, `train_state.pt`, `STATE_SAVED`, `val_metrics.json`, `COMPLETE` (14.6.6)
- `eval/epoch{e}/val/` (e = 0–4), `eval/epoch{b}/test/` (b ∈ {best, last}): M11이 쓰는 폴더 (M11 15.2, M12 16.2와 같은 경로), `eval/selection.json` (14.6.7)
- M12 `TrainRecorder`(16.6.2)에 넘기는 행: micro-batch마다 1행, optimizer step마다 `StepRecord` 1행, epoch마다 1행. 파일 쓰기는 M12가 `records.flush_every_steps` step마다·epoch 끝·checkpoint 저장 직전·종료 때만 (`records/train/`, `[SPEC §4]`)
- `run.log` 요약 줄 (k step마다), `meta.json`에 학습 시각·경과 시간·최대 GPU 메모리 (M0 기록 함수)
- `TrainSummary` (M12 `run_summary.json`, M14 비교 표)

### 14.3 논문 근거
- 목적식은 응답 토큰 로그우도의 최대화이고 최적화 대상은 instance-wise LoRA Δϕ `[논문 p.5 §3.1 식 8]` → LLM 가중치 고정 `(암시)`. gate W_g는 "trainable transformation matrix" `[논문 p.6 §3.2.2 식 11]`
- projector: "the projector parameters are updated using the same learning rate curve integrated with iLoRA during training" `[논문 App E p.17]`. text-only(easy) 손실에는 projector가 없음 `[LLaRA 간접 §4.3 p.6 식 7-8]`
- 식 8은 합. 토큰 평균·micro-batch 평균 정규화는 논문에 없음 → `[코드]` (14.4)
- lr: "a warm-up strategy for the learning rate, which begins the training process with an initial rate set at a fraction of the maximum learning rate", MovieLens 최대 1e-4 `[논문 App E p.17]` (D5). 시작 = 최대의 1/100 `[v1 App E p.16]` `[LLaRA §5.1.2 p.7]` (D6)
- "We adopt a cosine annealing scheduler to adjust the learning rate across training steps" `[논문 App E p.17]`, "over steps" `[v1 App E p.16]` `[LLaRA §5.1.2 p.7]` → optimizer step 단위 일정
- "Each experiment is trained for a maximum of five epochs, with a batch size of 128." `[v1 App E p.16]`, 같은 내용 `[LLaRA §5.1.2 p.7]`, "a total of five epochs" `[LLaRA §5.4 p.9]` (D11)
- "we apply a weight decay factor of 1e-5 to prevent overfitting." `[v1 App E p.16]`, 적용 범위 미기재 (D19)
- "half-precision computation (mixed-precision training)" `[논문 App E p.17]` (D9), "a single Nvidia A100" `[논문 App B p.16]`
- LLM 학습의 optimizer·β·ε·gradient clipping·checkpoint 선택은 iLoRA·LLaRA 모두 없음 (App E의 Adam·lr 0.001은 전통 기준선 문장) → `[코드]`, `[결정]`

### 14.4 공개 코드에서 가져올 것
- **optimizer:** `torch.optim.Adam` 기본값 + 그룹별 `weight_decay` (coupled L2, AdamW 아님) `U model/model_interface.py:227-236`, `L model/model_interface.py:153-156` (코드 계보 일치)
  - torch 2.0.0 기본값: betas (0.9, 0.999), eps 1e-8, amsgrad False (`torch2.0 optim/adam.py:14-15`). L2는 `grad + wd·param` (`:334-335`, foreach 경로 `:441-442`)
  - weight decay는 projector·router 그룹만, LLaMA(LoRA) 그룹은 키가 없어 0 `U :228-232`
- **gradient clipping 없음:** `Trainer.from_argparse_args(args)` `U main.py:57`, `L main.py:57`. parser에 `gradient_clip_val`이 없음 (`U main.py:77-138`, `L main.py:77-130`) → Trainer 기본값 (코드 계보 일치)
- **gradient checkpointing 없음:** U의 `gradient_checkpointing_enable`은 호출되지 않는 kbit 준비 함수 안뿐 `U model/peft/utils/other.py:95`, L에는 없음
- **lr 일정:** `U optims.py:20-36` (blob `3d7269a6`), w = max_steps // 20 `U model/model_interface.py:247-248`, 매 micro-batch 첫 줄에서 `scheduler.step(global_step, current_epoch, max_steps)` `U :135-136`
  - 대응: Lightning `global_step`은 끝난 optimizer step 수라 창 안 micro-batch가 모두 같은 s를 보고, 창 끝 step은 lr(s)를 씀 → 재구현은 창 s의 첫 micro-batch 전에 lr(s)를 한 번 설정 (같은 동작)
  - `cur_epoch == 0` 조건 `U optims.py:22`은 w = 129 < 517(epoch 0의 step 수)이라 효과 없음 → 식에서 뺌
  - 공개 코드의 T는 절단값 2,583 `U main.py:50`. 재구현은 D24의 2,585라 lr이 조금 다름 (s = 1,292에서 5.461093e-05 vs 5.455111e-05, 계산 `design/tmp/w5_counts.py`)
  - cosine은 warmup 끝에서 이어짐 (`cur_step - warmup_steps` `U optims.py:25`). 이동하지 않는 LLaRA판 `L optims.py:46-52`은 쓰지 않음 (D6)
- **데이터 순서:** train DataLoader `shuffle=True, drop_last=True` `U data/data_interface.py:137-143`, `L data/data_interface.py:127-133` → 순서는 M4·M5 `BatchPlan`, drop_last 유지
- **fp16 AMP:** `--precision 16`, `--amp_backend native` `U main.py:79-80` = Lightning 1.8.6 native AMP (fp16 autocast + `torch.cuda.amp.GradScaler`)
  - GradScaler 값은 torch 2.0.0 기본값: init_scale 65,536 (2^16), growth_factor 2.0, backoff_factor 0.5, growth_interval 2,000 (`torch2.0 cuda/amp/grad_scaler.py:113-118`). Lightning 1.8.6 native AMP는 precision 16일 때 GradScaler를 인자 없이 만듦: `scaler = torch.cuda.amp.GradScaler()` (`PL1.8.6 plugins/precision/native_amp.py:56`), 그 plugin은 `NativeMixedPrecisionPlugin(self._precision_flag, device)` (`PL1.8.6 trainer/connectors/accelerator_connector.py:724`). VESSL 환경에서 실제 값을 출력하는 확인은 14.8 #18
  - LLaRA 2024-03 공개 checkpoint(precision 16, PL 1.8.6)의 scaler 상태는 학습 끝 scale 32,768 (이전 감사 기록, 실행 기록 보조 근거)
- **매 epoch validation·전부 저장:** `check_val_every_n_epoch=1` `U main.py:87`, `ModelCheckpoint(monitor='metric', save_top_k=-1, every_n_epochs=1)` `U main.py:23-32`
- **micro-batch 손실** = HF `out.loss` (label 토큰 평균) `U model/model_interface.py:143-144, :289-294`. accumulation 나눗셈은 Lightning이 함 (D24)
  - Lightning 1.8.6은 micro-batch마다 손실을 `accumulate_grad_batches`(16, `U train_movielens.sh:6`)로 나눔: `closure_loss / normalize` (`PL1.8.6 loops/optimization/optimizer_loop.py:88`), normalize = `trainer.accumulate_grad_batches` (`:416`)
  - epoch 마지막 batch에서도 step함 (`PL1.8.6 loops/epoch/training_epoch_loop.py:333-337` "Lightning steps on the final batch"). 그래서 epoch 끝 9개 창의 gradient는 (9개 손실의 합)/16
  - 재구현도 모든 micro-batch 손실을 `grad_accum`으로 나눔 (14.6.4 5.2단계)

### 14.5 버릴 것 (알려진 문제)
- **max_steps 절단:** `66,126 × 5 // 128 = 2,583` `U main.py:50`인데 실제 step은 2,585 → epoch 4가 8,240/8,265 micro-batch에서 멈추고 epoch 4 validation이 없음. fork 로그 `Epoch 4: 100%|▉| 8240/8277`, `epoch=04-metric=0.479.ckpt`는 epoch 3 값 (`F` 6c7ebac 결과 보관 파일) → D24
- **projector requires_grad 토글 + 저장 필터:** text-only micro-batch마다 projector를 `requires_grad=False`로 바꾸고 `U model/model_interface.py:137-142`, 저장 때 `requires_grad`가 False인 parameter와 buffer를 지움 `U :296-309` (기본 `save='part'` `U main.py:121`)
  - 그 epoch 마지막 micro-batch가 text-only면 projector가 checkpoint에서 빠지고, `strict=False` 로드 `U main.py:45`가 조용히 넘김
  - 재구현: requires_grad를 학습 중 바꾸지 않음 (쓰이지 않은 parameter는 grad가 None, 14.6.3), 저장은 고정 이름 목록, 로드는 strict
- **optimizer 상태 버림:** `checkpoint.pop('optimizer_states')` `U :298` → 재개 불가. 재구현은 optimizer·scaler·난수 상태까지 저장
- **router lr ×0.3** `U :230, :253-267`: BatchNorm CNN router용 값 → D19 변형(`variant_released_gate_lr`)으로만
- **기록 부족:** loss·lr만 진행 표시줄로 `U :145-147`. GradScaler skip·grad norm은 기록되지 않음
- **EarlyStopping** `U main.py:16-21` (patience 10)은 5 epoch에서 효과 없음 → 두지 않음
- **반올림된 파일 이름 지표** `'{epoch:02d}-{metric:.3f}'` `U main.py:26` → 선택은 정수 정답 수로 (14.6.7)

### 14.6 설계

#### 14.6.1 공개 API (`rebuild/ilora/train.py`)
```python
class TrainingError(RuntimeError): ...

GROUP_ORDER: Final = ("projector", "gate", "lora")

@dataclass(frozen=True)
class StepCounts:                    # 14.6.2, 표본 수에서 계산
    n_samples: int; micro_batch: int; accum: int; epochs: int
    micro_per_epoch: int             # n_samples // micro_batch (drop_last)
    tail: int                        # micro_per_epoch % accum (0이면 부분 창 없음)
    steps_per_epoch: int             # ceil(micro_per_epoch / accum)
    total_steps: int                 # T
    warmup_steps: int                # w = T // warmup_divisor
    total_micro: int                 # epochs × micro_per_epoch (D3의 T)
    def window_sizes(self) -> tuple[int, ...]   # 한 epoch의 창 크기: accum 반복, 마지막은 tail (D24). M13 smoke·M14 18.8 #5가 씀

@dataclass(frozen=True)
class StepRecord: ...                # M12 step_fields(K) 순서의 행 (14.6.8, 16.6.2. 계약 추가 제안). ω 평균은 K개 tuple 필드 omega_mean 하나, as_row()가 omega_mean_0..K−1 열로 펼침
@dataclass(frozen=True)
class TrainSummary: ...              # 14.6.7 (계약 추가 제안)

def step_counts(n_samples: int, tc: TrainConfig) -> StepCounts
def lr_at(step: int, sc: StepCounts, tc: TrainConfig) -> float
def build_optimizer(groups: Mapping[str, Sequence[tuple[str, nn.Parameter]]], tc: TrainConfig, *,
                    model: nn.Module | None = None) -> torch.optim.Adam   # model이 있으면 그룹 밖 학습 parameter도 거부 (14.6.3)
class SkipGuard:                     # 연속 skip 감시, 순수 로직
    def __init__(self, limit: int) -> None
    def update(self, step: int, skipped: bool, scale_before: float, scale_after: float, *,
               grad_norm: float | None = None) -> None   # grad_norm: 중단 메시지의 마지막 grad norm (14.6.8)
def save_checkpoint(ckpt_dir: Path, model, optimizer, scaler, state: Mapping[str, object]) -> dict[str, str]
def load_checkpoint(ckpt_dir: Path, model, optimizer, scaler, cfg: Config, *,
                    provenance: Mapping[str, object] | None = None) -> dict[str, object]   # 비교할 현재 값, 기본 current_provenance(cfg, model)
def read_trainable_state(ckpt_dir: Path) -> dict[str, Tensor]   # 마커·sha256 확인 뒤 trainable.pt 전체 (optimizer 없이). M11 eval 명령·M12 export도 씀
def load_model_state(ckpt_dir: Path, model) -> str               # read_trainable_state → generated_frozen.* bit 비교 → model.load_trainable_state(나머지, strict=True). trainable sha256 반환
def select_epochs(val: Mapping[int, SplitMetrics]) -> tuple[int, int]      # (best, last), D11
def train(cfg: Config, run_dir: Path, *, repo_root: Path, llm_path: Path | None,
          resume: bool = False) -> TrainSummary
```
- autocast 문맥과 GradScaler는 M0 precision 도구(4.2)로 만들고 값은 `train` 구역에서 읽음: `autocast_for(precision, device)`(M11 15.6.3도 같은 함수)와 `make_grad_scaler(tc, device)` (`fp16_amp`일 때만 `enabled=True`). 두 함수는 M0 `rebuild/ilora/precision.py`에 있음 (3.1 precision 도구, 14.6.10 제안 5가 반영됨). `fp16_amp`인데 장치가 CUDA가 아니면 두 함수가 `ValueError`를 냄 (torch 2.0이 autocast·scaler를 경고만 내고 끄는 대신) `R ilora/precision.py autocast_for`
- `llm_path`는 `model.backbone = llama2_7b`일 때만 필수 (M9 13.6.1). `TrainingError` 밖의 오류(M1–M9, M11, M12, M15)는 그대로 올림
- 위 블록에 더한 인자(`model=`, `grad_norm=`, `provenance=`)는 keyword 전용이고 기본값이 있음. 다른 모듈이 쓰는 보조 공개 이름 `[결정]`: `StepCounts.window_sizes()`, `epochs_to_test(best, last)`(test 순서, M13 smoke), `segment_problems(segments, resumes)`(14.6.10, M10-15 시험), `current_provenance(cfg, model, *, repo_root, llm_path, tokenizer_dir)`(14.6.6 읽기), `accumulate_window`·`train_window`(14.6.4 5단계), `frozen_checksum`, `tensor_digest`(14.6.8), `find_resume_point`(14.6.10), 상수 `STATE_KEYS`·`TRAIN_STATE_FORMAT` `R ilora/train.py`

#### 14.6.2 학습 길이와 lr 일정
- `step_counts` (full, 계산 `design/tmp/w5_counts.py`)
  - micro_per_epoch = 66,126 // 8 = 8,265 (epoch마다 6개 표본 제외, 어느 6개인지는 그 epoch의 permutation이 정함)
  - 8,265 = 16 × 516 + 9 → 창 517개 (마지막 창 9개), T = 5 × 517 = 2,585, w = 2,585 // 20 = 129, total_micro = 41,325
  - epoch e: τ = 8,265e + 1 … 8,265(e + 1), s = 517e … 517e + 516
  - micro_batch > n_samples, steps_per_epoch < 1, w < 1이면 `TrainingError` (smoke 오설정)
- `lr_at(s)`, s = 0부터 센 optimizer step (건너뛴 step 포함)
  - s < w: `start + (peak − start) · s / w`
  - s ≥ w: `floor + (peak − floor) · 0.5 · (1 + cos(π · (s − w) / (T − w)))`
  - paper_spec 값: s = 0 → 1e-6, 64 → 5.011628e-05, 128 → 9.923256e-05, 129 → 1e-4, 516 → 9.405772e-05, 1,292 → 5.461093e-05, 2,584 → 1.000040e-06
- 그룹 lr = `lr_scale_{g} × lr_at(s)` (g ∈ {projector, gate, lora}, 14.6.9). 식이 (시작, 최대, 최솟값)에 선형이라 공개 코드처럼 세 값에 배수를 곱한 것과 같음 `U model/model_interface.py:253-267`
- 창 s의 첫 micro-batch 전에 모든 그룹 lr을 설정하고 창 안에서는 바꾸지 않음
- GradScaler가 건너뛴 step도 s를 1 올림 `[결정]`: lr·τ가 수치 사건과 무관하게 계획대로 진행

#### 14.6.3 optimizer와 precision
- 그룹 (D19): M9 `param_groups()`(14.6.10)가 준 이름 정렬 목록 그대로, 순서 `GROUP_ORDER`

| 그룹 | 내용 (parameter 수) | lr 배수 | weight decay |
|---|---|---|---|
| `projector` | M6 두 층 (17,047,552) | 1.0 | 1e-5 |
| `gate` | W_g [4, 64] (256). dynamic에만 | 1.0 | 1e-5 |
| `lora` | 224 모듈의 expert A·B (19,988,480) | 1.0 | 0 |

- cluster_hard·K = 1에는 `gate` 그룹이 없음 (`[SPEC §11]` gate parameter를 optimizer에서 제외)
- 합 (계산 `design/tmp/w5_numbers.py`): dynamic 37,036,288, cluster_hard·K = 1 37,036,032. fork `trainable_params` 37,184,148과의 차이 147,860 = CNN router 90,772 + 안 쓰인 모듈별 Dense gate 224 × 256 − W_g 256 (`U model/peft/tuners/gating.py:5-16`, `U model/peft/tuners/moelora.py:604-605`)
- `torch.optim.Adam(groups, betas=(tc.adam_beta1, tc.adam_beta2), eps=tc.adam_eps, amsgrad=False)`. 그룹 dict에 `"name"` 키 (state_dict에 남음). foreach는 torch 기본 (CUDA면 foreach)
- 생성 전 검사 (어기면 `TrainingError`)
  - 학습 parameter가 정확히 한 그룹에만, 모두 fp32·`requires_grad=True`·같은 device (GradScaler는 fp16 grad의 unscale을 거부함 `torch2.0 cuda/amp/grad_scaler.py:211-212`)
  - 그 밖의 모든 parameter는 `requires_grad=False` (base fp16·SASRec fp32 여부는 M9·M3 검사)
  - 그룹 이름 집합 = {projector, lora} ∪ ({gate} if dynamic)
- precision 모드 (`train.precision`)
  - `fp16_amp` (paper_spec, D9): forward만 `torch.autocast("cuda", dtype=torch.float16)` 안. backward·unscale·step은 밖. GradScaler 사용
  - `bf16_amp` (D9 대안, 편차로 기록): autocast bfloat16, GradScaler 끔
  - `fp32` (CPU smoke 전용): autocast·GradScaler 없음. device가 CPU가 아니면 설정 오류
  - SASRec·gate 입력 z는 M9가 autocast를 끄고 fp32로 계산 (D23)
- 손실: M9 `train_forward(batch).loss`는 fp32 0차원 tensor (M9가 label 위치 logits를 `float()`로 바꿔 `F.cross_entropy` 평균, 13.6.5). 아니면 `TrainingError`
- `optimizer.zero_grad(set_to_none=True)` `[결정]`: 모든 micro-batch가 text-only인 창에서는 projector grad가 None으로 남아 Adam이 그 parameter의 moment·weight decay·갱신을 건너뜀 (`torch2.0 optim/adam.py:77`). torch 2.0 기본값(`torch2.0 optim/optimizer.py:435`)이라 공개 코드와 같은 동작이고, easy 손실에 projector가 없다는 식 7과 맞음

#### 14.6.4 학습 준비와 epoch 루프
**0. 준비 (새 학습, 순서 고정)** `[결정]`
1. M0 공통 준비(실행 폴더, `meta.json`, 환경 점검, `candidates.self_check()` (M2 6.6.5))는 `run_command`가 이미 했으므로 M10은 다시 부르지 않음 (3.1 명령 분기). `data = load_dataset(cfg, repo_root)` (M1), `templates = load_templates(run.resolve_path(cfg.prompt.template_path, repo_root))` (M4, 경로 해석은 3.1), `bundle = load_tokenizer(tokenizer_dir, cfg.batching.tokenizer_sha256)` (M5; `tokenizer_dir = commands.tokenizer_dir_for(cfg, llm_path)`: `llm_path`, CPU 시험·smoke는 `ILORA_TEST_TOKENIZER`, M5 9.2) `[결정]`: `train()`에 tokenizer 폴더 인자가 없어 M0 규칙을 그대로 씀. `run_command`에 준 `tokenizer_dir`는 `train()`에 전달되지 않음 (M11 eval 명령도 같음) `R ilora/train.py _Trainer.prepare`
2. cluster_hard만 (M15 19.6.7, `[SPEC §9, §11]`): `artifacts = load_cluster_artifacts(cfg.cluster.artifact_dir, expected_ids=train·val·test 표본 id 전체, sasrec_sha256=변환 state 파일 sha256, data_sha256=M0 meta 입력 해시, ccfg=cfg.cluster)` → `mapping = parse_expert_mapping(cfg.cluster.expert_mapping, n_clusters=artifacts.n_clusters, num_experts=K)` → `hard_gate = ClusterHardGate(artifacts, mapping, K)`. dynamic·none은 M15를 부르지 않음 `[SPEC §22]` (M7 11.8 #8)
3. `model = build_model(cfg, bundle, data.catalog, llm_path=llm_path, hard_gate=hard_gate)` (M9, cluster_hard 밖은 `hard_gate=None`). cluster_hard면 이어서 `verify_live_encoder(artifacts, model.sasrec, data.train, device=model.device, n=cfg.cluster.live_check_n, atol=cfg.cluster.live_check_atol)` (M15, 난수 없음). 그 결과 dict를 `meta.json`의 `cluster_live_check`에 씀 `[결정]`: 19.6.7·19.8 #12가 meta 기록만 정하고 키를 정하지 않음. M13 S5와 M15-12 시험이 이 키를 읽음 `R ilora/train.py _Trainer.prepare`
4. 그룹 이름이 routing 방식과 맞는지 확인(`gate`는 dynamic만, `[SPEC §11]`)한 뒤 `optimizer = build_optimizer(model.param_groups(), cfg.train, model=model)` (그룹 밖 학습 parameter 거부, 14.6.3), `scaler = make_grad_scaler(cfg.train, model.device)` (M0 precision 도구). 재개 때 비교할 현재 출처 값 `current_provenance(...)`(14.6.6)도 여기서 한 번 계산
5. `recorder = TrainRecorder(run_dir, cfg.records, num_experts=K, total_micro_batches=sc.total_micro, total_steps=sc.total_steps)`. `records.activation_max`면 `monitor = ActivationMaxMonitor(model, MoELoRALinear)` → `monitor.attach()` → `recorder.attach_monitor(monitor)` (M12 16.6.2–16.6.3, M8 12.6.1. 기록기가 모듈 index를 이름으로 바꾸고 `close`에서 모듈별 최대를 씀)
6. 고정 parameter checksum과 SASRec 내용 digest(M3 `state_digest(model.sasrec.state_dict())`, `model.sasrec.state_digest`와 같아야 함)의 시작 값 계산 (14.6.6), 그다음 `seed_torch_global(seed, "torch_global")` (14.6.5)
7. 학습·선택·test 전체를 `try/finally`로 감쌈: `finally`에서 `monitor.detach()`, `recorder.close(status, summary)` (status는 `completed`/`failed`, summary는 그때까지의 `TrainSummary` 필드, M12 16.6.2)

**epoch 루프 (epoch e)**
1. `plan = build_batch_plan(data.train, e, cfg)` (M4·M5). 검사: 길이 = micro_per_epoch, τ가 tau_done + 1부터 연속, `sample_indices` 길이 B, epoch 안 중복 없음, `hybrid`는 bool
2. 로그: `plan.digest()`(M4 8.6.6), plan의 hybrid micro-batch 수와 기대값 Σ p(τ) (epoch 0–4: 826.6, 2,479.6, 4,132.6, 5,785.6, 7,438.6, 계산 `design/tmp/w5_numbers.py`). 비교만 하고 판정하지 않음
3. `batches = make_train_loader(plan, data, cfg, e, bundle=bundle, templates=templates)` (M5 9.6.1): `TrainBatch`를 plan 순서로 냄
4. `model.train()` 뒤 `model.sasrec_is_eval()`이 아니면 `TrainingError` (D12). monitor가 있으면 `monitor.resume()`
5. plan을 앞에서부터 A개씩 창으로 자름 (마지막 창은 tail개). 창마다:
   1. n = 창 크기. 모든 그룹 lr 설정 (14.6.2). `c0 = model.omega_source.gate_calls` (M9 13.6.1 속성 이름, M7 `Router`), cluster_hard면 `l0 = hard_gate.lookups`
   2. micro-batch마다: `batch = next(batches)`
      - `batch.tau`·`batch.hybrid`가 plan 항목과 다르거나, `batch.sample_ids`가 `[data.train[j].sample_id for j in sample_indices]`와 다르거나, 응답 토큰이 없는 행이 있으면(`(labels != -100).any(1)`이 거짓, CPU에서 검사) `TrainingError`
      - autocast 안에서 `out = model.train_forward(batch)` (device 이동은 M9)
      - `scaler.scale(out.loss / A).backward()` (D24: A = `grad_accum` = 16. 마지막 창의 9개도 16으로 나눔, 14.4) `R ilora/train.py accumulate_window`
      - monitor가 있으면 `act = monitor.end_micro_batch(batch.tau)` (장치 위 0차원 tensor 2개, 동기화 없음, M12 16.6.3)
      - 기록용 손실·ω는 detach해 device에 모으고 창 끝에 grad norm과 함께 한 번 CPU로 옮김 (창마다 동기화 1회). act는 장치 위 0차원 tensor 그대로 기록기에 넘기고 flush 때 옮김 (M12 16.6.3)
   3. `scaler.unscale_(optimizer)` → 그룹별·전체 grad L2 norm (fp32, grad가 None인 parameter 제외)
   4. grad 유무 검사: projector grad 있음 ⇔ 창에 hybrid micro-batch가 있음, dynamic이면 gate grad 항상 있음, LoRA grad 항상 있음. 어긋나면 `TrainingError`
   5. `before = scaler.get_scale()`; `scaler.step(optimizer)`; `scaler.update()`; `after = scaler.get_scale()`; `skipped = after < before` (fork 측정 도구와 같은 판정 `F measurement/schema.py:75`)
   6. GradScaler가 꺼진 모드에서 grad norm이 유한하지 않으면 5단계의 `scaler.step` 전에 `TrainingError` `[결정]`: 비유한 갱신이 parameter에 닿기 전에 멈춤 `R ilora/train.py train_window`
   7. `optimizer.zero_grad(set_to_none=True)`. CUDA면 `max_memory_allocated`·`max_memory_reserved`를 읽은 뒤 `reset_peak_memory_stats` (step 구간 최대, M12 `max_mem_gib`)
   8. 창의 micro-batch 행 n개와 `StepRecord` 1행을 `recorder.log_micro_batch`·`log_optimizer_step`으로 넘김 (14.6.8). `gate_calls = model.omega_source.gate_calls − c0`, `cluster_lookups = hard_gate.lookups − l0`(cluster_hard 밖은 0): 학습 창 안의 증가분만 (평가 호출 제외). 그다음 `guard.update(..., grad_norm=)`와 학습 parameter 유한 확인(동기화 1회, 14.6.8의 '즉시') `[결정]`: 행을 먼저 넘겨야 연속 skip으로 멈추는 step의 행도 `close("failed")`가 파일에 남김 `R ilora/train.py _Trainer.run_epoch`. s += 1. s % k == 0이거나 epoch 마지막 step이면 `run.log` 요약
6. epoch 끝 검사: `batches`가 끝났는지 (더 내면 오류), s == steps_per_epoch·(e + 1), tau_done == micro_per_epoch·(e + 1) (paper_spec 517(e + 1), 8,265(e + 1)), 고정 parameter checksum과 SASRec `state_digest`가 학습 시작 값과 같음 (14.6.6, M3 7.8 #13). 학습 parameter 유한은 step마다 확인함 (5.8). 이때 다시 계산한 checksum을 그 epoch checkpoint의 `frozen_checksum_at_save`로 저장 (14.6.6)
7. `recorder.flush()` → 저장 → validation → `recorder.log_epoch(...)` → 마무리 (14.6.6, 14.6.7)

#### 14.6.5 난수 순서
- 새 학습: 모델·optimizer·scaler를 만든 뒤, 첫 micro-batch 직전에 M0 `seeding.seed_torch_global(seed, "torch_global")` 1회 (= `torch.manual_seed(derive_seed(seed, "torch_global"))`, CPU·CUDA 기본 generator, `rebuild/ilora/seeding.py`). `build_model` 다음이어야 함 (M9 13.6.2). 모델 초기화는 `("init", 이름)` key의 `torch_seed_scope`(CPU generator만 seed·복원)라 이 순서와 무관. cluster_hard 준비(`load_cluster_artifacts`, `verify_live_encoder`)는 난수를 쓰지 않음
- 학습 중 전역 torch 난수를 쓰는 것: LoRA dropout(M8)과 epoch마다 train DataLoader iterator가 만드는 base seed 1개 (`torch2.0 utils/data/dataloader.py:605`, CPU generator, worker 수와 무관). worker 안 난수는 쓰지 않음 (후보·템플릿·curriculum은 key, M2·M4)
- Python `random`·numpy 전역 난수는 쓰는 곳이 없어야 함 → seed하지도 저장하지도 않고 14.8 #12로 미사용을 확인
- 평가(M11)는 `torch.random.fork_rng`로 감싸 전역 상태를 바꾸지 않음 → 저장이 validation 앞이어도 재개가 같음
- 재개: 다시 seed하지 않고 `train_state.pt`의 CPU·CUDA 상태를 복원
- GPU 연산 비결정성은 허용 (D16)

#### 14.6.6 checkpoint (`checkpoints/epoch{e}/`)
- `trainable.pt`: `{이름: tensor}` (CPU, contiguous)
  - M9 `trainable_state()`의 학습 parameter 전부 (LoRA expert A·B, projector, gate). 이름 목록은 모델 생성 때 고정, 저장 때 `requires_grad`로 거르지 않음
  - `generated_frozen.*`: resize로 생긴 새 토큰 5개의 입력 임베딩·lm_head 행 (M9 `generated_frozen_state()`). 학습하지 않지만 우리 코드가 만든 값이라 넣음 (1.3)
- `train_state.pt`
  - `format_version` (`ilora-rebuild-train-state-v1`), `epoch` (끝난 epoch), `global_step` = steps_per_epoch·(e + 1), `tau_done` = micro_per_epoch·(e + 1) (paper_spec 517(e + 1), 8,265(e + 1))
  - `optimizer` (`state_dict()`), `group_param_names` (그룹별 이름 순서), `scaler` (`state_dict()`: scale, growth_factor, backoff_factor, growth_interval, _growth_tracker)
  - `rng`: `torch.get_rng_state()`, `torch.cuda.get_rng_state(device)` (CPU 실행이면 None)
  - `config_hash` (M0), `trainable_sha256`, `frozen_checksum` (학습 시작 때 고정 parameter마다 `torch.sum(p, dtype=torch.float64)`), `frozen_checksum_at_save` (그 epoch 끝 검사에서 다시 계산한 값. 시작 값과 같아야 저장되고 M13 smoke `frozen_sasrec`가 epoch마다 읽음, 17.6.7 #2), `input_hashes` (`inputs` = M0 meta의 입력 6개(데이터 4개, prompt, 원본 SASRec)와 변환 SASRec state·tokenizer model·LLaMA `config.json`(sha256, `torch_dtype`)·cluster manifest의 sha256)
  - `router_provenance` (M7 `provenance()`: routing_mode, K, z_dim, gate 모양 11.6.7), `sasrec_state_digest` (M3 7.6.8), cluster_hard면 `cluster` (manifest sha256, `expert_mapping`, `unit`, `n_clusters`, `fit_split`, M7 11.6.7·M15 19.6.7)
  - `counters`: 누적 skip·비유한 손실 수, 학습 창의 `gate_calls`·`cluster_lookups` 누적, 학습 시간 `train_s`, 연속 skip 감시 상태 `skip_guard`(재개 뒤에도 연속 수를 이어 셈)
  - `epoch_stats` (그 epoch 행의 값: plan digest, hybrid 수, 손실 평균, skip 수, activation 최대, 메모리, 학습 시간. `STATE_SAVED`만 있는 epoch를 재개할 때 epoch 행을 다시 만듦), `step_counts` (14.6.2)
- 쓰기 `[결정]`: 두 파일 모두 메모리에서 `torch.save`한 byte를 임시 이름에 쓰고 fsync → `os.replace` (`torch.load(weights_only=True)`로 읽힘). torch 2.0은 경로로 저장하면 파일 이름(마지막 확장자를 뺀 것)을 zip archive 접두어로 넣어 같은 상태도 임시 이름에 따라 byte·sha256이 달라짐 (측정: `trainable.pt` → `trainable/`, `trainable.pt.tmp` → `trainable.pt/`, 메모리 → `archive/`, 같은 상태 두 번은 같은 byte, `design_sync/g4/save_name_probe.py`). 그래서 sha256을 쓰기 전에 알고, `load_model_state`·`load_checkpoint`는 로드한 모델로 같은 byte를 다시 만들어 파일 sha256과 비교함 `R ilora/train.py _torch_bytes`. 저장 폴더가 비어 있지 않으면 `TrainingError`. 두 파일 뒤 `STATE_SAVED`, validation 뒤 `val_metrics.json`, 마지막에 `COMPLETE`. 셋 다 UTF-8 JSON(LF)이고 덮어쓰지 않음 `R ilora/train.py write_marker`: `STATE_SAVED` = {epoch, files {파일: sha256} 2개, seconds, utc}, `val_metrics.json` = {epoch, ckpt_tag, metrics (M11 `SplitMetrics`), eval_dir, records_sha256, seconds {train, save, val}, max_mem_gib, max_mem_reserved_gib}, `COMPLETE` = {epoch, files 3개, utc}
- 크기 (계산): trainable 약 141 MiB, Adam 상태 약 283 MiB → epoch당 약 424 MiB, 5 epoch 약 2.07 GiB. 실행 폴더에만 (`runs/`는 git 제외 `[SPEC §0.10]`)
- 읽기는 strict (어기면 `TrainingError`)
  - 마커와 sha256 일치, `format_version`, `config_hash`, `input_hashes`, `router_provenance`, `sasrec_state_digest`, `cluster`가 현재 값과 같음
  - trainable 이름 집합·모양·dtype이 모델과 정확히 같음 (빠짐·남음 모두 오류). `generated_frozen.*`가 새로 만든 모델 값과 bit 단위로 같음
  - `group_param_names`가 현재 그룹과 같을 때만 `optimizer.load_state_dict` (torch는 위치로 대응)
  - `frozen_checksum`(학습 시작 값)이 지금 모델의 값과 같음. 고정 parameter = 학습 그룹 밖의 모든 parameter(LLaMA base·입력 임베딩·lm_head, SASRec) `[결정]`: 문서가 정의하지 않았고, 다른 base 모델이나 SASRec 위에서 이어 학습하지 않게 함 (14.7) `R ilora/train.py frozen_checksum`
  - GradScaler 상태의 유무가 현재 scaler의 켜짐과 맞고, CUDA 실행이면 저장된 CUDA 난수 상태가 있음. `train_state.pt` 키가 하나라도 없으면 거부 (형식이 다른 이전 checkpoint는 재개하지 않음)
  - 비교할 현재 값(config hash, 입력 해시, provenance, SASRec digest, cluster 출처)은 `train()`이 준비 때 한 번 계산해 `load_checkpoint(..., provenance=)`로 넘김 `R ilora/train.py current_provenance`
  - 로드 뒤 trainable sha256을 다시 계산해 비교
- 모델 상태만 읽는 두 함수 (optimizer·scaler·난수 없음, test·M11 eval 명령·M12 export가 씀)
  - `read_trainable_state(ckpt_dir)`: `COMPLETE`(test·eval 명령) 또는 `STATE_SAVED`(재개 중 validation)와 `trainable.pt` sha256을 확인하고 `{이름: tensor}` 전체를 돌려줌 (`generated_frozen.*` 포함). M12 export는 이 중 `omega_source.` 접두어 키만 씀 (M9 13.6.8)
  - `load_model_state(ckpt_dir, model)`: `generated_frozen.*`를 `model.generated_frozen_state()`와 bit 단위로 비교한 뒤 나머지를 `model.load_trainable_state(..., strict=True)` (M9 13.6.8). 다시 계산한 trainable sha256 반환

#### 14.6.7 validation, 선택, test (D11)
- epoch e: `STATE_SAVED` 뒤 `monitor.pause()` → `evaluate_split(model, bundle, data, "val", cfg, run_dir, ckpt_tag=f"epoch{e}", ckpt_meta=…, templates=templates)` (M11 15.6.1) → `monitor.resume()`. `ckpt_meta` = {`source_run`, `checkpoint_dir`, `epoch`, `tags`, `trainable_sha256`}: M11 eval 명령(15.6.11)과 같은 키, `tags`는 validation `[]`, test는 해당하는 `best_val`·`last` `R ilora/train.py _Trainer.ckpt_meta`. M11이 `eval/epoch{e}/val/`에 씀 (M11 15.6.8). `EvalResult.metrics`를 `recorder.log_epoch`(M12 `epochs.csv`)와 `val_metrics.json`에 쓰고 마지막에 `COMPLETE`
  - 재개 때 `eval_dir(run_dir, f"epoch{e}", "val")`이 완료 상태(`metrics.json` 있음)면 `load_eval_result`로 읽고 다시 생성하지 않음 (M11 15.6.11)
- 5 epoch 뒤 `select_epochs`
  - best = argmax_e (정답 수_e, e). CombinedMetric = 정답/94라 N이 같아 정수로 비교 (epoch마다 N이 다르면 `TrainingError`). 동률이면 늦은 epoch. `hr × valid_ratio` 부동소수 곱(`U model/model_interface.py:181`)으로 비교하지 않음
  - last = 마지막 epoch (4)
- test: {best, last}를 오름차순으로 `load_model_state(checkpoints/epoch{e}, model)`(14.6.6)로 strict 로드한 뒤 `evaluate_split(..., "test", ckpt_tag=f"epoch{e}")` → `eval/epoch{e}/test/`. best == last면 1회
- test는 마지막 epoch `COMPLETE` 전에는 부르지 않음 (D11 "test로 고르지 않음"). 어기면 `TrainingError`
- `eval/selection.json` (M11 eval 명령·M12 export가 읽는 키 고정): `rule`(규칙 문장), `val` {epoch: {`n_total`, `n_valid`, `n_correct`, `valid_ratio`, `hr_at_1`, `combined_metric`}}, `best_val_epoch`, `last_epoch`, `test_dirs` {`best_val`: `eval/epoch{b}/test`, `last`: …}, `primary: "best_val"`. 재개한 실행에 이미 있으면 다시 계산한 내용과 같아야 하고 덮어쓰지 않음 (다르면 `TrainingError`)
- `TrainSummary` (frozen dataclass, M12 `run_summary.json`의 `train_summary`에 그대로): `status`, `routing_mode`, `num_experts`, `counts`(`StepCounts`), `epochs_complete`, `best_val_epoch`, `last_epoch`, `val` {e: `SplitMetrics`}, `test` {e: {`tags`, `metrics`, `aux_protocols`, `eval_dir`}}, `trainable_params` {`projector`, `gate`(dynamic만), `lora`, `total`}, `seconds` {`train`, `validation`, `save`, `test`, `total`}, `skipped_steps`, `nonfinite_loss`, `gate_calls`, `cluster_lookups`, `epoch_max_mem_gib`, `unused_experts`(cluster_hard에서 어느 군집도 받지 않은 expert, 19.6.7), `resumes` → `[SPEC §12]` TrainableParams·ElapsedTime
  - `seconds`는 끝난 epoch마다 `val_metrics.json`에 남긴 학습·저장·validation 시간과 test 평가 시간(이미 끝난 폴더를 다시 읽을 때도 그 평가 시간)의 합 `[결정]`: 재개를 거쳐도 이어지고, 멈춰 있던 시간과 중단된 epoch의 계산은 들지 않음 (18.6.7 ElapsedTime) `R ilora/train.py _Trainer.summary`

#### 14.6.8 기록과 중단 규칙
- 열 이름·순서·단위와 파일 쓰기는 M12 `TrainRecorder`(16.6.2) 한 곳에서 정함. M10은 값을 계산해 행으로 넘기고 step마다 파일을 쓰지 않음 `[SPEC §4]`
  - micro-batch 행 (`MICRO_FIELDS`): τ, epoch, 속한 step, hybrid, curriculum 확률, B, L, `TrainForward.n_label_tokens`, `TrainForward.loss`(나눗셈 전), 유한 여부, activation 최대값과 모듈 (monitor가 꺼져 있으면 None)
  - `StepRecord` (`step_fields(K)`): `step, epoch, tau_first, tau_last, n_micro, n_hybrid, loss_mean, loss_hybrid_mean, loss_text_mean, nonfinite_loss, lr_lora, lr_projector, lr_gate, grad_norm_lora, grad_norm_projector, grad_norm_gate, grad_norm_total, projector_has_grad, scale_before, scale_after, skipped, omega_mean_0..omega_mean_{K−1}, omega_entropy_mean, gate_calls, cluster_lookups, max_mem_gib, max_mem_reserved_gib, time_s`
  - epoch 행 (`EPOCH_FIELDS`): 마지막 step, τ, plan digest, hybrid 수, 학습 loss 평균, skip 수, val (N, 유효, 정답, 세 지표), activation 최대, epoch 안 step 구간 메모리 최대 (allocated·reserved), 학습·validation·저장 시간
  - 없는 값(gate 없음, 그 창에 text-only 없음, CPU의 메모리)은 None
- `run.log` 요약 (k = `log_every_steps`): step/T, epoch, τ와 그 τ의 hybrid 확률 (`linear`이면 p(τ) = τ/41,325, D3), 구간 평균 loss (hybrid·text-only 따로), 그룹별 lr, scale, 구간·누적 skip, grad norm (마지막·구간 최대), hybrid micro-batch 수, ω 평균, micro-batch/s, 구간 최대 메모리
- epoch 요약: 학습·validation·저장 시간, skip 수, 최대 메모리 (step마다 읽고 초기화한 값의 epoch 최대, 14.6.4 5.7), val 지표
- 학습 시작·끝 UTC와 epoch별 경과 시간은 M0 `run.update_meta`로 `meta.json`의 `train` 블록에 씀 `[결정]`: `segments`(프로세스마다 {`started_utc`, `resume`, `ended_utc`, `status`}, 14.6.10), `epochs` {e: 학습·저장·validation 초, `max_mem_gib`, `max_mem_reserved_gib`, `ended_utc`}, `step_counts`, `trainable_params`, `parameter_report`(M9), `generated_frozen_sha256`(새 토큰 행 3개 tensor의 `key|dtype|shape`와 raw byte에 대한 sha256. 13.8 #14가 meta 기록을 요구하나 쓰는 곳을 정하지 않음, bf16에서도 계산됨), `summary`(best·last, 끝난 epoch 수, `seconds`) `R ilora/train.py tensor_digest`
- **중단 규칙** `[결정]`: 연속 skip이 `max_consecutive_skips` = 16에 닿으면 `TrainingError` (step 범위, scale 변화, 마지막 grad norm을 메시지에)
  - 16 = log2(init_scale): 16번 연속 반으로 줄면 scale이 직전의 1/65,536이 되므로 그 뒤의 overflow는 loss scaling이 아니라 순전파·역전파 값 자체의 문제
  - 참고 측정: fork의 lr 1e-4 fp16 1 epoch 진단에서 skip 4회 (step 1, 2, 77, 98), 최대 연속 2회 (`F experiments/IMPLEMENTATION_NOTES.md:535`)
- 비유한 손실은 기록만 하고 역전파는 그대로 (GradScaler가 그 step을 건너뜀). 학습 parameter가 비유한이 되면 즉시 `TrainingError`

#### 14.6.9 설정 (`train` 구역)
| 키 | 타입 | paper_spec 값 | 라벨 | 설명 |
|---|---|---|---|---|
| `epochs` | int ≥ 1 | 5 | `[v1 App E p.16]` `[LLaRA §5.1.2 p.7]` D11 | epoch 수 |
| `micro_batch_size` | int ≥ 1 | 8 | `[결정]` 값은 `[코드]` `U train_movielens.sh:5` (`U main.py:82` 기본값 4와 다름, 4.4) | micro-batch 크기, drop_last |
| `grad_accum` | int ≥ 1 | 16 | `[결정]` 값은 `[코드]` `U train_movielens.sh:6` (`U main.py:86` 기본값 32와 다름). 곱 8 × 16 = 128은 `[v1 App E p.16]` `[LLaRA §5.1.2 p.7]` | 창 크기 A |
| `lr_peak` | float > 0 | 1e-4 | `[논문 App E p.17]` D5 | 최대 lr |
| `lr_warmup_start` | float > 0 | 1e-6 | `[v1 App E p.16]` `[LLaRA §5.1.2 p.7]` D6 | warmup 시작 |
| `lr_floor` | float ≥ 0 | 1e-6 | `[코드]` `U main.py:90` D6 | cosine 끝값 |
| `warmup_divisor` | int ≥ 1 | 20 | `[코드]` `U model/model_interface.py:248` D6 | w = T // 값 |
| `lr_scale_projector` | float > 0 | 1.0 | `[논문 App E p.17]` D19 | projector 그룹 lr 배수 |
| `lr_scale_gate` | float > 0 | 1.0 | `[논문 p.5 §3.2 (2)]` `(암시)` D19 | gate 그룹 lr 배수. `variant_released_gate_lr`는 0.3 |
| `lr_scale_lora` | float > 0 | 1.0 | `[논문 App E p.17]` D19 | LoRA 그룹 lr 배수 |
| `weight_decay_projector` | float ≥ 0 | 1e-5 | `[v1 App E p.16]` 값, 적용 범위 `[코드]` `U model/model_interface.py:228` D19 | Adam coupled L2 |
| `weight_decay_gate` | float ≥ 0 | 1e-5 | `[v1 App E p.16]` 값, 적용 범위 `[코드]` `U model/model_interface.py:230` D19 | Adam coupled L2 |
| `weight_decay_lora` | float ≥ 0 | 0.0 | `[코드]` `U model/model_interface.py:232` (키 없음 → Adam 기본 0) D19 | Adam coupled L2 |
| `adam_beta1` | float [0, 1) | 0.9 | `[코드]` torch 2.0.0 기본값, U·L 기본값 사용 | Adam β1 |
| `adam_beta2` | float [0, 1) | 0.999 | `[코드]` 같음 | Adam β2 |
| `adam_eps` | float > 0 | 1e-8 | `[코드]` 같음 | |
| `precision` | str (`fp16_amp`, `bf16_amp`, `fp32`) | `fp16_amp` | `[논문 App E p.17]` D9 | `fp32`는 CPU smoke 전용 |
| `scaler_init_scale` | float > 0 | 65536.0 | `[코드]` torch 2.0.0 GradScaler 기본값 (PL 1.8.6 native AMP) | |
| `scaler_growth_factor` | float > 1 | 2.0 | `[코드]` 같음 | |
| `scaler_backoff_factor` | float (0, 1) | 0.5 | `[코드]` 같음 | |
| `scaler_growth_interval` | int ≥ 1 | 2000 | `[코드]` 같음 | |
| `max_consecutive_skips` | int ≥ 1 | 16 | `[결정]` | 연속 skip 한계 (14.6.8) |
| `log_every_steps` | int ≥ 1 | 10 | `[결정]` | 요약 간격. 2,585 step에 요약 263줄 (10 step마다 258 + epoch 끝 5) |

- 그룹 값을 mapping 한 키로 두지 않고 그룹마다 스칼라 키 `[결정]`: M0 설정 로더(`rebuild/ilora/config.py` `_coerce`)는 bool·int·float·str·tuple·Optional만 받음. Adam β도 같은 이유로 스칼라 키 2개 (`adam_beta1`, `adam_beta2`, 최종 검토에서 목록 키를 나눔)
- `validate()`: `adam_beta1`·`adam_beta2` ∈ [0, 1), `lr_warmup_start ≤ lr_peak`, `lr_floor ≤ lr_peak`, `precision`과 `model.torch_dtype`·`model.device` 조합은 M9 13.6.11의 구역 사이 검사
- 설정 아님 (고정, 변형 없음): Adam(AdamW 아님)·clipping 없음·gradient checkpointing 없음 `[코드]`, 모든 micro-batch 손실을 `grad_accum`으로 나눔(마지막 부분 창도, D24 `[코드]`), 매 epoch validation·저장과 선택 규칙 (D11), 학습 중 requires_grad 고정 `[결정]`
- gate가 없는 프로파일과 `bf16_amp`·`fp32`에서도 `lr_scale_gate`·`weight_decay_gate`·scaler 키는 형식 검사만 하고 쓰지 않음 (M0 규칙상 모든 키 필수, 쓰지 않음을 INFO로 남김)
- 변형: `variant_released_lr` (lr_peak 8e-4, lr_warmup_start 8e-6, lr_floor 8e-6, D5), `variant_released_gate_lr` (`lr_scale_gate` 0.3, D19)

#### 14.6.10 재개, 연결, 계약 추가 제안
- **재개:** `python rebuild/main.py train --resume <실행 폴더>` (`--config`·`--set` 불가, 그 폴더의 `config.yaml`)
  1. `COMPLETE`인 마지막 epoch e를 찾음. 다음 폴더에 `STATE_SAVED`만 있으면 그것을 로드해 validation부터 마침. 마커가 없는 폴더는 `epoch{k}.incomplete-<UTC>`로 이름을 바꿔 보존 (이름이 겹치면 `-<n>`). `COMPLETE` epoch가 0..c−1로 이어지지 않거나 `STATE_SAVED`만 있는 폴더가 epoch c가 아니면 `TrainingError`, 저장된 epoch가 하나도 없으면 처음부터 (새 학습처럼 seed) `[결정]` `R ilora/train.py find_resume_point`. `STATE_SAVED`만 있는 epoch에 `val_metrics.json`이 남아 있으면(`COMPLETE` 직전 중단) `val_metrics.incomplete-<UTC>.json`(이름이 겹치면 `-<n>`)으로 옮긴 뒤 새로 쓰고(덮어쓰기 금지. Linux `rename`은 있는 파일을 덮으므로 번호로 피함), epoch 행은 `train_state.pt`의 `epoch_stats`로 다시 만듦 `R ilora/train.py _Trainer.after_save`, `R ilora/train.py _keep_stale_val_metrics`
  2. 새 학습과 같은 순서로 준비(14.6.4 0단계 1–5, cluster_hard 산출물 검사 포함)하고 14.6.6 규칙으로 로드, 난수 상태 복원 (다시 seed하지 않음)
  3. 학습 기록: 0단계 5의 `TrainRecorder(...)` 대신 M12 `TrainRecorder.resume(..., tau_done, global_step, epochs_complete)`(16.6.2): 로드한 상태의 경계(τ ≤ `tau_done`, step < `global_step`, `COMPLETE`가 있는 epoch) 밖의 행을 `*.incomplete-<UTC>.csv`로 옮기고 이어서 씀 (중복 행 없음)
  4. epoch e + 1부터 (s = steps_per_epoch·(e + 1), τ = micro_per_epoch·(e + 1) + 1). 모든 epoch가 끝난 실행이면 남은 선택·test만 (완료된 평가 폴더는 다시 생성하지 않음, M11 15.6.11)
  - epoch 중간 재개는 하지 않음 `[결정]`: 저장 상태를 epoch 경계에만 두어 단순하게
  - 살아 있는 실행은 다시 열지 않음 `[결정]`: M0 `run_command`는 명령 내내 `<실행 폴더>/run.lock`에 배타적 OS 잠금(Linux `fcntl.flock`, Windows `msvcrt.locking`)을 쥐고, `--resume`은 `meta.json`을 읽거나 쓰기 전에 그 잠금을 잡음. 다른 살아 있는 프로세스가 쥐고 있으면 `CommandError`이고 폴더는 그대로. 이유: 셸이 끊겨도 tmux 안의 학습은 살아 있고, 같은 폴더에 붙은 두 번째 프로세스가 기록 CSV를 경계까지 다시 쓰고 checkpoint 폴더를 옮기면 실행이 망가짐. 프로세스가 어떻게 끝나든 커널이 잠금을 풀어 오래 남는 잠금이 없음. `run.lock`은 지우지 않고 잡은 프로세스마다 한 줄(UTC, host, pid, 명령)을 덧붙임. 잠금을 지원하지 않는 파일 시스템이면 `run.log`에 경고를 남기고 진행 `R ilora/run.py hold_run_lock`, `R ilora/commands.py run_command`
  - `meta.json`의 `train.segments`는 프로세스마다 1개이고 `finally`에서 `ended_utc`·`status`를 적음. 재개는 이전 segment 가운데 `running`으로 남은 것(SIGKILL·OOM으로 `finally` 없이 끝난 프로세스. 잠금이 풀렸으므로 이미 끝남)을 `interrupted`로 바꾸고 재개 시각 `detected_utc`를 적음. 그 `ended_utc`는 측정하지 않은 값이라 null로 둠 `[SPEC §0.11]` `[결정]`. 끝난 실행의 segment 규칙(모두 `started_utc`, completed·failed는 `ended_utc`, interrupted 수 ≤ `running` 폴더의 재개 수, 마지막은 completed)은 `R ilora/train.py segment_problems`이고 M10-15 시험이 씀
- **메모리·시간 (예측하지 않음)**
  - fork 실행 로그 = 결과 보관 파일(sha256 `ede386ce…0398`)의 `logs/run_baseline_movielens_20261023T163202Z.log` (`F` 6c7ebac)
  - 고정 부분 (계산): base fp16 6,738,415,616 × 2 byte = 12.55 GiB (로그 `:100` "all params: 6,758,502,400" − 학습 20,045,824 − 새 토큰 행 40,960), 학습 parameter·grad·Adam 2개 = 37,036,288 × 16 byte ≈ 0.55 GiB
  - activation은 prompt 길이에 좌우 → `smoke_gpu_100`에서 epoch별 최대 메모리 측정 (VESSL_REQUIRED). 메모리가 모자라도 gradient checkpointing을 조용히 켜지 않고 오류로 멈춤
  - 참고 측정: fork (batch 8 × 16, gradient checkpointing 없음, A100-SXM4-80GB)의 "training elapsed: 30875s" (로그 `:855`). epoch 4가 8,240 micro-batch에서 끊긴 학습 + validation 4회. 우리 시간은 측정 뒤에만 기록 `[SPEC §0.14]`
- **연결:** M1 `LoadedData`, M2 `self_check`, M4·M5 `BatchPlan`·`TrainBatch`·`load_templates`·`load_tokenizer`, M9 모델, M11 `evaluate_split`·`EvalResult`·`eval_dir`·`load_eval_result`(15장), M12 `TrainRecorder`·`ActivationMaxMonitor`(16.6.2–16.6.3), M15 `load_cluster_artifacts`·`parse_expert_mapping`·`ClusterHardGate`·`verify_live_encoder`(cluster_hard만), M0 `derive_seed`·`seed_torch_global`·precision 도구·meta
- **계약 추가 제안**
  1. M9 모델 API: `param_groups() -> dict[str, list[tuple[str, nn.Parameter]]]` (이름 정렬, 키 `projector`, `gate`(dynamic만), `lora`), `train_forward(batch: TrainBatch) -> TrainForward(loss: Tensor[] fp32, n_label_tokens: int, omega: Tensor[B, K] fp32 detach, z)`, `trainable_state() -> dict[str, Tensor]`, `load_trainable_state(sd, strict=True)`, `generated_frozen_state() -> dict[str, Tensor]`, `sasrec_is_eval() -> bool` (M9 13.6.1과 같은 이름)
  2. M4·M5: `build_batch_plan(samples, epoch, cfg) -> BatchPlan`, `make_train_loader(plan, data, cfg, epoch, *, bundle, templates) -> Iterator[TrainBatch]` (M5 9.6.1. plan 순서, batch 안 표본 순서 = `sample_indices` 순서)
  3. 3.1에 `StepRecord`(14.6.8, 열은 M12 16.6.2)와 `TrainSummary`(14.6.7) 추가 (M10 → M12, M14)
  4. M0: `train --resume`이 기존 실행 폴더를 다시 여는 규칙 (상태가 `failed` 또는 끝나지 않은 `running`일 때만, 기존 파일 덮어쓰기 금지), `meta.json`에 `resumes` 목록 (시각, argv, 이전 상태·끝 시각·오류, 끝난 epoch, 시작 epoch). 그 폴더의 `run.lock`을 살아 있는 프로세스가 쥐고 있으면 거부 (위 '살아 있는 실행' 규칙)
  5. M0 precision 도구 두 함수 (4.2에 있으나 코드 없음): `autocast_for(precision: str, device: torch.device) -> ContextManager` (`fp16_amp` → `torch.autocast("cuda", float16)`, `bf16_amp` → bfloat16, `fp32` → 아무것도 안 함), `make_grad_scaler(tc, device) -> torch.cuda.amp.GradScaler` (`fp16_amp`일 때만 enabled, 값은 `scaler_*` 키). M10·M11이 같은 함수를 씀
  6. `read_trainable_state(ckpt_dir)`, `load_model_state(ckpt_dir, model)` (14.6.6): M11 eval 명령(15.6.11)과 M12 export(16.6.5)가 checkpoint를 읽는 유일한 경로. `eval/selection.json` 키 `best_val_epoch`·`last_epoch` 고정 (14.6.7, M11 15.6.14 제안 5·M12 16.6.11 제안과 같음)
  7. cluster_hard 준비 순서(14.6.4 0단계 2–3)를 M10·M11 eval 명령·M12 export가 같은 순서로 씀 (M7 11.6.9, M15 19.6.11)

### 14.7 결정
- 20장: D3 (plan을 그대로 소비), D5, D6, D9, D11, D12 (모드 확인), D16, D19, D24
- 이 장의 새 `[결정]`
  - 연속 skip 한계 16 (log2 init_scale), 요약 간격 10 step
  - 건너뛴 step도 lr·τ 일정에 셈
  - 학습 중 requires_grad를 바꾸지 않고 `zero_grad(set_to_none=True)`
  - 저장 → validation → `COMPLETE` 순서, 두 단계 마커, epoch 경계 재개만
  - 선택 비교는 정수 정답 수
  - checkpoint에 새 토큰 행과 고정 parameter checksum을 넣어 로드 때 확인, optimizer 그룹 순서·router provenance·SASRec digest·cluster 출처도 확인
  - `fp32` precision은 CPU smoke 전용, Python·numpy 전역 난수는 쓰지 않음
  - 학습 준비 순서 고정 (14.6.4 0단계): cluster_hard 산출물 검사 → `build_model(hard_gate)` → live z 대조 → optimizer → 기록기 → `torch_global` seed
  - 그룹별 lr 배수·weight decay는 스칼라 키 6개 (M0 로더가 mapping을 받지 않음)
  - 기록 행의 `gate_calls`·`cluster_lookups`는 학습 창 안 증가분 (평가 호출 제외), 메모리 최대값은 step마다 읽고 초기화
  - GradScaler가 없는 모드의 비유한 grad norm은 `scaler.step` 전에 멈추고, 기록 행을 넘긴 뒤 연속 skip을 감시하며, 학습 parameter 유한은 step마다 확인 (14.6.4 5단계)
  - checkpoint 파일은 메모리에서 직렬화한 byte로 써 sha256이 임시 이름과 무관, 고정 parameter = 학습 그룹 밖 전부, 남은 `val_metrics.json`은 옆 이름으로 보존 (14.6.6, 14.6.10)
  - 살아 있는 실행은 `run.lock`으로 재개를 거부하고, 닫히지 않은 segment는 `interrupted` (14.6.10)
  - `TrainSummary.seconds`는 끝난 epoch와 test의 계산 시간 합 (14.6.7)

### 14.8 완료 기준
1. (CPU) `step_counts(66,126)`: micro_per_epoch 8,265, tail 9, steps_per_epoch 517, T 2,585, w 129, total_micro 41,325. epoch별 τ·s 범위가 14.6.2와 같음
2. (CPU) `lr_at`이 14.6.2 값과 그 표기 자리(유효숫자 7자리)까지 같고, s ∈ [0, 2,584] 전부에서 독립 닫힌 식(warmup은 유리수 계산, cosine은 cos² 꼴)과 상대오차 1e-12 안이며 `U optims.py:20-36`를 옮긴 함수(cur_step = s, cur_epoch = s // 517, max_step = 2,585)와 정확히 같음. 모든 값이 [1e-6, 1e-4] 안
3. (CPU) optimizer: dynamic 그룹 (projector, gate, lora), wd (1e-5, 1e-5, 0), lr 배수 1, betas (0.9, 0.999), eps 1e-8, amsgrad False. cluster_hard·K = 1은 (projector, lora). fp16 parameter, 두 그룹에 든 parameter, 그룹 밖 학습 parameter를 각각 넣으면 `TrainingError`. Llama-2-7b 모양(hidden 4,096, intermediate 11,008, 32층)으로 센 그룹 크기 17,047,552 / 256 / 19,988,480. 설정 `train` 구역에 mapping 값(`{projector: …}`)을 넣으면 M0 로더가 `ConfigError`
4. (CPU, 초소형 모델 fp32) 크기 9 창의 누적 grad = (Σ loss_i)/16의 autograd grad (상대오차 1e-6). 크기 16 창도 같음 (나누는 수는 창 크기가 아니라 `grad_accum`, D24). `train_window`도 smoke_cpu의 크기 3 창(마지막 창)과 4 창을 모두 4로 나눔
5. (CPU) 모든 micro-batch가 text-only인 창 뒤 projector grad가 None, projector 값 bit 단위 그대로, Adam state step 그대로. hybrid가 하나라도 있으면 grad 있음. 대응이 어긋나게 만든 모의 모델은 `TrainingError`
6. (CPU) `SkipGuard(16)`: 15회 연속 skip 뒤 성공이면 통과, 16회 연속이면 `TrainingError`이고 메시지에 step 범위
7. (CPU) checkpoint 왕복: 저장 → 새 모델·optimizer·scaler에 로드 → 상태가 bit 단위로 같음. 키 빠짐·남음, 모양·dtype, `generated_frozen` 값, config hash, 그룹 순서, 고정 parameter checksum, sha256 불일치를 각각 넣으면 `TrainingError`
8. (CPU, 초소형, thread 1) 2 epoch 연속 학습과 1 epoch + 재개 + 1 epoch의 학습 parameter가 bit 단위로 같고, `records/train/` 세 CSV의 행도 같음 (재개 전에 경계 밖 행을 심어 두면 `*.incomplete-<UTC>.csv`로 옮겨짐). `STATE_SAVED`만 있는 폴더에서 재개하면 validation부터 이어감
9. (CPU) 선택: val 정답 수 [40, 45, 45, 44, 43] → best 2 (동률 중 늦은 것), last 4, test 호출 순서 [2, 4]. [40, 41, 42, 43, 45] → best = last = 4, test 1회
10. (CPU) test 평가는 마지막 epoch `COMPLETE` 뒤에만 호출됨 (호출 순서 기록으로 확인)
11. (CPU) `model.train()` 뒤 SASRec이 train 모드인 모의 모델은 `TrainingError`
12. (CPU) 1 epoch smoke 뒤 `random.getstate()`·`np.random.get_state()`가 시작과 같음. M11 평가 전후 torch CPU 난수 상태가 같음
13. (CPU) `StepRecord` 수 = 실행한 step 수, micro-batch 행 수 = 실행한 micro-batch 수, `n_hybrid` 합 = plan의 hybrid 수, k step마다 `run.log` 요약 줄. 행의 열이 M12 `MICRO_FIELDS`·`step_fields(K)`·`EPOCH_FIELDS`와 같음. dynamic의 `gate_calls` 합 = 학습 micro-batch 수 (validation 생성 호출 미포함)
14. (CPU, 초소형 + 합성 군집 산출물) cluster_hard 준비: `load_cluster_artifacts`·`ClusterHardGate`·`build_model(hard_gate=…)`·`verify_live_encoder` 순서로 불림 (호출 기록), optimizer에 gate 그룹 없음, 1 epoch 뒤 `gate_calls` 합 0이고 `cluster_lookups` 합 = 학습 표본 수(= micro-batch 수 × B). 산출물 해시를 1 byte 바꾸면 모델을 만들기 전에 실패. dynamic 실행에서는 M15 함수 호출 0회 `[SPEC §22]`
15. (VESSL_REQUIRED) fp16 A100: 첫 step `scale_before` 65,536, skip이 `StepRecord`에 남고 16회 연속 없음, 2,585 step·41,325 micro-batch 완료, epoch 5개 모두 validation(94명)과 `COMPLETE`, 시작·끝 시각과 epoch별 최대 메모리 기록
16. (VESSL_REQUIRED) grad hook으로 한 step의 grad에 NaN을 넣으면 그 step만 skip (scale 절반), 다음 step 정상
17. (VESSL_REQUIRED) epoch e checkpoint를 새 프로세스에서 로드해 validation을 다시 돌리면 학습 중 `eval/epoch{e}/val/records.jsonl`과 sha256이 같음
18. (VESSL_REQUIRED) fork 환경(PL 1.8.6)에서 `NativeMixedPrecisionPlugin(16, "cuda").scaler`의 (init_scale, growth_factor, backoff_factor, growth_interval)이 (65536, 2.0, 0.5, 2000)인지 출력 (14.4 계보 확인용, 결과와 무관). PL을 import할 수 없으면 이유를 `measurements.json`에 적고 문서화된 이유 `SKIP_NO_PL`로 skip (17.8 #2)
19. (VESSL_REQUIRED) CUDA fp16 실행의 재개: 17.6.12의 작은 GPU 학습을 2 epoch로 돌려 epoch 1 계획을 만들 때 실패시킨 뒤 `train --resume`하면, `load_checkpoint` 직후 GradScaler 상태(key 5개), cuda:0의 Adam 상태(step 값, moment bit 단위), CUDA·CPU 난수 상태가 `epoch0/train_state.pt`와 같음. 재개 뒤 첫 step의 `scale_before`가 저장된 scale과 같고 기록 행이 빈틈·중복 없이 이어짐. loss·grad norm·parameter는 GPU 실행끼리 비교하지 않음 (D16). CPU 시험(#7·#8)이 닿지 않는 CUDA 경로를 실제 중단 전에 확인함

---

## 15. M11 생성과 평가
상태: 테스트 통과(로컬)

### 15.1 목적과 범위
- **목적:** 모델(M9)로 validation·test split 전체를 생성 평가해 D17 지표(ValidRatio, HR@1, CombinedMetric)와 표본별 `EvalRecord`를 만든다. 같은 checkpoint와 같은 설정이면 같은 기록이 나와야 한다 (1.3, D16)
- **범위**
  - 평가 계획: split 전체를 행 순서로, 고정 후보(D10), 고정 템플릿(D15), hybrid prompt(D4, 변형 `text_only`), 왼쪽 padding(D22), `eval.batch_size`, drop_last 없음(D17)
  - 생성 호출(M9 `generate`)과 결과 검사, decode, 첫 줄 추출, 후보 판정(공개 코드 규칙 그대로), 지표, 보조 프로토콜(N = 92·88), 보조 통계
  - `EvalRecord` 구성(M2 flag, 생성에 쓰인 ω), z 전달, M12 기록 함수 호출, `EvalResult` 반환(M10)
  - batch 크기 무관성 점검(선택), 재현 점검, `main.py eval` 명령
- **범위 밖:** 생성 설정 값과 `generate` 내부(M9 `model.gen_*`), 후보와 flag 정의(M2), prompt·템플릿(M4), tokenization·padding·`EvalBatch`(M5), ω 계산(M7, M15), 파일 형식과 쓰기 함수(M12), checkpoint 저장과 선택 규칙(M10, D11), 실험 간 비교 표(M14)
- 파일 `rebuild/ilora/evaluate.py`, 설정 구역 `eval` (4.6.1, 4.6.2)
- 출처 표기는 M5 9.1과 같음: `T4.28 <파일>:<라인>` = transformers 4.28.0 소스, `torch2.0 <파일>:<라인>` = torch 2.0.0 소스

### 15.2 입력과 출력
**입력**
- 모델(M9 `ILoRAModel`): `generate(EvalBatch) -> GenerationOutput`, `omega_source`(M7 `Router`, 기록기 포함. M9 13.6.1의 속성 이름, parameter 이름 `omega_source.gate.proj.weight`, M7 11.6.7·11.6.9와 같음), `gen_config`, `sasrec`(M3, `hidden_size`·`state_digest`), `device`, `eval()`/`train()`
  - `GenerationOutput` (M9 13.6.1): `new_token_ids [B, T] int64` CPU (BOS 자리표 제거, EOS 뒤 `[PAD]`, 1 ≤ T ≤ 64), `omega [B, K]`, `z [B, d]` float32 CPU
- `TokenizerBundle`(M5): slow `LlamaTokenizer`(길이 32,005), `TokenIds` (`eos` 2, `pad` 32000)
- `LoadedData`(M1): `catalog`(`titles`, `title_mode`), `val` 94·`test` 95 (행 순서), smoke는 각 10 (D14)
- `Config`: `eval`(15.6.12), `run.seed`, `candidates`, `prompt`(`eval_prompt`, `template_choice`, `template_path`), `batching.num_workers`, `lora.num_experts`(K), `model.gen_*`(검사·기록), `train.precision`(autocast), `records.gate_rowsum_atol`
- `run_dir`, `ckpt_tag` (`epoch{e}`, e는 0부터(M4 8.7), 또는 `init`), 선택 `ckpt_meta` (checkpoint 폴더, `trainable.pt` sha256)

**출력**
- `EvalResult` (15.6.1): `SplitMetrics`(D17), 보조 프로토콜 지표, `EvalRecord` 목록(행 순서), `omega [N, K]`·`z [N, d]` float32 numpy
- 실행 폴더 `eval/{ckpt_tag}/{split}/` (M12 16.2와 같은 경로, `split`은 계약 값 `val`/`test`)

| 파일 | 형식 | 쓰는 함수 |
|---|---|---|
| `records.jsonl` | UTF-8, 1줄 = `EvalRecord` 1개, 필드 순서 15.6.8 | M12 `write_eval` |
| `metrics.json` | D17 지표 + 보조 블록 + meta (15.6.9). 마지막에 씀 = 완료 표시 | M12 `write_eval` |
| `representations.npz` | M12 NPZ 형식 (16.2): `sample_id, split, row, user_id, z, omega, argmax_expert, meta_json` | M12 `save_npz` |
| `batch_invariance.json` | `eval.batch_invariance_check`가 true일 때만 (15.6.10) | M11 |

- `main.py eval` 실행 폴더에는 추가로 `eval/summary.json` (15.6.11)
- 생성 중에는 파일을 쓰지 않음. split 하나가 끝난 뒤 한 번 씀 `[SPEC §4]`

### 15.3 논문 근거
- 지표: "ValidRatio [9] and HitRatio@1 are used as evaluation metrics, to separately quantify the ratios of valid responses over all sequences and relevant items over all candidate items" `[논문 p.6 §4]`
- 응답 하나와 유효 응답: "LLM-based recommenders, when provided with appropriate prompts, generate a single candidate item." "the ValidRatio metric to quantify the proportion of valid responses (i.e., items within the candidate set) across all sequences" `[논문 App B p.16]`
- 같은 내용: "With appropriate prompting, LLM-based recommenders can generate a single candidate item as required." "since LLaRA employs a generative paradigm for prediction, which may yield invalid responses such as nonsensical words or items outside the candidate sets, we introduce an additional metric — valid ratio. It quantifies the proportion of valid responses (i.e., items in the candidate set) across all sequences" `[LLaRA §5.1.3 p.7]`
- 해석
  - ValidRatio의 분모는 전체 sequence → split 전체, drop_last 없음 (D17). 공개 코드 test drop_last와 논문-코드 충돌 (15.5)
  - HR@1의 분모는 두 논문 모두 모호 ("relevant items over all candidate items") → 이름과 정의는 `[코드]` (D17)
  - "유효 = 후보 집합 안의 아이템"을 문자열로 판정하는 규칙(부분 문자열, 정확히 하나, 대소문자·공백 정규화)과 첫 줄 규칙은 논문에 없음 → `[코드]`
- 수치: Table 1 MovieLens iLoRA ValidRatio 0.9891, HitRatio@1 0.5275, LLaRA 0.9684, 0.4421 `[논문 p.7 Table 1]`. 역산하면 iLoRA = 48/91 (정답/유효, N = 92), LLaRA = 42/95 (정답/전체, N = 95)로 한 표 안에서 분모와 N이 다름 (D17, 감사 확인) → 세 지표와 보조 프로토콜을 모두 기록
- 평가 입력: hybrid `[논문 p.4 식 6-7, p.5 식 8 (암시)]`, "fully integrated with the hybrid item representation" `[LLaRA §4.3 p.6]` (D4). 템플릿 "randomly sampled" for training and testing `[LLaRA §5.1.2 p.7]` (D15). 후보 20 `[논문 App B p.16]` `[LLaRA §5.1.3 p.7]` (D10)
- 논문에 없음: decoding 방식과 길이, 특수 토큰 처리, 첫 줄 규칙, batch 크기, padding 방향 → `[코드]`, `[결정]`

### 15.4 공개 코드에서 가져올 것
`U model/model_interface.py`의 생성·판정 부분은 `L model/model_interface.py`와 같음 (코드 계보 일치).
- 생성 설정: greedy(`do_sample=False`, `num_beams=1`), `max_new_tokens=64`, `min_new_tokens=1`, `repetition_penalty=1.0`, `pad_token_id=[PAD]` `U model/model_interface.py:70, :72-88` (`L model/model_interface.py:43-57`). 값은 M9 `model.gen_*` 한 곳에 둠 (13.6.11). `temperature`·`length_penalty`는 greedy에서 쓰이지 않음 (13.4)
- decode: `batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)` 뒤 `text.strip()` `U :89-90` (`L :58-59`)
- 첫 줄: `generate.strip().split("\n")[0]` `U :165` (validation_step `:158-167`), `:200` (test_step `:193-202`), `L :92, :127`
- 판정 `calculate_hr1` `U :468-492` (`L :315-339`)
  - 생성문·정답·후보 제목을 모두 `strip().lower().strip()` `:476-478`
  - 후보 순서대로, 정규화 제목이 생성문의 부분 문자열이면 목록에 추가 `:479-482` (정규화 제목이 같은 후보 둘은 둘 다 들어감)
  - 목록 길이가 정확히 1이면 유효 `:483-484`, 그 제목이 정답 제목과 문자열로 같으면 정답 `:485-486`
  - `valid_ratio = valid/total`, `hr = correct/valid` (유효 0이면 0) `:487-491`, `metric = hr × valid_ratio` `:181, :216`
- 평가 순서 = 파일 행 순서 (`shuffle=False` `U data/data_interface.py:149, :156`), validation은 drop_last 없음 `:145-150`
- 평가 prompt는 학습 추첨과 무관하게 hybrid `U data/data_interface.py:40` (`or not self.train`) (D4)
- 표본별 기록 항목의 출발점: 생성문·정답·후보 제목 `U model/model_interface.py:176-179, :211-214`

### 15.5 버릴 것 (알려진 문제)
- **test drop_last (논문-코드 충돌):** `U data/data_interface.py:158`. 기본 batch 4(`U main.py:82`)면 92명, 스크립트 batch 8(`U train_movielens.sh:5`)이면 88명만 평가. 논문은 "across all sequences", LLaRA 코드는 drop_last 없음(`L data/data_interface.py:142-147`) → 95명 전부 (D17). 공개 코드의 N은 보조 프로토콜로만 계산 (15.6.6)
- **생성 오른쪽 padding** `U model/model_interface.py:316` → 왼쪽 (D22, M5 9.6.6)
- **평가 후보·템플릿이 매번 다름:** worker 난수와 batch 단위 템플릿 (`U data/movielens_data.py:27`, `U data/data_interface.py:32`) → key로 고정 (D10, D15; M2, M4)
- **결과 덮어쓰기와 식별자 없음:** `valid.csv`·`test.csv`를 매번 덮어씀 `U :179, :214`, 표본 id·템플릿·gate 없음 → checkpoint·split별 경로, 덮어쓰기 금지, 표본별 기록 (15.6.8)
- **부동소수 곱 지표:** `metric = hr × valid_ratio` `U :181` → 정수 비 `n_correct / n_total` (같은 양의 정확한 형태). checkpoint 선택은 정수 정답 수로 비교 (M10 14.6.7)
- **판정 규칙의 약점** (빈 제목은 모든 문자열에 일치, 짧은 제목의 부분 문자열 일치, 같은 제목 구별 불가, M2 6.5): 고치지 않고 표본마다 `flags`로 표시 (D20, D17 계보)
- **CNN router의 BatchNorm (`U model/router/nlpr.py:15-36, :48-80`):** 학습 모드에서는 batch 통계를 써서 한 표본의 ω가 같은 batch의 다른 표본에 따라 바뀜. 평가(eval 모드)는 running 통계를 쓰지만 그 buffer가 저장 필터에서 빠져(`U model/model_interface.py:300-305`, `get_parameter`가 buffer에서 AttributeError) 다른 프로세스 평가는 초기 통계로 계산됨 (M7 11.5) → 선형 gate라 학습·평가·재평가 모두 표본별 같은 ω (D1, M7 11.8 #4)
- **진행 표시줄 기록** (`prog_bar=True`) → split마다 요약 줄 (M0 4.6.6)

### 15.6 설계

#### 15.6.1 공개 API (`rebuild/ilora/evaluate.py`)
```python
class EvalError(RuntimeError): ...

EVAL_SPLITS: Final = ("val", "test")
FULL_SPLIT_SIZE: Final = {"val": 94, "test": 95}          # D17, M1 5.6.2
AUX_DROP_LAST_BATCH_SIZES: Final = (4, 8)                 # 15.6.6, test만
WILSON_Z: Final = statistics.NormalDist().inv_cdf(0.975)  # Φ⁻¹(0.975) = 1.9599639845400536 (repr), 보조 통계
CKPT_TAG_RE: Final = re.compile(r"epoch(0|[1-9][0-9]*)|init")

@dataclass(frozen=True)
class Judgement:
    matched: tuple[int, ...]          # 정규화 제목이 첫 줄에 들어 있는 후보 id (후보 순서)
    valid: bool                       # len(matched) == 1
    correct: bool                     # valid이고 그 후보의 정규화 제목 == 정규화 정답 제목

@dataclass(frozen=True)
class EvalRecord:                     # 3.1 계약. 필드 순서 = 기록 순서 (15.6.8)
    sample_id: str; split: str; user_id: int; template_id: int; hybrid: bool
    cand_ids: tuple[int, ...]; cand_titles: tuple[str, ...]
    target_id: int; target_title: str
    raw_text: str; first_line: str
    matched: tuple[int, ...]; valid: bool; correct: bool
    gates: tuple[float, ...]          # ω 한 행 (K개) → 기록 키 gate_0..gate_{K-1}
    flags: Mapping[str, bool]         # M2 candidate_flags, FLAG_NAMES 순서
    gen_token_ids: tuple[int, ...]    # 계약 추가 제안 (15.6.8)

@dataclass(frozen=True)
class SplitMetrics:                   # D17. 필드 이름 = M12 metrics.json 키
    split: str; n_total: int; n_valid: int; n_correct: int
    valid_ratio: float                # n_valid / n_total
    hr_at_1: float                    # n_correct / n_valid (n_valid = 0이면 0.0), 코드 hr
    combined_metric: float            # n_correct / n_total, 코드 metric

@dataclass(frozen=True, eq=False)    # numpy 배열 필드라 자동 __eq__를 쓰지 않음 (GeneratedSplit도 같음)
class EvalResult:
    split: str; ckpt_tag: str
    metrics: SplitMetrics
    aux: Mapping[str, SplitMetrics]   # test: drop_last_bs4, drop_last_bs8. val: 빈 dict
    records: tuple[EvalRecord, ...]   # 행 순서
    omega: np.ndarray                 # [N, K] float32
    z: np.ndarray                     # [N, d] float32
    out_dir: Path
    seconds: float

def first_line(raw_text: str) -> str
def judge(first_line: str, cand_ids: Sequence[int], cand_titles: Sequence[str],
          target_title: str) -> Judgement
def trim_generated(row: Sequence[int], *, eos_id: int) -> tuple[int, ...]
def decode_generated(bundle: TokenizerBundle, rows: Sequence[Sequence[int]]) -> list[str]   # 15.6.4 decode, M13 overfit 점검도 씀
def check_output(out, *, batch_size: int, num_experts: int, z_dim: int, max_new_tokens: int, eos_id: int,
                 pad_id: int, rowsum_atol: float, sample_ids: Sequence[str] | None = None) -> None   # 15.6.3
def compute_metrics(records: Sequence[EvalRecord], split: str) -> SplitMetrics
def aux_protocol_metrics(records: Sequence[EvalRecord], split: str) -> dict[str, SplitMetrics]
def wilson_interval(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float] | None
def eval_dir(run_dir: Path, ckpt_tag: str, split: str) -> Path
def generate_records(model, bundle: TokenizerBundle, data: LoadedData, split: str, cfg: Config, *,
                     batch_size: int, templates: TemplateSet, record_gates: bool) -> GeneratedSplit
def evaluate_split(model, bundle: TokenizerBundle, data: LoadedData, split: str, cfg: Config,
                   run_dir: Path, *, ckpt_tag: str, ckpt_meta: Mapping[str, object] | None = None,
                   templates: TemplateSet | None = None) -> EvalResult
def compare_records(a: Sequence[EvalRecord], b: Sequence[EvalRecord]) -> dict
def load_eval_result(out_dir: Path, *, num_experts: int) -> EvalResult
def eval_config_differences(source: Config, current: Config) -> tuple[list[str], list[str]]   # (금지, 허용), 항목은 "구역.키: a != b"
def resolve_eval_checkpoints(source: Path, tags: Sequence[str]) -> list[tuple[str, int | None, tuple[str, ...]]]   # 15.6.11 3단계
def run_eval_command(cfg: Config, run_dir: Path, *, repo_root: Path, llm_path: Path | None) -> dict
```
- `GeneratedSplit`은 모듈 내부 묶음 (기록, ω, z, 진단 수, 생성 호출 수). 파일을 쓰지 않음
- M10이 부르는 꼴 `evaluate_split(model, bundle, data, "val", cfg, run_dir, ckpt_tag=f"epoch{e}", ckpt_meta=…, templates=templates)` (14.6.7)을 그대로 받음. `templates`가 None이면 `prompts.load_templates(run.resolve_path(cfg.prompt.template_path))` (상대 경로는 M0 `run.py`의 `REPO_ROOT` 기준, 3.1) `R ilora/evaluate.py evaluate_split`
- `EvalRecord`·`SplitMetrics`·`EvalResult`는 `evaluate.py`에 둠. M12 `records.py`는 `EvalRecord`를 `read_eval_records` 안에서만 import하고, `write_eval`·`crosscheck_gates`는 필드를 이름으로 읽으며 필드 순서가 `records.EVAL_RECORD_FIELDS`와 같은지 확인함. `evaluate.py`도 M12 함수를 함수 안에서 import. 두 모듈 모두 상대를 module 수준에서 import하지 않아 순환이 없음 `[결정]` `R ilora/records.py read_eval_records`

#### 15.6.2 평가 계획 (`evaluate_split` 앞부분)
1. 검사 (`EvalError`): `split ∈ EVAL_SPLITS`, `CKPT_TAG_RE.fullmatch(ckpt_tag)`, `data.subset == "full"`이면 N == `FULL_SPLIT_SIZE[split]` (D17: val 94, test 95), smoke면 N == `data.smoke_users`의 해당 값 (D14), `torch.backends.cudnn.benchmark`가 False (15.6.10)
2. `samples = data.samples(split)`: M1 tuple 그대로 (행 오름차순), `expected_ids = [s.sample_id for s in samples]`
3. 표본별 기대값 (main process, 189표본이라 비용 무시) `[결정]`
   - 후보 `candidates_for(s, main_seed=cfg.run.seed, epoch=None, cfg=cfg.candidates)`: key `("candidates", split, row)` (D10, M2 6.6.2)
   - 템플릿 `choose_template(cfg.prompt, cfg.run.seed, n_templates, s, None)`: key `("template", split, row)` (D15, M4 8.6.3)
   - flag `candidate_flags(cands, s.target, catalog)` (M2 6.6.4)
   - `hybrid = eval_is_hybrid(cfg.prompt)`: paper_spec `hybrid` (D4), E3 변형 `text_only`
4. loader: `EvalBatches(samples, batch_size, catalog, templates, cand_fn, bundle, cfg.prompt, cfg.batching, cfg.run.seed)` + `make_loader(ds, cfg.batching.num_workers, pin_memory=(device.type == "cuda"))` (M5 9.6.8). `cand_fn`은 M10이 학습에 쓰는 것과 같은 pickle 가능한 `CandidateFn` (M5 9.6.11)
   - 묶음 `[i, min(i + bs, N))`, drop_last 없음, 왼쪽 padding + `[BOS]` + prompt (D22, M5 9.6.4). 결과는 worker 수와 무관 (M5)
   - paper_spec `eval.batch_size` 8: validation 12 batch (마지막 6개), test 12 batch (마지막 7개) (계산 `design/tmp/m11_wilson.py`)
5. batch마다 대조 (어긋나면 batch 번호와 sample_id를 담은 `EvalError`) `[결정]`
   - `sample_ids == expected_ids[i : i + B]`, 행마다 `cand_ids`·`template_ids`·`target_ids`가 3단계 값과 같음, `cand_titles == [catalog.titles[c] …]`, `target_titles == catalog.titles[target]`, `batch.hybrid == hybrid`
   - 이유: M5도 같은 함수로 만들지만, 기록된 후보·템플릿이 D10·D15 key 값이라는 것을 평가 경계에서 보장 (순서 뒤바뀜, 잘못된 epoch key를 잡음)
- cluster_hard의 표본 범위(세 split 전부 배정됨)는 `ClusterHardGate`를 만들기 전 M15 `load_cluster_artifacts(expected_ids=…)`가 이미 확인함 (M7 11.6.5, 19.6.7). M7에는 미리 조회하는 함수가 없음: `hard_gate`로 미리 조회하면 `lookups`가 늘어 E4 조회 수 기록이 틀어짐. 생성 중 모르는 sample_id는 M15의 `KeyError`가 그대로 올라옴

#### 15.6.3 생성 호출과 결과 검사 (`generate_records`)
```python
was_training = model.training
model.eval()                                            # LoRA dropout 꺼짐, SASRec은 항상 eval (D12)
router, dev = model.omega_source, model.device          # M9 13.6.1 속성 이름 (M7 Router)
recording = False
try:
    with torch.random.fork_rng(devices=[dev.index] if dev.type == "cuda" else []):
        if record_gates:
            router.start_recording(split); recording = True
        calls0 = router.gate_calls
        for i, batch in enumerate(make_loader(...)):
            check_batch(batch, ...)                      # 15.6.2 5단계
            with autocast_for(cfg.train.precision, dev):  # M0 precision 도구 (M10 14.6.10 제안 5)
                out = model.generate(batch)               # M9 13.6.6
            check_output(out, ...)                        # 아래
            rows = [trim_generated(r, eos_id=ids.eos) for r in out.new_token_ids.tolist()]
            texts = bundle.tokenizer.batch_decode(rows, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
            ...                                           # 15.6.4–15.6.5로 EvalRecord
        if record_gates:
            rec = router.stop_recording(expected_ids); recording = False
            check_gate_record(rec, ...)                   # 아래 "gate 수집 지점 하나" (기록할 때만)
except BaseException:
    if recording: router.discard_recording()              # M7 11.6.5
    raise
finally:
    model.train(was_training)
```
- **난수:** 평가 경로에는 난수를 쓰는 연산이 없음 (greedy, eval 모드). 다만 DataLoader iterator가 전역 CPU 생성기에서 base seed 하나를 뽑으므로 (`torch2.0 utils/data/dataloader.py:605`) 전체를 `fork_rng`로 감싸 CPU·CUDA 상태를 되돌림 (M10 14.6.5) `[결정]`
- **autocast** `[결정]`: 학습과 같은 계산 정밀도. `train.precision`이 `fp16_amp`면 `torch.autocast("cuda", dtype=torch.float16)`, `bf16_amp`면 bfloat16, `fp32`(CPU smoke)면 없음 (D9). z·ω는 M9가 autocast 밖 fp32로 계산 (D23). fp16 base에서 autocast 없이 부르면 M8이 오류를 냄 (12장)
- **생성 설정 확인** (`EvalError`) `[결정]`: `model.gen_config`의 `do_sample` False, `num_beams` 1, `max_new_tokens`·`min_new_tokens`·`repetition_penalty`가 `cfg.model.gen_*`와 같음, `max_time` None, `eos_token_id == ids.eos`(2, 하나만), `pad_token_id == ids.pad`(32000), `model.gen_strategy == greedy`. 종료 조건은 EOS와 최대 길이뿐 (15.6.4의 이유로 줄바꿈 정지 없음). 모델의 K(`omega_source.num_experts`)가 `lora.num_experts`와 다르면 `EvalError`
- **출력 검사** (`check_output`) `[결정]`: M9가 보장하는 형식을 평가 경계에서 다시 확인 (조용한 잘림·섞임 방지). 토큰 검사와 ω·z가 CPU tensor인지는 `EvalError`. ω·z 계약은 M12 `validate_gate`·`validate_z`가 검사해 그 `RecordsError`(`ValueError`)가 그대로 올라옴 (15.6.13). 메시지에는 그 batch의 sample_id를 씀 `R ilora/evaluate.py check_output`
  - `new_token_ids`: CPU int64 2차원, `shape[0] == B`, `1 ≤ T ≤ gen_max_new_tokens`
  - 행마다 첫 EOS 뒤는 모두 `[PAD]` (greedy가 끝난 행을 pad로 채움 `T4.28 generation/utils.py:2285-2288`)
  - `T < gen_max_new_tokens`이면 모든 행에 EOS가 있음 (모든 행이 끝나야 멈춤 `:2300-2305`)
  - ω: M12 `validate_gate(ω, batch=B, num_experts=K, rowsum_atol=cfg.records.gate_rowsum_atol)` (float32, `[B, K]`, 유한, 0 이상, 행 합), K = `lora.num_experts`
  - z: M12 `validate_z(z, batch=B, dim=d)`, d = `model.sasrec.hidden_size` (64를 코드에 쓰지 않음 `[SPEC §4]`)
- **`trim_generated`** `[결정]`: 첫 EOS까지(EOS 포함) 자르고 뒤의 `[PAD]`를 버림. EOS가 없으면 T개 전부 (그때 T = 64). 끝난 행의 pad 길이만 batch 구성에 따라 다르므로, 이 값은 표본 하나의 생성 결과
- **gate 수집 지점 하나** `[SPEC §4]`: `rec.sample_ids == expected_ids`, `rec.omega`·`rec.z`가 batch마다 받은 `out.omega`·`out.z`를 이어 붙인 것과 bitwise 같음 (같은 텐서의 CPU 사본). `router.gate_calls − calls0` = 생성 호출 수(dynamic), 0(cluster_hard·none) (M7 11.6.7). 어긋나면 `EvalError`

#### 15.6.4 decode와 첫 줄
- `raw_text = batch_decode(gen_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)` `[코드]` `U :89`
  - 4.28.0 slow tokenizer는 `all_special_ids`(0, 1, 2, 32000–32004)를 먼저 버리고 (`T4.28 tokenization_utils.py:883-913, :921-931`), 남은 조각을 `sp_model.decode`로 이음 (`models/llama/tokenization_llama.py:132-149`). 조각마다 앞 `▁`는 공백이 되고 맨 앞 공백은 빠짐
  - BOS 자리표(M9가 제거)와 EOS 뒤 `[PAD]`를 붙여 decode해도 같은 문자열 (측정 `design/tmp/m11_decode_check.py` D2: 예시 6개 모두 같음) → `U :89`가 `generate_ids` 전체를 decode한 것과 같음
  - 생성 중간의 특수 토큰(`[PH]`, `[HistoryEmb]`, `[CansEmb]`, `[ItemEmb]`, unk, BOS)도 빠짐 (측정 D3). `released_pair` 변형(D18) 모델이 BOS를 먼저 내도 문자열에 영향 없음. 공개 코드와 같은 처리
- `raw_text`는 strip 전 문자열 그대로 기록 (공개 코드의 생성문 = `raw_text.strip()` `U :90`)
- `first_line = raw_text.strip().split("\n")[0]` `[코드]` `U :90, :165, :200` (두 번째 strip은 효과 없음)
  - `"\n"`으로만 나눔. `splitlines()`가 아님 (`\r`, `\x85`, `\u2028`에서 나누지 않음). 끝에 남는 `\r`은 판정의 `match_norm`(strip)이 지움
  - 줄바꿈을 만드는 토큰은 `<0x0A>`(id 13) 하나뿐 (test tokenizer 32,000개 전수, 측정 D4). `\r`이 들어 있는 조각은 25개
- **첫 줄바꿈에서 생성을 멈추지 않음** `[결정]`
  - 학습 응답은 제목 + `"\n"`이고 EOS가 없어 (D18) 생성이 대개 64 토큰까지 갈 것으로 예상됨 (실측은 `n_no_eos`, 15.8 #22). 첫 `"\n"`에서 멈추면 빨라지지만 결과가 같지 않음
  - 공개 코드는 decode 전체를 먼저 `strip()`하므로, 줄바꿈으로 시작하거나 공백뿐인 앞부분은 사라지고 다음 줄이 첫 줄이 됨. 첫 줄바꿈에서 멈추면 그 빈 앞부분이 첫 줄이 됨
  - 예 (측정 D5): 생성 토큰 `[29871, 13, 16213, 9149, 13, 3868, 271, 13]` = `"\nStar Wars\nHeat\n"` → 공개 코드 첫 줄 `"Star Wars"`, 첫 13에서 멈추면 `""`. 빈 문자열은 빈 제목 후보(item 266)에만 일치하므로 판정이 달라짐. `[29871, 13, 7828, 9149, 13]`도 `"Star Wars"` 대 `""`
  - 앞부분에 공백이 아닌 글자가 있으면 판정은 같지만 저장되는 `first_line`은 끝 공백만큼 다름 (측정 D6: `"Star Wars  "` 대 `"Star Wars"`, `match_norm` 뒤 같음)
  - 그래서 EOS 또는 64 토큰까지 생성하고, 차이가 실제로 몇 번 생기는지는 `n_blank_leading_segment`로 셈 (15.6.7)
- 64 토큰 안에 줄바꿈이 없으면 첫 줄이 64 토큰에서 잘림. 공개 코드와 같은 조건이고 `n_cut_at_max_tokens`로 셈

#### 15.6.5 판정 (`judge`) `[코드]` `U :468-492`
```python
def judge(first_line, cand_ids, cand_titles, target_title):
    g = match_norm(first_line)                      # U :476
    real = match_norm(target_title)                 # U :477
    hits = [(c, n) for c, t in zip(cand_ids, cand_titles)
            if (n := match_norm(t)) in g]           # U :478-482, 후보 순서
    valid = len(hits) == 1                          # U :483
    correct = valid and hits[0][1] == real          # U :485
    return Judgement(tuple(c for c, _ in hits), valid, correct)
```
- `match_norm`은 M2 `candidates.match_norm`(= `strip().lower().strip()`) 하나만 씀: flag와 판정이 같은 정규화 (M2 6.6.10 계약 추가 제안)
- 입력 검사 `[결정]`: `len(cand_ids) == len(cand_titles) == 20`, 후보 id 중복 없음. 기록을 만들 때 `target_id ∈ cand_ids`와 `target_title == catalog.titles[target_id]`도 확인
- 정답은 공개 코드처럼 문자열 비교. 유효일 때 "문자열 정답 ⇔ `matched[0] == target_id`"가 항상 성립함 (정규화 제목이 정답과 같은 다른 후보가 있으면 둘 다 일치해 무효). 기록을 만들 때 이 동치를 확인하고 어긋나면 `EvalError` `[결정]` (정규화 구현 오류를 잡음)
- 빈 제목·짧은 제목·같은 제목의 영향은 고치지 않고 `flags`로 표시 (D20). M2 6.8 #6의 교차 확인 (`target_unscorable == not judge(정답 제목).correct`)이 이 함수를 씀

#### 15.6.6 지표 (D17)와 보조 프로토콜
- `compute_metrics(records)`: n = 표본 수, v = Σ`valid`, c = Σ`correct` (정수). n = 0이면 `EvalError`, `correct ⇒ valid` 위반도 `EvalError`
  - `valid_ratio = v / n`, `hr_at_1 = c / v` (v = 0이면 0.0 `U :488-491`), `combined_metric = c / n`
  - `combined_metric`은 정수 비를 한 번 나눈 값 `[결정]`. `hr × valid_ratio` 곱(`U :181`)은 같은 양이지만 마지막 자리가 다를 수 있음 (M12 16.6.4는 `c/n`과 1e-12 안을 요구)
- 이름 (D17, `[SPEC §12]`): ValidRatio = `valid_ratio`, HR@1 = `hr_at_1` (코드 `hr`), CombinedMetric = `combined_metric` (코드 `metric`). 주 지표는 split 전체 (val 94, test 95)
- **보조 프로토콜 (test만)** `[결정]` (D17 변형)
  - `drop_last_bs4` = 앞 ⌊N/4⌋·4개, `drop_last_bs8` = 앞 ⌊N/8⌋·8개 (행 순서). test 95 → 92, 88. 공개 코드 test loader(`shuffle=False`, `drop_last=True`, `U data/data_interface.py:152-160`)가 batch 4(`U main.py:82`)·8(`U train_movielens.sh:5`)일 때 평가하는 표본 집합과 같음. smoke 10이면 8·8
  - 같은 기록의 앞부분으로 계산하고 다시 생성하지 않음. 가정: 표본의 생성 결과가 batch 구성에 의존하지 않음
    - 정확한 산술에서는 성립: 왼쪽 padding과 mask 기반 position_ids (D22, M9 13.6.6), padding 열은 attention에서 가려짐, ω는 표본별 (D1)
    - 부동소수에서는 batch 모양에 따라 연산 순서가 바뀜. CPU fp32 작은 모델에서 첫 단계 logits가 batch와 단독 사이에 bitwise 같지 않았지만 (최대 |Δ| 2.98e-7) 생성 토큰은 95표본 모두 같았음 (15.6.10). GPU fp16에서는 차이가 더 클 수 있으므로 이 계산은 근사이고, 일치율을 `batch_invariance.json`으로 잼 (VESSL_REQUIRED)
    - `eval.batch_size` = 8이면 `drop_last_bs8`의 88개는 bs 8 + drop_last 실행과 batch 구성(앞 11 batch)까지 같아 이 가정이 필요 없음. 가정에 기대는 것은 `drop_last_bs4`뿐
  - 공개 코드와 달라지는 것은 표본 수뿐이고, padding 방향·후보·템플릿은 재구현 규칙을 따름. 해석 참고: iLoRA Table 1의 0.5275 = 48/91은 `drop_last_bs4`의 `hr_at_1` 꼴, LLaRA의 0.4421 = 42/95는 주 지표의 `combined_metric` 꼴 (D17 역산). 비교 표는 M14
  - validation에는 계산하지 않음: 공개 코드의 validation loader는 drop_last가 없어 94명 전부를 평가함 (`U data/data_interface.py:145-150`)
  - test flag 표본(M2 6.6.6의 rows 5, 11, 38, 39, 43)은 모두 앞 88개 안에 있음 (행 번호로 확인)

#### 15.6.7 보조 통계 (논문 지표 아님, 기록만) `[결정]`
- `[SPEC §12]`의 "존재하지 않는 metric을 논문 metric처럼 쓰지 않는다"에 따라 `metrics.json`에서 `supplementary` 아래에만 두고, checkpoint 선택(D11)과 비교 표의 주 열에 쓰지 않음
- **Wilson 95% 구간:** `p = k/n`, `center = (p + z²/2n)/(1 + z²/n)`, `half = z/(1 + z²/n)·√(p(1 − p)/n + z²/4n²)`, [0, 1]로 자름, n = 0이면 null. k = 0이면 하한을 정확히 0, k = n이면 상한을 정확히 1로 둠 `[결정]`: 닫힌 식은 반올림 잔차를 남겨 p가 자기 구간 밖에 놓일 수 있음 (계산: n = 10, k = 0의 하한 2.8e-17. n = 88, k = 88의 상한 0.9999999999999999) `R ilora/evaluate.py wilson_interval`. `wilson95`에는 쓴 z 값도 남김
  - `valid_ratio` (k = v, n = N), `hr_at_1` (k = c, n = v: 유효 수를 고정한 조건부 구간), `combined_metric` (k = c, n = N)
  - 기준값 (계산 `design/tmp/m11_wilson.py`, z = `NormalDist().inv_cdf(0.975)`, 닫힌 식과 2차식 근의 두 계산 차이 최대 1.1e-16): (42, 95) → (0.346422, 0.542289), (48, 91) → (0.425878, 0.626842), (0, 10) → (0, 0.277533), (10, 10) → (0.722467, 1)
  - 목적: N = 94·95에서 표본 1개가 약 0.011이고 seed가 하나(D2)라, 실험 간 차이를 읽을 때 참고. 검정·판정에 쓰지 않음
- **flag 집계:** flag별 표본 수, 하나 이상인 표본 수, flag가 하나도 없는 표본만의 `SplitMetrics` (`flag_free`, 그런 표본이 없으면 null). M2 6.6.9가 M11에 맡긴 정의
- **생성 진단 수** (표본 수)
  - `n_no_eos`: 64 토큰 안에 EOS 없음
  - `n_empty_first_line`: `first_line == ""`
  - `n_blank_leading_segment`: `"\n" in raw_text`이고 `raw_text.split("\n")[0].strip() == ""`. 첫 줄바꿈에서 멈췄다면 첫 줄이 달라졌을 표본 (15.6.4)
  - `n_cut_at_max_tokens`: EOS도 줄바꿈도 없이 64 토큰을 채움
  - `n_no_match`, `n_multi_match`: `matched` 0개, 2개 이상. `v = N − n_no_match − n_multi_match` 검사

#### 15.6.8 `EvalRecord`와 기록 파일
| 필드 (기록 순서) | 타입 | 값 |
|---|---|---|
| `sample_id` | str | `Sample.sample_id` (`"test:0"`) |
| `split` | str | `val`/`test` (계약 값) |
| `user_id` | int | `Sample.user_id` (D14 pseudo) |
| `template_id` | int | `EvalBatch.template_ids[b]` (D15 key) |
| `hybrid` | bool | `EvalBatch.hybrid` (D4) |
| `cand_ids` | list[int] 20 | M2 순서 (D10) |
| `cand_titles` | list[str] 20 | `catalog.titles[c]` (prompt와 같은 문자열, `data.title_mode`) |
| `target_id`, `target_title` | int, str | `Sample.target`, `catalog.titles[target]` |
| `raw_text` | str | decode 결과, strip 전 (15.6.4) |
| `first_line` | str | 15.6.4 |
| `matched` | list[int] | 15.6.5 (후보 id, 후보 순서) |
| `valid`, `correct` | bool | 15.6.5 |
| `gate_0` … `gate_{K−1}` | float | 생성에 쓰인 ω 행. float32 값을 Python float로 (JSON repr 왕복 정확) |
| `flags` | {str: bool} | M2 `candidate_flags`, `FLAG_NAMES` 순서 5개 |
| `gen_token_ids` | list[int] | 계약 추가 제안: 첫 EOS까지의 새 토큰 id (BOS 자리표 제외) |

- `gen_token_ids`를 두는 이유 `[결정]`: 같은 문자열로 decode되는 다른 토큰열(특수 토큰 끼임, 다른 분절)을 구별해 결정성·batch 무관성을 토큰 단위로 확인. 95 × 64개 이하라 크기 부담 없음. 계약에 넣지 않기로 하면 두 확인은 메모리 안 값으로만 함
- **쓰기 순서** (모두 `eval/{ckpt_tag}/{split}/`) `[결정]`: 1단계는 생성 전에, 2–4단계는 평가가 끝난 뒤 한 번
  1. 폴더: 없으면 만듦. 있고 `metrics.json`이 있으면 `EvalError` (덮어쓰기 금지, 4.6.3). 있고 `metrics.json`이 없으면 `{split}.incomplete-<UTC>`로 이름을 바꿔 보존한 뒤 새로 (M10 14.6.10과 같은 방식, 이름이 겹치면 `-<n>`). 생성 전에 하는 이유: 완료 폴더는 생성 시간을 쓰기 전에 거부하고, 실패한 평가가 남긴 빈 폴더를 다음 평가가 옮기게 함 (15.6.13, 15.8 #14) `R ilora/evaluate.py _prepare_out_dir`
  2. `representations.npz`: M12 `RepresentationTable(split, sample_ids, rows, user_ids, z, omega, argmax, meta)` → `save_npz`. meta: `source: "evaluate"`, `routing_mode`, `num_experts`, `z_dim`, `ckpt_tag`, `ckpt_meta`, `sasrec_state_digest`, `device`, `eval_batch_size`, `torch_version`. UTC는 넣지 않음 `[결정]`: 같은 checkpoint·설정의 두 평가가 같은 byte여야 함 (15.6.10, 15.8 #11). 시각은 `metrics.json` meta에 있음 `R ilora/evaluate.py _npz_meta`
  3. `eval.batch_invariance_check`면 `batch_invariance.json` (15.6.10)
  4. M12 `write_eval(out_dir, records, metrics, expected_ids=expected_ids, num_experts=K)` → `records.jsonl`, 마지막에 `metrics.json` (M12 16.6.4 검사 통과 후)
- 완료 표시는 `metrics.json` 존재. 중간에 끊긴 폴더는 1단계 규칙으로 옮겨짐
- SPEC 산출물 대응: `results/baseline/metrics.json` `[SPEC §20]` = E2 실행의 `eval/epoch{a}/test/metrics.json` (a = D11 주 결과 checkpoint). 복사·이름 붙이기는 M14

#### 15.6.9 `metrics.json` 내용
- M12 16.6.4 키: `split`, `epoch` (`ckpt_tag`의 e, `init`이면 null), `n_total`, `n_valid`, `n_correct`, `valid_ratio`, `hr_at_1`, `combined_metric`, `meta`
- 추가 블록 (계약 추가 제안)
  - `ckpt_tag`, `aux_protocols`: {`drop_last_bs4`: SplitMetrics 필드, `drop_last_bs8`: …} (val은 빈 객체)
  - `supplementary`: `wilson95` {`valid_ratio`: [lo, hi], `hr_at_1`: [lo, hi] 또는 null, `combined_metric`: [lo, hi]}, `flags` {`counts`, `any`, `flag_free`}, `generation` (15.6.7 진단 수)
- `meta` (M12 목록에 더해, `[SPEC §3]`의 시작·종료 시각과 경과 시간 포함)
  - 평가: `ckpt_meta`, `title_mode`, `eval_prompt`, `template_choice`, `eval_batch_size`, `n_batches`, `num_workers`, `precision`, `autocast_dtype`, `device`, `torch_num_threads` (평가 때 torch CPU thread 수, 15.6.10), `data_subset`, `smoke_users`, `started_utc`, `ended_utc`, `seconds`, `generation_seconds`, `batch_invariance_seconds`, `max_memory_allocated_gib` (CUDA만, CPU면 null: 평가 시작에 `reset_peak_memory_stats`, 끝에 읽고 다시 초기화해 M10 step 구간 측정과 섞이지 않게 함, 15.8 #18)
  - 생성·decode: `gen_strategy`, `max_new_tokens`, `min_new_tokens`, `repetition_penalty`, `pad_token_id`, `eos_token_id`, `skip_special_tokens: true`, `clean_up_tokenization_spaces: false`, 첫 줄 규칙 문자열 `first_line_rule` = `strip().split("\n")[0]`
  - 입력 지문: `tokenizer_sha256`, `candidates_sha256` (M2 6.6.5 직렬화 `json.dumps([[row, list(cands)], …], separators=(",", ":"))`의 sha256), `template_counts` (템플릿 index 순서의 목록), `gate_calls` (주 평가의 `Router.gate_calls` 증가분, batch 1 점검 분은 `batch_invariance.json`의 `gate_calls_batch1`), `sasrec_state_digest`
  - full·seed 1234의 `candidates_sha256`은 val `9a073206…`, test `1ffdb649…` (M2 6.6.5), `template_choice = random`이면 `template_counts`는 val 30/33/31, test 22/36/37 (M4 8.6.3)이어야 함 (15.8 #10)

#### 15.6.10 결정성, batch 무관성, 재현
- **결정성 (1.3, D16):** 같은 checkpoint·같은 설정·같은 하드웨어와 소프트웨어면 `records.jsonl`과 `representations.npz`의 내용이 같아야 함
  - 근거: 평가 경로에 난수 없음 (greedy, eval 모드, SASRec dropout 없음), 후보·템플릿은 key (D10, D15), batch 구성은 행 순서와 `eval.batch_size`로 고정, worker 수 무관 (M5), ω는 표본별 (D1)
  - `torch.use_deterministic_algorithms`는 쓰지 않음 (D16). `torch.backends.cudnn.benchmark`가 True면 `EvalError` `[결정]`: 실행마다 알고리즘 선택이 달라질 수 있음 (SASRec `Conv1d`는 cuDNN을 씀, M3)
  - CPU에서는 torch thread 수도 같아야 bit 단위로 같음 (측정: 1 thread로 학습·validation한 실행을 14 thread로 다시 평가하면 토큰은 같고 gate 값이 끝자리에서 달라 `records.jsonl` sha256이 다름). CPU 시험은 1 thread로 돌리고, M13 S2의 `identical`도 S1과 같은 thread 수를 전제로 함. thread 수는 `metrics.json` meta와 `meta.json` 환경 블록의 `torch_num_threads`에 적음 `[결정]`: 재평가가 `mismatch`면 먼저 두 값을 비교해 thread 차이인지 가름 `R ilora/evaluate.py _metrics_meta`, `R ilora/run.py environment_info`
  - 같은 모양 batch의 GPU fp16 결과가 실행마다 같은지는 VESSL에서 두 번 평가해 확인 (VESSL_REQUIRED, 15.8 #19). 학습 중 validation과 별도 프로세스 재평가도 같아야 함 (M10 14.8 #17)
- **batch 크기 무관성 점검** (`eval.batch_invariance_check = true`) `[결정]`
  - 주 평가(설정 batch 크기) 뒤 같은 split을 batch 1로 다시 생성. 기록기는 끔 (`record_gates=False`). 주 지표와 기록은 바꾸지 않음
  - `compare_records(주, batch1)`: 표본별 `gen_token_ids`, `first_line`, (`valid`, `correct`), `gate_*`(bit 단위), 기록 전체가 같은 수와 비율, ω 최대 |차이|. 목록에는 생성 결과(토큰, 첫 줄, 판정)가 다른 표본만 처음 다른 단계와 함께 넣고, 끝자리만 다른 gate는 세기만 함 `[결정]`: CPU fp32에서 batch 8 대 1은 토큰이 94/94 같아도 gate 행이 거의 모두 끝자리에서 달라(bit 단위로 같음 3/94, max |Δω| 2.4e-7, max |Δz| 3.8e-6, 구현 단계 측정) 그 목록이 생성 차이를 가림 `R ilora/evaluate.py compare_records`
  - `batch_invariance.json`: `split`, `batch_sizes` [`eval.batch_size`, 1], `n`, `records_equal`·`tokens_equal`·`first_line_equal`·`judgement_equal`·`gates_equal`과 `ratios`, `differences` [{`sample_id`, `first_diff_step`, `first_line_a`, `first_line_b`, `judgement_a`, `judgement_b`, `tokens_equal`, `max_abs_diff_omega`}], `metrics_batch1`, `aux_protocols_batch1`, `max_abs_diff_omega`, `max_abs_diff_z`, `n_batches_batch1`, `gate_calls_batch1`, `seconds_batch1`. batch 1 생성도 dynamic gate를 불러 `Router.gate_calls`가 split마다 N만큼 더 늘지만, `metrics.json`의 `gate_calls`(주 평가)와 M10 학습 창 증가분에는 들지 않음
  - 설계 확인 측정 (`design/tmp/m11_batch_single.py`, `m11_logit_diff.py`): 순수 작은 LLaMA (hidden 64, 2층, vocab 32,005, CPU fp32), 길이 165–310의 무작위 토큰 prompt 95개, greedy 64 토큰, stock `generate(inputs_embeds, attention_mask)`
    - batch 8·4와 batch 1의 새 토큰이 95/95 같음 (thread 14와 1 모두), batch 8 재실행도 95/95
    - 첫 단계 logits는 batch와 단독이 bitwise 같지 않음 (thread 14에서 0/95, thread 1에서 7/95, 최대 |Δ| 2.98e-7) → 토큰 일치는 산술 동치가 아니라 argmax 여유에 의한 측정된 일치
  - CPU fp32 작은 모델 시험은 모든 표본 토큰 일치를 요구함 (seed·thread 고정이라 결정적, 15.8 #12). GPU fp16은 100%를 요구하지 않고 일치율을 기록 (VESSL_REQUIRED). 이 값이 `drop_last_bs4` 근사의 크기를 알려 줌
- **재현 점검** (`main.py eval`, 15.6.11): 원 실행에 같은 `ckpt_tag`·split 기록이 있으면 `compare_records`와 `records.jsonl` sha256을 비교해 `summary.json`의 `reproduction`에 씀
  - 같음을 요구하는 조건 `[결정]`: 평가 결과를 바꿀 수 있는 허용 차이 키 `eval.batch_size`, `prompt.eval_prompt`, `data.subset`, `data.smoke_users`가 원 실행과 모두 같을 때만. 같지 않으면 비교·판정하지 않고 `reproduction`에 `"not_comparable"`과 다른 키 목록을 적음 (E3 `text_only`는 원 실행 hybrid와 prompt가 달라 항상 이 경우, M13 S6·M14 R9)
  - 조건을 만족하면 `reproduction`에 `"identical"` 또는 `"mismatch"`와 다른 표본·처음 다른 단계를 적음. 판정은 M13 `test_e2_reeval_reproduces`와 RUNBOOK R8이 이 값으로 함 (15.8 #19). 원 실행에 그 (ckpt_tag, split)의 완료 폴더가 없으면(`init` 등) `"absent"` `[결정]`: 문서가 이 경우의 상태를 정하지 않음 `R ilora/evaluate.py _reproduction`

#### 15.6.11 M10 연결과 `main.py eval`
- **M10 (학습 중, 14.6.7)**
  - epoch e: `STATE_SAVED` 뒤 `evaluate_split(model, bundle, data, "val", cfg, run_dir, ckpt_tag=f"epoch{e}", ckpt_meta=…)` → `EvalResult.metrics`를 `select_epochs`, `val_metrics.json`, `epochs.csv`(M12)에 씀. 정답 수 비교로 best 선택 (D11)
  - 마지막 epoch `COMPLETE` 뒤 best·last 오름차순으로 `evaluate_split(..., "test", ckpt_tag=f"epoch{b}")`
  - M10이 전후에 할 일: M12 `ActivationMaxMonitor.pause()`/`resume()` (16.6.3). 모델 모드와 전역 난수는 M11이 되돌림
  - 재개 (14.6.10): `eval_dir(...)`가 완료 상태면 `load_eval_result`로 읽어 쓰고 다시 생성하지 않음. 미완료면 `evaluate_split`이 옆으로 옮기고 다시 생성. `load_eval_result`는 기록과 NPZ의 표본·순서, `gate_*` = NPZ ω(bit 단위), 다시 한 첫 줄·판정, 다시 계산한 지표·보조 프로토콜이 `metrics.json`과 같은지 확인함 `R ilora/evaluate.py load_eval_result`
- **`main.py eval` (`run_eval_command`)** `[결정]`: `python rebuild/main.py eval --config <프로파일> --set eval.source_run=<학습 실행 폴더> [--set eval.checkpoints=[best_val,last]] [--set eval.splits=[val,test]]`
  1. M0 공통 준비: 새 실행 폴더, 환경 점검, `candidates.self_check()` (M2), meta에 `source_run`
  2. 설정 비교 `[결정]`: 원 실행 `config.yaml`과 현재 설정에서 허용하는 차이는 `eval` 구역, `prompt.eval_prompt` (E3), `batching.num_workers`, `data.subset`·`data.smoke_users`·`data.train_rows`, `records` 구역뿐. `run.seed`를 포함한 나머지가 다르면 다른 키 목록을 담아 `EvalError`. 원 실행 `meta.json`의 입력 해시가 현재 값과 같아야 함: `inputs`(M0 입력 6개: 데이터 4개, prompt, 원본 SASRec)와 `assets`의 변환 SASRec state sha256, LLaMA `config.json`의 sha256·`torch_dtype` `[결정]`: 'SASRec'이 원본과 변환본 중 무엇인지 정하지 않아 둘 다 비교 `R ilora/evaluate.py _input_hash_problems`
  3. checkpoint 해석: `best_val`·`last`는 원 실행 `eval/selection.json`의 `best_val_epoch`·`last_epoch` (M10 14.6.7), `epoch<k>`는 그 폴더, `init`은 불러오지 않음 (seed로 만든 초기 모델). 같은 epoch는 한 번. `init` 밖은 `COMPLETE` 마커가 있어야 함. 평가 순서는 `init`(seed로 만든 모델) 먼저, 그다음 epoch 오름차순 `[결정]`: 모델을 한 번 만들므로 `init`은 어떤 `load_model_state`보다 앞서야 함 `R ilora/evaluate.py resolve_eval_checkpoints`
  4. test는 `best_val`·`last`로만 (D11 "test로 고르지 않음"): `splits`에 test가 있는데 `checkpoints`에 다른 값이 있으면 설정 단계 `ConfigError` (15.6.12)
  5. 모델을 한 번 만듦: `load_dataset`·`load_templates`·`load_tokenizer` 뒤 cluster_hard면 M10 14.6.4 0단계 2–3과 같은 순서로 `load_cluster_artifacts`(원 실행과 같은 `cluster.artifact_dir`) → `parse_expert_mapping` → `ClusterHardGate` → `build_model(..., hard_gate=…)` → `verify_live_encoder` (M15 19.6.7; dynamic·none은 `hard_gate=None`, M15 미호출 `[SPEC §22]`). epoch 오름차순으로 M10 `load_model_state(원 실행/checkpoints/epoch{e}, model)` (= `read_trainable_state` → `generated_frozen.*` bit 비교 → M9 `load_trainable_state(strict=True)`, 계약 추가 제안) 뒤 split마다 `evaluate_split(..., ckpt_tag=f"epoch{e}")`
  6. `eval/summary.json`: `source_run`, `routing_mode`, `splits`, `allowed_differences`(허용된 차이, `구역.키: a != b`), `reproduction_keys`, `checkpoints` [{`ckpt_tag`, `source_run`, `checkpoint_dir`, `epoch`, `tags`, `trainable_sha256`}], `cluster_live_check`(cluster_hard만 값, 그 밖은 null), `results` {`<ckpt_tag>/<split>`: {`ckpt_tag`, `split`, `tags`, `metrics`, `aux_protocols`, `files` {이름: sha256}, `reproduction`}}, `created_utc`. `reproduction` = {`status`(`identical`, `mismatch`, `not_comparable`, `absent`), `source_dir`, `differing_keys`(not_comparable만), `records_sha256` {`source`, `eval`}과 `comparison`(identical·mismatch만)}. M13 S2·S6, RUNBOOK R8·R9, M14가 이 키를 읽음 `R ilora/evaluate.py run_eval_command`
- 이 명령이 쓰는 경우: 결정성 재확인 (15.8 #19), E3 text-only 평가 (`--set prompt.eval_prompt=text_only`), batch 무관성 점검 (`--set eval.batch_invariance_check=true`). 실행 순서와 명령은 M14 `RUNBOOK.md`

#### 15.6.12 설정 (`eval` 구역)
| 키 | 타입 | paper_spec 값 | 라벨 | 설명 |
|---|---|---|---|---|
| `batch_size` | int ≥ 1 | 8 | `[결정]` (계보 `[코드]` `U train_movielens.sh:5`, `U data/data_interface.py:145-160`) | 평가 batch 크기. drop_last 없음. 모든 실험 프로파일이 같은 값 |
| `batch_invariance_check` | bool | false | `[결정]` | true면 batch 1로 다시 생성해 비교 (15.6.10). 주 지표는 그대로 |
| `source_run` | Optional[str] | null | `[결정]` | `eval` 명령이 읽을 학습 실행 폴더. `eval` 명령에서 필수, 다른 명령은 읽지 않음 |
| `checkpoints` | tuple[str, ...] | [best_val, last] | `[결정]` D11 | `eval` 명령 전용. 원소는 `best_val`, `last`, `epoch<k>`, `init` |
| `splits` | tuple[str, ...] ⊆ {val, test} | [val, test] | `[결정]` D11·D17 | `eval` 명령 전용 |
- `validate()`: `batch_size ≥ 1`, 목록은 비어 있지 않고 중복 없음, `checkpoints` 형식, `splits`에 test가 있으면 `checkpoints ⊆ {best_val, last}`
- **`batch_size` 8의 이유** `[결정]`
  - 공개 스크립트 batch 8이 validation·test loader에도 그대로 쓰임 (`U data/data_interface.py:127, :145-160`, `U train_movielens.sh:5`) → 같은 batch 크기 (padding 방향만 다름, D22)
  - `drop_last_bs8` 보조 프로토콜이 batch 구성까지 같아짐 (15.6.6)
  - 메모리: KV cache = 2 × 32층 × B × (L + 64) × 4,096 × 2 byte. B = 8, L = 290(M5 9.6.4의 평가 입력 최장 측정값, hybrid·seed 1234)이면 1,484,783,616 byte ≈ 1.38 GiB (계산). 실측은 VESSL_REQUIRED
  - GPU fp16 결과는 batch 모양에 따라 미세하게 바뀔 수 있으므로 E1·E2·E4가 같은 값을 써야 비교가 맞음 (M14)
- 같은 의미의 키를 두지 않음: 생성 설정 `model.gen_*` (M9), 평가 prompt 형식 `prompt.eval_prompt` (M4), worker 수 `batching.num_workers` (M5), ω 허용오차 `records.gate_rowsum_atol` (M12), 판정 정규화는 M2 `match_norm`
- 코드 상수 (실험 의미를 바꾸지 않음): `AUX_DROP_LAST_BATCH_SIZES = (4, 8)` `[코드]` `U main.py:82`, `U train_movielens.sh:5`, `U data/data_interface.py:158`, `WILSON_Z`, `FULL_SPLIT_SIZE` (M1), decode 인자 `[코드]` `U model/model_interface.py:89`

#### 15.6.13 dtype·device·오류 처리
- 텐서: `EvalBatch`는 CPU int64 (M5), 장치 이동은 M9. `GenerationOutput`은 CPU (`new_token_ids` int64, ω·z float32). M11의 후처리는 CPU의 Python int·str
- ω·z는 float32 그대로: JSON에는 float32 값을 Python float로 (정확), NPZ에는 float32. 다른 dtype이면 M12 `validate_*`가 오류 (조용히 바꾸지 않음 `[SPEC §4]`)
- 장치: `model.device` (`cuda:0` 또는 `cpu`). `fork_rng`는 CUDA면 `devices=[0]`, CPU면 `[]`
- 오류 `[결정]`: 모두 원인·split·batch 번호·sample_id를 담은 `EvalError`. M2·M4·M5·M7·M9·M12의 오류는 그대로 올림
  - 예외 때: 기록기 `discard_recording()`, 모델 모드 복원, 난수 복원 (`fork_rng`), 파일은 쓰지 않음 (만든 빈 폴더는 다음 평가가 incomplete로 옮김)
  - 대체 동작 없음: 다시 생성, 표본 건너뛰기, 지표 추정, 잘린 후보로 판정을 하지 않음
- 로그 (`run.log`) `[결정]`: 시작 줄 (split, ckpt_tag, N, batch 크기, batch 수, prompt 형식), 끝 줄 (유효·정답 수, 세 지표, 보조 프로토콜, `n_no_eos`, `n_blank_leading_segment`, 경과 시간). batch마다 줄을 쓰지 않음 (M0 4.6.6)

#### 15.6.14 다른 모듈과의 연결
- M1: `LoadedData.samples(split)`, `catalog` (`titles`, `title_mode`)
- M2: `candidates_for(..., epoch=None)`, `candidate_flags`, `match_norm`, `FLAG_NAMES`, `self_check` (eval 명령 진입)
- M4: `load_templates`, `choose_template(..., None)`, `eval_is_hybrid`
- M5: `EvalBatches`, `make_loader`, `TokenizerBundle`, `CandidateFn`
- M7: `Router.start_recording`·`stop_recording`·`discard_recording`·`gate_calls`, `GateRecord` (`model.omega_source`로 접근)
- M9: `generate`, `GenerationOutput`, `gen_config`, `device`, `omega_source`, `sasrec.hidden_size`·`state_digest`, `build_model`·`load_trainable_state`·`generated_frozen_state` (eval 명령, M10 `load_model_state` 경유)
- M10: 학습 중 호출 (15.6.11), `eval/selection.json`(`best_val_epoch`, `last_epoch`), checkpoint 마커, `read_trainable_state`·`load_model_state`, cluster_hard 준비 순서 (14.6.4 0단계)
- M15 (cluster_hard eval 명령만): `load_cluster_artifacts`, `parse_expert_mapping`, `ClusterHardGate`, `verify_live_encoder`
- M12: `write_eval`, `read_eval_records`, `RepresentationTable`, `save_npz`·`load_npz`, `validate_gate`·`validate_z`. export 대조는 `records.jsonl`의 gate 값을 씀 (16.6.5)
- M14: `metrics.json`, `summary.json`, `batch_invariance.json` → 비교 표와 RUNBOOK. M13: 15.8을 테스트로
- **계약 추가 제안**
  1. 3.1 `EvalRecord` 끝에 `gen_token_ids` 추가. 3.1에 `SplitMetrics`(M11 → M10, M12, M14)와 `EvalResult`(M11 → M10) 행 추가
  2. 3.1 실행 폴더 행: 평가 경로 `eval/{ckpt_tag}/{split}/` (`ckpt_tag` = `epoch{e}` 또는 `init`)와 파일 4개 (15.2). M10 14.2·14.8 #17과 M12 16.2도 이 경로로 맞춤 (리뷰에서 반영됨)
  3. M12 `write_eval`: `metrics.json`의 추가 블록(15.6.9) 통과와 검사 (보조 프로토콜 수 = 앞 N개 기록의 합), `epoch` null(`init`) 허용, `records.jsonl` byte 결정성 (키 순서·separators·`ensure_ascii=False` 고정, sha256 비교용), `records.jsonl` 다음 `metrics.json` 순서, 임시 파일 + `os.replace` (M12 16.6.4에 반영됨)
  4. (해결됨) M7의 `pop_running_mean` ω 누적은 M7 11.7에서 없앰. 학습 ω 통계는 M10이 `TrainForward.omega`로 계산
  5. M10: `read_trainable_state(ckpt_dir)`·`load_model_state(ckpt_dir, model)` (M10 14.6.6, M12 export도 같은 함수), `eval/selection.json`의 키 `best_val_epoch`·`last_epoch` (M10 14.6.7, M12 16.6.5도 같은 경로·키로 맞춤)
  6. M9: `ILoRAModel.device`와 `gen_config`를 읽기 전용 공개 속성으로 (검사·meta 기록용)
  7. M12 export 대조에 z 추가: `eval/.../representations.npz`의 z와 export z의 최대 |차이|를 판정 없이 기록 (GPU 평가 z와 CPU export z의 차이 측정, M12 16.6.5 7단계에 반영됨)
  8. (해결됨, 최종 검토) 속성 이름은 M9 13.6.1·M7 11.6.9의 `ILoRAModel.omega_source`, checkpoint key 접두어 `omega_source.`로 통일 (M10·M12도 같은 이름)

### 15.7 결정
- 20장: D4 (평가 hybrid, `text_only` 변형), D9 (평가도 같은 autocast), D10 (고정 후보), D11 (validation 매 epoch, test는 best·last), D14 (`user_id`), D15 (평가 템플릿 key), D16 (평가 결정성), D17 (지표·분모·N, 보조 프로토콜), D20 (판정 규칙 유지, flag), D22 (왼쪽 padding)
- 이 장의 새 `[결정]`
  - `eval.batch_size` = 8: 공개 스크립트의 평가 batch와 같고, `drop_last_bs8`이 batch 구성까지 같아지며, 메모리 여유가 큼 (15.6.12)
  - 첫 줄바꿈에서 생성을 멈추지 않음: 앞쪽 빈 줄에서 공개 코드와 결과가 달라짐 (15.6.4). 차이 가능 표본 수를 `n_blank_leading_segment`로 기록
  - 보조 프로토콜은 test만, 같은 기록의 앞 N개로 계산 (다시 생성하지 않음)
  - 보조 통계 (Wilson 95%, flag 없는 표본 지표, 생성 진단 수)는 `supplementary`에만, 선택·비교 주 열에 쓰지 않음
  - `combined_metric` = 정수 비 `c/n`, 판정의 문자열 정답과 id 정답의 동치 확인
  - 평가 경계에서 후보·템플릿·hybrid를 key 값과 다시 대조, gate 수집 지점과 기록값 bitwise 대조
  - 평가 전체를 `fork_rng`로 감싸고 모델 모드 복원, `cudnn.benchmark` 금지
  - 완료 표시 = `metrics.json`, 덮어쓰기 금지, 미완료 폴더 보존
  - `eval` 명령: 설정 비교의 허용 목록, test는 `best_val`·`last`로만, checkpoint는 M10 `load_model_state`로만 읽음, cluster_hard는 M10과 같은 M15 준비 순서
  - 재현 점검은 결과를 바꿀 수 있는 허용 차이 키(`eval.batch_size`, `prompt.eval_prompt`, `data.subset`, `data.smoke_users`)가 원 실행과 같을 때만 비교하고, 아니면 `not_comparable` (E3가 재현 실패로 잘못 판정되지 않게, 15.6.10)
  - 평가 최대 GPU 메모리는 평가 시작·끝에 peak 통계를 초기화해 따로 잼 (M10 step 구간 값과 섞지 않음)
  - `EvalRecord`에 `gen_token_ids` (계약 추가 제안)
  - 평가 폴더는 생성 전에 준비(완료 폴더 거부, 미완료 폴더 보존), `representations.npz` meta에 UTC를 넣지 않음 (15.6.8)
  - Wilson 구간은 k = 0, k = n에서 끝값을 정확히 둠 (15.6.7)
  - batch 무관성 목록은 생성 결과가 다른 표본만, gate 끝자리 차이는 수로만 (15.6.10)
  - `eval` 명령: `init` 먼저, 입력 해시는 `inputs`와 변환 SASRec·LLaMA config, 원 실행에 없는 평가는 `absent` (15.6.10, 15.6.11)

### 15.8 완료 기준
- M13에서 테스트로 구현. tokenizer가 필요한 시험은 M5 9.8의 `tokenizer_dir` fixture를 씀
1. (CPU) `first_line`·`judge`가 `U model/model_interface.py:89-90, :165, :468-492`를 그대로 옮긴 참조 함수와 같음: 손으로 만든 사례 (빈 제목 후보, 정규화 제목이 같은 두 후보, 부분 문자열 제목, 대소문자, 앞뒤 공백과 `\r`, 줄바꿈으로 시작하는 생성문, 줄바꿈 없는 생성문)와 catalog 제목으로 만든 고정 seed 무작위 사례 10,000건에서 `valid`·`correct`가 모두 같음
2. (CPU) 유효인 모든 사례에서 문자열 정답 ⇔ `matched[0] == target_id`. 후보 19개·21개, 중복 id, 정답 없는 후보 → `EvalError`
3. (CPU, test tokenizer) decode: BOS 자리표·EOS 뒤 `[PAD]`의 유무와 무관하게 같은 문자열, 중간 특수 토큰 8종 제거, 줄바꿈을 포함하는 id는 13 하나 (설계 측정 D2–D4와 같음)
4. (CPU, test tokenizer) 줄바꿈 정지 반례: `[29871, 13, 16213, 9149, 13, 3868, 271, 13]` → `first_line == "Star Wars"`이고, 첫 13에서 자른 decode의 첫 줄은 `""` (D5). `model.gen_config`의 종료 id는 2 하나이고 다른 stopping criteria가 없음
5. (CPU) `trim_generated`: `[a, b, 2, 32000, 32000]` → `(a, b, 2)`, EOS 없는 64개 → 64개. `check_output`이 EOS 뒤 pad가 아닌 값, T = 0, T = 65, T < 64인데 EOS 없는 행, ω `[B, 1, K]`·NaN·float16, z 차원 불일치를 각각 거부
6. (CPU) 지표: 합성 기록 (n, v, c) = (95, 90, 40) → `valid_ratio == 90/95`, `hr_at_1 == 40/90`, `combined_metric == 40/95` (정확히 같음). (95, 0, 0) → `hr_at_1 == 0.0`. n = 0, `correct`인데 `valid` 아님 → `EvalError`
7. (CPU) 보조 프로토콜: test 95 기록에서 `drop_last_bs4` = 앞 92개, `drop_last_bs8` = 앞 88개의 지표, validation은 빈 dict, smoke 10은 8·8. 작은 모델에서 batch 8 평가의 앞 88개 기록 == 앞 88개만 batch 8로 평가한 기록 (bitwise)
8. (CPU) Wilson: 15.6.7 기준값과 1e-6 안에서 같음, n = 0 → null, 모든 구간이 [0, 1] 안, `WILSON_Z == statistics.NormalDist().inv_cdf(0.975)`
9. (CPU) batch 대조: 모의 loader로 행 순서 바꿈, 템플릿 id 하나 바꿈, 후보 순서 바꿈, `hybrid` 반대, 정답 id 다름, 후보 제목 다름 → 각각 `EvalError`
10. (CPU, `tiny_random` + 실제 SASRec·데이터, fp32) validation 94 평가: 파일 3개 생성, 기록 94줄 행 순서, `gate_*` K개, `flags` = M2 `candidate_flags`, M12 `write_eval` 검사 통과, `GateRecord`의 ω·z == 기록·NPZ 값 (bitwise), gate 호출 증가분 12, `candidates_sha256` = `9a07320651cadd0d…` (M2 6.6.5), `template_counts` 30/33/31 (M4 8.6.3). test도 같은 방식 (`1ffdb649…`, 22/36/37)
11. (CPU) 결정성: 10번을 새 실행 폴더에서 두 번, `num_workers` 0과 2 → `records.jsonl`·`representations.npz` sha256이 모두 같음
12. (CPU) batch 무관성: 10번 설정에서 `eval.batch_invariance_check = true` → `batch_invariance.json`의 토큰 일치 수 = 94 (설계 측정: 순수 작은 LLaMA 95/95, logits 최대 |Δ| 2.98e-7)
13. (CPU) 난수·모드: `evaluate_split` 전후 `torch.get_rng_state()`, `random.getstate()`, `np.random.get_state()`가 같음 (M10 14.8 #12). train 모드로 들어오면 train 모드로 돌아감
14. (CPU) 예외: 3번째 batch에서 예외를 넣으면 기록기가 비워져 다음 `start_recording`이 되고, 모델 모드·난수가 복원되며 `metrics.json`이 없음. 다시 평가하면 이전 폴더가 `val.incomplete-<UTC>`로 옮겨짐. 완료 폴더에 다시 평가하면 `EvalError`
15. (CPU) `text_only` 변형 평가: 기록 `hybrid` false, projector 호출 0 (M9 계수), 지표 정상. cluster_hard·K = 1 모델: ω가 one-hot·`ones [B, 1]`, gate 호출 증가분 0, 기록 키 `gate_0..gate_{K−1}`
16. (CPU) `eval` 명령: `prompt.eval_prompt`·`batching.num_workers`·`eval.*` 차이는 통과, `lora`·`router` 구역 값이나 `run.seed` 차이는 `EvalError`. `best_val`·`last`가 `eval/selection.json`의 `best_val_epoch`·`last_epoch`대로 해석됨, `splits = [test]`와 `epoch1` 조합은 `ConfigError`, `init` validation 평가 동작, 원 실행 기록과의 비교가 `summary.json`에 들어감 (같은 설정이면 `identical`, `prompt.eval_prompt = text_only` 또는 `eval.batch_size`가 다르면 `not_comparable`과 그 키). checkpoint는 M10 `load_model_state`로만 읽음 (`generated_frozen.*` 1 bit 바꾸면 실패). cluster_hard 원 실행(합성 군집 산출물)은 M15 준비 순서를 거쳐 one-hot ω로 평가되고, dynamic 원 실행에서는 M15 함수 호출 0회
17. (CPU) `load_eval_result`: 다시 계산한 지표가 `metrics.json`과 같음. 기록 한 줄을 지우거나 `valid`를 바꾼 폴더 → 오류
18. (VESSL_REQUIRED) paper_spec 학습: validation 5회(각 94)와 test(best·last, 각 95)의 폴더·파일과 N 검사, 지표는 TBD. split당 평가 시간과 최대 GPU 메모리 기록 (TBD)
19. (VESSL_REQUIRED) 결정성: best checkpoint를 `main.py eval`로 두 번 평가 → 두 `records.jsonl`과 학습 중 기록의 sha256이 모두 같음 (D16). 다르면 다른 표본과 처음 다른 단계를 기록하고, 원인을 밝히기 전에는 결과를 확정하지 않음
20. (VESSL_REQUIRED) `eval.batch_invariance_check = true`로 best checkpoint의 validation·test 평가: 토큰·첫 줄·판정 일치율 기록 (100%를 요구하지 않음, 값 TBD). `drop_last_bs4` 해석에 씀
21. (VESSL_REQUIRED) `$LLM_PATH` tokenizer로 3·4번 통과
22. (VESSL_REQUIRED) 학습된 모델의 `n_blank_leading_segment`, `n_no_eos`, `n_cut_at_max_tokens` 실측값 기록 (TBD): 줄바꿈 정지가 결과를 바꿨을 표본 수

---

## 16. M12 기록과 gate 분석
상태: 테스트 통과(로컬)

### 16.1 목적과 범위
- **범위**
  - 학습 기록: micro-batch·optimizer step·epoch 수치를 메모리에 모았다가 N step마다 CSV에 붙여 씀. step마다 파일을 쓰지 않음 `[SPEC §4]`
  - 평가 기록: M11의 `EvalRecord`와 지표를 split·checkpoint마다 끝에 한 번 씀
  - gate·표현 export: 학습이 끝난 checkpoint로 train 66,126·validation 94·test 95 표본 전부의 `z`와 `ω`를 LLM 없이(SASRec + gate) 계산해 NPZ·CSV로 저장 `[SPEC §4]`
  - gate 다양성 분석: SPEC §5 지표, 요약 json·csv, 해석 txt `[SPEC §5]`
  - 그림: SPEC §6 그림 4개, 논문 그림 3c 대응 그림, M15가 부르는 군집 그림 함수 `[SPEC §6, §6-1, §8]`
  - 선택 계측: MoE-LoRA 층 출력의 micro-batch별 최대 |값| (fp16 overflow 진단, D9)
- **범위 밖:** 지표 계산(M11), checkpoint 선택(M10, D11), KMeans와 군집 요약(M15), 실험 간 비교 표(M14), `meta.json`(M0)

### 16.2 입력과 출력
**입력**
- `Config.records`(16.6.9), 실행 폴더(M0)
- 학습 중 M10이 넘기는 값 (M10 14.6.4, 14.6.8): `BatchPlan`의 `tau`·`hybrid`, loss, 그룹별 lr, gradient norm, GradScaler scale, ω 통계, 학습 창의 gate 호출·군집 조회 증가분, step 구간 최대 메모리, activation 최대값
- 평가 뒤 M11이 넘기는 `EvalRecord` 목록과 지표, `RepresentationTable` (생성에 쓰인 ω와 z)
- export: 원 학습 실행 폴더(`config.yaml`, `checkpoints/epoch{e}/`, `eval/`), M1 `Sample` 전체, M3 `z`, M7 `Router` (세 모드 공통. cluster_hard는 M15 `ClusterHardGate`를 `hard_gate`로 받음, M7 11.6.1)

**출력** (실행 폴더 아래에만 씀. `records/train/`·`eval/`은 학습 실행 폴더, `records/export/`·`records/analysis/`는 export 실행 폴더. M0 규칙대로 명령마다 새 실행 폴더)

| 경로 | 내용 |
|---|---|
| `records/train/micro_batches.csv` | micro-batch마다 1행 (paper_spec 41,325행 = 5 × 8,265, D3) |
| `records/train/optimizer_steps.csv` | optimizer step마다 1행 (2,585행, D24) |
| `records/train/epochs.csv` | epoch마다 1행 (5행) |
| `records/train/activation_module_max.csv`, `columns.json`, `run_summary.json` | 모듈별 최대값, 열 정의, 실행 요약 |
| `eval/epoch{e}/{split}/records.jsonl`, `metrics.json` | `EvalRecord` 1줄씩, 지표 (D17). M11이 `write_eval`로 씀. 경로의 `split`은 계약 값 `val`/`test` (M11 15.2) |
| `eval/epoch{e}/{split}/representations.npz` | 생성에 쓰인 ω와 z (아래 NPZ 형식, `meta_json.source = "evaluate"`). M11이 `save_npz`로 씀 (3.2 평가 3단계). 같은 폴더의 `batch_invariance.json`은 선택 (M11 15.6.10) |
| `eval/selection.json` | M10이 씀 (`best_val_epoch`, `last_epoch`, M10 14.6.7). export가 읽음 |
| `records/export/epoch{e}/representations_{train,validation,test}.npz` | 아래 NPZ 형식 |
| `records/export/epoch{e}/gates_{validation,test}.csv` | `sample_id, user_id, split, gate_0..gate_{K-1}, argmax_expert, sasrec_0..sasrec_{d-1}` `[SPEC §4]` |
| `records/export/epoch{e}/crosscheck.json`, `manifest.json` | 평가 시점 ω(`eval/…/representations.npz`, `records.jsonl`)와의 대조 결과와 max \|Δz\|, 파일 sha256 |
| `records/analysis/epoch{e}/{split}/` | `gate_summary.json`, `gate_summary.csv`, `gate_interpretation.txt`, `figures/` `[SPEC §5, §6]` |

- **NPZ 형식** (`np.load(allow_pickle=False)`로 읽힘): `sample_id` `<U` [N], `split` `<U` [N], `row` int64 [N], `user_id` int64 [N], `z` float32 [N, d], `omega` float32 [N, K], `argmax_expert` int64 [N], `meta_json` 0차원 문자열(키 정렬 JSON. export는 routing_mode, K, d, checkpoint 폴더와 sha256, SASRec 가중치 sha256, thread·batch, torch 버전, UTC. M11 평가 NPZ는 UTC 없이 15.6.8 2단계의 키). z만 계산하면(M15) `omega`·`argmax_expert` 없음 `[결정]`
- d = M3가 가중치에서 읽은 은닉 차원(64), K = expert 수(4). 둘 다 실행 중 shape으로 다시 확인하고 64·4를 코드에 쓰지 않음 `[SPEC §4]`
- 표본 순서는 M1 행 순서. 파일 이름의 split은 SPEC 표기(`validation`), 파일 안 `split` 값은 계약 값(`val`) `[결정]`
- train은 NPZ만 씀 (CSV면 66,126행 × 72열) `[SPEC §4]` "대용량인 경우 NPZ를 우선 고려한다"

### 16.3 논문 근거
- 그림 3c: "3c exhibits the attention scores over four experts, for ten sequences." `[논문 p.7 그림 3]`, "Each horizontal bar represents a sequence, and the length of the segments within each bar indicates the percentage of attention scores assigned to each expert." `[논문 p.8 §4.1.2]`
- 그림 3c의 40개 값은 9.7–46.7%, argmax가 네 expert에 모두 나뉨 `[논문 p.7 그림 3c (암시)]`. 데이터셋·층·10개 선택 기준은 없음
  - 우리 계산(10행을 합 1로 정규화): 정규화 entropy 평균 0.9599 (0.9185–0.9937), top1−top2 평균 0.0929 (0.0040–0.2773), 평균 gate와의 JS 평균 0.0199 bit. 해석 참고값으로만 씀 (16.6.6, `design/tmp/reference_points.py`)
- "distinct sequences tend to yield different attention scores and activate different experts, while similar sequences incline toward analogous attention scores." `[논문 p.6 §3.2.2]`: 경향 서술이고 논문은 측정하지 않음
  - 식 11 gate는 z에 대해 ‖W_g‖₂/2-Lipschitz(softmax Jacobian의 최대 고유값 ≤ 1/2)라 "비슷한 z → 비슷한 ω"는 구조상 성립 `[논문 p.6 식 11 (암시)]`. 측정할 것은 ω가 표본 사이에서 실제로 얼마나 퍼지는가 `[SPEC §5]`
- softmax가 "preventing any single expert from disproportionately influencing the recommendation" `[논문 p.6 §3.2.2]`: softmax는 합 1만 보장하고 한 expert 쏠림(SPEC Case B)은 막지 않음 `(암시)` → collapse 진단 필요
- half precision 학습 `[논문 App E p.17]` → GradScaler·overflow 기록 (D9)
- 기록 형식과 분석 절차는 논문에 없음 → `[결정]`

### 16.4 공개 코드에서 가져올 것
- 공개 코드에는 gate·표현 기록과 분석 코드가 없음. 기록 항목의 출발점만 가져옴
  - 학습 중 loss, lr, step 기록 `U model/model_interface.py:145-147` (코드 계보 일치 `L model/model_interface.py:73-75`)
  - 표본별 평가 결과(생성문, 정답, 후보) 저장 `U model/model_interface.py:176-179, :211-214` → `EvalRecord` 기록으로 확장
  - 지표 정의와 이름(`hr` = 정답/유효, `metric` = `hr` × ValidRatio) `U model/model_interface.py:180-184, :487-491` (D17)
- 그림 3c와 RQ1 분석 코드는 공개되지 않음. LoRA gradient를 pickle로 모으는 `capture_and_store_gradients`가 정의만 있고 호출되지 않음 `U model/model_interface.py:115-132` (저장 경로 `capture_dir`는 `U main.py` 인자에 없음)
- 참고(근거 아님): fork의 gate 요약·그림 구성 `F analysis/analyze_gates.py`, 진단 A 측정값 `F experiments/IMPLEMENTATION_NOTES.md:535`

### 16.5 버릴 것 (알려진 문제)
- `U`: 진행 표시줄 기록(`prog_bar=True`) `U model/model_interface.py:145-147`과 TensorBoard 기록뿐 `U main.py:51` → 표 형식 기록 없음 (M0 4.6.6)
- `U`: `valid.csv`·`test.csv`를 실행마다 덮어쓰고 표본 id·gate가 없음 `U model/model_interface.py:179, :214`
- `F`: gate 검증 실패를 경고만 하고 저장을 계속함 `F model/model_interface.py:151-154`, 허용오차 1e-3 `F model/routing_utils.py:93` → 즉시 중단, 1e-5
- `F`: `user_id`를 `NOT_AVAILABLE`로 기록 `F model/model_interface.py:127` → pseudo 사용자 ID (D14)
- `F`: train 표본 gate가 없음 (생성 경로에서만 수집) `F experiments/IMPLEMENTATION_NOTES.md` §11j "Open item: gates of all users" → export가 세 split 모두 계산
- `F`: 해석 문구를 entropy 평균·utilization 폭 임계값(0.9, 0.1, 0.5)으로 고름 `F analysis/analyze_gates.py:146-159` → SPEC §5가 금지한 임계값 판정과 같음
- `F`: entropy를 log(p + 1e-12)로 계산 `F analysis/analyze_gates.py:63-64` → 0·log 0 = 0 규칙
- 유지할 한계: 진단 A는 micro-batch마다 최대 |fp16 출력|과 그 모듈만 기록(22,400, `layers.30.mlp.down_proj`, GradScaler skip 4회: step 1, 2, 77, 98) `F experiments/IMPLEMENTATION_NOTES.md:535`. 재구현도 모듈 위치까지만 기록하고 토큰·채널 위치는 기록하지 않음 (비용, 16.6.3)

### 16.6 설계

#### 16.6.1 공개 API
```python
# rebuild/ilora/records.py (M12 기록)
class RecordsError(ValueError): ...                              # M12 오류 (16.6.2의 ValueError)
class GateCrosscheckError(RecordsError): ...                     # .result = 실패한 대조의 수치 (16.6.5 7단계)
MICRO_FIELDS: tuple[str, ...]; EPOCH_FIELDS: tuple[str, ...]
def step_fields(num_experts: int) -> tuple[str, ...]              # omega_mean_0..K-1가 K에 따라 늘어남

class TrainRecorder:
    def __init__(self, run_dir: Path, cfg: RecordsConfig, *, num_experts: int,
                 total_micro_batches: int, total_steps: int) -> None
    @classmethod
    def resume(cls, run_dir: Path, cfg: RecordsConfig, *, num_experts: int, total_micro_batches: int,
               total_steps: int, tau_done: int, global_step: int, epochs_complete: int) -> "TrainRecorder"
    def log_micro_batch(self, row: Mapping[str, object]) -> None   # act_max는 장치 위 0차원 tensor 허용
    def log_optimizer_step(self, row: Mapping[str, object]) -> None
    def log_epoch(self, row: Mapping[str, object]) -> None
    def flush(self) -> None
    def close(self, status: Literal["completed", "failed"], summary: Mapping[str, object]) -> None
    def attach_monitor(self, monitor: ActivationMaxMonitor) -> None   # 모듈 index → 이름, close의 모듈별 최대 (16.6.3)
    def elapsed(self) -> float                                        # time_s 값 (재개하면 남긴 행의 최대에서 이어짐)

class ActivationMaxMonitor:
    def __init__(self, model: nn.Module, wrapper_type: type[nn.Module]) -> None
    def attach(self) -> None; def pause(self) -> None; def resume(self) -> None; def detach(self) -> None
    def end_micro_batch(self, tau: int) -> tuple[Tensor, Tensor]   # (max_abs [], module_idx []) 장치 위, 동기화 없음. tau는 모듈별 최대가 나온 micro-batch 기록용
    module_names: tuple[str, ...]
    def module_maxima(self) -> tuple[np.ndarray, np.ndarray]           # 모듈별 최대 float32 [M], 그 τ int64 [M] (−1 = 관측 없음), 동기화 1회

@dataclass(frozen=True, eq=False)       # numpy 필드: 비교는 numpy로. __post_init__이 dtype·모양, sample_id = split:row, 행 오름차순, argmax = 행 최댓값의 첫 index를 검사
class RepresentationTable:
    split: str; sample_ids: tuple[str, ...]; rows: np.ndarray; user_ids: np.ndarray
    z: np.ndarray                    # float32 [N, d]
    omega: np.ndarray | None         # float32 [N, K]
    argmax: np.ndarray | None        # int64 [N]
    meta: Mapping[str, object]

def validate_z(z: Tensor, *, batch: int, dim: int, sample_ids: Sequence[str] | None = None) -> None
def validate_gate(omega: Tensor, *, batch: int, num_experts: int, rowsum_atol: float,
        sample_ids: Sequence[str] | None = None) -> None             # sample_ids: 오류 메시지의 처음 5행 이름 (16.6.8)
def compute_representations(samples: Sequence[Sample], encoder: SASRecEncoder,
        gate_fn: Callable[[Tensor, list[str]], Tensor] | None, *, num_experts: int,
        batch_size: int, num_threads: int, rowsum_atol: float) -> RepresentationTable   # gate_fn = M7 Router 또는 None
def save_npz(path: Path, table: RepresentationTable) -> str        # sha256 반환
def load_npz(path: Path) -> RepresentationTable
def write_gate_csv(path: Path, table: RepresentationTable) -> str
def read_gate_csv(path: Path) -> RepresentationTable
def write_eval(out_dir: Path, records: Sequence[EvalRecord], metrics: Mapping[str, object], *,
        expected_ids: Sequence[str], num_experts: int, rowsum_atol: float = 1e-5) -> None   # 기본 = M7 ROW_SUM_ATOL, M11은 records.gate_rowsum_atol을 넘김
def read_eval_records(path: Path) -> list[EvalRecord]
def crosscheck_gates(table: RepresentationTable, eval_table: RepresentationTable,
        records: Sequence[EvalRecord], *, atol: float) -> dict           # export ↔ 평가 NPZ ↔ records.jsonl (16.6.5 7단계)
def run_export(cfg: Config, run_dir: Path, *, repo_root: Path) -> None   # main.py export (M10 train·M11 eval 명령과 같은 꼴)

# rebuild/ilora/analysis.py (M12 gate 분석·그림, matplotlib Agg)
def normalized_entropy(p: np.ndarray) -> np.ndarray
def js_divergence_bits(p: np.ndarray, q: np.ndarray) -> np.ndarray
def gate_summary(omega: np.ndarray, *, user_ids: np.ndarray | None) -> dict
def reference_points(num_experts: int) -> dict
def write_gate_summary(out_dir: Path, summary: Mapping[str, object]) -> None   # json + csv
def write_interpretation(out_dir: Path, summary: Mapping[str, object]) -> None
def plot_gate_heatmap(stem: Path, omega: np.ndarray, rows: np.ndarray, *, title: str,
        formats: Sequence[str] = ("png",)) -> None                  # 모든 plot_*와 run_gate_analysis가 formats(records.figure_formats)를 받음
def plot_mean_utilization(stem: Path, mean: np.ndarray, *, title: str) -> None   # plot_argmax_counts,
def plot_entropy_hist(stem: Path, h: np.ndarray, *, title: str) -> None          # plot_ten_sequences도 같은 꼴 (labels = y 눈금 sample_id 추가)
def pca_2d(z_fit: np.ndarray) -> np.ndarray                        # 시각화 전용 [SPEC §8]
def plot_pca_clusters(stem: Path, coords: np.ndarray, labels: np.ndarray, *, n_clusters: int, title: str) -> None
def plot_cluster_gate_heatmap(stem: Path, means: np.ndarray, *, title: str) -> None
def run_gate_analysis(table: RepresentationTable, out_dir: Path, *, ckpt_tag: str, formats: Sequence[str] = ("png",)) -> None
```
- `SASRecEncoder`(M3 7.6.3), `Router`(M7 11.6.1), `RecordsConfig`·`ClusterConfig`(M0 `Config` 구역) 타입 이름은 해당 모듈 문서를 따름
- 위 블록에 더한 인자는 keyword 전용이고 `plot_ten_sequences`의 `labels` 밖은 기본값이 있음. 오류는 `records.RecordsError`와 `analysis.AnalysisError`(둘 다 `ValueError`) `[결정]`: 문서가 오류 클래스를 정하지 않음. 다른 모듈이 쓰는 보조 공개 이름: `read_rows(path, fields)`(기록 CSV를 열 타입대로 읽음, M13 smoke), `export_epochs(source_run, checkpoints)`, `export_router(cfg, state, *, z_dim, hard_gate=None)`, `SPEC_SPLIT_NAMES`(M15 파일 이름), `EVAL_RECORD_FIELDS`, `figure_style()`, `heatmap_order` `R ilora/records.py`, `R ilora/analysis.py`

#### 16.6.2 학습 기록 `TrainRecorder` `[결정]`
- **열** (순서 고정, 단위와 뜻은 `columns.json`. M10 14.6.8이 이 정의대로 행을 만듦)
  - `micro_batches.csv` (`MICRO_FIELDS`): `tau, epoch, step, hybrid, curriculum_p, batch_size, seq_len, n_label_tokens, loss, loss_finite, act_max_abs, act_max_module, time_s`
  - `optimizer_steps.csv` (`step_fields(K)`): `step, epoch, tau_first, tau_last, n_micro, n_hybrid, loss_mean, loss_hybrid_mean, loss_text_mean, nonfinite_loss, lr_lora, lr_projector, lr_gate, grad_norm_lora, grad_norm_projector, grad_norm_gate, grad_norm_total, projector_has_grad, scale_before, scale_after, skipped, omega_mean_0..omega_mean_{K−1}, omega_entropy_mean, gate_calls, cluster_lookups, max_mem_gib, max_mem_reserved_gib, time_s`
  - `epochs.csv` (`EPOCH_FIELDS`): `epoch, last_step, tau_last, plan_digest, n_hybrid, train_loss_mean, n_skipped_steps, val_n_total, val_n_valid, val_n_correct, val_valid_ratio, val_hr_at_1, val_combined_metric, act_max_abs, act_max_module, max_mem_gib, max_mem_reserved_gib, train_s, val_s, save_s, time_s`
- **정의** (계산은 M10, 정의는 여기서 고정)
  - `loss`: 그 micro-batch 응답 토큰 NLL 평균(accumulation 나눗셈 전, M9 `TrainForward.loss`). `loss_mean`: 그 step의 `n_micro`개 평균 (D24: 16, epoch 끝 9). `loss_hybrid_mean`·`loss_text_mean`: 형식별 평균 (그 형식이 없으면 빈 칸). `nonfinite_loss`: 비유한 손실 micro-batch 수
  - `curriculum_p` = 그 τ의 hybrid 확률: `linear` τ/T (D3), `all_hybrid` 1, `reverse` 1 − τ/T (M4 8.6.5). `hybrid`는 `BatchPlan` 추첨 결과
  - `grad_norm_*`: `scaler.unscale_` 뒤·`optimizer.step` 전 fp32 gradient L2 norm. clipping 없음. 그 step에 gradient가 없는 그룹(text-only만 있는 step의 projector, cluster_hard·K = 1의 gate)은 빈 칸. `projector_has_grad`: 0/1
  - `skipped` = `scale_after < scale_before` (GradScaler는 inf/NaN이면 scale을 줄이고 step을 건너뜀). GradScaler가 꺼진 모드(`bf16_amp`, `fp32`)에서는 `get_scale()`이 1.0이라 `scale_before` = `scale_after` = 1.0, `skipped` 0
  - `omega_mean_k`: 그 step 창 표본들의 ω_k 평균, `omega_entropy_mean`: 정규화 entropy H(ω)/ln K의 평균 (0·log 0 = 0, K = 1이면 빈 칸)
  - `gate_calls`·`cluster_lookups`: 그 step 창 안에서 늘어난 M7 `Router.gate_calls`와 M15 `ClusterHardGate.lookups` (학습 호출만, 평가 생성 호출 제외) `[SPEC §11]`. 합계: dynamic `gate_calls` = 학습 micro-batch 수 (E2 41,325), cluster_hard `cluster_lookups` = 학습 표본 수 (E4 5 × 8,265 × 8 = 330,600), 나머지 0. 평가 호출 수는 M11 `metrics.json`의 `gate_calls` 증가분
  - `max_mem_gib`·`max_mem_reserved_gib`: 직전 step 이후 `torch.cuda.max_memory_allocated()`·`max_memory_reserved()` / 2³⁰ (읽은 뒤 `reset_peak_memory_stats`), CPU면 빈 칸. epoch 행은 그 epoch step 값의 최대 (validation 메모리는 M11 `metrics.json`)
  - `time_s`: recorder 생성 뒤 `time.perf_counter()` 경과. `train_s`·`val_s`·`save_s`: epoch 안 구간 시간
- **표기:** bool은 `0/1`, None은 빈 칸, 실수는 `repr(float(x))`(왕복 정확), 문자열은 CSV 인용 규칙. UTF-8, 줄바꿈 `\n`
- **buffer:** 행을 메모리 목록에 둠. `step % flush_every_steps == 0`인 step 행을 받은 직후(step이 0부터라 step 0, N, 2N, … 뒤 `[결정]`: 문자 그대로 따름), `log_epoch` 뒤, M10의 checkpoint 저장 직전, `close`에서만 세 CSV에 붙여 쓰고 `flush()` + `os.fsync()` `[SPEC §4]`
  - header는 파일을 만들 때 한 번. 행의 키가 열 정의와 다르면(누락·추가·타입 불일치) 즉시 `ValueError`. 번호도 검사함 `[결정]`: τ는 1부터, step·epoch는 0부터 빈틈 없이 이어지고, step 행은 앞 step 뒤에 받은 micro 행을 정확히 닫으며(`tau_first`, `tau_last`, `n_micro`), epoch 행은 그 epoch 마지막 step 뒤에 오고, τ·step은 `total_micro_batches`·`total_steps`를 넘지 않음. 새 기록기는 이미 있는 기록 파일을 거부함 (재개는 `resume`) `R ilora/records.py TrainRecorder`
  - 강제 종료(SIGKILL)면 마지막 flush 뒤 최대 20 step(약 320 micro-batch) 기록을 잃음 (허용)
- **resume** `[결정]` (M10 14.6.10 재개): 세 CSV의 header가 현재 열 정의와 같아야 함. 로드한 checkpoint의 경계 밖 행(τ > `tau_done`인 micro-batch 행, step ≥ `global_step`인 step 행, epoch ≥ `epochs_complete`인 epoch 행)은 `<이름>.incomplete-<UTC>.csv`로 옮기고 본 파일은 경계까지로 다시 씀. 경계 안 행 수가 기대(`tau_done`, `global_step`, `epochs_complete`)와 다르면 `ValueError`. 이어 쓰는 행은 중복 없이 연속. `columns.json`도 현재 정의와 같아야 하고, 이전 `run_summary.json`·`activation_module_max.csv`도 `<이름>.incomplete-<UTC>.<확장자>`로 옮김 (덮어쓰기 금지, 이름이 겹치면 `-<n>`). `time_s`는 남긴 행의 최대 `time_s`에서 이어짐. 모듈별 activation 최대(`activation_module_max.csv`)는 재개 뒤의 τ만 다루고(`run_summary.json`의 `activation_max.module_maxima_from_tau`), 전체 최대는 `micro_batches.csv`의 모든 행에서 계산 `[결정]`: 모듈별 최대는 프로세스 메모리에만 있어 중단된 구간 값이 남지 않고, 전체 최대는 `micro_batches.csv` 행으로 다시 셀 수 있음 `R ilora/records.py TrainRecorder.resume`
- **close:** M10의 `finally`에서 반드시 호출. 남은 buffer를 쓰고 `activation_module_max.csv`, `run_summary.json`을 씀: `status`, `routing_mode`(summary의 값, 없으면 `config.yaml`), `num_experts`, `micro_batches`·`optimizer_steps`·`epochs`(쓴 행 수)와 `expected`, `skipped_steps`(번호)·`n_skipped_steps`, `nonfinite_loss_micro_batches`, 학습 `gate_calls`·`cluster_lookups` 합계, `activation_max` {`enabled`, `monitor`, `hook_modules`, `max_abs`, `module`, `tau`, `module_maxima_from_tau`}, `flushes`, `time_s`, `closed_utc`, `train_summary`(M10 `TrainSummary` 필드 전부: best·last, epoch별 val, test 지표, 그룹별 학습 parameter 수 `trainable_params`, 경과 시간 `seconds`, 14.6.7), `files`(sha256). M14는 `train_summary.trainable_params`·`train_summary.seconds.total`을 읽음 (18.6.7). 쓰기 오류는 원래 예외에 이어서 그대로 올림

#### 16.6.3 activation 최대값 계측 `ActivationMaxMonitor` `[결정]`
- **대상:** M8 MoE-LoRA 래퍼 클래스의 모든 instance (paper_spec 224개, D7), `named_modules()` 순서. 찾은 수가 0이면 오류 (조용히 끄지 않음)
- **hook:** forward hook이 출력(기본 경로 + LoRA 합)을 읽기만 함: `mn, mx = torch.aminmax(out.detach())` → `torch.maximum(-mn, mx)`를 모듈별 장치 buffer(float32, [M])에 최대값으로 누적. NaN·Inf는 그대로 전파되어 기록됨
  - gradient checkpointing은 쓰지 않으므로(M9 13.6.5, M10 14.4) 한 micro-batch의 forward hook은 모듈마다 한 번
- `end_micro_batch(tau)`: buffer의 최대값과 모듈 index를 장치 위 0차원 tensor로 돌려주고, 모듈별 전체 최대값과 그 `tau`(인자)를 장치 위에서 갱신한 뒤 buffer를 비움. `.item()`이 없어 동기화가 없음. 값은 flush 때 한 번에 CPU로 옮김. 모듈 index를 이름으로 바꾸고 `close`에서 모듈별 최대(`module_maxima()`)를 쓰려면 기록기가 monitor를 알아야 하므로 M10이 `recorder.attach_monitor(monitor)`를 부름 `[결정]`: 16.6.1의 `TrainRecorder` 생성 인자에 monitor가 없어 따로 넘김. 대상 모듈이 여러 장치에 있으면 `attach()`가 오류
- **켜는 범위:** 학습 micro-batch만. 평가·생성 동안 `pause()` (hook이 바로 반환)
- **비용:** 모듈 출력을 한 번 더 읽는 reduction 224회, 추가 메모리는 [224] float32 buffer뿐 → micro-batch 시간 대비 작을 것으로 예상, 실측은 `VESSL_REQUIRED`
- `attach()` 때 대상 dtype·장치로 `aminmax`를 한 번 시험해 지원되지 않으면 즉시 실패 (torch 2.0 CPU는 Half `aminmax` 미구현을 확인함, CUDA fp16은 `VESSL_REQUIRED`)
- hook은 아무것도 반환하지 않고 난수를 쓰지 않음 → loss·gradient가 hook 유무와 같아야 함 (16.8 #3)

#### 16.6.4 평가 기록 `write_eval`
- M11이 (checkpoint, split)마다 평가를 끝낸 뒤 한 번 부름 `[SPEC §4]`. 같은 폴더의 `representations.npz`는 M11이 그 전에 `save_npz`로 씀 (M11 15.6.8)
- `records.jsonl`: `EvalRecord` 계약의 필드 순서 그대로 1줄 1표본 (M11 15.6.8 표, 끝에 계약 추가 제안 `gen_token_ids`), `gate_0..gate_{K-1}`는 개별 키, 실수는 JSON 기본 repr. byte 결정성 `[결정]`: `json.dumps(obj, ensure_ascii=False, separators=(",", ":"))` + `"\n"`, 키 순서 = 필드 순서, UTF-8 (같은 기록이면 같은 sha256)
- `metrics.json`: `split, epoch(ckpt_tag의 e, init이면 null), n_total, n_valid, n_correct, valid_ratio, hr_at_1, combined_metric`(D17: ValidRatio, HR@1 = 정답/유효 = 코드 `hr`, CombinedMetric = 정답/전체 = 코드 `metric`), M11 추가 블록 `ckpt_tag`, `aux_protocols`, `supplementary` (M11 15.6.9)와 `meta`(`git_commit`, `git_dirty`, `seed`, `dataset: "movielens"`, `routing_mode`, `num_experts`, `lora_r`, `epochs`, `lr_peak`, `micro_batch_size`, `grad_accum`, `num_candidates`, `backbone`, `llm_path_basename`, `gpu`, `torch_version`, `transformers_version`, `peft: "not_used"`; M0 `meta.json`·`Config`에서 복사. 평가 시작·종료 시각·경과 시간은 M11이 넣음) `[SPEC §3]`
- 쓰기 순서: `records.jsonl` → `metrics.json` (각각 임시 이름에 쓰고 `os.replace`). `metrics.json` 존재가 완료 표시 (M11 15.6.8)
- **검사 (실패하면 쓰지 않고 오류):** 표본 id = `expected_ids`(validation 94, test 95, drop_last 없음, D17; smoke는 그 수)이고 순서도 같음 `[결정]`: `aux_protocols` 검사가 행 순서의 앞 ⌊N/b⌋·b개를 셈, 중복 없음, `valid` = (`matched` 1개), `matched` ⊆ `cand_ids`, `meta`는 mapping, 정해진 밖의 `metrics.json` 키 거부, gate 필드 K개·유한·행 합 허용오차, `correct ⇒ valid`, 개수 일치, `valid_ratio = n_valid/n_total`, `hr_at_1 = n_correct/n_valid`(n_valid = 0이면 0, `U model/model_interface.py:487-491`), `combined_metric = n_correct/n_total` (절대 오차 1e-12). `aux_protocols`의 각 블록은 기록 앞 ⌊N/b⌋·b개로 다시 센 수와 같음 (b = 4, 8, test만, val은 빈 객체)

#### 16.6.5 gate·표현 export `run_export` (명령 `export`)
1. export 실행 폴더를 새로 만들고(M0) `meta.json`에 원 실행 경로를 적음. `records.source_run`의 `config.yaml`과 현재 설정의 `data`·`sasrec`·`router`·`lora`·`cluster` 구역이 같은지 확인 (다르면 다른 키를 적고 실패)
2. `records.export_checkpoints`(`best_val`, `last`)를 원 실행 `eval/selection.json`의 `best_val_epoch`·`last_epoch`(M10 14.6.7)로 epoch 번호에 대응시킴 (D11). 같은 epoch면 한 번만
3. M1 `Sample`(split별 행 순서), M3 `SASRecEncoder`를 CPU·eval·`no_grad`·autocast 밖으로 준비 (D12, D23), `torch.set_num_threads(records.export_num_threads)`
4. epoch마다 `gate_fn` = M7 `Router` (세 모드 공통, CPU fp32, M7 11.6.1·11.6.9) `[결정]`
   - dynamic, K ≥ 2: `Router(cfg.router, num_experts=K, z_dim=d, main_seed=seed, hard_gate=None)`에 M10 `read_trainable_state(checkpoints/epoch{e})`의 `omega_source.` 접두어 키(접두어를 떼면 `gate.proj.weight` 하나)를 strict 로드 (M9 13.6.8). 다른 키는 쓰지 않음
   - cluster_hard: M10 14.6.4 0단계 2–3과 같은 순서로 원 실행의 `cluster.artifact_dir`에서 `load_cluster_artifacts` → `parse_expert_mapping` → `ClusterHardGate` → `verify_live_encoder`(export의 CPU 인코더, `device=cpu`)를 거쳐 `Router(..., hard_gate=…)` (M15 19.6.7). live 대조 결과는 `manifest.json`의 `cluster_live_check` `[결정]`: 14.6.10 제안 7·19.6.11·3.1 군집 산출물 행이 live 대조까지 같은 순서를 요구하고, export에는 `build_model`이 없어 export 인코더로 대조함 `R ilora/records.py _cluster_hard_gate`. `ClusterHardGate`는 인자가 `(sample_ids, device)`라 `gate_fn` 자리에 직접 넣지 않음
   - K = 1 (`none`): `Router`가 모두 1인 [B, 1]
5. split마다 `compute_representations`: 행 순서로 `export_batch_size`개씩 연속 batch(마지막은 짧음) → M1 공유 함수 `data.pad_histories`(M1 5.6.13, `hist_ids [B, 10]`, `len_seq [B]`) → `z` → `validate_z` → `ω = gate_fn(z, sample_ids)` → `validate_gate` → `.detach().float().cpu()` 복사를 목록에 모음 → 끝에 이어 붙이고 표본 id 순서 = M1 순서인지 확인, `argmax` 계산
6. 저장: NPZ 3개, `gates_validation.csv`·`gates_test.csv`(실수는 `'%.9g'`, float32 왕복 정확). 파일은 split마다 끝에 한 번 씀 `[SPEC §4]`
7. **대조:** validation·test의 ω를 원 실행의 `eval/epoch{e}/{split}/representations.npz`(M11이 쓴 생성 시점 ω·z, float32 그대로)와 sample_id로 맞춰 비교하고, 같은 폴더 `records.jsonl`의 `gate_*`와 그 NPZ ω가 bit 단위로 같은지도 확인. 표본 집합이 같고 max |Δω| ≤ `gate_crosscheck_atol`이어야 함, argmax 불일치 수와 max |Δz|(판정 없이 기록, GPU 평가 z와 CPU export z의 차이, M11 15.6.14 제안 7)를 `crosscheck.json`에 씀. 평가 기록이 없거나 어긋나면 실패. 생성에 실제 쓰인 ω(공유 gate를 부른 한 지점의 값)와 export 값이 같음을 보임 `[SPEC §4]`
   - max |Δω|가 허용오차를 넘어도 두 split을 모두 대조하고 `crosscheck.json`을 먼저 쓴 뒤 실패함 `[결정]`. split마다 `passed`가 있고, 실패한 split은 `passed` false와 그 수치(max |Δω|와 그 표본 id, argmax 불일치 수, max |Δz|)를 가짐. `manifest.json`과 분석은 쓰지 않음 `R ilora/records.py _export_epoch`
   - 오류 문구도 max |Δz|, argmax 불일치 수, 최악 표본 id를 적음 `R ilora/records.py crosscheck_gates` (`GateCrosscheckError`). 이유: 반올림 오경보와 실제 gate 불일치를 문구만으로 가름. max |Δz|가 M3 CUDA 허용 1e-4(7.8 #12) 근처면 평가 z가 다른 것이고, 훨씬 작으면 gate가 다른 것 (16.7의 여유 측정)
8. K ≥ 2면 `records.analysis_splits`마다 `run_gate_analysis`. 끝으로 `manifest.json`: `source_run`, `epoch`, `routing_mode`, `num_experts`, `z_dim`, `rows`(split별), `checkpoint` {`dir`, `epoch`, `tags`, `trainable_sha256`}, `sasrec` {`state_path`, `state_sha256`, `state_digest`}, `cluster_live_check`(cluster_hard만 값, 그 밖은 null), `export` {`batch_size`, `num_threads`, `torch_version`}, `files`·`analysis_files`(sha256), `analysis`(K = 1이면 하지 않은 이유), `created_utc`. M15 dynamic 비교가 `routing_mode`·`epoch`를, M13 S3과 M12-16 시험이 `files`·`analysis_files`를 읽음 `R ilora/records.py _export_epoch`
- 비용: SASRec 꼴 forward 66,126개가 CPU 1 thread에서 2.4초 (임의 가중치 측정, `design/tmp/sasrec_cpu_timing.py`)

#### 16.6.6 gate 다양성 분석 `[SPEC §5]`
- 입력 ω(float32)를 float64로 바꿔 계산. K < 2면 "gate analysis requires K >= 2" 오류 (E1은 export만)
- **지표**
  1. expert 평균 utilization: `mean(ω_k)` [K]
  2. argmax 분포: 행 최댓값의 첫 index, count·ratio [K], 최댓값 동률 행 수 `argmax_tie_rows` (모두 균등이면 전부 expert 0이 되므로 함께 보고) `[결정]`
  3. 정규화 entropy `H(p)/ln K` (0·log 0 = 0, `scipy.special.entr`): mean, std, median, min, max
  4. expert별 표준편차 [K]
  5. 전체 평균 gate m̄와 각 표본의 JS divergence, 밑 2(0–1 bit) `[결정]`: `(½Σ rel_entr(p, M) + ½Σ rel_entr(m̄, M)) / ln 2`, `M = (p + m̄)/2` (`scipy.special.rel_entr`는 자연로그라 ln 2로 나눔, `design/tmp/reference_points.py`와 같음), 같은 5개 통계
  6. 보조 분산 통계 하나: top1 − top2 차이 (argmax 기반 요약이 얼마나 안정적인지, one-hot과의 거리) 5개 통계 `[결정]`
  7. collapse 진단: 판정값 없음. 참고점만 함께 기록 `reference_points`
     - Case A (모두 균등): entropy 1, JS 0, 표준편차 0, top1−top2 0
     - Case B 예 [0.90, 0.03, 0.03, 0.04] 전부: entropy 0.3130, JS 0, 표준편차 0, argmax 한 expert 100%, top1−top2 0.86. K ≠ 4면 한 expert 0.9, 나머지 0.1/(K − 1)씩 `[결정]`: SPEC 예가 K = 4뿐 `R ilora/analysis.py reference_points`
     - 논문 그림 3c 10행 (16.3, 선택 기준 미상). K = 4에서만 계산하고 다른 K는 '해당 없음'을 적음
     - Case C는 단일 참고점이 없음 (JS·표준편차가 0보다 큼으로 드러남)
- 표준편차는 ddof = 0 (관측 표본 전체를 기술) `[결정]`
- train만 `user_view` `[결정]`: expert별 사용자 간 분산 비율 η²_k = Σ_u n_u(m̄_uk − m̄_k)² / Σ_i(ω_ik − m̄_k)² (pseudo user, D14). 전체 분산 0이면 null과 이유. train은 사용자당 window가 3–506개라 "사용자마다 다른가"와 "같은 사용자 안에서도 달라지는가"를 나눠 볼 수 있음 (D25 해석 보조)
- **출력:** `gate_summary.json` (위 값 + split, n, K, routing_mode, checkpoint, 정의 문자열), `gate_summary.csv` (긴 형식 `metric, expert, statistic, value`, 같은 값), `gate_interpretation.txt` (UTF-8 고정 틀: 1 수치, 2 참고점, 3 지표 읽는 법 고정 문장, 4 "판정하지 않음, 해석은 사람이 작성") `[SPEC §5]`. 임계값으로 문장을 고르는 코드는 두지 않음

#### 16.6.7 그림 `[SPEC §6, §6-1]`
- `figure_style()` 문맥: matplotlib Agg, `font.family = sans-serif`(DejaVu Sans), `font.style = normal`, 굵기 normal, 흰 배경, grid 없음, `constrained_layout=True`, `savefig.dpi = 300`, 기본 색 순환(C0–C3)과 heatmap `viridis`, `axes.unicode_minus = False`(눈금의 음수 기호도 ASCII), 범례 테두리 없음. 그림은 pyplot 없이 `matplotlib.figure.Figure`와 `FigureCanvasAgg`로 만들어 전역 상태를 남기지 않음 `[결정]`. 모든 그림 함수는 `formats`(`records.figure_formats`)마다 저장하고 이미 있는 파일은 덮어쓰지 않음 (분석 폴더도 비어 있어야 함) `R ilora/analysis.py figure_style`
- 그림 안 글자는 영어 ASCII만 `[결정]`: 기본 sans-serif 글꼴에 한글 glyph가 없음. 제목은 짧은 사실 서술, 축 이름 간결, 축 안 주석·막대 위 숫자·셀 숫자 없음, 범례는 필요할 때만
- expert 표기는 그림에서 `Expert 1..K` (= `gate_0..gate_{K-1}`) `[SPEC §6-1]`, 파일·json은 0부터
- 파일 (`records.figure_formats`마다 저장)
  - `gate_heatmap`: 행 = 표본, 열 = Expert 1–K, 값 범위 [0, 1] 고정. 정렬 = argmax 오름차순 → 최대 가중치 내림차순 → 행 번호. 500행 초과면 정렬된 순서에서 `linspace`로 고른 500행만 그림 (통계는 전체), y tick 없음 `[결정]`
  - `expert_mean_utilization`: 직사각형 막대 K개 + 1/K 점선(범례 "1/K")
  - `argmax_expert_distribution`: 막대 K개 (표본 수)
  - `entropy_distribution`: [0, 1]을 20칸으로 나눈 histogram `[결정]`. 반올림으로 [0, 1]을 조금 벗어난 값은 그림에서만 끝 칸에 넣음 (통계는 그대로)
  - `gate_ten_sequences` (그림 3c 대응): test split 행 순서 첫 10개(`test:0`–`test:9`), 가로 누적 막대, 위에서 아래로 행 순서(y 눈금 = sample_id), x = gate 가중치 0–1, 범례 Expert 1–K, 막대 안 숫자 없음. 값은 `gate_ten_sequences.csv`로 따로 씀. 선택 규칙은 gate 값을 보지 않는 고정 규칙 `[결정]`
  - M15용: `pca_2d`(sklearn `PCA(n_components=2, svd_solver="full")`, 시각화 전용, 군집 입력 금지) `[SPEC §8]`, `plot_pca_clusters`(점 크기 작게, 행 순서대로 한 번에 그려 한 군집이 위를 덮지 않게, 범례 `Cluster 0..C-1`), `plot_cluster_gate_heatmap`(행 `Cluster c`, 열 `Expert k+1`, [0, 1] 고정)

#### 16.6.8 dtype·device·난수·오류 처리
- `z`·`ω`는 계약대로 float32만 받음. 다른 dtype이면 오류(조용히 바꾸지 않음), 확인 뒤 `.detach().float().cpu()` `[SPEC §4]`
- shape는 정확히 `[B, d]`, `[B, K]`. `[B, 1, K]` 같은 꼴을 squeeze하지 않고 오류 `[SPEC §4]`
- `validate_gate`: NaN·Inf 없음, 0 ≤ ω ≤ 1 + atol, |행 합 − 1| ≤ `gate_rowsum_atol`. 어긋나면 처음 5개 sample_id(`sample_ids` 인자, 없으면 행 번호)와 값을 메시지에 넣어 실패 (fp32 softmax K = 4 행 합 오차 측정 최대 2.0e-7, 1e6행)
- 통계는 float64 numpy. activation 최대값은 모듈 출력 dtype(fp16 autocast)으로 장치에서 계산하고 flush 때 float32로 CPU에 옮김. GradScaler 값은 step마다 `get_scale()` 한 번
- export는 CPU fp32, `no_grad`, autocast 밖, thread 수 고정. 1 thread에서 반복 결과가 bit 단위로 같았고 8 thread는 최대 7.2e-7 달랐음 (임의 가중치 측정)
- M12는 난수를 쓰지 않음: heatmap 표시 행은 `linspace`, 그림 3c 표본은 고정 규칙 → `derive_seed` key 추가 없음
- 오류는 잡아서 숨기지 않음. 경로 없음, checkpoint 없음, 설정 불일치, 대조 실패, 열 불일치는 모두 원인을 적은 예외

#### 16.6.9 설정 키 (`records` 구역)
| 키 | 타입 | paper_spec 값 | 라벨 | 설명 |
|---|---|---|---|---|
| `flush_every_steps` | int ≥ 1 | 20 | `[결정]` | 학습 기록을 optimizer step N번마다 파일에 씀 (epoch 끝·저장 직전·종료 때도) |
| `activation_max` | bool | true | `[결정]` | MoE-LoRA 층 출력의 micro-batch별 최대 \|값\| 계측 |
| `gate_rowsum_atol` | float > 0 | 1e-5 | `[결정]` | ω 행 합 허용오차 |
| `gate_crosscheck_atol` | float > 0 | 1e-5 | `[결정]` | export ω와 `EvalRecord` ω의 최대 차 |
| `source_run` | Optional[str] | null | `[결정]` | export가 읽을 학습 실행 폴더. `export`에서는 필수 (`--set`) |
| `export_checkpoints` | tuple[str, ...] ⊆ {best_val, last} | [best_val, last] | `[결정]` D11 | export·분석할 checkpoint |
| `export_num_threads` | int ≥ 1 | 1 | `[결정]` | export CPU thread 수 (결정적 z) |
| `export_batch_size` | int ≥ 1 | 1024 | `[결정]` | export batch 크기 |
| `analysis_splits` | tuple[str, ...] ⊆ {test, val, train} | [test, val, train] | `[결정]` | gate 분석 split |
| `figure_formats` | tuple[str, ...] ⊆ {png, pdf} | [png] | `[SPEC §6]` | 그림 형식 |

- 코드 상수(의미를 바꾸지 않음): heatmap 최대 500행, entropy 20칸, 그림 3c 표본 10개 `[논문 p.7 그림 3]`, DPI 300 `[SPEC §6-1]`

#### 16.6.10 다른 모듈과의 연결
- M10 → `TrainRecorder`: `BatchPlan`의 `tau`·`hybrid`, loss, lr, norm, scaler, ω 통계, 학습 창의 `gate_calls`·`lookups` 증가분, 메모리 (M10 14.6.4, 14.6.8). M10은 학습 시작에 `ActivationMaxMonitor(model, MoELoRALinear).attach()`(M8 래퍼)와 `recorder.attach_monitor(monitor)`, 평가 전후 `pause`/`resume`, 재개 때 `TrainRecorder.resume`, `finally`에서 `detach`·`close`
- M11 → `write_eval(EvalRecord, 지표)`, `save_npz(RepresentationTable)` (`eval/{ckpt_tag}/{split}/`)
- M1 `Sample`(`sample_id`, `user_id`, `row`, `history`), M3 `z`, M7 `Router`(cluster_hard는 M15 `ClusterHardGate`를 품음) `ω`, M10 `read_trainable_state`·`eval/selection.json` → export
- M15 ← `compute_representations(gate_fn=None)`(z만), E2 export NPZ의 ω, `pca_2d`·`plot_*`
- M14 ← `metrics.json`, `run_summary.json`(`TrainableParams`, 시간), 분석 결과. SPEC 산출물 대응 `[SPEC §20]`: `results/baseline/gates_test.csv` = `records/export/epoch{a}/gates_test.csv`, `representations_train.npz`, `gate_summary.json`, 그림 4개 = `records/analysis/epoch{a}/test/` (a = D11 주 결과 checkpoint)
- M0: 명령 `export`, 실행 폴더, `meta.json`(읽기만)

#### 16.6.11 계약 추가 제안
- 실행 폴더 하위 경로 `records/train/`, `records/export/epoch{e}/`, `records/analysis/epoch{e}/{split}/`, `eval/epoch{e}/{split}/` (3.1 실행 폴더 행). D11의 epoch별 validation과 (a)·(b) test 평가 기록은 학습 실행 폴더의 `eval/`에 씀 (M10·M11) → export가 이 경로로 대조
- 모듈 간 파일 `RepresentationTable` NPZ (16.2 형식): M12가 정의하고 M11(평가 시점 ω·z)·M12 export·M15(z만)가 쓰고 M14·M15가 읽음
- M3: SASRec 인코더가 가중치에서 읽은 은닉 차원 `hidden_size: int`를 공개 (M3 7.6.3에 있음)
- 공유 함수 `pad_histories(histories, pad_id, length) -> (hist_ids [N, 10] int64, len_seq [N] int64)`(오른쪽 padding 1682)를 `TrainBatch`(M5)·M3 `encode_all`·export가 같이 씀. 최종 검토에서 M1 `data.py`(5.6.13)로 정함
- M10: `eval/selection.json`의 `best_val_epoch`·`last_epoch`(M10 14.6.7)와 `read_trainable_state(ckpt_dir)`(M10 14.6.6, export는 `omega_source.` 키만 씀)
- M7·M15: 누적 계수 `Router.gate_calls`, `ClusterHardGate.lookups`. M10이 학습 창마다 증가분을 기록 (16.6.2)
- 3.2 평가 3단계: 생성에 쓰인 ω·z는 M11이 `eval/{ckpt_tag}/{split}/representations.npz`(M12 형식)와 `EvalRecord.gate_*`로 저장하고, export(LLM 없음, CPU)는 같은 checkpoint로 세 split 전체를 다시 계산해 validation·test에서 대조 (16.6.5 7단계)
- `TrainRecorder` 행 정의(`MICRO_FIELDS`, `step_fields(K)`, `EPOCH_FIELDS`)를 3.1의 `StepRecord` 계약으로 (M10 14.6.10 제안 3)

### 16.7 결정
- 관련 확정 결정: D9(fp16 GradScaler 기록), D11(export할 checkpoint), D12·D23(결정적 z), D14(pseudo user_id), D17(지표 이름·분모), D24(step 행 수)
- 이 장의 새 `[결정]` (20장 등록 후보, 이유는 해당 절)
  - 기록: 단계별 CSV + 요약 json, 20 step마다 flush (step별 I/O 금지와 강제 종료 손실의 절충). activation 계측 기본 켬(224층, `aminmax`, 동기화 없음). 열 정의는 M10 `StepRecord`와 하나 (`step_fields(K)`), `gate_calls`·`cluster_lookups`는 학습 창 증가분, 재개 때 경계 밖 행은 옆 파일로 옮김
  - export의 ω는 세 모드 모두 M7 `Router`로 계산 (dynamic은 `omega_source.` 키만 strict 로드), 대조는 평가 NPZ·`records.jsonl` 둘 다. 대조가 실패해도 `crosscheck.json`(split별 `passed`와 수치)을 먼저 쓰고, 오류 문구에 max |Δz|·argmax 불일치 수·최악 표본 id를 적음 (16.6.5 7단계)
  - 검증: ω 허용오차 1e-5 (측정 오차 2.0e-7의 50배), export ω 대조 1e-5, CSV 실수 `'%.9g'` (`'%.8g'`는 300,000개 중 3,524개 불일치)
    - export ω 대조 1e-5의 여유 (CPU 대리 측정, 2026-11-06 검토 `review/verify-model/v1_gpu_cpu_omega.py`, `v2_budget.py`): 실제 SASRec 가중치와 val·test 189행의 현실적인 fp32 z 오차(평가 batch 8 대 export batch 1,024, fp64, 다른 fp32 구현: max |Δz| 1.9e-6–7.6e-6)에서는 W_g를 초기값의 3배로 키워도 max |Δω| ≤ 1e-6이고, 그 오차 모양을 키우면 |Δz|가 약 1e-4(W_g ×3 1.1e-4, ×1 3.4e-4: M3 CUDA 허용 7.8 #12가 잡는 크기)에 닿아야 대조가 실패함 (GPU 평가 z와 CPU export z의 실제 차는 VESSL_REQUIRED, 16.8 #16, `crosscheck.json`의 max |Δz|)
  - export: 별도 명령·실행 폴더, CPU 1 thread, batch 1024, 행 순서 (반복 실행 bit 단위 동일)
  - 분석: JS 밑 2, ddof 0, argmax 첫 index + 동률 수, 보조 통계 top1−top2 하나, train `user_view` η², 판정 없는 해석 txt, split은 test(SPEC 산출물)·validation·train
  - 그림: 영어 글자, heatmap `viridis` [0, 1], 그림 3c 대응은 test 첫 10행
  - M11 평가 NPZ meta에는 UTC를 넣지 않음 (16.2)
  - `write_eval`은 기록 순서도 `expected_ids`와 같아야 함, flush 규칙은 0부터 센 step에 문자 그대로 (16.6.2, 16.6.4)
  - 재개 때 이전 요약 파일 보존, `time_s` 이어 셈, 모듈별 activation 최대는 재개 뒤 τ만, 기록기에 monitor를 `attach_monitor`로 넘김 (16.6.2, 16.6.3)
  - cluster_hard export도 live z 대조를 함 (16.6.5 4단계), K ≠ 4의 Case B 참고점 (16.6.6), 오류 클래스 `RecordsError`·`AnalysisError` (16.6.1)

### 16.8 완료 기준
1. (CPU) `TrainRecorder`: 열 순서·이름이 정의와 같고 (`step_fields(4)`에 `omega_mean_0..3`, `step_fields(1)`에 `omega_mean_0`), 키 누락·추가·타입 불일치 행은 `ValueError`. 쓴 값을 다시 읽으면 실수가 bit 단위로 같음
2. (CPU) 64 step 동안 파일 쓰기 횟수 = flush 횟수 (20 step마다 + epoch 끝 + 저장 직전 + close). micro-batch마다 쓰지 않음. 중간 예외를 넣어도 `close("failed")` 뒤 모든 행과 `run_summary.json`(status failed)이 남음
3. (CPU) 초소형 fp32 모델: 한 래퍼 출력에 심은 값(예: 1234.5)이 `act_max_abs`와 그 모듈 이름으로 기록됨. hook 유무에 따라 loss·gradient·난수 상태가 bit 단위로 같음. `detach()` 뒤 hook 0개. 래퍼가 없는 모델에 `attach()` → 오류
4. (CPU) `validate_gate`: `[B, 1, K]`, K 불일치, NaN, Inf, 음수, 행 합 1 + 2e-5, float16을 각각 거부하고 fp32 softmax 출력은 통과. `validate_z`: `[B, 1, d]`, d 불일치, float16 거부
5. (CPU) 합성 SASRec(d = 8과 64)과 합성 gate로 `compute_representations`: 순서 = 입력 행 순서, 열 이름이 `sasrec_0..sasrec_{d-1}`(d를 코드에 쓰지 않음), NPZ 왕복 bit 동일·`allow_pickle=False`로 읽힘, CSV 왕복 float32 동일, CSV 열 순서가 SPEC §4와 같음
6. (CPU) 변환된 실제 SASRec 가중치로 z만 export: train 66,126·validation 94·test 95행, d = 64를 가중치에서 읽음, 유한값, 두 번 실행이 bit 단위로 같음
7. (CPU) `crosscheck_gates`: 같은 값 통과, 한 표본 +2e-5 → 실패, 표본 하나 빠짐 → 실패, `records.jsonl` gate와 평가 NPZ ω가 1 bit 다르면 실패, max |Δz|가 `crosscheck.json`에 기록됨. export의 dynamic `Router`가 `read_trainable_state`의 `omega_source.` 키만 strict 로드하고, cluster_hard는 `Router(hard_gate=ClusterHardGate)`로 one-hot ω를 냄
8. (CPU) `gate_summary` 알려진 답: 모두 균등 → entropy 1·JS 0·표준편차 0·top1−top2 0·`argmax_tie_rows` = n. 모두 [0.90, 0.03, 0.03, 0.04] → entropy 0.313045, JS 0, argmax 전부 0. one-hot 행 → entropy 0 (NaN 없음). 두 점 혼합의 JS가 손계산과 1e-12 안에서 같음. η² 손계산 일치
9. (CPU) `gate_summary.json`과 `.csv` 값이 같고, `gate_interpretation.txt`에 참고점 수치가 있고 판정 문장(collapse 여부 TRUE/FALSE 등)이 없음
10. (CPU) 그림: 지정 형식 파일 생성, PNG DPI 300, 모든 축에 `ax.texts`가 비어 있음, 글꼴 italic·bold 없음, train heatmap 행 ≤ 500, `gate_ten_sequences`의 막대 10개가 `test:0..test:9`
11. (CPU) K = 1 실행: export는 `gate_0` = 1로 성공하고 분석은 명시적 오류
12. (CPU) `write_eval`: 표본 수 94/95 불일치, 중복 id, `correct`인데 `valid` 아님, 지표와 개수 불일치, `aux_protocols` 수가 앞 92·88개 기록과 다름을 각각 거부. `epoch: null`(`init`)과 M11 추가 블록(`ckpt_tag`, `aux_protocols`, `supplementary`)은 통과. 같은 기록을 두 번 쓰면 `records.jsonl` sha256이 같음
13. (CPU) `TrainRecorder.resume`: 경계 밖 행(τ·step·epoch)을 심은 세 CSV에서 경계 밖 행이 `*.incomplete-<UTC>.csv`로 옮겨지고 본 파일 행 수가 (`tau_done`, `global_step`, `epochs_complete`)와 같음. header가 다르거나 경계 안 행이 모자라면 `ValueError`
14. (VESSL_REQUIRED) A100 fp16에서 `aminmax` 동작 확인, hook 유무 micro-batch 시간 차 측정값 기록(TBD), hook 유무 loss 동일성
15. (VESSL_REQUIRED) E2 전체 학습 기록: `micro_batches.csv` 41,325행, `optimizer_steps.csv` 2,585행(epoch 끝 step의 `n_micro` = 9), `epochs.csv` 5행, `gate_calls` 합 41,325, 건너뛴 step 목록과 최대 activation(TBD)
16. (VESSL_REQUIRED) E2 export: validation·test 대조 통과(max |Δω| ≤ 1e-5), NPZ·CSV·분석·그림 생성. 수치는 TBD

---

## 17. M13 테스트
상태: 테스트 통과(로컬)

### 17.1 목적과 범위
- **목적:** M0–M12·M15 장의 완료 기준(n.8) 187개를 빠짐없이 pytest 시험으로 옮기고, 로컬(Windows, CPU)과 VESSL(Linux, A100 1장)에서 무엇을 어떤 명령으로 확인하는지 정한다 `[SPEC §0.13, §21]`
- **범위**
  - 시험 폴더·파일 구성, fixture, marker·skip 규칙, 기준 번호 표시와 대응 누락 검사
  - 공개 코드 동작을 옮긴 참조 구현(대조 전용, 출처 표기)
  - 동등성 시험 6종: MoE-LoRA ↔ fp64 naive, K = 1 ↔ 표준 LoRA, SASRec ↔ 원본 기준 출력, checkpoint 저장·로드 ↔ 같은 출력, batch 생성 ↔ 한 표본씩 생성, worker 0/2/8 무관
  - smoke 3단계: (1) CPU 전체 경로, (2) GPU 100명 (D14), (3) GPU 8–16 표본 과적합. 명령 `main.py smoke`
  - 완료 기준 대응표(17.6.9)와 자동 판정할 수 없는 항목(17.6.10)
- **범위 밖:** 각 모듈 기능 자체(M0–M12, M15), 실험 프로파일 값과 VESSL 실행 순서(M14. 단 smoke·과적합 프로파일의 시험 목적 값은 이 장에서 정하고 M14가 파일로 적음), 결과 해석
- **원칙**
  - 시험은 저장소의 fork·upstream 코드(`model/`, `data/`, `recommender/` 등)를 import하지 않음 (1.3). 대조할 공개 코드 동작은 `tests/refs_upstream.py`에 출처를 적어 옮김
  - 시험은 `tmp_path`에만 씀. 저장소 파일은 읽기만 함 (17.8 #4)
  - 측정만 하고 판정 기준이 설계되지 않은 VESSL 값은 기록하고 TBD로 둠 `[SPEC §0.11]`

### 17.2 입력과 출력
**입력**
- 저장소 파일 (M0 해시 확인 대상): `data/ref/movielens/*`, `prompt/movie.txt`, `rec_model/movielens.pt`
- 시험 tokenizer 폴더: `ILORA_TEST_TOKENIZER` (로컬은 LLaMA SentencePiece 시험 tokenizer, `tokenizer.model` sha256 `9e556afd…d347`. VESSL은 `$LLM_PATH`) (M5 9.8)
- 변환 SASRec 산출물: `ILORA_SASREC_STATE` 또는 기본 위치 `runs/rebuild/assets/sasrec/` (M3 7.6.1 도구 출력)
- 작은 무작위 LLaMA (`model.backbone = tiny_random`, M9 13.6.9), 실제 데이터의 작은 부분집합 (smoke 2명/10/10, 앞쪽 표본 조각)
- VESSL만: `LLM_PATH`, `ILORA_VESSL=1`, 실험 실행 폴더 경로 `ILORA_RUN_<이름>` (M14 RUNBOOK이 설정)

**출력**
- pytest 결과 (통과·실패·skip 이유). 시험 산출물은 `tmp_path` 아래에만
- smoke: `runs/rebuild/<프로파일>_<UTC>/smoke_report.json` (17.6.6–17.6.8)과 단계별 하위 실행 폴더 `steps/`
- VESSL 측정 시험: `runs/rebuild/vessl_tests_<UTC>/measurements.json` (측정값, 판정 없는 값은 기록만)

### 17.3 논문 근거
- 시험 방법은 iLoRA·LLaRA 논문에 없음 → 시험 구성은 모두 `[결정]`. 아래 불변식만 논문에서 옴
- **expert 분할:** r* = r/K, 전체 parameter 수 = 표준 LoRA `[논문 p.5 §3.2.1 식 9]` → M8 #1·#4
- **gate:** ω = Softmax(W_g z), W_g ∈ R^{K×d} → 행 합 1, 0 이상 `[논문 p.6 식 11]` → M7 #1·#2, `GateContext` 검사
- **혼합과 K = 1:** ΔW = Σ_k ω_k B_k A_k `[논문 p.6 식 12]`. K = 1이면 B_1 = B, A_1 = A, ω = 1이라 표준 LoRA h' = Wh + (α/r)BAh `[논문 p.3 식 3]`와 같음 (대수). 그림 4a의 K = 1 값이 LLaRA 행과 같아 보임 `[논문 p.8 그림 4a (암시)]` → 동등성 시험 2
- **z의 입력:** z = SR-EMB(i<n), 정답 i_n 미포함 `[논문 p.5 식 10]` → M3 입력 계약, M9 #7 (정답 id를 바꿔도 z 같음)
- **학습 목표:** 응답 토큰 로그우도 `[논문 p.3 식 2]`, `[LLaRA 간접 §4.2 p.5]` → M5 #2·#6, M9 #4
- **curriculum:** p(τ) = τ/T, τ = T에서 1 `[LLaRA 간접 §4.3 p.6 식 9]`, text-only 손실에 Θp 없음 `[LLaRA 간접 §4.3 p.6 식 7]` → M4 #6, M6 #5·#6, M10 #5
- **지표:** ValidRatio는 "over all sequences" `[논문 p.6 §4]`, `[LLaRA §5.1.3 p.7]` → 평가 N = split 전체 (M11 #6·#10)
- **sequence마다 ω 하나:** `[논문 p.7 그림 3c (암시)]` → gate 값 수집 지점 하나 (M7 #12, M11 #10)
- 로컬은 정적 검증·CPU 시험만, GPU는 VESSL `[SPEC §0.13]`. 요구 시험 목록 `[SPEC §21]` (17.6.12의 대응표)

### 17.4 공개 코드에서 가져올 것
- 공개 iLoRA에는 시험이 사실상 없음: 유일한 `U model/peft/tuners/test_moelora.py`(96줄)는 모양만 확인하고(`:41, :47, :54`) 가져올 것이 없음 (17.5). `L`에는 평가 스크립트 `test_*.sh`뿐
- **대조 기준 동작** (`tests/refs_upstream.py`에 옮김, 각 함수 docstring에 출처) `[코드]`
  - 첫 줄·판정·지표: `U model/model_interface.py:89-90, :165, :468-492` (M11 #1, M2 #6)
  - prompt 치환 연쇄: `U data/data_interface.py:43-48, :56-63` (M4 #3)
  - 쌍 인코딩 label mask: `U model/model_interface.py:38-42` + 4.28.0 `tokenizer([[p, a]])` (M5 #2)
  - 행동 토큰 삽입 반복: `U model/model_interface.py:453-461` (M9 #5)
  - lr 일정: `U optims.py:20-36` (M10 #2)
  - expert별 einsum과 gate 합: `U model/peft/tuners/moelora.py:750-783` (M8 #2)
  - fork 표본 번호: 필터 후 위치 `F data/movielens_data.py:30` (M1 #8, 측정 대조용)
- **방법 참고 (근거 아님):** `F scripts/test_moelora_equivalence.py`의 비교 방식(같은 가중치·입력으로 fp64·fp32 출력과 gradient 비교, dropout 같은 mask, 생성 1토큰 단계, 저장 byte 비교). 재현값은 M8 12.4

### 17.5 버릴 것 (알려진 문제)
- **upstream 시험 파일이 실행되지 않음:** `U model/peft/tuners/test_moelora.py:57`이 `unittest.main()import torch`(두 파일이 붙음)로 구문 오류 (`ast.parse` 확인), `:36`의 `MoELoRA` 클래스는 `moelora.py`에 없음(있는 것은 `MoeLoraModel`, `Linear` 등), `:32` `from moelora import *`는 패키지 밖 import → 시험으로 쓰지 않음
- **fork 점검 스크립트 방식:** 독립 스크립트가 결과를 출력하고 종료 코드만 냄, fork 모듈을 import함 (`F scripts/smoke_test_routing.py:29` `from model.routing_utils import …`), 시험 시점에 git에서 upstream 파일을 꺼내 import함 (`F scripts/test_moelora_equivalence.py`) → pytest 시험, 참조 동작은 출처를 적어 복사, 저장소 코드 import 없음
- **개수만 보는 GPU smoke:** fork bootstrap의 GPU smoke는 test 95개 평가와 `gates_test.csv` 95행만 확인 (`F experiments/VESSL_GUIDE.md:49`) → 상태 저장·재로드 일치, 기록 행 수, 수집 지점 대조까지 확인 (17.6.7)
- **경고로 넘어가는 검사** (예: gate 검증 경고 후 계속, `F model/model_interface.py:151-154`) → 시험과 smoke 검사는 모두 assert 또는 예외

### 17.6 설계

#### 17.6.1 폴더·파일 구성과 실행 명령 `[결정]`
```
rebuild/
  pytest.ini                 testpaths = tests, markers 등록, addopts = -p no:cacheprovider -ra --strict-markers
  ilora/smoke.py             main.py smoke의 3단계 계획과 검사 (계약 추가 제안 17.6.12)
  ilora/commands.py          명령 분기 run_command (M0 소유, 계약 추가 제안 17.6.12)
  tests/
    conftest.py              fixture (17.6.2), marker skip 규칙 (17.6.3)
    refs_upstream.py         공개 코드 동작 사본 (17.6.4). 시험 파일이 아님
    test_m0_seeding.py       (있음)
    test_m0_config_run.py    (있음, 시험 추가)
    test_m0_sections.py      실제 설정 schema·프로파일·main.py --help
    test_m1_data.py  test_m2_candidates.py  test_m3_convert.py  test_m3_sasrec.py
    test_m4_prompts.py  test_m5_batching.py  test_m6_projector.py  test_m7_router.py
    test_m8_moe_lora.py  test_m9_model.py  test_m10_train.py  test_m11_evaluate.py
    test_m12_records.py  test_m12_analysis.py  test_m15_cluster.py
    test_workers.py          worker 0/2/8 무관 (M0·M2·M4·M5·M11 공통)
    test_m13_coverage.py     기준 번호 대응 누락·저장소 import·compileall (17.6.3)
    test_m14_profiles.py  test_m14_runbook.py  test_m14_compare.py   (M14 18.8)
    test_smoke_cpu.py        smoke 1단계 (17.6.6)
    vessl/
      conftest.py            ILORA_VESSL=1과 CUDA가 없으면 전부 skip
      test_vessl_env.py  test_vessl_sasrec.py  test_vessl_model.py  test_vessl_train.py
      test_run_outputs.py    실험 실행 폴더 검사 (M14 RUNBOOK 단계 사이)
```
- **시험 실행기:** pytest 9.2.0 (공개 requirements에 없음 → 로컬 시험 환경 판으로 고정, VESSL도 같은 판 설치 M14). Python 3.10, torch 2.0.0, transformers 4.28.0, numpy 1.24.2, pandas 1.5.3, scikit-learn 1.2.2, scipy 1.10.1, sentencepiece 0.1.97, matplotlib 3.7.1 (`U requirements.txt`와 같은 판, 로컬 확인). 로컬의 joblib 1.6.0·threadpoolctl 3.8.0은 `U requirements.txt:49, :127`(1.2.0, 3.1.0)과 다름: M15는 scikit-learn 판만 고정하므로 시험에는 영향 없음, VESSL은 requirements 판
- **명령** (저장소 루트에서, `-I`로 `PYTHONPATH`·사용자 site 무시. 경로는 `conftest.py`가 `rebuild/`를 `sys.path`에 넣음)
  - 로컬 빠른 묶음: `python -I -m pytest rebuild/tests -m "not slow"`
  - 로컬 전체: `python -I -m pytest rebuild/tests` (vessl 시험은 "VESSL_REQUIRED"로 skip)
  - VESSL CPU 묶음 (Linux 재확인, 실제 tokenizer): `ILORA_TEST_TOKENIZER=$LLM_PATH python -I -m pytest rebuild/tests`. `tests/vessl`도 모으지만 `ILORA_VESSL`이 없으므로 그 시험은 모두 `VESSL_REQUIRED`로 skip됨. 이 묶음에서 `VESSL_REQUIRED` 밖의 skip은 0 (`-ra` 요약으로 확인) `[결정]`: 명령은 로컬 전체와 같게 두고(그 시험은 GPU 묶음이 돌림) 판정 문구를 맞춤 (M14 18.6.6 R3) `R RUNBOOK.md R3`
  - VESSL GPU 묶음: `ILORA_VESSL=1 python -I -m pytest rebuild/tests/vessl -k "not run_outputs"`
  - 실험 뒤 실행 폴더 검사: `ILORA_VESSL=1 python -I -m pytest rebuild/tests/vessl/test_run_outputs.py -k <선택어>` (M14 18.6.6). 선택어와 시험 이름 머리: `smoke_gpu`, `overfit`, `e2_run`(E2 학습 실행), `e2_reeval`, `e2_export`, `cluster`, `e4`, `e1`
- Windows: DataLoader worker는 spawn이라 dataset·후보 함수가 pickle 가능해야 함 (M5 9.6.7). 하위 프로세스는 부모의 `sys.path`를 받으므로 `-I`에서도 `ilora`를 찾음

#### 17.6.2 fixture와 환경 변수
| fixture | 범위 | 내용 | 없을 때 |
|---|---|---|---|
| `repo_root`, `rebuild_dir` | session | 저장소 루트, `rebuild/` (있음) | – |
| `tokenizer_dir` | session | `ILORA_TEST_TOKENIZER` 폴더 (있음) | skip "test tokenizer not available" |
| `sasrec_state_path` | session | `ILORA_SASREC_STATE`, 없으면 기본 위치 `runs/rebuild/assets/sasrec/sasrec_movielens_state.pt` (계약 추가 제안: 기본 위치 추가) | skip |
| `sasrec_reference_path` | session | state 파일과 같은 폴더의 `sasrec_movielens_reference.pt` | skip |
| `profile_cfg(name, *sets)` | 함수 | `load_config(rebuild/configs/{name}.yaml, sets)` (M0). SASRec 경로는 `ILORA_SASREC_STATE`가 있을 때만 그 값으로 `--set` (기본 위치는 프로파일 값과 같음) `[결정]`: skip하는 `sasrec_state_path`에 기대면 자산이 없을 때 프로파일 시험까지 skip됨 `R tests/conftest.py sasrec_overrides` | – |
| `paper_cfg`, `cpu_cfg` | session | `paper_spec`, `smoke_cpu` 프로파일 (M14) | – |
| `full_data` | session | `load_dataset(paper_cfg, repo_root)` (M1, 실제 배포 파일) | – |
| `smoke_data` | session | `select_smoke(full_data, 1234, (2, 10, 10), train_rows=())` = `smoke_cpu`의 데이터 (17.6.6) | – |
| `bundle` | session | `load_tokenizer(tokenizer_dir, paper_cfg.batching.tokenizer_sha256)` (M5) | `tokenizer_dir` skip |
| `templates` | session | `load_templates(repo_root / "prompt/movie.txt")` (M4) | – |
| `tiny_model(*sets, hard_gate=None)` | 함수 (factory) | `build_model(cpu_cfg + sets, bundle, smoke_data.catalog, llm_path=None, hard_gate=…)` (M9 13.6.9: hidden 64, 2층, CPU fp32) | 위 둘 skip |
| `fake_hard_gate(table, K, *, fault=None)` | 함수 | `HardGate` 프로토콜(M7 11.6.1)을 만족하는 시험용 객체: dict 조회 → one-hot, `lookups`·`calls` 계수, 모르는 id → `KeyError`. `fault`로 잘못된 출력: `dtype`(float64), `shape`(`[B, K + 1]`), `batch_axis`(`[B, 1, K]`), `two_hot`, `half`(0.5·0.5), `nan`, `negative`, `rowsum`, `zeros`, `device`(CPU 요청에 `meta`) `[결정]`: M7-6의 다른 장치·`[B, 1, K]`·0.5 사례를 CPU만 있는 기계에서도 만듦 `R tests/conftest.py FakeHardGate` | – |
| `smoke_cluster_dir` | session | `run_cluster(cpu_cfg, <tmp 실행 폴더>, repo_root=repo_root)`(M15)가 만든 `cluster/` (smoke train 156표본) | SASRec skip |
| `converted_sasrec` | session (쓰는 시험이 `slow` 표시: pytest 9는 fixture의 mark를 무시) | `tools/convert_sasrec.py --repo <repo_root> --out <tmp>` 실행 결과 폴더 (M3 7.6.1) | git 없음·shallow면 skip (`SKIP_NO_GIT_HISTORY`) |
| `git_objects_dir` | session | 저장소의 git object 폴더(`git rev-parse --git-path objects`). M3-1의 임시 저장소가 `objects/info/alternates`로 읽기만 함 (commit을 만들지 않음) | git 없음·shallow면 skip (`SKIP_NO_GIT_HISTORY`) |
| `cpu_smoke_run` | session (쓰는 시험이 `slow` 표시) | `smoke.run_smoke(smoke_cpu, tmp 실행 폴더)` 결과 `SmokeReport` (17.6.6). M13 smoke 시험(`test_smoke_cpu.py`)과 M14 비교 시험이 산출물을 재사용 (M10·M11·M12의 실제 실행 시험은 17.6.9 머리말) | 위 자산 skip |
| `llama7b_skeleton` | session | 4.28.0 `LlamaConfig()` 기본값(hidden 4096, 11008, 32층, 32 head, vocab 32000)의 `LlamaForCausalLM`을 meta 장치에 생성 (가중치 없음) | – |
| `torch_threads1` | 함수 | `torch.set_num_threads(1)` 후 복원 (bitwise 비교 시험) | – |
| VESSL `llm_path`, `llama_bundle`, `llama_model` | session | `LLM_PATH`, 그 tokenizer, `build_model(paper_cfg, …)` (cuda:0, fp16) | skip |
| VESSL `run_dir_env(name)` | 함수 | `ILORA_RUN_<NAME>` 실행 폴더 (M14 18.6.6 이름) | skip "run not available" |
| VESSL `measurements` | session | 측정값 dict, 끝에 값이 있으면 `runs/rebuild/vessl_tests_<UTC>/measurements.json`(M0 `create_run_dir`) | – |
| VESSL `lightning` | session | env의 `pytorch_lightning` (M10-18 계보 확인 전용). import가 안 되면 이유를 `measurements`에 남기고 skip | `SKIP_NO_PL` |

- 시험의 합성 입력은 `seeding.np_rng(0, "test", <이름>)`·`seeding.torch_seed_scope(0, "test", <이름>)`로 만듦 `[결정]`: M0의 "난수원은 derive_seed로만" 규칙을 시험에도 적용해 시험 값이 실행마다 같음
- 환경 변수 요약: `ILORA_TEST_TOKENIZER`, `ILORA_SASREC_STATE`(선택), `ILORA_VESSL`, `LLM_PATH`, `ILORA_RUN_*`. token 등 비밀값은 읽지 않음

#### 17.6.3 marker, skip, 기준 번호
- `pytest.ini`에 등록 (`--strict-markers`로 오타 marker는 수집 오류)
  - `slow`: CPU에서 수십 초 이상 (전체 데이터 순회, 학습 2회, 66,315개 SASRec, silhouette). 기본 실행에 포함, `-m "not slow"`로 제외
  - `vessl`: GPU·`LLM_PATH`·VESSL 실행 폴더가 필요. `vessl/conftest.py`가 `ILORA_VESSL != "1"` 또는 CUDA 없음이면 "VESSL_REQUIRED"로 skip
  - `criterion(*ids)`: 시험이 구현하는 완료 기준 번호 (`"M5-2"` = M5 9.8의 2번). 한 시험이 여러 번호를 가질 수 있음
- `test_m13_coverage.py`
  - `test_every_criterion_has_a_test`: `rebuild/DESIGN.md`에서 "## c. Mn" 장마다 "### c.8 완료 기준" 아래 `^\d+\. ` 줄을 세어 번호 집합(M0-1 … M15-14, M13·M14 포함)을 만들고, `rebuild/tests/**/*.py`를 `ast`로 훑어 `pytest.mark.criterion(...)` 인자를 모은 집합과 비교. 문서에 있는데 시험이 없는 번호, 시험에 있는데 문서에 없는 번호 모두 실패. 17.6.9 `합계` 줄의 모듈별 수도 각 장의 기준 수와 같아야 함 (M13·M14는 합계 밖). 정적 분석이라 `-k`·`-m` 선택과 무관
  - `test_no_repo_imports`: `rebuild/` 아래 모든 `.py`(tests 포함)의 import 최상위 이름이 `ilora`, 표준 라이브러리, 설치된 패키지뿐 (저장소 폴더 이름 `model`, `data`, `recommender`, `optims`, `SASRecModules_ori`, `debug`, `analysis`, `scripts`, `measurement`, `main` 금지). 예외는 `tools/convert_sasrec.py`가 임시 폴더에서 import하는 git blob 원본(M3 7.6.1 5단계)
    - 구현 `[결정]`: 금지 이름에 저장소 루트의 모든 `.py` 파일과 `.py`를 가진 폴더 이름을 더함. 설치된 패키지 = 설치 배포판 이름, 저장소 밖에서 `importlib.util.find_spec`이 찾는 이름(Python 3.10의 `packages_distributions`가 scipy·threadpoolctl을 놓침), 저장소 `requirements.txt`에 고정된 이름(VESSL env에만 있는 `pytorch_lightning`, M10-18). 시험 지역 모듈 `refs_upstream`·`conftest`는 `tests/` 안에서만 허용: 17.6.4 참조 구현을 시험이 import해야 함. 상대 import는 `ilora` 안에서만 `R tests/test_m13_coverage.py test_no_repo_imports`
  - `test_compileall`: `compileall.compile_dir(rebuild/, quiet=1)` 성공 `[SPEC §21]`
  - `test_skip_reasons_documented`: 시험 코드의 `pytest.skip`·`pytest.importorskip`·`mark.skip`·`mark.skipif` 이유가 두 conftest의 `DOCUMENTED_SKIP_REASONS` 상수 이름이거나 그 문구와 같음 (17.8 #2)
  - `test_repo_guard_autouse`: `repo_guard`가 session autouse fixture이고 teardown에서 `git status --porcelain`과 `runs/rebuild/` 목록을 비교함 (17.8 #4)
  - `test_refs_upstream_sources`: `refs_upstream.py`의 함수마다 docstring에 0.2 형식 출처(`U`/`F` 경로:라인)와 "대조 전용"이 있고 `ilora/`가 그 모듈을 import하지 않음 (17.8 #5)

#### 17.6.4 공개 코드 참조 구현 (`tests/refs_upstream.py`) `[결정]`
- 함수마다 원문 위치(`U`/`F` 경로:라인)와 "대조 전용, 실행 경로에서 쓰지 않음"을 docstring에 적음. 원문의 결함도 그대로 둠 (대조 대상이므로)
- `first_line_and_hr1(generate, cand_titles, target_title)` (`U model/model_interface.py:89-90, :165, :468-492`), `code_prompt(template, history_titles, cand_titles, hybrid)` (`U data/data_interface.py:43-48, :56-63`), `code_pair_input_and_labels(tok, prompts, answers)` (`U model/model_interface.py:38-42`, 4.28.0 쌍 인코딩), `wrap_emb_loop(...)` (`U model/model_interface.py:453-461`), `upstream_lr(cur_step, cur_epoch, max_step, …)` (`U optims.py:20-36`), `moelora_einsum(x, A, B, gate, scaling, r, K)` (`U model/peft/tuners/moelora.py:750-783`), `fork_sample_idx(df)` (`F data/movielens_data.py:30`)
- 구현에서 정한 인자 `[결정]`: `wrap_emb_loop(input_embeds, input_ids, his_item_embeds, cans_item_embeds, len_seq, len_cans, *, his_token_id, cans_token_id)`(원문처럼 제자리 수정, zip 절단 유지), `upstream_lr(cur_step, cur_epoch, max_step, *, init_lr, min_lr, warmup_start_lr, warmup_steps)`, `code_pair_input_and_labels(tok, prompts, answers, *, terminator)`(answers는 끝 줄바꿈 없이). `moelora_einsum(..., *, dropout=None)`은 LoRA 항만 돌려주고(base는 호출자가 더함) 2차원 gate `[B, K]`에 공유 router의 가운데 축을 넣음: `U model/router/nlpr.py:80`이 `[B, 1, K]`를 내고, 2차원 그대로면 `:780-783`에서 잘못 broadcast됨. 보조 함수 `code_calculate_hr1`, `upstream_group_lr_lists`, `upstream_scaling`이 더 있음. `code_prompt`는 인용 범위 밖의 `[TargetHere]` 분기(`U data/data_interface.py:49-51`)를 옮기지 않음 `R tests/refs_upstream.py`

#### 17.6.5 동등성 시험 6종
| 시험 | 파일::이름 | 비교와 허용 오차 | 근거 |
|---|---|---|---|
| MoE-LoRA ↔ fp64 naive | `test_m8_moe_lora.py::test_matches_naive_and_upstream_fp64` | fp64에서 출력과 x·A·B gradient 상대오차 < 1e-12, ω(fp32) gradient < 1e-6. 기준 둘: expert별 Σ_k ω_k s B_k(A_k x)와 `refs_upstream.moelora_einsum`. dropout 0·0.1(같은 mask), `[3, 7, d]`·`[2, 1, d]` | M8 #2 |
| K = 1 ↔ 표준 LoRA | `test_m8_moe_lora.py::test_k1_equals_plain_lora`, `test_m9_model.py::test_k1_model_equals_plain_lora` | 층: `base(x) + 4·B(A(dropout(x)))`와 bitwise (두 scaling 규칙). 모델: `tiny_model("lora.num_experts=1", "router.routing_mode=none")`의 logits = 같은 A·B를 단순 LoRA 래퍼(시험 안 20줄 구현)에 넣은 모델의 logits (bitwise, fp32 CPU) | 식 3·9·12, M8 #3 |
| SASRec ↔ 기준 출력 | `test_m3_sasrec.py::test_matches_reference_bitwise`, `::test_encode_all_matches_reference` | split별 4,096개 묶음·thread 1에서 `torch.equal(z, z_ref)`, batch 8은 최대 ≤ 1e-5, `embed_items` bitwise. 기준은 원본 `cacul_h` 출력(M3 도구) | M3 #5·#10 |
| checkpoint 저장·로드 | `test_m9_model.py::test_checkpoint_round_trip`, `test_m10_train.py::test_checkpoint_round_trip_state`, `test_m10_train.py::test_resume_equals_uninterrupted` | 새 모델에 불러온 뒤 logits·생성 bitwise, optimizer·scaler·난수 상태 bitwise, 중단·재개 학습 parameter bitwise (CPU thread 1) | 1.3, M9 #11, M10 #7·#8 |
| batch 생성 ↔ 한 표본씩 | `test_m9_model.py::test_batch_generation_equals_single`, `test_m11_evaluate.py::test_batch_invariance_tiny` | CPU fp32: 생성 토큰 전부 같음 (M11 시험은 val 94/94, M9 시험은 smoke val·test 20표본을 batch 8과 하나씩). GPU fp16은 일치율만 기록 (VESSL, M11 #20) | D22, M9 #9, M11 #12 |
| worker 0/2/8 무관 | `test_workers.py::test_train_batches_identical_across_workers[0-2-8]`, `::test_eval_batches_identical_across_workers[0-2]` | smoke 계획 epoch 0 처음 30 batch의 `TrainBatch` 모든 텐서·목록, 평가 val 전체 `EvalBatch` 모든 필드가 같음. Linux 8 worker 50 batch는 VESSL | M0 #4, M2 #2, M4 #8, M5 #8·#12 |

#### 17.6.6 smoke 1단계: CPU 전체 경로 `[결정]`
- **명령** (로컬 Windows PowerShell, 저장소 루트):
```powershell
$env:ILORA_TEST_TOKENIZER = "<LLaMA SentencePiece 시험 tokenizer 폴더>"   # tokenizer.model sha256 9e556afd…d347
python -I rebuild/tools/convert_sasrec.py --repo . --out runs/rebuild/assets/sasrec   # 처음 한 번 (M3 7.6.1)
python -I rebuild/main.py smoke --config rebuild/configs/smoke_cpu.yaml
```
  - pytest 형태(같은 계획을 tmp 폴더에): `python -I -m pytest rebuild/tests/test_smoke_cpu.py`
- **계획 선택:** `smoke.smoke_plan(cfg)` = `overfit` (`data.train_rows`가 비어 있지 않음), `cpu` (`model.backbone == tiny_random`), `gpu100` (그 밖). 프로파일 값으로만 정함 (명령행 인자 추가 없음). `data.subset`이 `smoke`가 아닌 프로파일은 `SmokeError` (full 데이터로 smoke를 돌리지 않음). `run_smoke`의 첫 검사라 `main.py smoke`에서는 M0가 실행 폴더를 만들고 환경 점검을 한 뒤에 나므로 그 폴더는 `failed`로 남음 (GPU가 없는 기계에서는 환경 점검이 먼저 실패) `R ilora/smoke.py run_smoke`
- **프로파일 `smoke_cpu`** (M14 18.6.5): paper_spec에서 `data.subset = smoke`, `data.smoke_users = [2, 10, 10]`, `model` = (`tiny_random`, `float32`, `cpu`), `train` = (epochs 2, micro-batch 4, accumulation 4, `warmup_divisor` 4, `fp32`), `batching.num_workers = 0`, `eval.batch_invariance_check = true`, `records.flush_every_steps = 5`, `cluster.live_check_n = 64`만 다름
  - 이유: train 2명이면 156표본이라 학습 경로 전부(부분 창 D24, text-only만 있는 창, hybrid 창)가 몇십 micro-batch 안에 나옴. `warmup_divisor` 4는 T = 20에서 w ≥ 1을 만족시키기 위함(M10 14.6.2는 w < 1이면 오류). worker 수 무관은 `test_workers.py`가 따로 확인하므로 smoke는 0. `live_check_n` 64는 train 156개보다 작게
- **계산된 기대값** (seed 1234, M1·M4·M10 식 그대로, 계산 `design/tmp/m1314/smoke_sizes.py`, `curriculum_counts.py`)
  - train 사용자 117, 129 → window 156개. val 행 [1, 7, 15, 20, 26, 44, 59, 61, 77, 90], test 행 [4, 9, 14, 20, 50, 66, 71, 72, 82, 92] (M1 5.6.7과 같음)
  - 2 epoch: NB 39, tail 3, epoch당 step 10, T = 20, w = 5, micro-batch 78, epoch별 hybrid 10 / 28. epoch 0의 step 0–2는 hybrid가 없는 창 (projector grad None 경로), 각 epoch 마지막 창은 micro-batch 3개
  - 1 epoch 실행(S5, S7): T = 10, w = 2, micro-batch 39, hybrid 23
- **단계** (`run_smoke`가 각 단계를 `commands.run_command`(17.6.12 제안 3)로 실행: 설정 = `load_config(프로파일, 기존 --set + 단계 --set)`, 하위 실행 폴더 `steps/S<k>_<이름>_<UTC>/`, `<이름>`은 `check_env`, `train`, `eval`, `export`, `cluster`, `train_e4`, `eval_e3`, `train_e1` `[결정]` `R ilora/smoke.py PLAN_STEPS`)

| 단계 | 명령 | 추가 `--set` | 확인 (모두 assert, 하나라도 실패하면 smoke 실패) |
|---|---|---|---|
| S0 | check-env | – | M0 환경 점검(GPU 생략), 입력 해시, `candidates.self_check()`, SASRec `verify_against_reference` CPU 차이 0 |
| S1 | train (E2 경로) | – | 상태 `completed`. `epoch0`·`epoch1`에 5개 파일과 `COMPLETE`. `trainable.pt` key 36개 (LoRA 28, projector 4, gate 1, generated_frozen 3), 두 epoch 모두 projector 포함. `micro_batches.csv` 78행, `optimizer_steps.csv` 20행(epoch 끝 `n_micro` 3), `epochs.csv` 2행, `n_hybrid` 합 38, `gate_calls` 합 78, `projector_has_grad` = 창에 hybrid가 있음과 일치. `plan_digest` = `make_batch_plan` 재계산값. val 10명 평가 2회, test(best·last, 같은 epoch이면 한 번) 10명, `batch_invariance.json` 토큰 일치 10/10 |
| S2 | eval (재로드) | `eval.source_run=<S1>` | `summary.json` 재현 점검이 (best·last, 같은 epoch이면 한 번) × (val·test) 모두 `identical`이고 `records.jsonl` sha256 같음, 불러온 checkpoint의 `trainable_sha256` = `trainable.pt` sha256 (checkpoint 왕복). 작은 무작위 모델에서는 val 정답 수가 같아 동률 규칙(14.6.7)으로 best = last가 되기 쉬움 (그때 결과 2개) |
| S3 | export | `records.source_run=<S1>` | `crosscheck.json` 통과 (\|Δω\| ≤ 1e-5), NPZ 행 수 train 156·val 10·test 10, 세 split 분석 폴더와 그림, `manifest.json`의 파일 sha256 |
| S4 | cluster | `cluster.dynamic_export_dir=<S3>/records/export/epoch<a>` (a = S1 `selection.json`의 `best_val_epoch`, RUNBOOK R11과 같음) | fit 156행 `train:`만, 빈 군집 없음, joblib 재로드 predict = CSV, E2 export z와 대조 ≤ 1e-5, dynamic 비교 항목 존재 |
| S5 | train (E4 경로) | `router.routing_mode=cluster_hard`, `cluster.artifact_dir=<S4>/cluster`, `train.epochs=1` | optimizer에 gate 그룹 없음, `gate_calls` 합 0, `cluster_lookups` 합 156 (39 × 4), live z 대조 통과, 평가 기록의 gate가 one-hot이고 = mapping[배정 CSV 군집], key 35개 |
| S6 | eval (E3 경로) | `eval.source_run=<S1>`, `prompt.eval_prompt=text_only` | 기록 `hybrid` 모두 false, 지표 파일 존재, 재현 점검은 "비교 대상 아님"으로 표시 (17.6.12 불일치 3) |
| S7 | train (E1 경로) | `lora.num_experts=1`, `router.routing_mode=none`, `train.epochs=1` | 그룹 {projector, lora}, 평가 기록 `gate_0` = 1.0, step 행 `omega_mean_0` = 1.0, key 35개 |
| S8 | 보고 | – | `smoke_report.json`(`format` `ilora-rebuild-smoke-report-v1`): 계획, 단계별 `id`·이름·명령·`--set`·실행 폴더·상태(`completed`·`failed`·`not_run`)·초·오류, 검사 목록(`id` = `<단계>.<이름>`, 단계, 설명, `passed` true·false·null(판정 없이 기록), 측정값, 기대값), `counts`, `failed_steps`, `failed_checks`, 전체 통과 여부. `main.py smoke`면 smoke 실행 `meta.json`에도 `smoke` 블록 `[결정]` `R ilora/smoke.py _SmokeRun.finish` |

- S1–S7은 학습 저장 상태가 조용히 빠지거나(저장 필터), 다시 불러온 출력이 달라지거나, 수집 지점이 둘이 되는 경로를 CPU에서 모두 지남. 정확도 판정은 없음 (무작위 작은 모델)
- 학습 단계(S1·S5·S7, G1·G5·G6, O1)는 표의 확인 외에 공통 검사를 더 함 `[결정]`: optimizer 그룹과 `group_param_names`, 그룹별 Adam 상태 수(17.6.7 #2), scaler 상태 key와 CPU·CUDA 난수 상태, epoch마다 `frozen_checksum_at_save`와 SASRec digest(17.6.7 #2), 연속 skip, τ·step 빈틈 없음, 평가 폴더별 표본 수 `R ilora/smoke.py _training_checks`. batch 무관성은 S1(CPU E2)에서만 판정하고 S5·S7과 GPU 단계는 기록만 (M11 #20)
- 기대값은 코드에 숫자로 두지 않고 각 단계 설정으로 M1·M4·M10 함수(`load_dataset`, `build_batch_plan`, `step_counts`)를 불러 실행 때 계산함 `[결정]`: 같은 코드가 cpu·gpu100·overfit 계획을 판정함. 17.6.6–17.6.8의 숫자 자체는 `test_smoke_cpu.py`, VESSL 보고 시험, M14 #5가 확인
- 실패 처리 `[결정]`: 검사 안의 예외는 오류 문구를 단 실패한 검사, 실패한 단계에 기대는 단계는 `not_run`, `smoke_report.json`은 항상 씀. 계획의 필수 검사(`required_checks(plan, cfg)`)가 빠지면 보고 단계의 `required_checks` 검사가 실패. `passed`는 모든 단계 `completed`, 필수 검사가 모두 있음, 실패한 검사 없음일 때만 true. 단계가 끝날 때마다 `gc.collect()`와 (CUDA면) `torch.cuda.empty_cache()`: GPU에 모델 하나씩 `R ilora/smoke.py _SmokeRun`

#### 17.6.7 smoke 2단계: GPU 100명 (D14, VESSL_REQUIRED) `[결정]`
- 명령: `python -I rebuild/main.py smoke --config rebuild/configs/smoke_gpu_100.yaml` (M14 RUNBOOK R5)
- 프로파일: paper_spec에서 `data.subset = smoke` (smoke_users 80/10/10은 paper_spec 값 그대로, D14), `train.epochs = 2`, `eval.batch_invariance_check = true`만 다름 (M14 18.6.5)
- 계산된 기대값: train 7,230표본, NB 903, tail 7, epoch당 step 57, T = 114, w = 5, micro-batch 1,806, epoch별 hybrid 218 / 670. 1 epoch 실행(E4·E1 경로): T = 57, w = 2, micro-batch 903, hybrid 428, E4 `cluster_lookups` 7,224 (903 × 8)
- 단계: G0 check-env(GPU 1장·75 GiB 이상, `$LLM_PATH` tokenizer sha256, transformers 파일 무결성 17.6.12, SASRec CUDA ↔ CPU 기준 ≤ 1e-4) → G1 train → G2 eval 재로드 → G3 export → G4 cluster → G5 E4 경로 1 epoch → G6 E1 경로 1 epoch → G7 보고. 단계별 `--set`은 17.6.6의 S1–S5와 같고 G6은 S7(E1 경로)과 같음 (E3 단계 없음). 하위 폴더 이름도 S 계획과 같고 G6은 `train_e1` `R ilora/smoke.py PLAN_STEPS`
- **통과 기준 (정확도 아님, 실행과 상태 보존):**
  1. 모든 단계 `completed`, 예외 없음. 연속 skip 16회 없음 (M10)
  2. 저장 상태: epoch마다 `trainable.pt` key 456개(LoRA 448, projector 4, gate 1, generated_frozen 3; E4·E1 경로 455개), `train_state.pt`의 그룹별 Adam 상태 수가 그 epoch까지 skip되지 않은 step에서 gradient를 받은 parameter 수와 같음 (Adam 상태는 한 번 생기면 남으므로 누적 `[결정]`. dynamic은 hybrid 창을 한 번 지난 뒤 453), scaler 상태 5개 key·CPU·CUDA 난수 상태 존재, epoch마다 epoch 끝에 다시 잰 고정 parameter checksum(`frozen_checksum_at_save`)이 epoch 0의 시작값(`frozen_checksum`)과 같고 SASRec digest가 `sasrec.state_digest`와 같음 (M10 14.6.6) `R ilora/smoke.py frozen_sasrec_values`
  3. 재로드: G2의 재평가 `records.jsonl`이 학습 중 기록과 sha256 같음 (best·last × val·test)
  4. 기록 행: micro 1,806, step 114 (epoch 끝 `n_micro` 7), epoch 2, `gate_calls` 합 1,806, `n_hybrid` 합 888
  5. 수집 지점: G3 export ω와 생성 시점 ω 차 ≤ 1e-5 (저장된 gate가 실제 쓰인 gate와 같음), G5 one-hot·`gate_calls` 0·`cluster_lookups` 7,224
  6. 측정 기록 (값 TBD): step 구간·평가 최대 GPU 메모리, 단계별 시간, skip 수, batch 무관성 일치율
- 실패하면 E1·E2·E4 학습을 시작하지 않음 (M14 18.6.6)

#### 17.6.8 smoke 3단계: GPU 과적합 (VESSL_REQUIRED) `[결정]`
- 명령: `python -I rebuild/main.py smoke --config rebuild/configs/overfit_gpu.yaml` (M14 RUNBOOK R6)
- 표본: train 16개 = 사용자 0–15 각자의 마지막 train window (행 71, 196, 377, 554, 797, 1001, 1129, 1234, 1595, 1651, 1905, 2070, 2311, 2693, 2765, 2874, 모두 len_seq 10, 정답 16개가 모두 다르고 item 266 없음. 계산 `design/tmp/m1314/smoke_sizes.py`, `overfit_rows.py`). val·test는 smoke 10/10
  - 선택 규칙은 gate·결과를 보지 않는 고정 규칙. 사용자마다 1개라 이력이 서로 겹치지 않음 (한 사용자의 연속 window는 이력 9개를 공유)
  - 이 16개는 M1 `data.train_rows`(M1 5.6.7·5.6.8, 최종 검토에서 M1이 채택, 기준 M1-13)로 고정. smoke 사용자 1명 선택은 seed 1234에서 사용자 117(132표본)이라 8–16개를 만들 수 없음 (계산)
- 프로파일: paper_spec에서 `data` = (smoke, [16, 10, 10], `train_rows` 16개), `prompt.train_curriculum = all_hybrid`, `train` = (epochs 25, micro-batch 4, accumulation 1). 계산: NB 4, T = 100 step, w = 5, micro-batch 100개 모두 hybrid, checkpoint 25개 (약 10.35 GiB, M10 14.6.6의 epoch당 424 MiB로 계산)
  - all_hybrid 이유: 평가 생성이 hybrid(D4)라 학습·생성 형식을 맞춰 "학습한 것을 생성 경로가 재현하는가"만 봄. text-only 경로는 1·2단계가 지남
  - lr 등 나머지는 paper_spec 값 그대로 (안정성 측정 기록을 그대로 쓰기 위함)
- 단계: O0 check-env → O1 train → O2 과적합 생성 점검 → O3 보고
  - O2는 하위 실행 폴더 없이 smoke 프로세스 안에서 돎: M9 `build_model`로 새 모델을 만들고 O1의 마지막 checkpoint(epoch 24 = `selection.json`의 `last_epoch`)를 M10 `load_model_state`로 불러 `overfit_generation_check(model, bundle, templates, data, cfg)`에 넘김 `[결정]`: 17.6.12 제안 2의 signature에 checkpoint 인자가 없음 `R ilora/smoke.py _SmokeRun._generation`. 함수는 16개마다 마지막 epoch의 학습 key로 후보 `candidates_for(s, main_seed=cfg.run.seed, epoch=24, cfg=cfg.candidates)`·템플릿 `choose_template(cfg.prompt, cfg.run.seed, 3, s, 24)`를 정하고 hybrid prompt를 `build_prompt` → `encode_eval` → M5 `collate_eval`(9.6.1, 왼쪽 padding)로 `EvalBatch`를 만들어(공개 API만 사용) `model.generate` → M11 `trim_generated`·`first_line`
  - M5 `EvalBatches`는 평가 key(epoch None)만 받으므로 train 표본에 쓰지 않음
  - 표본은 행 순서로 `eval.batch_size`(8)개씩 묶음. M4 `check_catalog_titles`, M5 `check_token_counts`, M11 `check_output`도 부르고, 생성은 모델 eval 모드(끝나면 원래 모드)·`autocast_for(train.precision)`·`torch.random.fork_rng` 안에서 함. 반환 key: `epoch`, `n`, `n_match`, `all_match`, `batch_size`, `n_batches`, `hybrid`, 표본별 `samples`(첫 줄, 일치, 새 토큰 수, EOS 여부), `seconds` `R ilora/smoke.py overfit_generation_check`
- **통과 기준:**
  1. 학습 `completed`, 25 epoch `COMPLETE`, micro 100행 모두 hybrid, 연속 skip 16회 없음
  2. 마지막 epoch micro-batch loss 평균 ≤ 첫 epoch 평균 × 0.1 `[결정]`: gradient가 LoRA·projector·gate에 닿고 lr 일정이 동작한다는 거친 신호 (값은 기록, TBD). 평균은 그 epoch의 모든 micro 행으로 내고, 비유한 손실 때문에 비가 유한하지 않으면 실패 (비유한 행 수는 기록) `[결정]`: fp16에서 생길 수 있는 비유한 손실을 조용히 빼지 않음 `R ilora/smoke.py _train_overfit`
  3. 16개 모두 `match_norm(first_line) == match_norm(target_title)` `[결정]`: loss가 낮은데 이것이 실패하면 학습과 생성 경로가 어긋난 것(padding·position, label 정렬 D18, 자리 대응)으로 보고 E1·E2·E4를 시작하지 않음
- 정확도 숫자(일치 수 외)는 보고하지 않음. 이 실행은 실험 결과가 아님

#### 17.6.9 완료 기준 대응표 (M0–M12, M15)
- 열: 번호 = 해당 장 n.8의 번호, 시험 = `rebuild/tests/` 기준 `파일::이름` (`::이름`은 그 행 앞에서 가장 가까이 적힌 같은 모듈의 시험 파일 `test_m<n>_*.py`, VESSL 행은 가장 가까운 `vessl/` 파일. M0의 '(있음)' 이름은 `test_m0_config_run.py`), fixture는 17.6.2 이름. 모든 시험은 `criterion("Mn-k")`를 붙임
- (CPU) 시험은 VESSL의 CPU 묶음(17.6.1)에서도 다시 돌아 Linux 확인을 겸함
- 표는 기준마다 대표 시험을 적음. 같은 번호를 붙인 보조 시험(경계 사례, 검토 뒤 회귀 시험)이 더 있을 수 있음. 17.6.5에만 이름이 있는 시험은 그 근거 열의 번호를 붙임 (`test_m9_model.py::test_k1_model_equals_plain_lora` → M8-3, `test_workers.py::test_eval_batches_identical_across_workers` → M0-4·M5-8). M10-15는 로컬 precision 도구 시험(`test_m0_config_run.py::test_autocast_for_modes`, `::test_make_grad_scaler`)에도 붙음 (4.8에 precision 도구 기준이 없음)
- fixture 열과 다른 점 `[결정]`: M7-7·M10-10·M10-13·M10-14와 M11-16·M12-7의 실제 실행 부분은 `cpu_smoke_run` 대신 `test_m10_train.py`의 module fixture(`reference_run`, `hard_run`)가 같은 설정(S1·S5)의 학습을 `commands.run_command`로 tmp 폴더에 만들어 씀 (`::test_eval_command_on_trained_run`, `::test_export_on_trained_run`): M10 시험이 M13 smoke보다 먼저 만들어졌고, 비트 단위로 비교하는 실행은 CPU thread 1로 돌려야 함 (CPU 재평가는 같은 thread 수에서만 `records.jsonl` sha256이 같음). M10-12는 1 epoch module 실행 `skip_run`, M8-7은 시험 안에서 13.6.9의 tiny LLaMA를 직접 만듦(M9 없이), M3-1은 `git_objects_dir` `R tests/test_m10_train.py`

**M0 (4.8)**
| # | 시험 | 확인 | fixture | 구분 |
|---|---|---|---|---|
| 1 | `test_m0_config_run.py::test_valid_config_round_trip`, `::test_wrong_types_and_choices_are_rejected`, `::test_unknown_and_missing_keys_are_rejected` (있음), `test_m0_sections.py::test_profile_round_trip[<프로파일>]` | 합성 schema와 실제 `sections.Config`의 모든 프로파일에서 저장→재적재 dict·`config_hash` 같음. 알 수 없는 키, 빠진 키, `"False"`, int 자리 `8.0`, 선택지 밖 → `ConfigError` | `tmp_path`, `profile_cfg` | CPU |
| 2 | `test_m0_sections.py::test_every_key_has_label` | `iter_items()` 전부 `LABEL_RE` 라벨과 비어 있지 않은 설명, 구역 14개 = 3.1 `Config` 목록 | – | CPU |
| 3 | `test_run_dir_is_never_reused`, `test_meta_status_and_no_secrets`, `test_secret_values_are_blocked` (있음), `test_m0_config_run.py::test_meta_has_no_environment_dump` | 같은 이름 폴더 거부, 필수 키, token 형태 값 없음, 심어 둔 환경 변수 `ILORA_DUMMY_ENV`의 값이 meta 어디에도 없음 | `tmp_path`, `monkeypatch` | CPU |
| 4 | `test_m0_seeding.py` 전부 (있음), `test_workers.py::test_train_batches_identical_across_workers[0-2-8]` | derive_seed 재현·분리, worker 수와 무관한 batch 순서·후보·템플릿·curriculum | `cpu_cfg`, `smoke_data`, `bundle`, `templates` | CPU `slow` |
| 5 | `test_environment_check_reports_problems`, `test_environment_check_detects_hash_mismatch` (있음), `test_m0_config_run.py::test_environment_check_gpu_count[0-2]`, `::test_environment_check_torch_version`, `::test_environment_check_missing_llama_path`, `::test_environment_check_library_pins[numpy-pandas-sentencepiece-sklearn]`, `::test_environment_check_patched_transformers` | 모의 조건마다 `EnvironmentCheckError`와 해당 문구 (GPU는 `torch.cuda.is_available`·`device_count`·`get_device_properties` monkeypatch) | `tmp_path`, `monkeypatch` | CPU |
| 6 | `test_meta_status_and_no_secrets` (있음), `test_m0_config_run.py::test_keyboard_interrupt_marks_failed`, `::test_killed_process_stays_running`, `::test_run_lock_has_one_holder`, `::test_resume_refuses_a_live_run`, `::test_resume_reopens_failed_runs_only` | 예외·`KeyboardInterrupt` → `failed`와 메시지. 자식 프로세스를 `Popen.kill()`로 끝내면 `running`·`ended_utc` null로 남아 구분됨. `run.lock`은 쥔 동안 두 번째를 거부하고 파일은 줄만 늘어남. 살아 있는 실행의 `train --resume` → `CommandError`, `meta.json`·`config.yaml`·`run.log` byte 그대로, 프로세스를 죽인 뒤에는 다시 열림(`previous_status` running). `completed` 실행은 다시 열지 않음 | `tmp_path` | CPU |

**M1 (5.8)**
| # | 시험 | 확인 | fixture | 구분 |
|---|---|---|---|---|
| 1 | `test_m1_data.py::test_hash_checked_before_unpickle` | 4개 파일을 tmp 저장소로 복사하고 `Val_data.df` 1 byte 변경 → `DataError`, `pandas.read_pickle` spy 호출 0회 | `repo_root`, `tmp_path`, `monkeypatch` | CPU |
| 2 | `::test_row_and_sample_counts` | 68,388/94/95, 필터 후 66,126/94/95, `len(history) == len_seq`, 길이 범위 | `full_data` | CPU |
| 3 | `::test_user_blocks` | 블록 754, 빈 이력 시작, window 길이 순서, 블록마다 3행 제거, val 754–847, test 848–942 | `full_data` | CPU |
| 4 | `::test_fingerprints_and_projection` | fingerprint 3개, 투영 열, `Sample`·`LoadedData`에 cluster 정보 없음 | `repo_root` | CPU |
| 5 | `::test_code_titles_reproduce_defects` | 5.5 결함 문자열 그대로, 중복 22개·45 id | `full_data` | CPU |
| 6 | `::test_fixed_titles_differ_in_12_ids` | 다른 id 12개 집합, 빈 제목 0, 중복 22/45 | `repo_root` | CPU |
| 7 | `::test_item_file_crlf_invariance` | CRLF·LF `u.item`에서 같은 catalog·정규화 해시 | `tmp_path` | CPU |
| 8 | `::test_row_maps_to_fork_sample_idx` | `refs_upstream.fork_sample_idx`로 train 66,126개 `row = idx + 3(user_id + 1)`, val·test `row = idx` | `full_data` | CPU |
| 9 | `::test_smoke_selection_seed1234` | 80명·7,230, val·test 행 목록, 두 번 같음. (2, 10, 10)은 사용자 117·129, 156표본 | `full_data` | CPU |
| 10 | `::test_parse_rows_rejects[<11개 사례>]` | 합성 DataFrame 사례마다 `DataError` (왼쪽·가운데 padding을 따로 셈) | – | CPU |
| 11 | `::test_sample_rejects[<사례>]` | 길이 2·11, bool id, 범위 밖 → `DataError`, `sample_id` 형식 | – | CPU |
| 12 | `vessl/test_vessl_env.py::test_data_on_vessl` + VESSL CPU 묶음의 1–11번 | VESSL 작업 트리 해시, pandas 1.5.3·numpy 1.24.2, fingerprint·표본 수 | `repo_root` | VESSL_REQUIRED |
| 13 | `test_m1_data.py::test_train_rows_subset`, `::test_train_rows_rejects[<7개>]` | `overfit_gpu` 설정: train 16표본·사용자 0–15·val/test 행 = 9번. 필터된 행·train 밖 번호·사용자 수 불일치 → `DataError`, 중복·내림차순·65개·`subset = full` → `ConfigError` | `full_data`, `profile_cfg` | CPU |
| 14 | `::test_pad_histories` | 오른쪽 padding 1682, `[N, 10]` int64·`len_seq`, 길이 0·11·범위 밖·bool → `DataError` | – | CPU |

**M2 (6.8)**
| # | 시험 | 확인 | fixture | 구분 |
|---|---|---|---|---|
| 1 | `test_m2_candidates.py::test_candidate_contract` | val·test 전부, train 처음 2,000 × epoch 0·1: 20개, 정답 1, 중복 없음, 이력과 겹침 없음, 0–1681 | `full_data`, `paper_cfg` | CPU |
| 2 | `::test_deterministic_and_epoch_dependent`, `test_workers.py::test_train_batches_identical_across_workers` | 같은 key 같음, train row 3의 epoch 0 ≠ 1, worker 0·2·8 `cand_ids` 같음 | 〃 | CPU |
| 3 | `::test_golden_rows_and_split_sha256`, `::test_self_check_passes` | 6.6.5 tuple 2개, sha256 `9a073206…`, `1ffdb649…`, `self_check()` 통과 | `full_data` | CPU |
| 4 | `::test_smoke_candidates_equal_full` | smoke val·test 20표본 후보 = full 같은 행 | `full_data`, `smoke_data` | CPU |
| 5 | `::test_flags_single_trigger[<flag>]`, `::test_flag_counts_seed1234` | 손 catalog로 flag마다 최소 발동 사례와 근소 미발동 사례 (정의상 단독 발동은 앞 세 flag만: `target_title_duplicated`·`target_unscorable`은 늘 다른 flag와 함께 남, 6.6.4), 6.6.6 표 | `full_data` | CPU |
| 6 | `::test_unscorable_matches_judge` | val·test 189 × 제목 모드 2: `target_unscorable == not judge(정답 제목).correct` (M11 `judge`) | `full_data` | CPU |
| 7 | `::test_invalid_calls_rejected[<6개>]` | 각 `CandidateError`, `num_candidates=21`은 설정 단계 `ConfigError` | – | CPU |
| 8 | `::test_uniformity_chi_square` | train row 10, epoch 0–9,999, 아이템·위치 카이제곱 p > 0.001 (값 기록, 설계 계산 0.380·0.398) | `full_data` | CPU |
| 9 | `vessl/test_vessl_env.py::test_numpy_and_golden` | numpy 1.24.2, `self_check()` | – | VESSL_REQUIRED |

**M3 (7.8)**
| # | 시험 | 확인 | fixture | 구분 |
|---|---|---|---|---|
| 1 | `test_m3_convert.py::test_tool_refuses[blob-pickle_hash-foreign_global-nonempty_out-missing_commit]` | 사례마다 0이 아닌 종료와 원인 문구. blob·commit 사례는 저장소 object를 읽기만 하는 임시 git 저장소(commit을 만들지 않음. blob 사례는 기준 commit을 fork 6c7ebac로 바꿔 bf16 사본 blob을 가리킴), global 사례는 시험용 pickle의 해시로 `PICKLE_SHA256`·`PICKLE_BLOB`을 monkeypatch | `tmp_path`, `git_objects_dir` (git 전체 이력, 없으면 `SKIP_NO_GIT_HISTORY`) | CPU |
| 2 | `::test_state_file` | 텐서 20개 모양, 129,664값, `dropped_keys`, digest `6a78697b…1dae`, `weights_only=True`, `arch` Python 타입 | `converted_sasrec` | CPU `slow` |
| 3 | `::test_manifest_observed` | 7.8 #3 관찰값과 global 18개 | `converted_sasrec` | CPU `slow` |
| 4 | `::test_reference_file` | 66,315행(66,126/94/95), `sample_id`, z·item_emb 모양, 자체 점검 차이 0 | `converted_sasrec` | CPU `slow` |
| 5 | `test_m3_sasrec.py::test_matches_reference_bitwise`, `::test_batch8_close`, `::test_embed_items_bitwise` | 17.6.5 표 | `sasrec_state_path`, `sasrec_reference_path`, `torch_threads1` | CPU |
| 6 | `::test_always_eval` | 부모 `.train()`·`encoder.train(True)` 뒤 모든 하위 모듈 eval, `nn.Dropout` 없음 | `sasrec_state_path` | CPU |
| 7 | `::test_frozen_and_gate_grad` | parameter·z `requires_grad` False, gate 손실 backward가 W_g에만, `requires_grad_(True)` 뒤 forward `RuntimeError` | 〃 | CPU |
| 8 | `::test_fp32_under_autocast` | CPU bf16 autocast 안 z float32·밖과 bitwise, fp16 parameter → `RuntimeError` | 〃 | CPU |
| 9 | `::test_input_validation[<8개>]` | 사례마다 `ValueError` | 〃 | CPU |
| 10 | `::test_encode_all_matches_reference`, `::test_pad_histories_matches_reference`, `::test_export_z_close_to_reference` | bitwise와 thread 수 복원, `hist_ids`·`len_seq` 같음, `compute_representations` 최대 차 ≤ 1e-5 (값 기록) | 〃, `full_data` | CPU `slow` |
| 11 | `::test_check_env_sasrec_failures[digest-source_sha256-pad_id]` | 사례마다 점검 실패 | `profile_cfg`, `monkeypatch` | CPU |
| 12 | `vessl/test_vessl_sasrec.py::test_cuda_matches_cpu_reference` | CUDA z ↔ 기준 ≤ 1e-4 (값 기록), TF32를 켠 차이도 기록, 반복 bitwise, 호출 뒤 TF32 flag 원래 값, CUDA `embed_items` bitwise | 〃 | VESSL_REQUIRED |
| 13 | `vessl/test_run_outputs.py::test_e2_run_sasrec_digest` | E2 `epoch0..4/train_state.pt`의 `sasrec_state_digest` = 시작값 (M10이 epoch 끝마다 재계산·비교) | `run_dir_env("E2")` | VESSL_REQUIRED |

**M4 (8.8)**
| # | 시험 | 확인 | fixture | 구분 |
|---|---|---|---|---|
| 1 | `test_m4_prompts.py::test_templates_exact`, `::test_templates_crlf_invariance` | 세 문자열, CRLF·LF 같은 결과와 해시 | `repo_root`, `tmp_path` | CPU |
| 2 | `::test_parse_templates_rejects[<7개>]`, `::test_load_templates_missing_or_modified`, `::test_template_path_choice` | `PromptError`. `prompt.template_path=prompt/game.txt` → `ConfigError` | 〃, `profile_cfg` | CPU |
| 3 | `::test_render_matches_code_chain` | val 94 + test 95 × 템플릿 3 × 형식 2 = 1,134개가 `refs_upstream.code_prompt`와 같고 태그 수 일치 | `full_data`, `templates` | CPU |
| 4 | `::test_build_prompt_rejects[<7개>]` | 사례마다 `PromptError` | `full_data` | CPU |
| 5 | `::test_template_counts_seed1234`, `::test_template_choice_errors` | 22,033/22,031/22,062, 30/33/31, 22/36/37, fixed 0, epoch 오류 | `full_data`, `paper_cfg` | CPU |
| 6 | `::test_curriculum_counts_seed1234[linear-reverse-all_hybrid]`, `::test_curriculum_boundaries` | 8.6.5 수, τ = 1 u 0.260466, τ = T, 오류 | – | CPU |
| 7 | `::test_batch_plan_golden`, `::test_build_batch_plan_wrapper` | 8.6.6 첫 batch·index 수·digest 5개·재생성, 오류, wrapper의 digest·`len`·순회·`plan[i]`, val 섞임·순서 오류 | `full_data`, `paper_cfg` | CPU |
| 8 | `test_workers.py::test_train_batches_identical_across_workers` | `(tau, hybrid, template_ids, sample_ids)` 순서 같음 | 17.6.5 | CPU `slow` |
| 9 | VESSL CPU 묶음에서 6·7번 시험 (`criterion("M4-6", "M4-7", "M4-9")`) | Linux torch 2.0.0+cu117에서 digest·hybrid 수 같음 | – | VESSL_REQUIRED |

**M5 (9.8)**
| # | 시험 | 확인 | fixture | 구분 |
|---|---|---|---|---|
| 1 | `test_m5_batching.py::test_load_tokenizer_ids`, `::test_load_tokenizer_rejects[sha-fast-transformers_ver-sentencepiece_ver-sp_model_kwargs]`, `test_m9_model.py::test_build_rechecks_token_ids` | 길이 32,005·id·지문 2개, 사례마다 `BatchingError`. 마지막 절은 M9 build 1단계의 `bundle.ids` 재대조 (M5 9.8 #1, 최종 검토에서 개정) | `tokenizer_dir`, `monkeypatch` | CPU |
| 2 | `::test_matches_code_pair_encoding` | 6,378 / 6,378 (`refs_upstream.code_pair_input_and_labels`) | `full_data`, `bundle`, `templates` | CPU `slow` |
| 3 | `::test_eval_encoding_matches_code` | 같은 6,378건 | 〃 | CPU `slow` |
| 4 | `::test_token_count_checks[<사례>]` | 형식별 수, 오류 사례 | 〃 | CPU |
| 5 | `::test_answer_tokens_all_titles` | 1,682개 끝 13, 길이 2–23(1241, 1331), 266 → `[29871, 13]` | `full_data`, `bundle` | CPU |
| 6 | `::test_train_and_eval_padding` | mask·label·int64·모양, 평가 = HF `padding_side="left"` | 〃 | CPU |
| 7 | `::test_constructed_worst_case`, `::test_real_max_lengths`, `::test_too_long_rejected` | 631/632/639(+1) ≤ 1024, train epoch 0 최장 334·평가 290, 1,025 → 오류 | 〃 | CPU (`slow`: 2번째) |
| 8 | `test_workers.py::test_train_batches_identical_across_workers`, `test_m5_batching.py::test_dataset_pickle_round_trip`, `::test_candidate_fn_keyword_only` | worker 0·2 같음, pickle 왕복 encoding 같음, `fn(s, 0)` → `TypeError` | 〃 | CPU |
| 9 | `::test_eval_loader_test_split` | 12 batch(마지막 7), 행 순서, 누락·중복 없음 | 〃 | CPU |
| 10 | `::test_make_train_loader_checks` | 오류 3종, `len(plan)`개를 계획 순서로 | `smoke_data`, 〃 | CPU |
| 11 | VESSL CPU 묶음(`ILORA_TEST_TOKENIZER=$LLM_PATH`)의 1–10번, `vessl/test_vessl_env.py::test_llm_tokenizer_sha256` | 실제 tokenizer sha256 = 설정값, 1–10 통과 | `llm_path` | VESSL_REQUIRED |
| 12 | `vessl/test_vessl_env.py::test_workers_8_linux` | epoch 0 처음 50 batch, worker 8 = 0 | 〃 | VESSL_REQUIRED |

**M6 (10.8)**
| # | 시험 | 확인 | fixture | 구분 |
|---|---|---|---|---|
| 1 | `test_m6_projector.py::test_param_counts_and_shapes` | 17,047,552, key·모양·fp32·requires_grad | – | CPU |
| 2 | `::test_init_deterministic_and_isolated` | 같은 seed bitwise·다른 seed 다름, 전역 CPU 난수 상태 전후 같음, gate 생성 여부 무관 | – | CPU |
| 3 | `::test_init_ranges` | \|w\| ≤ 1/8, 1/64 | – | CPU |
| 4 | `::test_forward_and_input_checks` | 모양·dtype, 오류 사례 | – | CPU |
| 5 | `test_m9_model.py::test_projector_only_for_hybrid` | text-only: forward hook 0회·grad None, hybrid: grad 있음 | `tiny_model`, `smoke_data` | CPU |
| 6 | `test_m10_train.py::test_text_only_window_leaves_projector` | 값·Adam state bitwise 그대로 | 〃 | CPU |
| 7 | `test_m10_train.py::test_optimizer_groups` | projector wd 1e-5, 모든 s에서 lr = LoRA lr | 〃 | CPU |
| 8 | `test_m10_train.py::test_checkpoint_keys_after_text_only_window`, `test_m9_model.py::test_checkpoint_round_trip` | text-only 창 직후 저장에도 projector 4 key, 재로드 출력 bitwise | 〃, `tmp_path` | CPU |
| 9 | `vessl/test_vessl_model.py::test_projector_fp16_autocast`, `vessl/test_run_outputs.py::test_e2_run_records` | fp16 출력·유한, step 행에 `skipped` 기록 | `llama_model`, `run_dir_env("E2")` | VESSL_REQUIRED |

**M7 (11.8)**
| # | 시험 | 확인 | fixture | 구분 |
|---|---|---|---|---|
| 1 | `test_m7_router.py::test_linear_gate_init` | `[4, 64]` fp32, bias 없음, 256개, \|w\| ≤ 1/8, 재현, 전역 난수 상태 같음 | – | CPU |
| 2 | `::test_dynamic_omega_matches_fp64` | 행 합 ≤ 1e-5, 0 이상, fp64 softmax와 ≤ 1e-6 | – | CPU |
| 3 | `::test_omega_fp32_under_autocast` | bf16 autocast 안팎 bitwise, float32 | – | CPU |
| 4 | `::test_batch_independence` | 혼자·함께 차이 ≤ 1e-6 | – | CPU |
| 5 | `::test_gradient_only_to_wg` | W_g grad 있음, SASRec grad 없음 | `sasrec_state_path` | CPU |
| 6 | `::test_cluster_hard_omega[fake-real]`, `::test_hard_gate_output_checks[<5개>]`, `::test_unknown_sample_id` | 0/1·행 합 1·mapping, 잘못된 `HardGate` 출력마다 `RouterError`, 모르는 id → `KeyError` | `fake_hard_gate`, `smoke_cluster_dir` | CPU |
| 7 | `::test_cluster_hard_has_no_gate`, `test_m10_train.py::test_cluster_hard_training` | gate None·빈 목록·빈 state_dict, 1 epoch 뒤 `gate_calls` 0·`lookups` = 표본 수, gate 그룹 없음, parameter 차 256 | `tiny_model`, `cpu_smoke_run` | CPU `slow` |
| 8 | `::test_hard_gate_mode_mismatch[<3개>]`, `test_m10_train.py::test_dynamic_never_calls_cluster` | `RouterError`, dynamic 학습에서 `ilora.cluster` 공개 함수 spy 호출 0 | `monkeypatch` | CPU |
| 9 | `::test_none_mode`, `::test_mode_k_combinations[<3개>]` | ω = ones, 조합 오류 | – | CPU |
| 10 | `::test_forward_input_checks[<4개>]`, `::test_nan_gate_weight_caught` | `RouterError`, NaN W_g → 출력 검사가 잡음 | – | CPU |
| 11 | `::test_recorder_errors[<6개>]`, `::test_recorder_normal_path`, `::test_router_writes_no_files` | 오류 사례, N행·호출 순서·CPU float32·discard 뒤 재시작, `router.py`를 `ast`로 훑어 `open`·`np.save`·`torch.save`·`to_csv` 호출 없음 | – | CPU |
| 12 | `test_m11_evaluate.py::test_gate_record_single_collection_point` | `GateRecord` ω·z = batch별 `GenerationOutput` (bitwise), checkpoint gate + M3 z로 다시 만든 `Router` 출력과도 같음 | `tiny_model`, `smoke_data` | CPU |
| 13 | `::test_state_dict_and_provenance` | dynamic key 1개, 나머지 빈 dict, provenance 불일치 → 오류 | – | CPU |
| 14 | `vessl/test_run_outputs.py::test_e2_run_gate_counts` | 학습 `gate_calls` 합 41,325, 평가 `metrics.json`의 증가분 = 그 평가의 batch 수(val 12, test 12), 기록 gate 94·95행·행 합 ≤ 1e-5·유한 | `run_dir_env("E2")` | VESSL_REQUIRED |
| 15 | `vessl/test_run_outputs.py::test_e4_run_no_gate_calls_and_one_hot` | E4 학습 `gate_calls` 합 0, `cluster_lookups` 합 330,600, 평가 gate one-hot = mapping[배정] | `run_dir_env("E4")`, `run_dir_env("CLUSTER")` | VESSL_REQUIRED |

**M8 (12.8)**
| # | 시험 | 확인 | fixture | 구분 |
|---|---|---|---|---|
| 1 | `test_m8_moe_lora.py::test_llama7b_lora_counts` | 224 모듈, 448 텐서, 19,988,480(모듈 종류별 수), 합계 37,036,288, gate 비율 0.000691% | `llama7b_skeleton` | CPU |
| 2 | `::test_matches_naive_and_upstream_fp64[<dropout>-<모양>]` | 17.6.5 표 | – | CPU |
| 3 | `::test_k1_equals_plain_lora[expert_rank-total_rank]` | scaling 4.0, bitwise | – | CPU |
| 4 | `::test_scaling_rules_uniform_omega` | 16.0/4.0, 균등 ω에서 표준 LoRA·그 1/4 (fp64 상대오차 < 1e-12) | – | CPU |
| 5 | `::test_no_expert_output_saved` | `saved_tensors_hooks`로 모은 모양에 `[B, L, K, d_out]` 없음 | – | CPU |
| 6 | `::test_zero_b_initial_state` | 출력 = base bitwise, ω·A grad 0, B grad ≠ 0 | – | CPU |
| 7 | `::test_lora_init` | 범위, 같은 seed bitwise·다른 seed 다름, 전역 난수 상태 같음 | `build_tiny_llama` | CPU |
| 8 | `::test_bf16_autocast_proxy` | bf16 출력, A·B·ω grad fp32, base grad 없음, autocast 밖 bf16 입력 오류 | – | CPU |
| 9 | `::test_errors[<9개>]` | 사례마다 예외, 뒤에 `GateContext` 비어 있음 | – | CPU |
| 10 | `test_m9_model.py::test_gate_use_calls`, `test_m9_model.py::test_host_reads_do_not_grow_with_layers`, `test_m8_moe_lora.py::test_layer_call_reads_nothing_back[K4-K1]`, `::test_tiny_llama_forward_and_generate_calls` | forward 14, 생성 14 × T. 감싼 층 하나의 forward·backward `aten::item` 0회, 학습 forward의 host 읽기 수가 2층과 3층에서 같음. `::test_tiny_llama_forward_and_generate_calls`는 M9 없이 13.6.9 모양의 작은 LLaMA를 직접 만듦 (M8-7과 같음) | `tiny_model` | CPU |
| 11 | `vessl/test_vessl_model.py::test_moe_lora_layer_fp16` | fp16 출력, fp32 grad, fp32 기준 차이 기록 | CUDA | VESSL_REQUIRED |
| 12 | `vessl/test_vessl_model.py::test_longest_batch_memory` | 5 epoch 계획에서 L 최대인 batch(CPU에서 M4·M5로 계산, 계산값 기록)의 forward+backward `max_memory_allocated` 기록, < 76 GiB | `llama_model` | VESSL_REQUIRED |

**M9 (13.8)**
| # | 시험 | 확인 | fixture | 구분 |
|---|---|---|---|---|
| 1 | `test_m9_model.py::test_tiny_build[dynamic-cluster_hard-none]` | 14 모듈, LoRA 28 텐서 17,408, projector 8,320, gate 256(dynamic만), vocab 32,005, 그룹 키·이름 정렬 | `tiny_model`, `fake_hard_gate` | CPU |
| 2 | `::test_new_token_rows` | 같은 seed bitwise·다른 seed 다름, 앞 32,000행 불변, fp16 CPU 모델도 같음 | `tiny_model` | CPU |
| 3 | `::test_initial_logits_equal_unwrapped` | B = 0에서 logits bitwise | 〃 | CPU |
| 4 | `::test_loss_equals_hf_loss` | 차이 < 1e-6 | 〃 | CPU |
| 5 | `::test_embed_matches_reference_wrap_emb`, `::test_text_only_embeddings` | 같은 projector 출력을 넣은 `refs_upstream.wrap_emb_loop`와 bitwise, 공개 코드 배치(padding 포함 projector)와 최대 차이 ≤ 1e-5 (13.6.3), text-only는 projector 0회·`embed_tokens` 그대로 | 〃 | CPU |
| 6 | `::test_batch_checks[<10개>]` | 사례마다 예외 | 〃 | CPU |
| 7 | `::test_gradient_flow`, `::test_z_ignores_target` | 0이 아닌 grad 위치, base·SASRec grad 없음, text-only projector grad None. 정답 id를 바꾼 batch의 z가 같음 (식 10) | 〃 | CPU |
| 8 | `::test_sasrec_stays_eval` | `model.train()` 뒤 eval, z 두 번 bitwise | 〃 | CPU |
| 9 | `::test_generate_contract`, `::test_batch_generation_equals_single`, `::test_use_cache_required` | BOS 제거, T ≤ 64, batch = 단독, calls 14 × T, 학습 모드 오류, 예외 뒤 `GateContext` 빈 상태, 반복 같음, `use_cache` False → build 오류 | 〃 | CPU |
| 10 | `::test_position_ids_left_padding` | 실제 토큰 위치 0, 1, 2, … | 〃 | CPU |
| 11 | `::test_checkpoint_round_trip`, `::test_router_from_checkpoint_keys` | 13.8 #11 전부 | 〃, `tmp_path` | CPU |
| 12 | `::test_bf16_autocast_proxy` | logits bf16, loss fp32, grad fp32 | 〃 | CPU |
| 13 | `::test_llama7b_parameter_counts` | 6,738,415,616 → 6,738,456,576, LoRA 19,988,480, 37,036,288 / 37,036,032, `parameter_report()`가 같은 식 | `llama7b_skeleton` | CPU |
| 14 | `vessl/test_vessl_model.py::test_load_llama2_fp16` | 로드 정보 허용 범위, 전부 fp16, `rope_scaling` None, 32,005, 새 행 sha256 두 번 같음, 로드 시간·최대 호스트 메모리(`resource.getrusage`) 기록 | `llm_path` | VESSL_REQUIRED |
| 15 | `::test_fp16_train_microbatch` | logits fp16, loss fp32 유한, HF loss와 차이 기록, 첫 step ω·`lora_A` grad 0, `lora_B` grad ≠ 0 | `llama_model` | VESSL_REQUIRED |
| 16 | `::test_stock_generate_fp16` | 패치 없는 `generate` 동작, 첫 열 BOS, 두 번 같음, batch·단독 일치율 기록 | 〃 | VESSL_REQUIRED |
| 17 | `::test_longest_batch_memory` (M8-12와 같은 시험) | 13.6.7의 12.1 GiB 감소 예상과 측정값을 나란히 기록 | 〃 | VESSL_REQUIRED |

**M10 (14.8)**
| # | 시험 | 확인 | fixture | 구분 |
|---|---|---|---|---|
| 1 | `test_m10_train.py::test_step_counts` | 8,265·9·517·2,585·129·41,325, epoch별 τ·s 범위 | – | CPU |
| 2 | `::test_lr_matches_upstream` | 14.6.2에 적은 값과 유효숫자 7자리(적힌 값이 7자리로 반올림됨: s = 64는 5.011628e-05 대 5.0116279069767e-05, 상대 1.9e-8), 닫힌 식과 상대 1e-12, s 0–2,584 전부 `refs_upstream.upstream_lr`와 같음, [1e-6, 1e-4] 안 | – | CPU |
| 3 | `::test_optimizer_groups`, `::test_optimizer_rejects[<3개>]`, `::test_group_sizes_llama7b_shape`, `::test_mapping_config_rejected` | 그룹·wd·배수·Adam 값, 오류, 17,047,552/256/19,988,480, mapping 값 → `ConfigError` | `tiny_model`, `llama7b_skeleton` | CPU |
| 4 | `::test_accumulation_matches_mean_loss[9-16]`, `::test_train_window_divides_by_grad_accum[3-4]` | 누적 grad = (Σloss)/16의 grad (상대 1e-6, 9개 창도 16, D24), `train_window`의 grad = (Σloss)/4 (smoke_cpu 마지막 창 3개도) | `tiny_model`, `torch_threads1` | CPU |
| 5 | `::test_text_only_window_leaves_projector` | M6-6과 같은 시험, 어긋나게 만든 모의 모델 → `TrainingError` | 〃 | CPU |
| 6 | `::test_skip_guard` | 15회 뒤 성공 통과, 16회 → 오류와 step 범위 | – | CPU |
| 7 | `::test_checkpoint_round_trip_state`, `::test_checkpoint_load_rejects[<8개>]` | 상태 bitwise, 불일치마다 `TrainingError` (14.8 #7의 8개: 키 빠짐·남음, 모양·dtype, `generated_frozen`, config hash, 그룹 순서, 고정 parameter checksum `frozen_checksum`, sha256) | `tiny_model`, `tmp_path` | CPU |
| 8 | `::test_resume_equals_uninterrupted`, `::test_resume_after_a_killed_process`, `::test_segment_rules`, `::test_stale_val_metrics_kept_with_counter` | 2 epoch 연속 학습과 (epoch 1 시작에서 `build_batch_plan`이 예외를 내게 monkeypatch → `--resume`)의 parameter·세 CSV 같음, 경계 밖 행 이동, `STATE_SAVED`만 있는 폴더 재개. 강제 종료된 프로세스 뒤 재개(구간 `interrupted` 표시), `segment_problems` 규칙, 남은 `val_metrics.json`을 번호를 붙여 보존(덮어쓰기 없음) | `cpu_cfg`, 자산, `torch_threads1` | CPU `slow` |
| 9 | `::test_select_epochs[<2개>]` | best 2·last 4·호출 [2, 4], best = last = 4·1회 | – | CPU |
| 10 | `::test_test_after_last_complete` | 2 epoch 작은 학습에서 `evaluate_split`과 마커 쓰기를 spy로 감싸 순서 기록: test 호출은 마지막 epoch `COMPLETE` 뒤에만 | `cpu_cfg`, 자산, `monkeypatch` | CPU `slow` |
| 11 | `::test_sasrec_train_mode_detected` | 모의 모델 → `TrainingError` | – | CPU |
| 12 | `::test_global_rng_untouched` | `random`·`np.random` 상태 전후 같음, 평가 전후 torch CPU 상태 같음 | `tiny_model` | CPU |
| 13 | `::test_record_counts` | S1 기록: micro 78, step 20, epoch 2, `n_hybrid` 합 38, `gate_calls` 합 78 (평가 제외), 열 = M12 정의, `run.log` 요약 줄 | `cpu_smoke_run` | CPU `slow` |
| 14 | `::test_cluster_hard_training`, `::test_cluster_artifact_tamper_fails_before_build` | S5 호출 순서·그룹·0·156, 산출물 1 byte 변경 → `build_model` 전 실패, dynamic M15 호출 0 | `cpu_smoke_run`, `smoke_cluster_dir` | CPU `slow` |
| 15 | `vessl/test_run_outputs.py::test_e2_run_complete` | 첫 step `scale_before` 65,536 (`fp16_amp`. bf16 대안이면 모든 step scale 1.0·skip 없음), skip 기록·16회 연속 없음, 2,585 step·41,325 micro, epoch 5개 `COMPLETE`·val 94, `meta.json` `train.segments`가 M10 `segment_problems` 규칙을 따름 (강제 종료 뒤 재개면 `interrupted`·`ended_utc` null), 시각·메모리 기록 | `run_dir_env("E2")` | VESSL_REQUIRED |
| 16 | `vessl/test_vessl_train.py::test_nan_step_skipped` | 17.6.12의 작은 GPU 학습에서 step 5의 loss에 NaN을 곱함(`train_forward` monkeypatch) → 그 step `skipped` 1·scale 절반·grad norm 비유한, step 6 grad norm 유한, 학습 완료 | `llm_path` | VESSL_REQUIRED |
| 17 | `vessl/test_run_outputs.py::test_e2_reeval_reproduces` | 새 프로세스 eval 2회(R8)의 `records.jsonl` sha256 = 학습 중 기록, `summary.json` 재현 `identical` | `run_dir_env("E2")`, `run_dir_env("E2_REEVAL")`, `run_dir_env("E2_REEVAL_BI")` | VESSL_REQUIRED |
| 18 | `vessl/test_vessl_env.py::test_pl_native_amp_scaler_defaults` | PL 1.8.6(requirements로 설치됨) `NativeMixedPrecisionPlugin(16, "cuda").scaler`의 4값 기록과 (65536, 2.0, 0.5, 2000) 비교. PL이 없으면 skip 이유 기록 | CUDA | VESSL_REQUIRED |
| 19 | `vessl/test_vessl_train.py::test_resume_restores_cuda_state` | 17.6.12의 작은 GPU 학습 2 epoch(fp16, epoch당 micro 16·step 8)이 epoch 1 계획에서 실패 → `--resume`: `load_checkpoint` 직후 scaler dict 5개 key·CPU·CUDA 난수·Adam step·moment(cuda:0)가 `epoch0/train_state.pt`와 bitwise, step 8 `scale_before` = step 7 `scale_after` = 저장 scale, 기록 32·16·2행의 결정적 열(τ, hybrid, curriculum 확률, `n_micro`, `n_hybrid`, 그룹별 lr) = 계획·`lr_at` | `llm_path` | VESSL_REQUIRED |

**M11 (15.8)**
| # | 시험 | 확인 | fixture | 구분 |
|---|---|---|---|---|
| 1 | `test_m11_evaluate.py::test_judge_matches_upstream[<손 사례>]`, `::test_judge_random_10000` | `refs_upstream.first_line_and_hr1`과 `valid`·`correct` 모두 같음 | `full_data` | CPU |
| 2 | `::test_correct_iff_target_id`, `::test_judge_rejects[<4개>]` | 동치, `EvalError` | – | CPU |
| 3 | `::test_decode_rules` | 자리표·pad 유무 무관, 특수 토큰 8종 제거, 줄바꿈 id는 13 하나 | `bundle` | CPU |
| 4 | `::test_newline_counterexample`, `::test_gen_config_stops` | `"Star Wars"` 대 `""`, 종료 id 2 하나 | `bundle`, `tiny_model` | CPU |
| 5 | `::test_trim_generated`, `::test_check_output_rejects[<8개>]` | 자르기, 거부 사례 | – | CPU |
| 6 | `::test_compute_metrics[<사례>]` | 정확히 같은 비, 오류 | – | CPU |
| 7 | `::test_aux_protocols`, `::test_drop_last_bs8_prefix_bitwise` | 92·88개, val 빈 dict, smoke 8·8, 앞 88개 기록 bitwise | `tiny_model`, `full_data` | CPU (`slow`: 2번째) |
| 8 | `::test_wilson` | 기준값 1e-6, null, [0, 1], `WILSON_Z` | – | CPU |
| 9 | `::test_batch_crosscheck_rejects[<6개>]` | 모의 loader 사례마다 `EvalError` | – | CPU |
| 10 | `::test_full_split_eval_tiny[val-test]` | 파일 3개, 94·95줄, gate K개, flags, `write_eval` 검사, `GateRecord` = 기록·NPZ, 증가분 12, `candidates_sha256`·`template_counts` | `tiny_model`, `full_data` | CPU `slow` |
| 11 | `::test_eval_deterministic` | 새 폴더 두 번·worker 0과 2 → sha256 같음 | 〃 | CPU `slow` |
| 12 | `::test_batch_invariance_tiny` | 토큰 일치 94 | 〃 | CPU `slow` |
| 13 | `::test_rng_and_mode_restored` | 난수 3종 전후 같음, 모드 복원 | 〃 | CPU |
| 14 | `::test_exception_cleanup` | 3번째 batch 예외 뒤 기록기·모드·난수 복원, `metrics.json` 없음, `val.incomplete-<UTC>`, 완료 폴더 재평가 오류 | 〃 | CPU |
| 15 | `::test_eval_variants[text_only-cluster_hard-none]` | `hybrid` false·projector forward hook 0회, one-hot·`ones`, 증가분 0, 키 `gate_0..` | `tiny_model`, `fake_hard_gate` | CPU |
| 16 | `::test_eval_command[<사례>]` | 15.8 #16 사례 전부 | `cpu_smoke_run` | CPU `slow` |
| 17 | `::test_load_eval_result` | 재계산 = 파일, 변조 → 오류 | `tmp_path` | CPU |
| 18 | `vessl/test_run_outputs.py::test_e2_run_eval_folders` | val 5회·test best·last 폴더, N 94·95, 시간·메모리 기록 | `run_dir_env("E2")` | VESSL_REQUIRED |
| 19 | `vessl/test_run_outputs.py::test_e2_reeval_reproduces` (M10-17과 같은 시험) | 두 eval 실행과 학습 중 기록의 sha256 같음. 다르면 다른 표본·처음 다른 단계를 출력하고 실패 | `run_dir_env("E2")`, `run_dir_env("E2_REEVAL")`, `run_dir_env("E2_REEVAL_BI")` | VESSL_REQUIRED |
| 20 | `vessl/test_run_outputs.py::test_e2_reeval_batch_invariance` | val·test `batch_invariance.json` 존재, 일치율 기록 (요구값 없음) | `run_dir_env("E2_REEVAL_BI")` | VESSL_REQUIRED |
| 21 | `vessl/test_vessl_env.py::test_decode_rules_llm_tokenizer` (VESSL CPU 묶음의 3·4번 시험도 실제 tokenizer로 다시 돎) | `$LLM_PATH` tokenizer로 3·4번 규칙: 자리표·EOS 뒤 pad 무관, 특수 id 8개 제거, 줄바꿈은 id 13 하나, `"Star Wars"` 대 `""`, 종료 id 2 하나 | `llama_bundle`, `paper_cfg` | VESSL_REQUIRED |
| 22 | `vessl/test_run_outputs.py::test_e2_run_generation_diagnostics` | `supplementary.generation`의 진단 수 존재 (값 TBD) | `run_dir_env("E2")` | VESSL_REQUIRED |

**M12 (16.8)**
| # | 시험 | 확인 | fixture | 구분 |
|---|---|---|---|---|
| 1 | `test_m12_records.py::test_columns_and_row_validation` | 열 순서, `step_fields(1)`·`(4)`, 잘못된 행 `ValueError`, 실수 왕복 bitwise | `tmp_path` | CPU |
| 2 | `::test_flush_schedule`, `::test_close_failed_keeps_rows` | 파일 쓰기 횟수 = flush 횟수(쓰기 함수 spy), 예외 뒤 행과 `run_summary.json` status failed | 〃 | CPU |
| 3 | `::test_activation_monitor` | 심은 1234.5 기록, hook 유무 loss·grad·난수 bitwise, detach 뒤 hook 0, 래퍼 없음 → 오류 | `tiny_model` | CPU |
| 4 | `::test_validate_gate_rejects[<7개>]`, `::test_validate_z_rejects[<3개>]` | 거부 사례, fp32 softmax 통과 | – | CPU |
| 5 | `::test_compute_representations_synthetic[8-64]` | 순서, 열 이름, NPZ·CSV 왕복, SPEC §4 열 순서 | `tmp_path` | CPU |
| 6 | `::test_real_sasrec_z_export` | 66,126/94/95행, d = 64를 가중치에서, 유한, 두 번 bitwise | `sasrec_state_path`, `full_data` | CPU `slow` |
| 7 | `::test_crosscheck_gates[<4개>]`, `::test_export_router_loading[dynamic-cluster_hard]` | 통과·실패 사례, \|Δz\| 기록, gate key만 strict 로드, cluster_hard one-hot | `smoke_cluster_dir` | CPU |
| 8 | `test_m12_analysis.py::test_gate_summary_known_answers` | 균등·Case B·one-hot·JS 손계산·η² | – | CPU |
| 9 | `::test_summary_files_and_interpretation` | json·csv 같은 값, 참고점 수치 있음, 판정 패턴 없음: `TRUE`·`FALSE` 낱말, "collapse … yes/no/true/false" 꼴, 임계값과 비교해 결론을 적은 줄 (정규식 목록은 시험 상수. 참고점 설명에 쓰는 "Case B"·"collapse" 낱말 자체는 허용, 17.6.10) | `tmp_path` | CPU |
| 10 | `::test_figures` | PNG DPI 300, 모든 축 `ax.texts` 비어 있음, italic·bold 없음, train heatmap 행 ≤ 500, 그림 3c 대응 막대 10개 = test 행 순서 첫 10개 | 〃 | CPU |
| 11 | `::test_k1_export_and_analysis_error` | export `gate_0` = 1, 분석 명시적 오류 | 〃 | CPU |
| 12 | `test_m12_records.py::test_write_eval_rejects[<5개>]`, `::test_write_eval_deterministic` | 거부 사례, `init`·추가 블록 통과, 두 번 sha256 같음 | 〃 | CPU |
| 13 | `::test_recorder_resume` | 경계 밖 행 이동, 행 수, header·부족 → `ValueError` | 〃 | CPU |
| 14 | `vessl/test_vessl_model.py::test_activation_monitor_fp16` | CUDA fp16 `aminmax`, hook 유무 loss bitwise, micro-batch 시간 차 기록 | `llama_model` | VESSL_REQUIRED |
| 15 | `vessl/test_run_outputs.py::test_e2_run_records` | 41,325·2,585(epoch 끝 `n_micro` 9)·5행, `gate_calls` 합 41,325, skip 목록·최대 activation 기록 | `run_dir_env("E2")` | VESSL_REQUIRED |
| 16 | `vessl/test_run_outputs.py::test_e2_export` | best·last의 val·test 대조 통과(≤ 1e-5), NPZ·CSV·분석·그림 존재 | `run_dir_env("E2_EXPORT")` | VESSL_REQUIRED |

**M15 (19.8)**
| # | 시험 | 확인 | fixture | 구분 |
|---|---|---|---|---|
| 1 | `test_m15_cluster.py::test_parse_mapping[<사례>]` | 허용 2개, 거부 9개 | – | CPU |
| 2 | `::test_cluster_hard_gate` | 모양·dtype·값·행 합·mapping·장치·`lookups`, 모르는 id `KeyError` | – | CPU |
| 3 | `::test_fit_uses_train_only` | `build_fit_set` = M1 train 순서, val 섞임 실패, `KMeans.fit` spy 1회·행 = train 수·열 = d | `full_data`, `monkeypatch` | CPU |
| 4 | `::test_fit_deterministic_one_thread` | 합성 66,126 × 64(`np_rng(0, "test", "z")`) 두 번 bitwise, joblib 재로드 = CSV, `centroids.npz` | `tmp_path` | CPU |
| 5 | `::test_predict_input_checks`, `::test_sklearn_version_pinned` | float64·비연속 거부, 1.2.1 흉내 → 실패 | `monkeypatch` | CPU |
| 6 | `::test_convergence_warning_is_error` | 서로 다른 점 3개 → 실패 | – | CPU |
| 7 | `::test_user_level_fit_set` | 754행 = 블록별 최대 행, 표본이 사용자 군집을 받음 | `full_data` | CPU |
| 8 | `::test_summary_synthetic` | 크기 합, 비율 합 1, 분할표 합, 행 합 1(1e-6), NMI ∈ [0, 1], NOT_RUN·heatmap 없음 | `tmp_path` | CPU |
| 9 | `::test_load_artifacts_rejects[<6개>]`, `::test_full_artifacts_for_smoke`, `::test_live_check_reads_train_npz` | 거부 사례, full 산출물을 smoke에 통과, NPZ 1 byte 변경 → 해시 거부 | `smoke_cluster_dir`, `tmp_path` | CPU |
| 10 | `::test_dynamic_export_crosscheck[<3개>]` | 같음 통과, 1e-4 차 실패, cluster_hard export 실패 | `tmp_path` | CPU |
| 11 | `::test_cluster_command_real_sasrec` | fit 66,126, 두 실행 z·중심·배정 bitwise (시각·silhouette·그림 제외), silhouette·시간 기록 (값 TBD) | `sasrec_state_path`, `full_data` | CPU `slow` |
| 12 | `vessl/test_run_outputs.py::test_e4_run_live_check` | E4 meta·로그의 `verify_live_encoder` 결과 통과, \|Δz\| 기록 ≤ 1e-4 | `run_dir_env("E4")` | VESSL_REQUIRED |
| 13 | `vessl/test_run_outputs.py::test_e4_run_no_gate_calls_and_one_hot` (M7-15와 같은 시험) | | 〃 | VESSL_REQUIRED |
| 14 | `vessl/test_run_outputs.py::test_cluster_dynamic_comparison` | dynamic 비교 항목·heatmap 존재, z 대조 통과 (수치 TBD) | `run_dir_env("CLUSTER")` | VESSL_REQUIRED |

- 합계 187개 (M0 6, M1 14, M2 9, M3 13, M4 9, M5 12, M6 9, M7 15, M8 12, M9 17, M10 19, M11 22, M12 16, M15 14): CPU 155, VESSL_REQUIRED 32. 모두 시험이 있음 (부분 자동은 17.6.10)

#### 17.6.10 자동 판정할 수 없는 부분과 이유
- 시험이 전혀 없는 기준은 없음. 아래는 시험이 일부만 판정하는 절
  - **M12-9 해석 문장:** 금지 표현이 없고 참고점 수치가 있는지만 자동. 문장이 "보수적 해석"인지는 사람이 읽어야 함 `[SPEC §5]`
  - **M12-10 그림 양식:** DPI, 그림 안 글자, italic·bold, 행 수, 막대 수는 자동. SPEC §6-1의 장식 금지·학술 그래프 모양 같은 시각적 규칙은 사람이 그림을 봐야 함
  - **M11-19 뒷절** ("원인을 밝히기 전에는 결과를 확정하지 않음"): 운영 규칙. RUNBOOK이 이 시험 통과를 다음 단계의 조건으로 둠 (M14 18.6.6)
  - **측정만 하는 값** (판정 기준이 설계되지 않음 → 값 존재·유한만 확인, 값은 TBD): M3-12 TF32 켠 차이, M8-11 fp32 기준 차이, M9-14 로드 시간·호스트 메모리, M9-16 일치율, M9-17·M8-12 메모리 값(76 GiB 미만만 판정), M10-15 시각·메모리, M11-18 시간·메모리, M11-20 일치율, M11-22 진단 수, M12-14 hook 시간 차, M12-15 skip 목록·activation, M15-11 silhouette·시간
  - **M10-18:** fork 계보 확인용. requirements로 설치되는 PL 1.8.6에 의존하고 GradScaler 속성이 CUDA에서만 생기므로 VESSL에서만 판정. 결과는 학습과 무관
- 로컬에서 판정할 수 없어 VESSL로 미룬 32개는 대응표 구분 열에 표시함 `[SPEC §0.13]`

#### 17.6.11 설정
| 키 | 타입 | paper_spec 값 | 라벨 | 설명 |
|---|---|---|---|---|
| (M13 소유 키 없음) | | | | smoke 계획과 값은 기존 구역 값으로 정함 (17.6.6–17.6.8) |
- smoke·과적합 계획이 쓰는 `data.train_rows`(tuple[int, ...], paper_spec `[]`)는 M1 소유 키 (M1 5.6.7·5.6.8, 최종 검토에서 채택). 이 장은 `overfit_gpu`의 값만 정함 (17.6.8)

#### 17.6.12 다른 모듈과의 연결, 계약 추가 제안, 발견한 불일치
- **연결:** 모든 모듈의 공개 API를 부름. smoke는 M0 명령 함수(train·eval·export·cluster·check-env)를 같은 순서·같은 준비로 부르고, 하위 실행 폴더는 M0 `create_run_dir(root=<smoke 폴더>/steps)`로 만듦. M14는 smoke·과적합 프로파일 파일을 적고 RUNBOOK에서 이 장의 명령을 씀
- **VESSL 작은 학습 (M10-16·M10-19용):** `smoke_gpu_100` + `--set data.smoke_users=[1,10,10]`, `train.epochs=1`, `train.grad_accum=2`, `train.warmup_divisor=4` → train 132표본(사용자 117), NB 16, step 8, w 2 (계산). `warmup_divisor` 20이면 w = 0이라 M10이 거부하므로 4로 둠. M10-19는 같은 값에 `train.epochs=2`, `eval.batch_invariance_check=false` → epoch당 NB 16·step 8, T 16, w 4 (계산)
- **`SPEC §21` 요구 대응:** compileall → `test_m13_coverage.py::test_compileall`, import 확인 → 같은 파일 `test_no_repo_imports`와 모든 시험의 import, CLI `--help` → `test_m0_sections.py::test_main_help` (`python -I rebuild/main.py --help`가 0으로 끝나고 명령 6개를 보임), routing 합성 시험(dynamic·cluster-hard 모양, one-hot, 행 합, expert 범위, NaN 거부) → M7-2·6·9·10, mapping 파서·거부 → M15-1, KMeans 저장·로드 → M15-4, 분석 합성 CSV → M12-5·8, shell 정적 검토 → rebuild에는 shell 스크립트가 없고 RUNBOOK 명령은 M14 18.8 #4가 검사
- **`SPEC §22` 대응:** dynamic 경로가 군집 산출물을 읽지 않음 → M7-8, M10-14, M11-16
- **계약 추가 제안**
  1. `rebuild/pytest.ini`, `rebuild/tests/refs_upstream.py`, `rebuild/tests/vessl/` (4.6.1 폴더 구성에 추가)
  2. `rebuild/ilora/smoke.py` (M13 소유): `smoke_plan(cfg) -> Literal["cpu", "gpu100", "overfit"]`, `run_smoke(profile_path, overrides, run_dir, *, repo_root, llm_path, tokenizer_dir) -> SmokeReport`, `overfit_generation_check(model, bundle, templates, data, cfg) -> dict`, `SmokeReport(plan, run_dir, steps, checks, passed)`, `SmokeError`. `main.py smoke`는 보고서를 쓴 뒤 하나라도 실패면 0이 아닌 종료 코드. 구현이 더한 것: `SmokeError`는 `ValueError`의 하위, 보고 값 `SmokeStep`·`SmokeCheck`, 계획 표 `StepSpec`·`PLAN_STEPS`·`KIND_CHECKS`, `required_checks(plan, cfg)`, `SmokeReport.step(id)`·`step_dir(id)`·`check(id)`·`failed_checks()` (M14 비교 시험이 `step_dir`로 단계 폴더를 찾음). `passed`가 false면 M0 `run_command`가 `CommandError`를 내 실행이 `failed`로 남고 `main.py`가 1로 끝남 `R ilora/smoke.py`
  3. M0: 명령 분기를 패키지 모듈 `rebuild/ilora/commands.py`의 `run_command(command, cfg, overrides, *, profile, root) -> Path`(M0 공통 준비 = 실행 폴더·`config.yaml`·`meta.json`·환경 점검·`tracked_run` + 명령 함수)로 두고 `main.py`는 인자 해석만 함. smoke가 하위 단계를 같은 준비로 실행하고, 패키지가 `main.py`를 import하지 않음 (`test_no_repo_imports`의 `main` 금지와 맞음). 실행 폴더 경로를 로그 첫 줄에 `run_dir=<경로>`로 남김. 구현은 keyword-only 기본값 인자 `repo_root`, `argv`, `llm_path`, `tokenizer_dir`, `profile_path`(smoke에 필수), `resume`(train만), `stamp`를 더함. smoke 단계는 `profile="S<k>_<이름>"`, `root=<smoke 폴더>/steps`로 부르고 단계마다 자기 폴더의 `run.lock`을 쥠 `R ilora/commands.py run_command`
  4. M0 환경 점검(4.6.5) 추가: numpy 1.24.2·pandas 1.5.3·sentencepiece 0.1.97(M1·M2·M5 제안)과 scikit-learn 1.2.2(M15), **transformers 파일 무결성**: 설치된 transformers의 모든 `.py`가 wheel `RECORD`의 sha256과 같음 (로컬 stock 4.28.0에서 1,265개 모두 일치, `generation/utils.py` `fe55556e…48d9`, `models/llama/modeling_llama.py` `61affbfa…89fe`. fork 패치 파일 `debug/utils.py`·`debug/modeling_llama.py`는 `1a974f61…`·`bd7ea126…`로 다름. 계산 `design/tmp/m1314/tf_record_check.py`). D13의 "내부 파일 교체 제거"를 실행 환경에서 확인하는 유일한 방법
  5. (채택됨) M1 `data.train_rows` (M1 5.6.7·5.6.8, 기준 M1-13)
  6. conftest `sasrec_state_path`의 기본 위치(`runs/rebuild/assets/sasrec/…`)와 `sasrec_reference_path` fixture
  7. (채택됨) M15 `verify_live_encoder`: `n`이 train 표본 수보다 크면 `ValueError` (M15 19.6.9에 반영. `smoke_cpu`는 64로 피함)
- **발견한 장 사이 불일치** (1–4는 최종 검토에서 각 장에 반영됨)
  1. **gate 속성·key 이름:** M7 11.6.7·M9 13.6.1·13.6.8은 `ILoRAModel.omega_source`와 `omega_source.gate.proj.weight`, M10 14.6.4·14.6.6(`model.router.gate_calls`, "`router.` 접두어 키")·M11 15.2(`model.router`)·M12 16.6.5(`router.` 접두어)는 `router`. → (반영됨) M10·M11·M12를 `omega_source`로 맞춤
  2. **M5 9.8 #1의 `SpecialTokenIds.from_tokenizer`(M9):** M9 13.6.13은 "M9에 별도 id 클래스 없음"으로 바뀜 → (반영됨) M5 #1의 마지막 절을 M9 build 1단계의 `bundle.ids` 재대조로 바꿈 (대응표 M5-1)
  3. **M11 15.6.10 재현 점검과 E3:** `main.py eval`이 원 실행의 같은 (ckpt_tag, split) 기록과 같음을 "eval.batch_size가 같을 때" 요구하는데, E3(`prompt.eval_prompt = text_only`)는 원 실행(hybrid)과 prompt가 달라 반드시 다름 → 재현 요구 조건에 "`prompt.eval_prompt`(와 평가 결과를 바꾸는 허용 차이 키)도 같을 때"를 더하고, 아니면 `summary.json`에 "비교 대상 아님"으로 적어야 함 (smoke S6, M14 R9) → (반영됨) M11 15.6.10이 `not_comparable`을 적음
  4. **M8 12.8 #12의 길이 근거:** "M5 측정 6,378건 최장 311"은 M5 개정 전 값. 현재 M5 9.6.3은 6,378건 163–314, train 전체 epoch 0 key 153–334. 시험(M8-12)은 5 epoch 계획 전체에서 실제 최장 batch를 계산해 씀 → (반영됨) M8 12.8 #12 문구 정정
  5. **D13 "VESSL `ilora` 환경 재사용"과 "transformers 내부 파일 교체 제거":** fork의 `ilora` env는 `F setup_vessl.sh:87-106`이 `generation/utils.py`·`models/llama/modeling_llama.py`를 `debug/` 사본으로 바꿈 → 그 env를 그대로 쓰면 D13의 둘째 항을 어김. M14는 Python 3.10 새 env `ilora-rebuild`를 같은 requirements로 만들고 패치 단계를 돌리지 않음, check-env가 제안 4로 확인 (D13 결정 자체는 그대로)

### 17.7 결정
- 관련 확정 결정: D14(smoke 100명), D16(학습 GPU 비결정성 허용 → GPU 학습 bitwise 요구 없음), D22(batch·단독 생성 비교), D12·D23(SASRec bitwise 대조 조건)
- 이 장의 새 `[결정]`
  - pytest 9.2.0, `rebuild/pytest.ini`, marker `slow`·`vessl`·`criterion`, `vessl/`은 `ILORA_VESSL=1`과 CUDA가 있을 때만: 로컬과 VESSL이 같은 시험 코드를 씀
  - 완료 기준 번호를 시험에 붙이고 `DESIGN.md`와 정적으로 대조: 기준이 바뀌거나 시험이 빠지면 바로 실패
  - 공개 코드 동작은 출처를 적은 사본(`refs_upstream.py`)으로만 대조: 저장소 코드 import 없음(1.3)
  - 시험 합성 입력도 `derive_seed` 기반 (`np_rng(0, "test", 이름)`)
  - smoke는 `main.py smoke` 하나, 계획은 프로파일 값(`data.train_rows`, `model.backbone`)으로 정함. 1·2단계는 상태 보존·재로드·수집 지점·기록 수만 판정, 정확도 판정 없음
  - `smoke_cpu` 값 (train 2명, micro-batch 4, accumulation 4, epoch 2, `warmup_divisor` 4, worker 0, flush 5, `live_check_n` 64, batch 무관성 점검 켬): 학습 경로 전부가 수십 micro-batch 안에 나오고 w ≥ 1
  - 과적합: 사용자 0–15의 마지막 window 16개, all_hybrid, micro-batch 4·accumulation 1·25 epoch, 통과 = 마지막/첫 epoch loss 비 ≤ 0.1과 16/16 첫 줄 일치. 실패하면 본 실험을 시작하지 않음
  - 측정만 하는 VESSL 값은 `measurements.json`에 기록하고 판정하지 않음
  - check-env에 transformers 파일 무결성(wheel `RECORD` 대조) 추가 제안
  - smoke 판정: 검사 예외는 실패한 검사로 모으고 보고서는 항상 씀, 기대값은 실행 때 M1·M4·M10 함수로 계산, 학습 단계마다 공통 상태 검사 (17.6.6)
  - 과적합 O2는 호출자가 마지막 checkpoint를 불러 넘김 (제안 2 signature 유지)
  - `main.py smoke` 자체는 시험 밖에서 확인 (17.8 #3과 #4)
  - M10·M11·M12의 실제 실행 시험은 `test_m10_train.py` 안의 thread 1 실행으로 (17.6.9 머리말)

### 17.8 완료 기준
1. (CPU) `test_m13_coverage.py::test_every_criterion_has_a_test`: `DESIGN.md`의 모든 완료 기준 번호(M0–M15)가 `criterion` marker로 시험에 대응되고, 문서에 없는 번호를 가진 시험이 없음 (187 + M13·M14 자체 기준)
2. (CPU) 로컬 `python -I -m pytest rebuild/tests`가 실패 0. skip은 문서화된 이유(`tests/conftest.py`의 `SKIP_NO_TOKENIZER`·`SKIP_NO_SASREC`·`SKIP_NO_GIT_HISTORY`, `tests/vessl/conftest.py`의 `VESSL_REQUIRED`·`SKIP_NO_RUN`·`SKIP_NO_LLM`·`SKIP_NO_PL`)뿐이고 자산이 있으면 CPU 시험 skip 0 (`test_m13_coverage.py::test_skip_reasons_documented`: 시험 코드의 모든 `pytest.skip`·`importorskip`·`mark.skip`·`skipif` 이유가 두 conftest의 `DOCUMENTED_SKIP_REASONS` 상수(이름 또는 같은 문구)인지 정적 검사. 실패 0은 실행 결과 자체)
3. (CPU) `test_smoke_cpu.py::test_cpu_smoke_passes` (같은 계획을 tmp 폴더에, 명령은 17.6.6): `python -I rebuild/main.py smoke --config rebuild/configs/smoke_cpu.yaml`가 0으로 끝나고 `smoke_report.json`의 `passed`가 true, 기록 수가 17.6.6 계산값(156표본, micro 78, step 20, hybrid 10/28, `gate_calls` 78, `cluster_lookups` 156)과 같음. 시험은 같은 계획을 `cpu_smoke_run`(`run_smoke`를 tmp 폴더에)으로 판정함 `[결정]`: `main.py smoke`는 `runs/rebuild/`에 폴더를 만들어 #4에 걸리므로 시험 밖에서 돌려 확인함. `passed` false → `CommandError`·`failed`·0이 아닌 종료 경로는 `test_m0_config_run.py::test_smoke_command_fails_when_checks_fail`과 `test_smoke_cpu.py::test_overfit_plan_runs_on_cpu`가 확인
4. (CPU) 시험 session 전후로 `git status --porcelain` 출력과 `runs/rebuild/`(`assets/`, VESSL 측정 폴더 `vessl_tests_*` 제외) 목록이 같음 (`conftest.py`의 session autouse fixture `repo_guard`가 teardown에서 확인, 등록 여부는 `test_m13_coverage.py::test_repo_guard_autouse`)
5. (CPU) `test_no_repo_imports`, `test_compileall`, `test_main_help` 통과. `refs_upstream.py`의 함수마다 `U`/`F` 출처와 "대조 전용"을 적은 docstring이 있음 (정적 검사 `test_m13_coverage.py::test_refs_upstream_sources`)
6. (CPU) `test_smoke_cpu.py::test_smoke_plan_selection`: `smoke_plan`이 `overfit_gpu` → `overfit`, `smoke_cpu` → `cpu`, `smoke_gpu_100` → `gpu100`이고 `paper_spec`(subset full)을 비롯한 smoke가 아닌 14개 → `SmokeError` (17개 프로파일 전부, 프로파일 값만으로 정함). `data.train_rows` 선택 자체는 M1-13
7. (VESSL_REQUIRED) VESSL CPU 묶음(`ILORA_TEST_TOKENIZER=$LLM_PATH`)과 GPU 묶음(`ILORA_VESSL=1`) 실패 0, `measurements.json` 생성 (실패 0과 파일은 RUNBOOK R3 확인 단계, 기준 번호 marker는 `vessl/test_vessl_env.py::test_measurements_writer`: `measurements` fixture의 쓰기 함수가 M0 `create_run_dir`로 새 폴더에 JSON을 씀)
8. (VESSL_REQUIRED) `vessl/test_run_outputs.py::test_smoke_gpu_report`, `::test_overfit_report`: smoke 2단계(`smoke_gpu_100`)와 3단계(`overfit_gpu`)의 `smoke_report.json` `passed` true이고 17.6.7·17.6.8의 검사 항목이 모두 들어 있음. 측정값(메모리, 시간, loss 비, 일치 수)은 기록만, TBD. 로컬에서는 `test_smoke_cpu.py::test_overfit_generation_check_keys`(대역 모델로 O2의 key·묶음·일치 판정)와 `::test_overfit_plan_runs_on_cpu`(작은 모델로 과적합 계획 전체, 판정 실패 → `CommandError` 경로)가 같은 번호를 붙임
9. (VESSL_REQUIRED) M14 RUNBOOK 각 단계 뒤 `test_run_outputs.py`의 해당 시험 통과 (모듈 전체에 `criterion("M13-9")`, 단계별 선택어는 17.6.1)

---

## 18. M14 실험 구성
상태: 테스트 통과(로컬)

### 18.1 목적과 범위
- **목적:** 재구현으로 돌릴 실험(E1–E4와 변형)을 프로파일 YAML로 고정하고, VESSL(A100 80GB 1장)에서 돌리는 순서·명령·단계 사이 확인을 `rebuild/RUNBOOK.md`로 정하고, 결과 표의 형식과 논문 기준값 대조 방법, 한계를 정한다
- **범위**
  - `rebuild/configs/`: `paper_spec.yaml`(E2), `llara_baseline.yaml`(E1), `cluster_hard.yaml`(E4), `variant_*.yaml` 11개, `smoke_cpu.yaml`, `smoke_gpu_100.yaml`, `overfit_gpu.yaml` (4.6.2)
  - `rebuild/RUNBOOK.md`: 환경, 변환, 시험, smoke, E2 → E3 → export → 군집 → E4 → E1, 비교, 보관
  - 결과 표 (`[SPEC §12]`, D17), 논문 기준값과의 대조 규칙, 한계 (`[SPEC §13]`), 비교 도구 `rebuild/tools/compare_runs.py`
- **범위 밖:** 키의 뜻·검사·기본 동작(각 모듈 장), 시험과 smoke 판정(M13), 결과 값 (측정 전에는 TBD·NOT_RUN `[SPEC §0.11]`)

### 18.2 입력과 출력
**입력**
- 모듈 설정 표: M1 5.6.8, M2 6.6.7, M3 7.6.7, M4 8.6.8, M5 9.6.9, M6 10.6.7, M7 11.6.8, M8 12.6.7, M9 13.6.11, M10 14.6.9, M11 15.6.12, M12 16.6.9, M15 19.6.10, 결정 D1–D26
- VESSL: 저장소 clone(전체 git 이력), `LLM_PATH`(Llama-2-7b-hf, fp16), GPU 1장

**출력**
- 프로파일 파일 17개, `RUNBOOK.md`
- 실험 실행 폴더 (M0 규칙 `runs/rebuild/<프로파일>_<UTC>/`), 경로 기록 `runs/rebuild/runbook_paths.sh`
- 비교 폴더 `runs/rebuild/compare_<UTC>/`: `comparison.csv`, `comparison.md`, `spec20_index.json`, `summary.md`

### 18.3 논문 근거
- **E2 (iLoRA):** 방법 식 9–12 `[논문 p.5-6]`, K = 4 "we adapt 4 experts as the default setting" `[논문 p.9 §4.3.2]`, 데이터·설정은 LLaRA와 같음 "We request the dataset from LLaRA [9] and maintain exactly the same experimental settings as described in the original paper." `[논문 p.6 §4]`. MovieLens 결과 ValidRatio 0.9891, HitRatio@1 0.5275 `[논문 p.7 Table 1]`
- **E1 (LLaRA 기준선):** K = 1이면 식 9·12가 표준 LoRA 식 3이 됨 `[논문 p.3 식 3, p.5 식 9, p.6 식 12]`. LLaRA(SASRec) MovieLens 0.9684, 0.4421 `[LLaRA Table 2 p.8]`. iLoRA Table 1의 LLaRA 행은 다시 돌린 값이 아니라 옮겨 온 값: "Our study builds upon the main table from the LLaRA paper, using the results reported in the LLaRA paper directly." `[논문 p.6 §4]` → 같은 표 안에서 행마다 평가 프로토콜이 다를 수 있음 (D17)
- **E3:** 평가 prompt 형식만 바꾼 변형 (D4). 근거는 코드 `U@0a69c15 data/data_interface.py:22, :40` (2024-08 평가 collater가 항상 text-only, M4 8.5)
- **E4:** 논문은 hard routing을 다루지 않음 "we do not extensively investigate the effects of using hard routing for recommendations with a large number of experts." `[논문 p.9 §6]` → 연구 확장 `[SPEC §7, §9, §11]`
- **지표:** "ValidRatio [9] and HitRatio@1 are used as evaluation metrics, to separately quantify the ratios of valid responses over all sequences and relevant items over all candidate items" `[논문 p.6 §4]`. 분모와 이름은 D17. Table 1의 "Imp.%"는 같은 열 값의 비: MovieLens 19.32% = 0.5275/0.4421 − 1 (계산)
- **seed·반복:** LLM 방법의 seed 없음 → 1234 한 번 (D2) `[SPEC §0.6]`
- 실행 환경은 "a single Nvidia A100" `[논문 App B p.16]`. 1장·CUDA:0만 `[SPEC §0.12]`

### 18.4 공개 코드에서 가져올 것
- 변형 값의 출처: `U train_movielens.sh:18-20` (lr 8e-6 / 8e-4 / 8e-6 → `variant_released_lr`, D5), `:5-6` (8 × 16), `:3` (`--router share`, D1), `:9` (후보 20), `:21` (5 epoch). 주 설정 값과 공개 코드 기본값의 차이 표는 4.4
- 평가 명령 `U test_movielens.sh:15`는 checkpoint 하나(`./checkpoints/movielens.ckpt`)를 그대로 평가: 선택 규칙이 코드에 없음 → D11 선택 규칙(학습 전 고정)
- **운영 지식 (근거 아님, fork에서 VESSL로 확인된 것):** Python 3.10 conda env 필요 (`F setup_vessl.sh:23-34`), 그 env에서 `requirements.txt` 설치 (`:44`), workspace Pause가 container 파일 시스템을 지움 (`F experiments/VESSL_GUIDE.md:3-8`), 저장소용 여유 디스크 약 100 GB (`:17`), 장시간 실행은 tmux (`:80`)

### 18.5 버릴 것 (알려진 문제)
- **숨은 기본값과 스크립트 불일치** (`U main.py` 기본값 ≠ `U train_movielens.sh`, 4.4·4.5): 프로파일에 모든 키를 적고, 변형은 다른 키만 덮어씀
- **결과 덮어쓰기** (`U model/model_interface.py:179, :214`) 와 fork의 저장소 `results/` 쓰기 (SPEC §20 경로): 재구현은 `runs/rebuild/` 아래 새 폴더에만 씀 (4.6.1). SPEC §20 이름은 색인 파일로 대응 (18.6.9)
- **fork VESSL env의 transformers 교체:** `F setup_vessl.sh:87-106`이 `debug/utils.py`·`debug/modeling_llama.py`를 설치된 transformers에 복사 → 그 env를 재구현에 쓰면 D13("transformers 내부 파일 교체 제거")을 어김. 재구현은 새 env에서 같은 requirements를 설치하고 패치 단계를 돌리지 않음 (18.6.6 R1, 18.6.12 불일치 1)
- **파일 존재로 "완료" 판단** (fork run 스크립트의 단계 건너뛰기): 재구현 단계는 완료 표시(M10 `COMPLETE`, M11 `metrics.json`, `meta.json` 상태, `smoke_report.json`)로만 판단 `[SPEC §17]`
- **논문 표의 혼합 프로토콜 비교:** Table 1의 iLoRA 행(N = 92, 정답/유효)과 LLaRA 행(N = 95, 정답/전체)을 나눈 Imp.% (D17) → 우리 실행끼리는 같은 프로토콜로만 비교 (18.6.7)

### 18.6 설계

#### 18.6.1 실험 목록
| ID | 이름 | 프로파일 | 명령 | E2와 다른 키 | 목적 |
|---|---|---|---|---|---|
| E2 | iLoRA 논문 명세 | `paper_spec.yaml` | train → eval ×2 → export | – | 주 재현. iLoRA Table 1과 대조 |
| E1 | LLaRA 기준선 (K = 1) | `llara_baseline.yaml` | train → export | `lora.num_experts` 1, `router.routing_mode` none | 같은 구현의 단일 LoRA. LLaRA Table 2 SASRec 행과 대조, E2와 짝 비교 |
| E3 | E2의 text-only 평가 | `variant_text_only_eval.yaml` | eval (`eval.source_run` = E2) | `prompt.eval_prompt` text_only (평가만) | 평가 prompt 형식의 영향 (D4) |
| E4 | cluster-hard | `cluster_hard.yaml` | cluster(`paper_spec`) → train → export | `router.routing_mode` cluster_hard, `cluster.artifact_dir` | train z KMeans 군집 → expert 고정, 처음부터 학습 `[SPEC §9, §11]` |
| V | 변형 10개 (`variant_*.yaml` 11개 중 E3 파일 밖) | `variant_*.yaml` | 18.6.4 | 18.6.4 표 | 선택 실험. 돌리지 않으면 NOT_RUN |

- 모든 실험 seed 1234 (D2). 추가 seed(`--set run.seed=…`)는 계획 밖 (NOT_RUN)
- 보조 평가 프로토콜 N = 92·88(D17)은 프로파일이 아니라 모든 test 평가가 함께 계산함 (M11 15.6.6)
- E1·E4·변형은 `extends: paper_spec.yaml`(또는 `cluster_hard.yaml`)로 다른 키만 적음 `[결정]`: "나머지는 E2와 같음"을 파일 구조로 보장. 해석된 설정 전체는 각 실행 폴더의 `config.yaml`에 남음 (M0)

#### 18.6.2 설정 키 전체 (paper_spec 값)
- 키 이름·타입·라벨은 각 모듈 표 그대로 옮김. 87개 (`run.seed` 포함. `data.train_rows`는 최종 검토에서 M1이 채택, Adam β는 스칼라 키 2개 `adam_beta1`·`adam_beta2`, M10 14.6.9)

| 키 | 타입 | paper_spec 값 | 라벨 | 설명 (정의한 곳) |
|---|---|---|---|---|
| `run.seed` | int | 1234 | `[SPEC §0.6]` (D2, 계보 `[코드] U main.py:84`) | 모든 `derive_seed`의 main seed. `run` 구역의 유일한 키 (계약 추가 제안) |
| `data.title_mode` | str {code, fixed} | code | `[코드] U data/movielens_data.py:61-66, :74` (D20) | 5.6.8 |
| `data.subset` | str {full, smoke} | full | `[결정]` D14 | 5.6.8 |
| `data.smoke_users` | int 3개 | [80, 10, 10] | `[결정]` D14 | 5.6.8. full에서는 범위 검사만 |
| `data.train_rows` | tuple[int, ...] | [] | `[결정]` | 5.6.8. 비어 있으면 미사용, `overfit_gpu`만 씀 |
| `candidates.num_candidates` | int {20} | 20 | `[논문 App B p.16]` `[LLaRA §5.1.3 p.7]` `[코드] U main.py:122` (D10) | 6.6.7 |
| `sasrec.source_path` | str | `rec_model/movielens.pt` | `[코드]` U·L 같은 blob `41c057dc` | 7.6.7 |
| `sasrec.source_sha256` | str | `6928a06778d0e5b64ead0641501f73f13beb2c61ca2d881ee15582630d101df9` | `[코드]` (계산값) | 7.6.7. M0 `EXPECTED_INPUTS`와 같아야 함 |
| `sasrec.state_path` | str | `runs/rebuild/assets/sasrec/sasrec_movielens_state.pt` | `[결정]` | 7.6.7. 변환 도구 출력 (R2) |
| `sasrec.state_digest` | str | `6a78697b92ce8994250842dc780eac723285e5f51741264c09706e0057c51dae` | `[결정]` (prototype 계산값) | 7.6.2 |
| `sasrec.reference_path` | str | `runs/rebuild/assets/sasrec/sasrec_movielens_reference.pt` | `[결정]` | 7.6.7 |
| `prompt.template_path` | str {prompt/movie.txt} | `prompt/movie.txt` | `[코드]` | 8.6.8 |
| `prompt.template_choice` | str {random, fixed} | random | `[LLaRA §5.1.2 p.7]` (D15) | 8.6.8 |
| `prompt.fixed_template_index` | int 또는 null | null | `[결정]` | 8.6.8. random이면 null |
| `prompt.train_curriculum` | str {linear, all_hybrid, reverse} | linear | `[LLaRA 간접 §4.3 p.6 식 9-11]` (D3) | 8.6.8 |
| `prompt.eval_prompt` | str {hybrid, text_only} | hybrid | `[논문 p.4 식 6-7 (암시)]` (D4) | 8.6.8 |
| `batching.label_mode` | str {answer_only, released_pair} | answer_only | `[결정]` D18 | 9.6.9 |
| `batching.num_workers` | int ≥ 0 | 8 | `[코드]` `U main.py:83` | 9.6.9. 결과와 무관 |
| `batching.max_seq_len` | int ≥ 1 | 1024 | `[결정]` | 9.6.9 |
| `batching.tokenizer_sha256` | str | `9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347` | `[결정]` | 9.6.9 |
| (`projector` 구역) | – | `{}` | – | 키 없음 (10.6.7) |
| `router.routing_mode` | str {dynamic, cluster_hard, none} | dynamic | `[논문 p.6 식 11]` (D1), `[SPEC §9]` | 11.6.8 |
| `lora.r` | int | 8 | `[코드]` D7 | 12.6.7 |
| `lora.alpha` | int | 32 | `[코드]` D7 | 12.6.7 |
| `lora.dropout` | float | 0.1 | `[코드]` D7 | 12.6.7 |
| `lora.num_experts` | int | 4 | `[논문 p.9 §4.3.2]` | 12.6.7. K의 유일한 정의 |
| `lora.scaling_rule` | str {expert_rank, total_rank} | expert_rank | `[결정]` D8 | 12.6.7 |
| `lora.target_modules` | str 목록 ⊆ 7개 | [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj] | `[코드]` D7 | 12.6.7 |
| `model.backbone` | str {llama2_7b, tiny_random} | llama2_7b | `[논문 App B p.16]`, base는 `[코드]` `U README.md:13` | 13.6.11 |
| `model.torch_dtype` | str {float16, bfloat16, float32} | float16 | `[결정]` D9 | 13.6.11 |
| `model.device` | str {cuda:0, cpu} | cuda:0 | `[SPEC §0.12]` | 13.6.11 |
| `model.gen_strategy` | str {greedy} | greedy | `[코드]` `U model/model_interface.py:70`, `L model/model_interface.py:43` | 13.6.11 |
| `model.gen_max_new_tokens` | int | 64 | `[코드]` `U …:70, :80` | 13.6.11 |
| `model.gen_min_new_tokens` | int | 1 | `[코드]` `U …:70, :81` | 13.6.11 |
| `model.gen_repetition_penalty` | float | 1.0 | `[코드]` `U …:70, :83` | 13.6.11 |
| `train.epochs` | int ≥ 1 | 5 | `[v1 App E p.16]` `[LLaRA §5.1.2 p.7]` D11 | 14.6.9 |
| `train.micro_batch_size` | int ≥ 1 | 8 | `[결정]` 값은 `[코드]` `U train_movielens.sh:5` | 14.6.9 |
| `train.grad_accum` | int ≥ 1 | 16 | `[결정]` 값은 `[코드]` `U train_movielens.sh:6`, 곱 128 `[v1 App E p.16]` `[LLaRA §5.1.2 p.7]` | 14.6.9 |
| `train.lr_peak` | float > 0 | 1.0e-4 | `[논문 App E p.17]` D5 | 14.6.9 |
| `train.lr_warmup_start` | float > 0 | 1.0e-6 | `[v1 App E p.16]` `[LLaRA §5.1.2 p.7]` D6 | 14.6.9 |
| `train.lr_floor` | float ≥ 0 | 1.0e-6 | `[코드]` `U main.py:90` D6 | 14.6.9 |
| `train.warmup_divisor` | int ≥ 1 | 20 | `[코드]` `U model/model_interface.py:248` D6 | 14.6.9 |
| `train.lr_scale_projector` | float > 0 | 1.0 | `[논문 App E p.17]` D19 | 14.6.9 |
| `train.lr_scale_gate` | float > 0 | 1.0 | `[논문 p.5 §3.2 (2)]` `(암시)` D19 | 14.6.9 |
| `train.lr_scale_lora` | float > 0 | 1.0 | `[논문 App E p.17]` D19 | 14.6.9 |
| `train.weight_decay_projector` | float ≥ 0 | 1.0e-5 | `[v1 App E p.16]` 값, 적용 범위 `[코드]` `U model/model_interface.py:228` D19 | 14.6.9 |
| `train.weight_decay_gate` | float ≥ 0 | 1.0e-5 | `[v1 App E p.16]` 값, 적용 범위 `[코드]` `U model/model_interface.py:230` D19 | 14.6.9 |
| `train.weight_decay_lora` | float ≥ 0 | 0.0 | `[코드]` `U model/model_interface.py:232` D19 | 14.6.9 |
| `train.adam_beta1` | float [0, 1) | 0.9 | `[코드]` torch 2.0.0 기본값, U·L 기본값 사용 | 14.6.9 |
| `train.adam_beta2` | float [0, 1) | 0.999 | `[코드]` 같음 | 14.6.9 |
| `train.adam_eps` | float > 0 | 1.0e-8 | `[코드]` 같음 | 14.6.9 |
| `train.precision` | str {fp16_amp, bf16_amp, fp32} | fp16_amp | `[논문 App E p.17]` D9 | 14.6.9 |
| `train.scaler_init_scale` | float > 0 | 65536.0 | `[코드]` torch 2.0.0 GradScaler 기본값 | 14.6.9 |
| `train.scaler_growth_factor` | float > 1 | 2.0 | `[코드]` 같음 | 14.6.9 |
| `train.scaler_backoff_factor` | float (0, 1) | 0.5 | `[코드]` 같음 | 14.6.9 |
| `train.scaler_growth_interval` | int ≥ 1 | 2000 | `[코드]` 같음 | 14.6.9 |
| `train.max_consecutive_skips` | int ≥ 1 | 16 | `[결정]` | 14.6.8 |
| `train.log_every_steps` | int ≥ 1 | 10 | `[결정]` | 14.6.9 |
| `eval.batch_size` | int ≥ 1 | 8 | `[결정]` (계보 `[코드]` `U train_movielens.sh:5`) | 15.6.12. 모든 실험 같은 값 |
| `eval.batch_invariance_check` | bool | false | `[결정]` | 15.6.12 |
| `eval.source_run` | str 또는 null | null | `[결정]` | 15.6.12. eval 명령에서 `--set` |
| `eval.checkpoints` | str 목록 | [best_val, last] | `[결정]` D11 | 15.6.12 |
| `eval.splits` | str 목록 ⊆ {val, test} | [val, test] | `[결정]` D11·D17 | 15.6.12 |
| `records.flush_every_steps` | int ≥ 1 | 20 | `[결정]` | 16.6.9 |
| `records.activation_max` | bool | true | `[결정]` | 16.6.9 |
| `records.gate_rowsum_atol` | float > 0 | 1.0e-5 | `[결정]` | 16.6.9 |
| `records.gate_crosscheck_atol` | float > 0 | 1.0e-5 | `[결정]` | 16.6.9 |
| `records.source_run` | str 또는 null | null | `[결정]` | 16.6.9. export에서 `--set` |
| `records.export_checkpoints` | 목록 ⊆ {best_val, last} | [best_val, last] | `[결정]` D11 | 16.6.9 |
| `records.export_num_threads` | int ≥ 1 | 1 | `[결정]` | 16.6.9 |
| `records.export_batch_size` | int ≥ 1 | 1024 | `[결정]` | 16.6.9 |
| `records.analysis_splits` | 목록 ⊆ {test, val, train} | [test, val, train] | `[결정]` | 16.6.9 |
| `records.figure_formats` | 목록 ⊆ {png, pdf} | [png] | `[SPEC §6]` | 16.6.9 |
| `cluster.unit` | str {sequence, user} | sequence | `[결정]` D25 | 19.6.10 |
| `cluster.n_clusters` | int ≥ 2 | 4 | `[SPEC §7]` | 19.6.10 |
| `cluster.random_state` | int | 1234 | `[SPEC §7]` | 19.6.10. `derive_seed` 예외 |
| `cluster.n_init` | int ≥ 1 | 10 | `[SPEC §7]` 값 `[결정]` | 19.6.10 |
| `cluster.init` | str {k-means++} | k-means++ | `[결정]` | 19.6.10 |
| `cluster.algorithm` | str {lloyd} | lloyd | `[결정]` | 19.6.10 |
| `cluster.max_iter` | int ≥ 1 | 300 | `[결정]` | 19.6.10 |
| `cluster.tol` | float > 0 | 1.0e-4 | `[결정]` | 19.6.10 |
| `cluster.fit_num_threads` | int ≥ 1 | 1 | `[결정]` | 19.6.10 |
| `cluster.expert_mapping` | str | "0:0,1:1,2:2,3:3" | `[SPEC §9]` | 19.6.10. YAML에서 반드시 따옴표 (아래) |
| `cluster.artifact_dir` | str 또는 null | null | `[결정]` | 19.6.10. cluster_hard에서 `--set` 필수 |
| `cluster.dynamic_export_dir` | str 또는 null | null | `[결정]` | 19.6.10. cluster 명령에서 `--set` |
| `cluster.z_crosscheck_atol` | float > 0 | 1.0e-5 | `[결정]` | 19.6.10 |
| `cluster.live_check_n` | int ≥ 1 | 256 | `[결정]` | 19.6.10 |
| `cluster.live_check_atol` | float > 0 | 1.0e-4 | `[결정]` | 19.6.10 |

- **YAML 표기 규칙** `[결정]` (PyYAML 6.0 = YAML 1.1, 확인함)
  - 실수는 소수점과 지수를 함께 적음 (`1.0e-4`): `1e-4`는 문자열로 읽힘 (M0 `_coerce`가 받아 주지만 파일에서는 피함)
  - `cluster.expert_mapping`은 따옴표: 쉼표 없는 `1:1`은 60진수 int 61로 읽힘. 명령행에서도 `--set 'cluster.expert_mapping="0:3,1:2,2:1,3:0"'`처럼 YAML 문자열로 줌 (따옴표가 없으면 M0이 `ConfigError`로 거부해 조용히 틀리지는 않음)
  - 해시는 따옴표 문자열, `cuda:0`도 따옴표
  - 목록 값의 `--set`은 셸 glob을 피하려고 작은따옴표로 감쌈 (`--set 'eval.checkpoints=[best_val,last]'`)
- 경로 값(`sasrec.*_path`, `prompt.template_path`, `*.source_run`, `cluster.artifact_dir`, `cluster.dynamic_export_dir`)은 절대 경로이거나 저장소 루트 기준 상대 경로 (계약 추가 제안 18.6.12). RUNBOOK은 저장소 루트에서 실행

#### 18.6.3 `rebuild/configs/paper_spec.yaml` (E2)
```yaml
# E2 iLoRA paper_spec. Every key is listed (DESIGN 4.6.2); evidence in DESIGN 18.6.2.
run:
  seed: 1234
data:
  title_mode: code
  subset: full
  smoke_users: [80, 10, 10]
  train_rows: []              # empty = not used (DESIGN 5.6.7); only overfit_gpu sets rows
candidates:
  num_candidates: 20
sasrec:
  source_path: rec_model/movielens.pt
  source_sha256: "6928a06778d0e5b64ead0641501f73f13beb2c61ca2d881ee15582630d101df9"
  state_path: runs/rebuild/assets/sasrec/sasrec_movielens_state.pt
  state_digest: "6a78697b92ce8994250842dc780eac723285e5f51741264c09706e0057c51dae"
  reference_path: runs/rebuild/assets/sasrec/sasrec_movielens_reference.pt
prompt:
  template_path: prompt/movie.txt
  template_choice: random
  fixed_template_index: null
  train_curriculum: linear
  eval_prompt: hybrid
batching:
  label_mode: answer_only
  num_workers: 8
  max_seq_len: 1024
  tokenizer_sha256: "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"
projector: {}
router:
  routing_mode: dynamic
lora:
  r: 8
  alpha: 32
  dropout: 0.1
  num_experts: 4
  scaling_rule: expert_rank
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
model:
  backbone: llama2_7b
  torch_dtype: float16
  device: "cuda:0"
  gen_strategy: greedy
  gen_max_new_tokens: 64
  gen_min_new_tokens: 1
  gen_repetition_penalty: 1.0
train:
  epochs: 5
  micro_batch_size: 8
  grad_accum: 16
  lr_peak: 1.0e-4
  lr_warmup_start: 1.0e-6
  lr_floor: 1.0e-6
  warmup_divisor: 20
  lr_scale_projector: 1.0
  lr_scale_gate: 1.0
  lr_scale_lora: 1.0
  weight_decay_projector: 1.0e-5
  weight_decay_gate: 1.0e-5
  weight_decay_lora: 0.0
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_eps: 1.0e-8
  precision: fp16_amp
  scaler_init_scale: 65536.0
  scaler_growth_factor: 2.0
  scaler_backoff_factor: 0.5
  scaler_growth_interval: 2000
  max_consecutive_skips: 16
  log_every_steps: 10
eval:
  batch_size: 8
  batch_invariance_check: false
  source_run: null
  checkpoints: [best_val, last]
  splits: [val, test]
records:
  flush_every_steps: 20
  activation_max: true
  gate_rowsum_atol: 1.0e-5
  gate_crosscheck_atol: 1.0e-5
  source_run: null
  export_checkpoints: [best_val, last]
  export_num_threads: 1
  export_batch_size: 1024
  analysis_splits: [test, val, train]
  figure_formats: [png]
cluster:
  unit: sequence
  n_clusters: 4
  random_state: 1234
  n_init: 10
  init: k-means++
  algorithm: lloyd
  max_iter: 300
  tol: 1.0e-4
  fit_num_threads: 1
  expert_mapping: "0:0,1:1,2:2,3:3"
  artifact_dir: null
  dynamic_export_dir: null
  z_crosscheck_atol: 1.0e-5
  live_check_n: 256
  live_check_atol: 1.0e-4
```
- 파일 안 주석은 영어 ASCII만 (Windows 기본 코드 페이지 편집기에서 깨지지 않게). 값의 근거는 18.6.2 표와 각 모듈 설정 표, 코드 주석(M0 `item(label, …)`)이 가짐
- 구역 사이 조합은 M0 `Config.validate`가 검사: `routing_mode == cluster_hard` ⇔ `cluster.artifact_dir` not null, `routing_mode == none` ⇔ `lora.num_experts == 1` (M7 11.6.8), `model.torch_dtype` ↔ `train.precision`, `tiny_random` ⇔ (`cpu`, `float32`) (M9 13.6.11)

#### 18.6.4 E1·E4·변형 overlay
```yaml
# llara_baseline.yaml (E1)
extends: paper_spec.yaml
lora:
  num_experts: 1
router:
  routing_mode: none
```
```yaml
# cluster_hard.yaml (E4). cluster.artifact_dir is given only on the command line:
#   --set cluster.artifact_dir=<cluster run>/cluster   (ConfigError without it)
extends: paper_spec.yaml
router:
  routing_mode: cluster_hard
```
- `cluster.artifact_dir`를 파일에 두지 않는 이유 `[결정]`: 군집 실행 폴더 이름에 UTC가 들어가 미리 알 수 없고, 빈 값이면 M0 검사가 적재 단계에서 거부하므로 잘못된 산출물을 조용히 쓰는 경로가 없음. 실제 값은 학습 실행 `config.yaml`·`meta.json`(덮어쓴 항목)에 남음

| 파일 | extends | 다른 키 (값) | D | 명령 |
|---|---|---|---|---|
| `variant_all_hybrid.yaml` | paper_spec | `prompt.train_curriculum: all_hybrid` | D3 | train |
| `variant_reverse_curriculum.yaml` | paper_spec | `prompt.train_curriculum: reverse` | D3 | train |
| `variant_text_only_eval.yaml` (E3) | paper_spec | `prompt.eval_prompt: text_only` | D4 | eval (`eval.source_run` = E2) |
| `variant_released_lr.yaml` | paper_spec | `train.lr_peak: 8.0e-4`, `train.lr_warmup_start: 8.0e-6`, `train.lr_floor: 8.0e-6` | D5, D6 | train |
| `variant_released_gate_lr.yaml` | paper_spec | `train.lr_scale_gate: 0.3` | D19 | train |
| `variant_released_label.yaml` | paper_spec | `batching.label_mode: released_pair` | D18 | train |
| `variant_total_rank_scaling.yaml` | paper_spec | `lora.scaling_rule: total_rank` | D8 | train |
| `variant_fixed_template.yaml` | paper_spec | `prompt.template_choice: fixed`, `prompt.fixed_template_index: 0` | D15 | train |
| `variant_fixed_titles.yaml` | paper_spec | `data.title_mode: fixed` | D20 | train |
| `variant_user_level_cluster.yaml` | cluster_hard | `cluster.unit: user` | D25 | cluster(`paper_spec` + `--set cluster.unit=user`) → train |
| `variant_bf16.yaml` | paper_spec | `model.torch_dtype: bfloat16`, `train.precision: bf16_amp` | D9 | train. fp16이 불안정할 때만 (M10 중단 규칙에 걸리거나 비유한 parameter), 쓰면 편차로 기록 |

- 각 파일은 `extends`와 위 키만 가짐. D17 보조 프로토콜과 D2 추가 seed는 파일이 없음 (18.6.1)
- 해석된 설정이 기준 파일과 표의 키에서만 다름을 M14 시험이 확인 (18.8 #2)

#### 18.6.5 smoke·과적합 프로파일 (값의 이유와 판정은 M13 17.6.6–17.6.8)
```yaml
# smoke_cpu.yaml (M13 smoke level 1, local CPU, tiny random LLaMA)
extends: paper_spec.yaml
data:
  subset: smoke
  smoke_users: [2, 10, 10]
batching:
  num_workers: 0
model:
  backbone: tiny_random
  torch_dtype: float32
  device: cpu
train:
  epochs: 2
  micro_batch_size: 4
  grad_accum: 4
  warmup_divisor: 4
  precision: fp32
eval:
  batch_invariance_check: true
records:
  flush_every_steps: 5
cluster:
  live_check_n: 64
```
```yaml
# smoke_gpu_100.yaml (M13 smoke level 2, D14: 80 + 10 + 10 users)
extends: paper_spec.yaml
data:
  subset: smoke
train:
  epochs: 2
eval:
  batch_invariance_check: true
```
```yaml
# overfit_gpu.yaml (M13 smoke level 3: last train window of users 0-15)
extends: paper_spec.yaml
data:
  subset: smoke
  smoke_users: [16, 10, 10]
  train_rows: [71, 196, 377, 554, 797, 1001, 1129, 1234, 1595, 1651, 1905, 2070, 2311, 2693, 2765, 2874]
prompt:
  train_curriculum: all_hybrid
train:
  epochs: 25
  micro_batch_size: 4
  grad_accum: 1
```
- 계산된 학습 길이 (M10 14.6.2 식, seed 1234, 계산 `design/tmp/m1314/smoke_sizes.py`, `curriculum_counts.py`)

| 프로파일 | train 표본 | NB | tail | step/epoch | T | w | micro-batch | hybrid/epoch |
|---|---|---|---|---|---|---|---|---|
| smoke_cpu (2 epoch) | 156 | 39 | 3 | 10 | 20 | 5 | 78 | 10, 28 |
| smoke_cpu 하위 1 epoch (E4·E1 경로) | 156 | 39 | 3 | 10 | 10 | 2 | 39 | 23 |
| smoke_gpu_100 (2 epoch) | 7,230 | 903 | 7 | 57 | 114 | 5 | 1,806 | 218, 670 |
| smoke_gpu_100 하위 1 epoch | 7,230 | 903 | 7 | 57 | 57 | 2 | 903 | 428 |
| overfit_gpu (25 epoch) | 16 | 4 | 0 | 4 | 100 | 5 | 100 | 4씩 (all_hybrid) |
| paper_spec (E1·E2·E4) | 66,126 | 8,265 | 9 | 517 | 2,585 | 129 | 41,325 | 846 … 7,436 (M4 8.6.5) |

#### 18.6.6 `rebuild/RUNBOOK.md`: VESSL 실행 순서와 명령
- 모든 명령은 저장소 루트에서, conda env `ilora-rebuild`, `CUDA_VISIBLE_DEVICES=0` `[SPEC §0.12]`. GPU 작업은 한 번에 하나
- 각 단계는 "명령 → 산출물 → 확인(다음 단계의 조건)". 확인이 실패하면 다음 단계로 가지 않고 원인을 기록함. 결과 수치는 측정 뒤에만 적음 `[SPEC §0.11]`
- 실행 폴더 경로는 그 폴더를 만든 명령 뒤에 `&&`로 이어 `latest <프로파일>`로 잡고, `done_ok <폴더> <명령>`이 `meta.json` 상태 `completed`와 `command`를 확인한 뒤 `rec`로 `runs/rebuild/runbook_paths.sh`에 덧붙임 `[결정]`: eval·export·cluster·check-env 실행도 `--config` 이름(`paper_spec_<UTC>`)을 쓰므로 `latest`만으로는 다른 명령의 폴더를 고를 수 있고, 명령이 폴더를 만들기 전에 실패하면(`ConfigError`, 종료 코드 2) 사슬이 멈춰 앞 단계 폴더를 기록하지 않음
- R8 이후 명령은 `rec`가 기록한 `ILORA_*` 이름만 읽고 같은 사슬 앞에서 `need <이름>...`으로 기록 여부를 확인함 `[결정]`: 셸이 끊겨도 R0의 `source`로 경로가 돌아오고, 빈 경로가 `--set`에 들어가 엉뚱한 폴더나 모두 NOT_RUN인 비교표를 만들지 않음. `E2`·`R`·`CL` 같은 짧은 변수는 한 줄 안의 임시 이름 `R tests/test_m14_runbook.py test_paths_survive_a_new_shell`
- 긴 실행(R5–R15)은 tmux 안에서 함. SSH·셸이 끊겨도 tmux 안 실행은 계속되므로 다시 접속하면 `tmux attach -t rebuild`로 붙어 끝났는지 먼저 봄 (재개 조건은 R7). Hugging Face token은 셸 환경 변수 `HF_TOKEN`으로만 쓰고 파일·명령 기록에 적지 않음

**R0. 매 셸 준비** (workspace에서 처음이면 R1로 clone과 env를 먼저 만듦)
```bash
cd ~/iLoRA                                         # 저장소 루트
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ilora-rebuild
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0
export LLM_PATH=/root/models/Llama-2-7b-hf         # 실제 위치
export ILORA_TEST_TOKENIZER="$LLM_PATH"
mkdir -p runs/rebuild && touch runs/rebuild/runbook_paths.sh && source runs/rebuild/runbook_paths.sh
latest() { ls -1d runs/rebuild/"$1"_[0-9]*Z 2>/dev/null | sort | tail -n 1; }
done_ok() { python -I -c 'import json,sys; m=json.load(open(sys.argv[1]+"/meta.json")); print(sys.argv[1], m["status"], m.get("command")); sys.exit(m["status"]!="completed" or m.get("command")!=sys.argv[2])' "$1" "$2"; }
rec() { echo "export $1=$2" >> runs/rebuild/runbook_paths.sh; export "$1=$2"; }
need() { for n in "$@"; do test -n "${!n:-}" || { echo "need: $n is not recorded (do its step first)" >&2; return 1; }; done; }
outputs() { ILORA_VESSL=1 python -I -m pytest rebuild/tests/vessl/test_run_outputs.py -q -k "$1"; }
```

**R1. 환경 (workspace마다 한 번)** `[결정]`
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
- fork의 `setup_vessl.sh`·`bootstrap_vessl.sh`는 이 env에서 돌리지 않음 (transformers 파일 교체, 18.5). 같은 이유로 fork의 `ilora` env를 쓰지 않음
- 확인: torch 2.0.0, CUDA 11.7, transformers 4.28.0. 파일 무결성은 R4 check-env가 확인 (M13 17.6.12 제안 4)
- clone은 전체 이력(`--depth` 없음, R2가 확인). 실행할 commit이 `rebuild/`를 포함해야 하고 R2의 `git status --porcelain`이 비어 있어야 `meta.json`의 commit이 실행한 코드를 대표함. `conda`가 없으면 Miniforge, `tmux`가 없으면 설치 `R RUNBOOK.md R1`
- LLaMA 가중치가 `$LLM_PATH`에 없을 때만 `[결정]`: 내려받기용 별도 env `hf-download`에서 `huggingface_hub`의 `snapshot_download`로 `meta-llama/Llama-2-7b-hf`의 `*.json`·`*.safetensors`·`tokenizer.model`만 받음 (token은 셸의 `HF_TOKEN`): `ilora-rebuild`의 고정 판을 건드리지 않기 위함. 처음 workspace에서는 R1이 R0보다 먼저라 그 블록 첫 줄에서 `LLM_PATH`를 R0과 같은 값으로 export함. 확인은 R4 check-env (`config.json` `torch_dtype` float16, `tokenizer.model` sha256)

**R2. 코드 기록과 SASRec 변환 (한 번)**
```bash
git rev-parse HEAD && git status --porcelain       # meta.json에도 기록됨
test "$(git rev-parse --is-shallow-repository)" = false
git cat-file -e 28e47371192121b6ced44401d9e149fedfae13de^{commit}
python -I rebuild/tools/convert_sasrec.py --repo . --out runs/rebuild/assets/sasrec
```
- 산출물: `runs/rebuild/assets/sasrec/{sasrec_movielens_state.pt, sasrec_movielens_reference.pt, manifest.json}` (M3 7.6.1)
- 확인: manifest의 digest = `sasrec.state_digest` (`6a78697b…1dae`). 폴더가 이미 있으면 도구가 거부함 → 이전 변환본을 그대로 쓰고 R4가 digest를 확인
- 변환은 VESSL 기계에서 함: CPU 기준 출력의 bitwise 일치는 같은 기계에서 만든 기준 파일에서만 보장됨 (M3 7.6.5). 군집(R11) 뒤에는 다시 변환하지 않음: E4는 군집 `manifest.json`의 state sha256과 같은 파일을 요구함 (M15 19.6.7)

**R3. 시험**
```bash
python -I -m pytest rebuild/tests -q                                   # CPU 묶음 (Linux, 실제 tokenizer)
ILORA_VESSL=1 python -I -m pytest rebuild/tests/vessl -q -k "not run_outputs"
```
- 산출물: pytest 요약, `runs/rebuild/vessl_tests_<UTC>/measurements.json` (M13 17.6.2)
- 확인: 두 묶음 모두 실패 0. skip은 문서화된 이유만 (M13 17.8 #2, #7). CPU 묶음은 `tests/vessl`도 모으지만 `ILORA_VESSL`이 없어 그 시험은 모두 `VESSL_REQUIRED`로 skip되므로, CPU 묶음에서 `VESSL_REQUIRED` 밖의 skip은 0 (요약 끝 `SKIPPED` 줄로 봄). VESSL 묶음의 skip은 0 (`SKIP_NO_PL`이면 그 이유가 `measurements.json`에 남음)

**R4. 환경 점검**
```bash
python -I rebuild/main.py check-env --config rebuild/configs/paper_spec.yaml \
  && C=$(latest paper_spec) && done_ok "$C" check-env && rec ILORA_RUN_CHECK_ENV "$C" && python -m pip freeze > "$C/pip_freeze.txt"
```
- 산출물: `check_env.json`, `meta.json`, env 판 기록 `pip_freeze.txt` (18.8 #9)
- 확인: 종료 코드 0 (GPU 1장·75 GiB 이상, 판 고정, transformers 무결성, `LLM_PATH`·`torch_dtype` float16, 입력 해시, SASRec digest·기준 대조, tokenizer sha256, `candidates.self_check()`)

**R5. smoke 2단계 (GPU 100명)**
```bash
python -I rebuild/main.py smoke --config rebuild/configs/smoke_gpu_100.yaml \
  && S=$(latest smoke_gpu_100) && done_ok "$S" smoke && rec ILORA_RUN_SMOKE_GPU "$S" && outputs smoke_gpu
```
- 산출물: `smoke_report.json`, `steps/` (학습 2 epoch, 재평가, export, 군집, E4·E1 경로 각 1 epoch)
- 확인: `passed` true (M13 17.6.7 통과 기준). 측정한 메모리·시간을 기록 (TBD)

**R6. smoke 3단계 (과적합)**
```bash
python -I rebuild/main.py smoke --config rebuild/configs/overfit_gpu.yaml \
  && O=$(latest overfit_gpu) && done_ok "$O" smoke && rec ILORA_RUN_OVERFIT "$O" && outputs overfit
```
- 확인: `passed` true (loss 비 ≤ 0.1, 16/16 첫 줄 일치. M13 17.6.8). 실패하면 R7 이후를 시작하지 않음

**R7. E2 학습** (tmux 안에서: `tmux new -s rebuild` 후 R0)
```bash
python -I rebuild/main.py train --config rebuild/configs/paper_spec.yaml \
  && E2=$(latest paper_spec) && done_ok "$E2" train && rec ILORA_RUN_E2 "$E2" && outputs e2_run
```
- 산출물: `checkpoints/epoch0..4/`, `eval/epoch0..4/val/`, `eval/epoch{a,b}/test/`, `eval/selection.json`, `records/train/` (M10–M12)
- 확인: `test_run_outputs.py`의 `test_e2_run_*` (M13 대응표 M3-13, M6-9, M7-14, M10-15, M11-18·22, M12-15) 통과
- 중단되면 `[결정]`: 재개는 학습 프로세스가 끝났을 때만 함 (예외 → `failed`, OOM killer·`kill -9`·tmux 세션 종료 → `running`이 남음). SSH·셸이 끊긴 것만으로는 tmux 안 학습이 멈추지 않으므로 먼저 `tmux attach -t rebuild`로 붙어 보고, 돌고 있으면 재개하지 않음. 재개 블록은 `pgrep -af 'rebuild/main.py'`(출력 없음)와 `nvidia-smi --query-gpu=memory.used --format=csv,noheader`(수 MiB)로 확인한 뒤 `R=$(latest paper_spec) && python -I rebuild/main.py train --resume "$R" && done_ok "$R" train && rec ILORA_RUN_E2 "$R" && outputs e2_run` (M10 14.6.10, `--config`·`--set` 없음). 살아 있는 프로세스가 그 폴더의 `run.lock`을 쥐고 있으면 `train --resume`이 거부하고 아무것도 바꾸지 않음 (`... is in use by a live process`, M0). 강제 종료된 프로세스의 `train.segments` 항목은 재개 때 `interrupted`(`ended_utc` null) `R ilora/train.py segment_problems`. 재개 뒤 같은 확인
- fp16이 중단 규칙(GradScaler skip 연속 16회, 비유한 parameter)에 걸리면 이 절 끝의 bf16 대안으로 E2부터 다시 함

**R8. E2 재평가 2회 (결정성, batch 무관성)**
```bash
need ILORA_RUN_E2 && python -I rebuild/main.py eval --config rebuild/configs/paper_spec.yaml --set eval.source_run="$ILORA_RUN_E2" \
  && R=$(latest paper_spec) && done_ok "$R" eval && rec ILORA_RUN_E2_REEVAL "$R"
need ILORA_RUN_E2 && python -I rebuild/main.py eval --config rebuild/configs/paper_spec.yaml --set eval.source_run="$ILORA_RUN_E2" \
  --set eval.batch_invariance_check=true && R=$(latest paper_spec) && done_ok "$R" eval && rec ILORA_RUN_E2_REEVAL_BI "$R" \
  && outputs e2_reeval
```
- 확인: 두 실행 `eval/summary.json`의 재현 점검이 모두 같음 (M10-17, M11-19). 다르면 R9 이후 결과를 확정하지 않고 다른 표본과 처음 다른 생성 단계를 기록 (M11 15.8 #19). 일치율 기록 (M11-20, 값 TBD)

**R9. E3 (E2 checkpoint의 text-only 평가)**
```bash
need ILORA_RUN_E2 && python -I rebuild/main.py eval --config rebuild/configs/variant_text_only_eval.yaml --set eval.source_run="$ILORA_RUN_E2" \
  && E3=$(latest variant_text_only_eval) && done_ok "$E3" eval && rec ILORA_RUN_E3 "$E3"
need ILORA_RUN_E3 && python -I -c 'import json,sys; r=json.load(open(sys.argv[1]+"/eval/summary.json"))["results"]; s={k: v["reproduction"]["status"] for k, v in r.items()}; print(s); sys.exit(set(s.values())!={"not_comparable"})' "$ILORA_RUN_E3"
need ILORA_RUN_E3 && python -I -c 'import glob,json,sys; h=[json.loads(x)["hybrid"] for p in glob.glob(sys.argv[1]+"/eval/*/*/records.jsonl") for x in open(p, encoding="utf-8")]; print(len(h), "records, hybrid values", set(h)); sys.exit(not h or any(h))' "$ILORA_RUN_E3"
```
- 확인: 상태 completed, 기록 `hybrid` false. 재현 점검은 prompt가 달라 `summary.json`의 `reproduction`이 `not_comparable`이어야 함 (M11 15.6.10, 최종 검토에서 반영)

**R10. E2 export와 gate 분석**
```bash
need ILORA_RUN_E2 && python -I rebuild/main.py export --config rebuild/configs/paper_spec.yaml --set records.source_run="$ILORA_RUN_E2" \
  && E2X=$(latest paper_spec) && done_ok "$E2X" export && rec ILORA_RUN_E2_EXPORT "$E2X" \
  && A=$(python -I -c 'import json,sys; print(json.load(open(sys.argv[1]+"/eval/selection.json"))["best_val_epoch"])' "$ILORA_RUN_E2") \
  && rec ILORA_E2_BEST "$A" && outputs e2_export
```
- 산출물: `records/export/epoch{a,b}/` (NPZ 3개, `gates_{validation,test}.csv`, `crosscheck.json`, `manifest.json`), `records/analysis/epoch{a,b}/{test,val,train}/`
- 확인: 대조 통과(\|Δω\| ≤ 1e-5), 파일 sha256이 manifest와 같음, 그림·요약 존재 (M12-16)
- 실패하면 R11을 시작하지 않음. 대조가 실패해도 `crosscheck.json`은 먼저 쓰임 (M12 16.6.5 7단계): 실패한 export 폴더(상태 `failed`)의 `records/export/epoch{e}/crosscheck.json`에서 `passed`가 false인 split의 `max_abs_diff_omega`·`max_abs_diff_omega_sample_id`·`argmax_mismatches`·`max_abs_diff_z`를 기록함. `max_abs_diff_z`가 1e-4 근처면 평가 z가 다른 것(CUDA 반올림·TF32), 훨씬 작으면 gate가 다른 것 (16.7) `R RUNBOOK.md R10`

**R11. 군집 (train만 fit)**
```bash
need ILORA_RUN_E2_EXPORT ILORA_E2_BEST && python -I rebuild/main.py cluster --config rebuild/configs/paper_spec.yaml \
  --set cluster.dynamic_export_dir="$ILORA_RUN_E2_EXPORT/records/export/epoch$ILORA_E2_BEST" \
  && CL=$(latest paper_spec) && done_ok "$CL" cluster && rec ILORA_RUN_CLUSTER "$CL" && outputs cluster
```
- 산출물: `cluster/` (`kmeans_k4_seed1234.joblib`, `centroids.npz`, 배정 CSV 3개, `cluster_summary.json`, 그림 2개, `manifest.json`) (M15 19.2)
- 확인: manifest `fit_split = train`, fit 66,126행, 빈 군집 없음, E2 export z와 대조 ≤ 1e-5, dynamic 비교 항목 존재 (M15-14) `[SPEC §0.8]`

**R12. E4 학습과 export**
```bash
need ILORA_RUN_CLUSTER && python -I rebuild/main.py train --config rebuild/configs/cluster_hard.yaml \
  --set cluster.artifact_dir="$ILORA_RUN_CLUSTER/cluster" \
  && E4=$(latest cluster_hard) && done_ok "$E4" train && rec ILORA_RUN_E4 "$E4"
need ILORA_RUN_CLUSTER ILORA_RUN_E4 && python -I rebuild/main.py export --config rebuild/configs/cluster_hard.yaml \
  --set cluster.artifact_dir="$ILORA_RUN_CLUSTER/cluster" --set records.source_run="$ILORA_RUN_E4" \
  && E4X=$(latest cluster_hard) && done_ok "$E4X" export && rec ILORA_RUN_E4_EXPORT "$E4X" && outputs e4
```
- 확인: 학습 시작 때 live z 대조 통과(M15-12, `meta.json` `cluster_live_check`), 학습 `gate_calls` 합 0·`cluster_lookups` 합 330,600, 평가 gate one-hot = mapping[배정] (M7-15, M15-13), export ω가 세 split 모두 mapping[배정]의 one-hot이고 대조 통과 (`test_e4_export`, 모듈 번호 M13-9). E2 학습 가중치를 쓰지 않고 처음부터 학습함 `[SPEC §9, §25]`
- 중단되면 R7의 조건에서만 `R=$(latest cluster_hard) && python -I rebuild/main.py train --resume "$R" && done_ok "$R" train && rec ILORA_RUN_E4 "$R"`, 그다음 export 줄부터. 군집 산출물이나 SASRec state가 바뀌었으면 학습 전에 `ClusterError`로 멈춤: 군집을 다시 fit하지 않고 원인(오류가 파일 이름을 댐)을 기록

**R13. E1 학습과 export**
```bash
python -I rebuild/main.py train --config rebuild/configs/llara_baseline.yaml \
  && E1=$(latest llara_baseline) && done_ok "$E1" train && rec ILORA_RUN_E1 "$E1"
need ILORA_RUN_E1 && python -I rebuild/main.py export --config rebuild/configs/llara_baseline.yaml --set records.source_run="$ILORA_RUN_E1" \
  && E1X=$(latest llara_baseline) && done_ok "$E1X" export && rec ILORA_RUN_E1_EXPORT "$E1X" && outputs e1
```
- 확인: `test_e1_run_outputs` (M14 18.8 #8): 학습 실행의 그룹 {projector, lora}, 기록 `gate_0` = 1.0, 학습 기록 수는 E2와 같음 (41,325·2,585·5), export는 gate 분석 없이 성공 (M12-11)
- 중단되면 R7의 조건에서만 `R=$(latest llara_baseline) && python -I rebuild/main.py train --resume "$R" && done_ok "$R" train && rec ILORA_RUN_E1 "$R"`, 그다음 export 줄부터

**R14. 비교 표**
```bash
need ILORA_RUN_E1 ILORA_RUN_E2 ILORA_RUN_E3 ILORA_RUN_E4 ILORA_RUN_E1_EXPORT ILORA_RUN_E2_EXPORT ILORA_RUN_E4_EXPORT ILORA_RUN_CLUSTER \
  && python -I rebuild/tools/compare_runs.py --run E1="$ILORA_RUN_E1" --run E2="$ILORA_RUN_E2" --run E3="$ILORA_RUN_E3" \
  --run E4="$ILORA_RUN_E4" --export E1="$ILORA_RUN_E1_EXPORT" --export E2="$ILORA_RUN_E2_EXPORT" \
  --export E4="$ILORA_RUN_E4_EXPORT" --cluster "$ILORA_RUN_CLUSTER" \
  && CMP=$(latest compare) && done_ok "$CMP" compare && rec ILORA_RUN_COMPARE "$CMP"
```
- 산출물: 18.6.8. 확인: 값이 모두 실행 파일에서 왔고(출처 sha256 열), 없는 값은 NOT_RUN과 이유(`comparison.md` 끝). `compare error: ...`(종료 코드 1)는 실행 파일이 스스로 모순될 때만 나므로 그 파일을 고치지 않고 기록함. 빠진 실행은 오류가 아니라 NOT_RUN

**R15. 변형 (선택, 예산이 있을 때)**
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
- 변형 결과는 R14를 `--run <이름>=<폴더>`를 더해 다시 돌려 표에 넣음 (아래 블록, `ILORA_RUN_COMPARE_VARIANTS`). 돌리지 않은 변형은 NOT_RUN
- 예시 변형은 `all_hybrid` `[결정]`: `released_lr`(lr 8e-4)은 fork가 같은 A100·fp16 AMP에서 첫 validation 유효 예측 0, 둘째 epoch 내내 loss NaN을 기록함 (`F experiments/IMPLEMENTATION_NOTES.md` §11i). 고르기 전에 그 GPU 시간을 예산에 넣음. 같은 일이 생기면 GradScaler skip 16회 연속 뒤 `TrainingError`로 멈추고, bf16으로 다시 돌리지 않으며(D5와 D9가 섞임) 실패 폴더를 `--run V_released_lr="$(latest variant_released_lr)"`로 비교에 넘김 (그 칸 NOT_RUN과 `TrainingError` 이유)
```bash
need ILORA_RUN_E1 ILORA_RUN_E2 ILORA_RUN_E3 ILORA_RUN_E4 ILORA_RUN_E1_EXPORT ILORA_RUN_E2_EXPORT ILORA_RUN_E4_EXPORT ILORA_RUN_CLUSTER \
  "ILORA_RUN_V_$V" ILORA_RUN_V_user_level_cluster \
  && python -I rebuild/tools/compare_runs.py --run E1="$ILORA_RUN_E1" --run E2="$ILORA_RUN_E2" --run E3="$ILORA_RUN_E3" \
  --run E4="$ILORA_RUN_E4" --run V_$V="$(printenv ILORA_RUN_V_$V)" --run V_user_level_cluster="$ILORA_RUN_V_user_level_cluster" \
  --export E1="$ILORA_RUN_E1_EXPORT" --export E2="$ILORA_RUN_E2_EXPORT" --export E4="$ILORA_RUN_E4_EXPORT" \
  --cluster "$ILORA_RUN_CLUSTER" && CMP=$(latest compare) && done_ok "$CMP" compare && rec ILORA_RUN_COMPARE_VARIANTS "$CMP"
```

**R16. 보관 (Pause 전)**
```bash
tar -czf ~/rebuild_runs_$(date -u +%Y%m%dT%H%M%SZ).tgz runs/rebuild
```
- Pause·Stop은 container 파일 시스템을 지움 (`F experiments/VESSL_GUIDE.md:3-8`) → 내려받은 뒤에만 Pause. 실행 중에는 Pause하지 않음. 삭제 명령은 RUNBOOK에 두지 않음 (사용자 판단)

**bf16 대안 (R7의 실패 규칙에서만)** `[결정]`
- fp16 E2가 R7의 실패 규칙에 걸렸을 때만 씀 (D9, 18.6.4 `variant_bf16`, 편차로 기록). E2를 `variant_bf16.yaml`로 처음부터 다시 학습하고(`latest variant_bf16`), 짝 비교가 같은 조건이도록 E3·E4·E1과 R15 변형도 bf16(`--set model.torch_dtype=bfloat16 --set train.precision=bf16_amp`)으로 돌림. R8 재평가는 학습과 같은 설정이어야 하므로(M11 15.6.11) `variant_bf16.yaml`로 함. R10 export와 R11 군집은 precision과 무관해 본문 그대로 (export는 `data`·`sasrec`·`router`·`lora`·`cluster` 구역만 비교하고 LLaMA를 만들지 않음). 실패한 fp16 E2 폴더는 기록으로 남김
- M10-15의 fp16 항목(첫 step scale 65,536)은 bf16 실행에 해당하지 않음: `outputs e2_run`은 scale 1.0·skip 없음을 확인함. 비교 도구는 precision을 한계에 적음 (18.6.10 #9) `R RUNBOOK.md bf16 대안`

#### 18.6.7 결과 표 (`[SPEC §12]`, D17)와 논문 기준값
- 지표 정의 (D17): ValidRatio = 유효/전체, HR@1 = 정답/유효 (코드 `hr`), CombinedMetric = 정답/전체 = HR@1 × ValidRatio (코드 `metric`, upstream이 checkpoint 감시에 실제로 씀 `U main.py:23-32` → `[SPEC §12]`의 "실제 사용하는 경우"에 해당). 주 프로토콜 = test 95명 전부
- **표 A. 주 결과** (test, D11 (a) validation 최고 epoch, 전체 프로토콜 N = 95). 모든 값 TBD

| Method | Routing | ValidRatio | HR@1 | CombinedMetric | Seed | Epoch | num_moe | lora_r | TrainableParams | ElapsedTime | WallClock |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaRA baseline (E1) | none | TBD | TBD | TBD | 1234 | TBD | 1 | 8 | TBD (계산 기대 37,036,032) | TBD | TBD |
| iLoRA Dynamic (E2) | dynamic | TBD | TBD | TBD | 1234 | TBD | 4 | 8 | TBD (계산 기대 37,036,288) | TBD | TBD |
| iLoRA text-only eval (E3) | dynamic | TBD | TBD | TBD | 1234 | E2 (a) | 4 | 8 | E2와 같음 | TBD (평가 시간) | TBD |
| Cluster Hard (E4) | cluster_hard | TBD | TBD | TBD | 1234 | TBD | 4 | 8 | TBD (계산 기대 37,036,032) | TBD | TBD |

- **표 B.** 같은 형식, D11 (b) 마지막 epoch (4). **표 C.** 보조 프로토콜 `drop_last_bs4`(N = 92), `drop_last_bs8`(N = 88)의 세 지표 (E1, E2, E4). 모두 TBD
- 열 출처: 지표 = `eval/epoch{a}/test/metrics.json`, Epoch = `eval/selection.json`, TrainableParams = `records/train/run_summary.json`(그룹별 수). 기대값(계산)은 M9 13.8 #13·M10 14.6.3과 같고, 실행 기록과 다르면 그 차이를 적음 (계산값은 `model.backbone` llama2_7b에만 있음. 작은 무작위 모델은 계산값 없음으로 적음 `R tools/compare_runs.py trainable_check`)
- **ElapsedTime** `[결정]`: 학습 실행은 `records/train/run_summary.json`의 `train_summary.seconds.total` = 학습·validation·저장·test 시간 (14.6.7 `TrainSummary`). 끝난 epoch마다 checkpoint의 `val_metrics.json`에 남긴 시간과 test 시간의 합이라 재개를 거쳐도 이어지고, 실패와 `--resume` 사이 멈춘 시간(과 중단된 epoch에서 버려진 계산)은 들지 않음. 평가 실행(E3, 재개 없음)은 그 `meta.json`의 `elapsed_s` `R tools/compare_runs.py RunView.elapsed`
  - 이유: `meta.json`의 `elapsed_s`는 첫 `started_utc`부터 잼 (`R ilora/run.py finish_meta`, 재개도 `started_utc`를 그대로 둠 `R ilora/commands.py _reopen_for_resume`). 그래서 재개한 실행에서는 멈춘 시간까지 들어가 계산 시간과 다름. `[SPEC §12]`는 ElapsedTime을 "실제로 기록 가능할 경우"로만 두고 `[SPEC §0.14]`는 실제 측정값만 허용함: 두 값 모두 측정값
  - **WallClock**: `meta.json`의 `elapsed_s`(첫 시작부터 마지막 끝까지, 멈춘 시간 포함)와 재개 횟수(`resumes` 항목 수)를 따로 적음 `R tools/compare_runs.py RunView.wall_clock`
  - `meta.json`의 학습 segment 시간을 더하지 않음: 강제 종료된 프로세스의 segment는 끝 시각이 측정되지 않아 `ended_utc`가 null `R ilora/train.py segment_problems`
- **표 D. 논문 기준값 (우리 결과 아님)**

| 출처 | 행 | ValidRatio | HitRatio@1 | 정수 역산 (N, 유효, 정답) | HitRatio@1의 분모 | 우리 쪽 대조 값 |
|---|---|---|---|---|---|---|
| `[LLaRA Table 2 p.8]` | LLaRA (SASRec), MovieLens | 0.9684 | 0.4421 | (95, 92, 42) | 정답/전체 | E1 전체 프로토콜: `valid_ratio` ↔ 0.9684, `combined_metric` ↔ 0.4421 |
| `[논문 p.7 Table 1]` | iLoRA, MovieLens | 0.9891 | 0.5275 | (92, 91, 48) | 정답/유효 | E2 `drop_last_bs4`: `valid_ratio` ↔ 0.9891, `hr_at_1` ↔ 0.5275 |
| `[논문 p.7 Table 1]` | LLaRA, MovieLens | 0.9684 | 0.4421 | LLaRA Table 2와 같은 값 (옮겨 온 값, 18.3) | – | – |

  - 역산: N 80–95, 네 자리 반올림에서 각 행의 정수해는 위 하나뿐 (계산 `design/tmp/m1314/backsolve.py`, D17과 같음). iLoRA 행은 N = 92(batch 4 + drop_last, M11 15.6.6), LLaRA 행은 N = 95
  - 논문 Imp.% 19.32% = 0.5275/0.4421 − 1은 분모·N이 다른 두 수의 비. 우리는 같은 프로토콜 값끼리만 비교함
  - 대조 판정 기준은 두지 않음 `[결정]`: 단일 seed(D2), 표본 하나 = 1/95 ≈ 0.0105, 학습 GPU 비결정성(D16), 기준값의 실행 조건 차이(18.6.10) 때문에 "재현 성공" 경계를 근거 있게 정할 수 없음. 차이를 정답 수(표본 단위)로 적고 Wilson 구간은 M11 `supplementary`에만 둠
- **표 E. 우리 실행끼리 짝 비교** (같은 프로토콜, test 같은 95표본·같은 후보·템플릿: D10·D15, M2 6.6.6. best_val checkpoint(D11 (a))의 전체 프로토콜 `records.jsonl`이고, 두 기록의 표본·순서·후보·템플릿·정답이 같은지 먼저 확인 `[결정]`. B의 CombinedMetric이 0이면 비는 NOT_RUN): E2 vs E1, E4 vs E2, E3 vs E2의 정답 수 차, 한쪽만 맞힌 표본 수(불일치 쌍), CombinedMetric 비 − 1 (Imp.과 같은 꼴). 검정·판정 없이 수만 적음 `[결정]`. 모두 TBD
- **표 F. gate·군집 요약** (값 TBD): E2 test `gate_summary.json`의 expert 평균, argmax 분포, entropy·JS 통계 (M12 16.6.6), E4 평가 gate의 argmax 분포(one-hot 확인), 군집 크기·silhouette·NMI (M15 19.6.6)

#### 18.6.8 비교 도구 `rebuild/tools/compare_runs.py` `[결정]`
- 명령: `python -I rebuild/tools/compare_runs.py --run <이름>=<실행 폴더> … [--export <이름>=<폴더> …] [--cluster <폴더>] [--root <폴더>]`. M0 `create_run_dir("compare")`로 `<root>/compare_<UTC>/`(기본 `runs/rebuild/compare_<UTC>/`)를 만들고 M0 `start_meta`·`tracked_run`으로 `meta.json` 상태를 남김 (R14의 `done_ok` 확인)
  - `--root` `[결정]`: 시험이 비교 폴더를 tmp에 써야 함 (17.8 #4). 실행 폴더 값은 절대 경로 또는 저장소 루트 기준(M0), 빈 값은 주지 않은 것. 종료 코드: 인자 오류(`--run` 없음, 이름 중복·형식) 2, `CompareError` 1, 그 밖 0이고 새 폴더 경로를 출력. `meta.json`: `command` `compare`, `compare_inputs`, `compare`(출력 파일 sha256, NOT_RUN 이유), seed null, `config_hash` = 입력 목록의 sha256. Python API `run_compare(runs, exports=None, cluster=None, *, root, repo_root, argv) -> Path`, `build_report`, `build_parser`, `main`, `CompareError` `R tools/compare_runs.py run_compare`
- 읽는 파일만으로 표를 만듦: 학습 실행은 `config.yaml`, `meta.json`, `eval/selection.json`, `eval/epoch{a,b}/test/metrics.json`, `records/train/run_summary.json`, `eval/epoch{a}/test/records.jsonl`(짝 비교). eval 실행(E3)은 `eval/summary.json`과 원 실행의 `selection.json`, 자기 `eval/epoch{a,b}/test/{metrics.json, records.jsonl}`(값과 출처, `summary.json`의 `metrics.json` sha256과 대조)와 `meta.json`(ElapsedTime), TrainableParams는 원 학습 실행의 `run_summary.json`. 표 F와 18.6.9를 위해 export 실행의 `records/analysis/epoch{a}/test/gate_summary.json`·`gate_interpretation.txt`, 군집 실행의 `cluster/cluster_summary.json`·`manifest.json`(`kmeans_file`)
  - 실행마다 `meta.json` `command`가 역할과 맞아야 함 (`--run`은 train 또는 eval, `--export`는 export, `--cluster`는 cluster). E2 export는 그 `meta.json` `source_run`이 `--run E2`의 폴더일 때만 읽음 (아니면 NOT_RUN과 이유) `[결정]`: 다른 학습 실행의 export를 섞지 않음
- 출력
  - `comparison.csv`: 열 `method, run, routing_mode, num_moe, lora_r, seed, checkpoint, epoch, protocol, n_total, n_valid, n_correct, valid_ratio, hr_at_1, combined_metric, trainable_params, elapsed_s, wall_clock_s, resumes, source_path, source_sha256`, 행 = (실험, checkpoint ∈ {best_val, last}, 프로토콜 ∈ {full, drop_last_bs4, drop_last_bs8})
    - `elapsed_s` = 18.6.7의 ElapsedTime (학습 실행 `train_summary.seconds.total`, E3 `meta.json` `elapsed_s`), `wall_clock_s` = `meta.json`의 `elapsed_s`, `resumes` = `meta.json` `resumes` 항목 수 `R tools/compare_runs.py comparison_row`
    - 행은 E1–E4가 늘 먼저(주지 않으면 NOT_RUN), 그 뒤 다른 `--run` 이름(변형). 칸은 값 또는 정확히 `NOT_RUN` `[결정]`: 열 목록에 이유 열이 없음. 이유는 `comparison.md`의 NOT_RUN reasons, `summary.md`, `meta.json` `compare.not_run_reasons`. 실수는 `repr`로 적어 정확히 왕복
  - `comparison.md`: 표 A–F (표 A·B에 ElapsedTime과 WallClock(재개 횟수와 함께) 열), 정의(D17, ElapsedTime·WallClock 뜻), 표 D(논문 값은 이 표에만), 한계 요약(18.6.10). 절은 `## Runs`, `## Definitions (D17)`, `## Table A.`–`## Table F.`, `## NOT_RUN reasons`, `## Limitations (DESIGN 18.6.10)`. 표 A 아래에 실행마다 TrainableParams와 계산값의 차이(18.6.7), 표 C는 E3 밖. 비교한 실행의 precision이 `float16`·`fp16_amp`가 아니면 한계에 precision 항목을 더함 (18.6.10 #9). `comparison.md`·`summary.md`는 `[SPEC §19]` 절 이름을 따라 영어 `[결정]`
  - `spec20_index.json`: 18.6.9 대응 (파일 경로와 sha256, 없으면 NOT_RUN)
  - `summary.md`: `[SPEC §19]` 구성(설정, E2 재현, gate 다양성, 군집, cluster-hard, 비교, 한계)을 파일 값으로만 채움. 설정 표와 실험별 줄에 ElapsedTime·벽시계 시간·재개 횟수
- 규칙: 없는 파일·키는 그 칸을 `NOT_RUN`으로 (추정·0 채움 없음). 논문 숫자를 측정 열에 쓰지 않음. 실행 `meta.json` 상태가 `completed`가 아니면 그 실행의 모든 칸 `NOT_RUN`과 이유. 출력 숫자는 `metrics.json` 값을 그대로 (다시 계산한 값과 1e-12 안에서 같은지 확인) `[SPEC §0.11, §19]`
- 스스로 모순인 실행 파일은 NOT_RUN 칸이 아니라 `CompareError`로 비교 전체를 멈춤 (`meta.json` `failed`) `[결정]`: 경고하고 계속하지 않음 (17.5). 해당: `metrics.json`의 비가 개수와 1e-12 넘게 다름, `metrics.json`의 split·epoch·`ckpt_tag`가 `selection.json`과 다름, `records.jsonl`의 행·정답 수가 `metrics.json`과 다름, eval `summary.json`의 `metrics.json` sha256이 파일과 다름, JSON·기록 형식 오류. 파일·키가 없는 것은 NOT_RUN `R tools/compare_runs.py read_metrics`

#### 18.6.9 `[SPEC §20]` 산출물 대응 (재구현은 저장소 `results/`에 쓰지 않음)
| SPEC §20 파일 | 재구현 파일 (a = E2의 `best_val_epoch`, a4 = E4의 것) |
|---|---|
| `results/baseline/metrics.json` | `$E2/eval/epoch{a}/test/metrics.json` |
| `results/baseline/gates_test.csv` | `$E2X/records/export/epoch{a}/gates_test.csv` |
| `results/baseline/gate_summary.json` | `$E2X/records/analysis/epoch{a}/test/gate_summary.json` |
| `results/baseline/figures/{gate_heatmap, expert_mean_utilization, argmax_expert_distribution, entropy_distribution}.png` | `$E2X/records/analysis/epoch{a}/test/figures/` 같은 이름 (M12 16.6.7) |
| `artifacts/kmeans_k4_seed1234.joblib` | `$CL/cluster/` 아래 군집 `manifest.json`의 `kmeans_file` (paper_spec이면 `kmeans_k4_seed1234.joblib`. 이름은 `cluster.n_clusters`·`random_state`를 따름) |
| `results/clustering/cluster_summary.json` | `$CL/cluster/cluster_summary.json` |
| `results/clustering/figures/{sasrec_pca_clusters, cluster_dynamic_gate_heatmap}.png` | `$CL/cluster/figures/` 같은 이름 |
| `results/cluster_hard/metrics.json` | `$E4/eval/epoch{a4}/test/metrics.json` |
| `results/final_comparison.{csv,md}` | `$CMP/comparison.{csv,md}` |
| `results/today_summary.md` | `$CMP/summary.md` |

- 표의 `$E2`·`$E2X`·`$CL`·`$E4`·`$CMP`와 a는 RUNBOOK이 `runbook_paths.sh`에 기록한 `ILORA_RUN_E2`·`ILORA_RUN_E2_EXPORT`·`ILORA_RUN_CLUSTER`·`ILORA_RUN_E4`·`ILORA_RUN_COMPARE`와 `ILORA_E2_BEST`의 값. 짧은 셸 변수는 한 줄 안의 임시 이름이라 새 셸에는 남지 않음 (18.6.6)
- `spec20_index.json`: `created_utc`, `rule`, `epochs`(`E2_best_val`, `E4_best_val`), `files`(SPEC 이름 → `path`, `sha256`, 없으면 둘 다 NOT_RUN과 `reason`). 그림은 SPEC 이름대로 `.png` `R tools/compare_runs.py write_outputs`

#### 18.6.10 한계 (`[SPEC §13]`, 결과 해석 때 함께 적음)
1. **활성 용량 (E4):** r = 8, K = 4라 expert 하나는 rank r* = 2 `[논문 p.5 식 9]`. one-hot ω에서는 층마다 rank-2 expert 하나만 쓰이고, E2의 soft ω는 네 expert(합 rank 8)를 섞음. E4와 E2의 차이를 routing 방식과 활성 용량으로 나눌 수 없음 `[SPEC §13]`. 주 실험에서 r·K를 바꾸지 않음. 후속 후보(expert당 full rank, top-k, 활성 rank 맞춤)는 이 계획에 프로파일을 두지 않음 (NOT_RUN)
2. **D8 scaling:** 주 설정 α/r* = 16 (expert_rank). one-hot에서 활성 expert의 갱신은 16·B_k A_k, 균등 soft ω(각 0.25)에서 expert 하나의 갱신은 4·B_k A_k라 활성 expert 하나의 무게가 4배 (M15 19.3). 균등 ω에서 E2의 전체 갱신은 같은 A, B의 표준 LoRA(α/r = 4)와 정확히 같고(20.1), total_rank 해석에서는 그 1/4. E1·E2·E4 비교의 결론이 D8에 기대므로 `variant_total_rank_scaling` 결과가 없으면 그 사실을 함께 적음
3. **seed 하나, 작은 N:** D2. test 95 → 표본 하나 ≈ 0.0105. 학습 재실행 편차(D16)는 측정하지 않음 (NOT_RUN)
4. **제목 판정 결함 (D20):** 고정 후보에서 정답 출력으로도 맞출 수 없는 표본 val 1, test 2 → CombinedMetric 상한 93/94, 93/95 (M2 6.6.6 계산). flag 없는 표본 지표는 보조로만
5. **논문 표의 프로토콜:** iLoRA 행은 N = 92·정답/유효, LLaRA 행은 N = 95·정답/전체 (D17), LLaRA 행은 iLoRA가 다시 돌린 값이 아님 `[논문 p.6 §4]`
6. **E1 ≠ LLaRA 코드 실행:** E1은 이 재구현(fp16 base 로드 D9, Lightning 없음, max_steps 절단 없음(D24·14.5: `L main.py:49`·`U main.py:50`의 max_steps 식으로 학습이 끝나기 전에 멈추는 것을 따르지 않음. 부분 창을 `grad_accum`으로 나누는 것은 Lightning과 같음), 저장 필터·CNN router 없음 D26)의 K = 1이라, LLaRA Table 2와의 차이를 K 하나로 설명하지 않음
7. **군집 (E4):** 표본(sequence) 단위 KMeans(D25), 군집 번호에는 의미가 없음 `[SPEC §9]`. 사용자 단위는 변형
8. **E3:** 평가 prompt만 바꿈. 모델은 curriculum(D3)으로 text-only도 학습했으므로 E3은 "text-only로 학습한 모델"의 결과가 아님
9. **precision (D9):** 비교한 실행 중 `model.torch_dtype`·`train.precision`이 주 설정(`float16`·`fp16_amp`)이 아닌 것이 있으면 적음. 모두 같으면 기록된 편차(`variant_bf16`, CPU smoke), 서로 다르면 짝 차이에 precision 차이도 섞임. 비교 도구가 각 실행의 `config.yaml`에서 읽어 필요할 때만 한계에 더함 `R tools/compare_runs.py precision_note`

#### 18.6.11 시간·비용·디스크
- 시간: 측정 전 TBD. 실행마다 `meta.json`(`started_utc`, `ended_utc`, `elapsed_s`)과 `epochs.csv`(`train_s`, `val_s`, `save_s`)로 기록 `[SPEC §0.14]`
- 비용: VESSL 청구 정보가 없으면 적지 않음 (추정값을 결과처럼 쓰지 않음) `[SPEC §0.14]`
- 디스크 (M10 14.6.6의 epoch당 checkpoint 약 424 MiB로 계산): 학습 실행 하나(5 epoch) 약 2.07 GiB → E1·E2·E4 약 6.2 GiB, `overfit_gpu` 25 epoch 약 10.35 GiB, `smoke_gpu_100` 4 epoch분 약 1.66 GiB, 변형 하나 약 2.07 GiB. 평가·export·군집 산출물은 이보다 작음 (크기 측정은 TBD). fork 안내의 여유 약 100 GB(`F experiments/VESSL_GUIDE.md:17`) 안
- GPU 메모리: smoke 2단계와 E2에서 측정 (M8-12, M9-17, M10-15), TBD

#### 18.6.12 다른 모듈과의 연결, 계약 추가 제안, 발견한 불일치
- **연결:** 모든 모듈 설정 표(키 이름·라벨의 원본), M0 명령과 실행 폴더 규칙, M10 재개, M11 eval 명령·`metrics.json`·`summary.json`, M12 export·분석 파일, M15 군집 산출물, M13 smoke·시험 명령
- **계약 추가 제안**
  1. M0 4.6.2: `run` 구역 키는 `seed` 하나 (지금 4장에 표가 없음)
  2. M0 4.6.2 프로파일 목록 확정: `paper_spec`, `llara_baseline`, `cluster_hard`, `variant_{all_hybrid, reverse_curriculum, text_only_eval, released_lr, released_gate_lr, released_label, total_rank_scaling, fixed_template, fixed_titles, user_level_cluster, bf16}`, `smoke_cpu`, `smoke_gpu_100`, `overfit_gpu`
  3. M0: 경로 값 키(`sasrec.source_path`·`state_path`·`reference_path`, `prompt.template_path`, `eval.source_run`, `records.source_run`, `cluster.artifact_dir`, `cluster.dynamic_export_dir`)는 절대 경로 또는 저장소 루트(`run.REPO_ROOT`) 기준 상대 경로로, 한 함수에서 해석
  4. `rebuild/tools/compare_runs.py` (M14 소유, 18.6.8)
  5. VESSL env `ilora-rebuild` (R1)와 check-env의 transformers 파일 무결성 검사 (M13 17.6.12 제안 4)
  6. (채택됨) `data.train_rows` (M1 5.6.7·5.6.8): `overfit_gpu`에 필요
- **발견한 불일치**
  1. **D13 env 재사용:** D13의 "Python 3.10 (현재 VESSL `ilora` 환경 재사용)"을 그대로 따르면 `F setup_vessl.sh:87-106`이 교체한 transformers 파일로 실행되어 D13의 "transformers 내부 파일 교체 제거"와 충돌. 해결: Python 3.10·같은 requirements의 새 env (R1). D13의 판 결정은 그대로
  2. **M11 재현 점검과 E3:** M11 15.6.10은 원 실행 기록과의 같음을 `eval.batch_size`가 같을 때 요구하므로 E3(`prompt.eval_prompt` 다름)는 반드시 실패. 요구 조건에 평가 결과를 바꾸는 허용 차이 키(`prompt.eval_prompt`)도 같을 것을 넣고, 아니면 "비교 대상 아님"으로 기록해야 함 (R9, M13 17.6.12 불일치 3) → (반영됨) M11 15.6.10의 `not_comparable`

### 18.7 결정
- 관련 확정 결정: D2(seed 하나), D11(선택 규칙과 (a)·(b) 보고), D17(지표·보조 프로토콜), D26(값 수준 변형만), 변형 열의 D3·D4·D5·D8·D9·D15·D18·D19·D20·D25
- 이 장의 새 `[결정]`
  - E1·E4·변형은 `extends` overlay로 다른 키만: "나머지는 E2와 같음"을 파일 구조로 보장
  - `cluster.artifact_dir`는 명령행으로만: UTC 경로를 미리 알 수 없고, 빠지면 적재 단계에서 거부
  - `run` 구역은 `seed` 하나
  - VESSL은 새 env `ilora-rebuild` (같은 requirements, 패치 없음) + pytest 9.2.0
  - 실행 순서: 시험 → check-env → smoke 2단계 → 과적합 → E2 → 재평가 2회 → E3 → E2 export → 군집 → E4 → E1 → 비교 → 변형(선택). E2를 먼저 하는 이유: 주 재현이고 E4가 E2 export와 군집에 의존
  - 경로는 그 폴더를 만든 명령 뒤에 `&&`로 이어 `latest`로 잡고, `done_ok`가 `meta.json`의 상태와 `command`를 확인한 뒤 `runbook_paths.sh`에 덧붙임. R8 이후 명령은 기록된 `ILORA_*` 이름만 읽고 `need`로 먼저 확인
  - 재개는 학습 프로세스가 끝났을 때만 (tmux 재접속, `pgrep`·`nvidia-smi` 확인, 살아 있는 프로세스는 `run.lock`이 막음)
  - bf16 대안은 E2·E3·E4·E1·변형을 모두 같은 precision으로. R15 예시 변형은 `all_hybrid`
  - 비교 도구: `--root`, 스스로 모순인 실행 파일은 `CompareError`, NOT_RUN 이유는 CSV 밖에
  - 결과는 비교 도구가 파일에서만 만듦, 논문 값은 기준값 표에만, 논문과의 대조는 같은 프로토콜 열끼리 (E1: 전체 N = 95의 CombinedMetric, E2: `drop_last_bs4`의 HR@1), 판정 경계 없음
  - 우리 실행끼리의 짝 비교는 정답 수·불일치 쌍 수만 (검정 없음)
  - 저장소 `results/`에 쓰지 않고 SPEC §20 색인을 만듦
  - YAML 표기 규칙 (지수 실수, mapping 따옴표)
  - `variant_bf16`은 fp16 실패 때만 씀

### 18.8 완료 기준
1. (CPU) `test_m14_profiles.py::test_profile_loads[<17개>]`: 모든 프로파일이 M0 `load_config`로 적재됨 (`cluster_hard`·`variant_user_level_cluster`는 `--set cluster.artifact_dir=<임의 경로>`와 함께, 없으면 `ConfigError`). `::test_profile_files`: 파일 17개, 주석까지 ASCII만, `paper_spec`이 87개 키를 모두 적음 (18.6.3)
2. (CPU) `::test_resolved_differences`: 해석된 설정을 `paper_spec`과 비교한 차이 키 집합이 정확히 18.6.4–18.6.5 표와 같음 (E1 = {`lora.num_experts`, `router.routing_mode`}, E4 = {`router.routing_mode`, `cluster.artifact_dir`}, 변형마다 표의 키)
3. (CPU) `::test_paper_spec_values_and_labels`: `paper_spec` 값이 18.6.2 표(시험 안 고정 dict)와 같고, `sections.py`의 각 키 라벨이 표의 라벨과 같음 (문서 ↔ 코드)
4. (CPU) `test_m14_runbook.py::test_commands_are_valid`: `RUNBOOK.md`의 bash 블록을 `shlex`로 나눠, 모든 `rebuild/main.py` 호출의 명령이 6개 중 하나이고 `--config` 파일이 있으며, 모든 `--set` 키가 schema에 있고 값이 그 타입으로 적재됨. `latest` 인자가 프로파일 이름(또는 `compare`)이고 같은 블록에서 폴더를 만든 직전 명령의 프로파일과 같음. 같은 번호의 시험이 RUNBOOK 규칙을 더 확인함: `outputs` 선택어가 `test_run_outputs.py`의 시험을 빠짐없이 고름(`::test_output_selectors`), 비밀값 없음, 새 셸에서도 경로가 돌아옴, `latest`가 폴더를 만든 명령과 `&&`로 이어지고 `done_ok`가 그 명령을 확인, R7 재개 조건, R1 `LLM_PATH`, R3 skip 문구, bf16 대안, R15 예시 변형
5. (CPU) `::test_derived_lengths`: 18.6.5 표의 계산값을 M1·M4·M10 함수로 다시 계산해 같음 (smoke_cpu 156·39·3·10·20·5·78·10/28, smoke_gpu_100 7,230·903·7·57·114·5·1,806·218/670, overfit 16·4·0·4·100·5. 경우 8개: 18.6.5 표 6행과 17.6.12의 VESSL 작은 학습 2개(M10-16: 132·16·0·8·8·2·16, M10-19: 132·16·0·8·16·4·32), E4 조회 수 7,224·330,600)
6. (CPU) `::test_overfit_rows`: `overfit_gpu`의 16행이 사용자 0–15의 마지막 train window(len_seq 10)이고 정답이 서로 다르며 item 266이 없음
7. (CPU) `test_m14_compare.py::test_compare_from_cpu_smoke`: M13 `cpu_smoke_run`의 S1(E2 경로)·S5(E4)·S6(E3)·S7(E1)을 넣은 비교 결과에서 값이 각 `metrics.json`과 같고, 빠진 실행을 넣으면 그 칸이 NOT_RUN, 표 D 밖에 논문 숫자(0.9684, 0.4421, 0.9891, 0.5275)가 없음, `spec20_index.json`의 파일 sha256이 맞음
8. (VESSL_REQUIRED) RUNBOOK R1–R14가 순서대로 끝나고 각 단계의 `meta.json` 상태 `completed`, 해당 `test_run_outputs.py` 시험 통과 (E1은 이 장의 `vessl/test_run_outputs.py::test_e1_run_outputs`: 그룹·`gate_0`·기록 수·export), `comparison.md`의 측정 칸이 파일 값으로 채워짐 (값은 측정 전 TBD)
9. (VESSL_REQUIRED) `vessl/test_vessl_env.py::test_check_env_passes`: R1 env에서 R4 check-env 통과 (transformers 파일 무결성 포함), `pip freeze` 결과를 실행 기록과 함께 보관

---

## 19. M15 SASRec 군집과 cluster-hard 배정
상태: 테스트 통과(로컬)

### 19.1 목적과 범위
- **범위**
  - E4 cluster-hard용 고정 군집: train 표본의 `z`로 KMeans fit (D25), validation·test는 predict만 `[SPEC §0.8, §7]`
  - 산출물: joblib 모델, 중심 npz, split별 배정 CSV, `cluster_summary.json`, 그림 2개 `[SPEC §7, §8]`
  - cluster_hard 학습·평가에서 `sample_id` → one-hot `ω` 공급 (M7 경유) `[SPEC §9, §11]`
  - cluster_hard 학습 시작 전 산출물·표현 일관성 검사
  - 변형 `variant_user_level_cluster` (pseudo 사용자 단위)
- **범위 밖:** `z` 계산 함수(M3), routing 분기·gate parameter 제외·호출 계수(M7, M10), `ω` 소비(M8), 그림 스타일(M12 `analysis.py`), 비교 표(M14)

### 19.2 입력과 출력
**입력**
- `Config.cluster`(19.6.10), `Config.records`의 export 설정(z를 M12와 같은 방법으로 계산)
- M1 `Sample` 세 split (`sample_id`, `user_id`, `row`, `history`)
- M3 SASRec 가중치(변환본) → `z [N, 64] float32` (M12 `compute_representations`, CPU)
- 선택: E2 export 폴더(`cluster.dynamic_export_dir`)의 NPZ (`z`, dynamic `ω`, D11 주 결과 checkpoint)

**출력** (군집 실행 폴더의 `cluster/`, 명령 `cluster`)

| 파일 | 내용 |
|---|---|
| `representations_{train,validation,test}.npz` | z만 (16.2 형식) |
| `kmeans_k4_seed1234.joblib` | 학습된 KMeans, 이름 `kmeans_k{C}_seed{random_state}.joblib` `[SPEC §7]` |
| `centroids.npz` | `cluster_centers` float32 [C, d], `inertia`, `n_iter`, `fit_ids_sha256` (pickle 없는 사본) |
| `{train,validation,test}_assignments.csv` | `sample_id, split, row, user_id, cluster_id, sq_dist_1, sq_dist_2` `[SPEC §7]`. `split` 값은 계약 값(`train`/`val`/`test`), 파일 이름은 SPEC 철자 (M12 16.2 `[결정]`과 같음) |
| `train_user_assignments.csv` | user 단위만: `user_id, fit_sample_id, cluster_id` |
| `cluster_summary.json` | 19.6.6 `[SPEC §7]` |
| `figures/sasrec_pca_clusters.png`, `figures/cluster_dynamic_gate_heatmap.png` | `[SPEC §8]`. `records.figure_formats`의 형식마다 (paper_spec은 png만), heatmap은 dynamic export가 있을 때만 |
| `manifest.json` | 산출물·입력 sha256, 설정, 버전 |

- 학습·평가 중 출력: `ClusterHardGate(sample_ids, device)` → `ω [B, K] float32` one-hot, 행 합 1 (3.1 `ω`)
- 배정 CSV는 mapping과 무관한 군집 id만 담음. 군집 → expert 대응은 학습 설정에서 적용 → 같은 산출물로 mapping 변형을 돌릴 수 있음 `[결정]`

### 19.3 논문 근거
- "z = SR-EMBθ(i<n). (10)" `[논문 p.5 §3.2.2 식 10]`, "Here z ∈Rd provides a holistic view of the user’s behavioral patterns and preferences." `[논문 p.6 §3.2.2]`
- SR-EMB는 고정 `[논문 p.4 그림 2 (암시)]`, "The sequential recommender is well-trained and frozen" `[LLaRA 그림 2 caption p.4]` → z는 학습 중 변하지 않음 (D12, D23). SPEC §10의 A(pretrained + frozen)에 해당
- SASRec 표현을 유클리드 거리로 나누는 것은 논문 RQ1 분석과 같은 계열: "The recommendation sequences are divided into multiple tasks using representations derived from the sequential recommendation model SASRec, with a dimension of 64." "We clustered these sequences into 8 sub-datasets using Euclidean distance." `[논문 App C p.17]` (Steam, 8개, 알고리즘 이름 없음 → `(암시)`)
- 논문 routing은 dense soft (식 11-12). "we do not extensively investigate the effects of using hard routing for recommendations with a large number of experts." `[논문 p.9 §6]` → E4는 논문 밖 연구 실험 `[SPEC §9]`
- K = 4 `[논문 p.9 §4.3.2]`, r* = r/K = 2 `[논문 p.5 식 9]` → one-hot이면 rank-2 expert 하나만 활성. D8 scaling 16에서 활성 expert의 기여는 균등 soft gate일 때 expert 하나 기여(ω = 1/4)의 4배, 활성 rank는 8 → 2 `(암시)` → 성능 차이를 routing 전략과 활성 용량으로 나눌 수 없다는 한계 `[SPEC §13]`

### 19.4 공개 코드에서 가져올 것
- 공개 코드에는 군집과 hard routing이 없음
- z의 정의(마지막 실제 위치 hidden, `ln_3` 뒤) `U recommender/A_SASRec_final_bce_llm.py:202-217` (M3)
- 같은 표현을 사용자 표현으로 부름: `encode_users` → `cacul_h` `U model/model_interface.py:422-429`, `user_embeds` `:451` (D25 근거)
- 라이브러리 버전: scikit-learn 1.2.2 `U requirements.txt:112`, joblib 1.2.0 `:49`, threadpoolctl 3.1.0 `:127`, matplotlib 3.7.1 `:58`, scipy 1.10.1 `:113`
- 참고(근거 아님): fork의 fit·배정·요약 구성 `F scripts/fit_kmeans.py`, `F analysis/cluster_sasrec.py`

### 19.5 버릴 것 (알려진 문제)
- `U` iLoRA판 `train_data.df`(blob `73b47263`)의 `cluster` 열: 행 단위 0–3(14,276 / 19,902 / 9,609 / 24,601), 사용자 블록 754개 중 230개만 값이 일정, upstream `.py` 어디서도 읽지 않음, 출처 미상 → 읽지 않음 `[SPEC §0.8]`. 참고로 len_seq < 3인 2,262행은 모두 3 (재계산, `design/tmp/cluster_column.py`)
- `F`: 기본 thread 수로 fit, algorithm 미지정 `F scripts/fit_kmeans.py:60` → thread 수에 따라 결과가 달라짐 (합성 데이터에서 1 thread와 14 thread의 label이 다르고 중심 최대 차 1.05e-3, `design/tmp/kmeans_probe.py`)
- `F`: sample_id = 필터 뒤 위치 번호 `F scripts/fit_kmeans.py:41`, `F experiments/IMPLEMENTATION_NOTES.md` §1 → 원래 행 번호 기반 `split:row` (3.1)
- `F`: silhouette를 5,000개로 추출하고 shape가 안 맞으면 조용히 None `F analysis/cluster_sasrec.py:66-69, :87-93`, 군집 gate 평균의 빈 값을 0으로 채움 `:114`, inner join으로 표본이 조용히 빠질 수 있음 `:111-112`, joblib을 glob 첫 파일로 고름 `:53-57` → 모두 명시적 검사와 오류로
- `U`: 학습 중 Lightning이 SASRec dropout을 켜서 gate 입력과 eval 추출 표현이 달랐음 → D12로 해소 (eval 고정)

### 19.6 설계

#### 19.6.1 공개 API (`rebuild/ilora/cluster.py`)
```python
@dataclass(frozen=True)
class ClusterArtifacts:
    dir: Path; unit: Literal["sequence", "user"]; n_clusters: int
    centers: np.ndarray                    # float32 [C, d]
    assignments: Mapping[str, int]         # sample_id -> cluster_id (세 split 전부)
    manifest: Mapping[str, object]

def parse_expert_mapping(text: str, *, n_clusters: int, num_experts: int) -> tuple[int, ...]
def build_fit_set(train: RepresentationTable, samples: Sequence[Sample], *, unit: str
        ) -> tuple[np.ndarray, tuple[str, ...]]           # (z_fit float32 [n, d], fit sample_id)
def fit_kmeans(z_fit: np.ndarray, ccfg: ClusterConfig) -> KMeans
def predict(km: KMeans, z: np.ndarray) -> np.ndarray      # float32 C-contiguous만, int64 [N]
def run_cluster(cfg: Config, run_dir: Path, *, repo_root: Path) -> None   # main.py cluster
def load_cluster_artifacts(artifact_dir: Path, *, expected_ids: Sequence[str], sasrec_sha256: str,
        data_sha256: Mapping[str, str], ccfg: ClusterConfig) -> ClusterArtifacts
class ClusterHardGate:
    def __init__(self, artifacts: ClusterArtifacts, mapping: tuple[int, ...], num_experts: int) -> None
    def __call__(self, sample_ids: Sequence[str], device: torch.device) -> Tensor   # [B, K] float32
    lookups: int                                                                    # 누적 조회 표본 수
def verify_live_encoder(artifacts: ClusterArtifacts, encoder: SASRecEncoder, samples: Sequence[Sample],
        *, device: torch.device, n: int, atol: float) -> dict
```
- `SASRecEncoder`는 M3 7.6.3, `RepresentationTable`·`compute_representations`는 M12 16.6.1, `ClusterConfig`는 M0 `Config.cluster` 구역
- `verify_live_encoder`의 저장 z는 `artifacts.dir / "representations_train.npz"`(manifest sha256으로 확인, M12 `load_npz`)에서 현재 실행의 train 행 순서 첫 `n`개 표본과 같은 sample_id의 행 `[결정]`: 같은 표본으로 만든 산출물이면 앞 `n`행과 같고, full 산출물을 smoke 실행에 쓰면(19.6.7) 앞 `n`행이 아님. 결과의 `npz_prefix`가 앞 `n`행이었는지 적음
- 구현이 더한 공개 이름 `[결정]`: 오류는 모두 `ClusterError`(`ValueError`의 하위. 모르는 sample_id만 `KeyError`). `ClusterHardGate`는 `artifacts`, `mapping`, `num_experts`, `unused_experts`(어느 군집도 가리키지 않는 expert)도 가짐. `verify_live_encoder`는 `passed`, `n`, `max_abs_diff`, `atol`, `device`, `train_npz`, `train_npz_sha256`, `npz_prefix`, `first_sample_id`, `last_sample_id`, `sample_ids_sha256`, `state_digest`, `seconds`를 돌려줌. `write_cluster_artifacts(cfg, out_dir, tables, samples, *, z_dim, sasrec_info, data_sha256, repo_root, timings=None)`는 z 뒤의 전부(합성 z 시험용). M13 smoke가 쓰는 도우미: `kmeans_file_name`, `representation_file`, `assignment_file`, `read_assignments`, `read_user_assignments`, `load_kmeans`, `ids_sha256`, 파일 이름 상수 `R ilora/cluster.py`

#### 19.6.2 z 준비
- M15가 z를 직접 계산: `records.compute_representations(gate_fn=None)`, CPU fp32, `records.export_num_threads`(1), `records.export_batch_size`(1024) — M12 export와 같은 함수·설정 `[결정]`: z는 학습과 무관해 E2를 기다리지 않고 군집을 만들 수 있음
- z는 고정 SASRec(eval, fp32, autocast 밖)의 결정적 함수 (D12, D23) → fit 시점 표현과 학습 중 표현이 같음 → 배정을 미리 계산해 sample_id로 조회 `[SPEC §10]`
- 표현 크기 먼저 확인 `[SPEC §7]`: train z의 차원별 평균·표준편차(ddof 0, 각각 최소·최대와 차원별 값. 최대/최소 비는 표준편차에만 `[결정]`: 평균은 음수·0 근처일 수 있어 비가 뜻이 없음), 행 L2 norm 평균·표준편차·최소·최대를 `z_scale`에 기록 `R ilora/cluster.py z_scale`. 값과 무관하게 주 실험은 raw z (StandardScaler·PCA 없음) `[SPEC §7]`. scaling은 후속 변형으로만
- `cluster.dynamic_export_dir`가 있으면 E2 export NPZ의 z와 sample_id별로 비교 (max |Δz| ≤ `z_crosscheck_atol`) → KMeans 입력 = dynamic gate 입력 확인 `[SPEC §10]`

#### 19.6.3 fit 집합과 누수 검사 `[SPEC §0.8, §7]`
- `unit = sequence`(D25): train 표본 전부, M1 행 순서 (full 데이터 66,126행). 이 중 25개는 앞 표본과 (이력 아이템, len_seq)가 같아 z가 같음 → 중복점, 막지 않음 (`design/tmp/windows_per_user.py`)
- `unit = user`: 19.6.8
- 검사 (하나라도 어긋나면 실패)
  - fit sample_id가 모두 `train:`이고 순서까지 M1 train 목록(또는 사용자별 마지막 window 목록)과 같음
  - z_fit은 raw `[n, d]` float32 (d = 가중치 은닉 차원 64, PCA 2차원 좌표가 아님), C-contiguous, 유한값
  - KMeans `fit`은 이 배열로 한 번만 호출. validation·test는 `predict`만 (`partial_fit`·재fit 없음)
  - `manifest.json`에 `fit_split = "train"`, fit 행 수, fit sample_id 목록 sha256 (id마다 줄바꿈을 붙여 UTF-8로 이은 값, `ids_sha256`)

#### 19.6.4 KMeans fit
- `KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=300, tol=1e-4, algorithm="lloyd", random_state=1234, copy_x=True)`: scikit-learn 1.2.2의 기본값(n_init은 경고 뒤 10)을 모두 명시 `[SPEC §7]`
- 실행 조건 `[결정]`
  - scikit-learn이 정확히 1.2.2가 아니면 실패 (1.4부터 n_init 기본 'auto', 1.3에서 algorithm 'auto'·'full' 제거)
  - `threadpoolctl.threadpool_limits(limits=cluster.fit_num_threads, user_api="openmp")` 안에서 fit. lloyd 갱신은 thread별 부분합을 잠금 순서대로 더해 thread 수에 따라 결과가 달라짐. 1 thread는 반복 fit이 bit 단위로 같았음 (합성 66,126 × 64, 0.7초). fit 뒤 모델의 OpenMP thread 수(`_n_threads`)가 `cluster.fit_num_threads`보다 크면 제한이 적용되지 않은 것으로 보고 실패 `[결정]`
  - `ConvergenceWarning`(끝난 뒤 서로 다른 군집 수 < n_clusters, 중복점 등 `cluster/_kmeans.py:1497-1505`)과 `FutureWarning`은 오류로 승격. Lloyd 반복 중 빈 군집은 sklearn이 경고 없이 다른 점으로 옮기므로 경고에 기대지 않고 19.6.5의 빈 군집 검사를 따로 둠. 승격된 경고는 원인을 단 `ClusterError`. 이 경고는 끝난 뒤 서로 다른 label 수가 `n_clusters`보다 적을 때만 남: float32 반올림으로 같은 점의 사본이 겹친 두 중심에 나뉘면 경고가 없음 (계산: 8차원 정규 점 3개 × 사본 10개, `n_clusters` 4 → 군집 크기 [10, 8, 10, 2], 경고 없음). 그래서 19.8 #6은 float32 연산이 정확한 작은 정수 점(3개 × 사본 8개)을 쓰고, 겹친 중심은 train 군집이 비지 않는 한 잡지 않음 `R tests/test_m15_cluster.py test_convergence_warning_is_error`
- 난수 순서 (sklearn 내부, M15의 유일한 난수원)
  1. fit 시작에 `RandomState(1234)` 하나 생성
  2. 초기화 10회를 차례로: 첫 중심 `randint(n_fit)` 1회 → 나머지 3개 중심마다 `uniform(size=3)` 1회(`n_local_trials = 2 + int(ln 4) = 3`), 누적 최근접 거리에서 후보를 골라 가장 좋은 것 선택 → Lloyd 반복(난수 없음)
  3. 앞선 최선보다 inertia가 작고 군집 분할이 다른 실행만 새 최선으로 바꿈 (`_is_same_clustering`, sklearn 1.2.2 `cluster/_kmeans.py:1483-1490`). 반올림 차이로 inertia만 조금 작은 같은 분할은 바꾸지 않음
  - fit 행 순서와 float32 값이 같으면 결과가 같음. 행 순서를 바꾸면 달라질 수 있어 M1 행 순서로 고정
- `random_state`는 `cluster.random_state`(1234)를 그대로 씀. `derive_seed`의 예외 (19.6.11)
- 군집 번호는 sklearn이 낸 그대로 (크기순 재번호 없음). 번호에 의미 없음 `[SPEC §9]`

#### 19.6.5 배정과 산출물
- `unit = sequence`: 세 split 모두 `predict`로 배정 (fit의 `labels_`를 쓰지 않음) `[결정]`: 모든 split에 같은 함수. `labels_`와의 불일치 수는 기록 (합성 데이터에서 0). `unit = user`의 train 배정은 19.6.8
- `predict` 입력은 float32 C-contiguous만. float64는 sklearn 1.2.2가 "Buffer dtype mismatch"로 실패함을 확인 → 먼저 명확한 메시지로 거부
- 대조: numpy float64로 중심까지 제곱거리를 계산해 최근접 군집을 다시 구하고 sklearn 결과와 비교. 불일치 수·sample_id, 가장 가까운 두 거리(`sq_dist_1`, `sq_dist_2`)를 CSV와 summary에 기록 (실수 `'%.9g'`)
- train 군집 중 하나라도 비면 실패 (그 expert가 학습되지 않음)
- 저장: joblib, `centroids.npz`, 배정 CSV. 저장 뒤 joblib을 다시 읽어 세 split을 predict해 CSV와 같아야 함 (`unit = user`는 train CSV가 사용자의 군집을 담으므로 `train_user_assignments.csv` = fit window의 predict, train CSV = 그 사용자 군집, validation·test CSV = predict로 대조 `[결정]`). 재로드 모델과 `centroids.npz`의 중심도 fit 결과와 bit 단위로 같아야 함 `R ilora/cluster.py verify_saved_model`
- `manifest.json`: 모든 산출물 sha256, SASRec 가중치(변환본) sha256·digest, 데이터 파일 4개(`data/ref/movielens/*`)의 sha256(M0 `meta.json` `inputs`의 같은 값. `prompt/movie.txt`와 원본 SASRec pickle은 넣지 않음 `[결정]`: SASRec은 변환본 sha256·digest로 묶음), `data.subset`(`full`/`smoke`)과 smoke면 `data.smoke_users`·`data.train_rows`, fit sample_id sha256, `cluster` 구역 설정, scikit-learn·threadpoolctl·numpy 버전(joblib·scipy·torch도), `threadpool_info()`. 그 밖에 `format`(`ilora-rebuild-cluster-v1`), split별 행 수 `rows`, `kmeans_file`(M14 비교 도구가 읽음), `data.seed`·`title_mode`, z 계산 설정, `dynamic_export`, `created_utc`

#### 19.6.6 `cluster_summary.json` `[SPEC §7]`
- 설정과 버전 (19.6.4), `fit`: split, 행 수, sha256, inertia, n_iter, `labels_`·predict 불일치 수, numpy 대조 불일치 수
- 크기: `train_cluster_size`·`train_cluster_ratio`, `validation_cluster_size`·ratio, `test_cluster_size`·ratio (합 = split 표본 수)
- `silhouette_score_train`: fit 집합 전체로 정확 계산 (`metric="euclidean"`, `sample_size=None`, `random_state` 미사용) `[결정]`: 합성 66,126 × 64에서 54초라 추출이 필요 없고 난수원이 줄어듦. BLAS thread에 따라 끝자리만 다를 수 있음
- `train_len_seq_by_cluster`: len_seq(3–10) × 군집 개수 표 `[결정]`: train은 len_seq 10이 92.03%(60,854/66,126)인데 test는 95개 모두, validation은 92/94가 10이라, 짧은 이력이 몰린 군집은 평가 표본을 거의 받지 못할 수 있음
- `z_scale` (19.6.2)
- fit 집합의 군집(silhouette의 label, PCA 그림의 색)은 `predict` 배정 `[결정]`: 실제로 쓰는 배정. `labels_`는 불일치 수만 `fit.labels_vs_predict_mismatches`에 기록 `R ilora/cluster.py fit_and_assign`
- dynamic 비교: `cluster.dynamic_export_dir`가 없으면 항목마다 `"NOT_RUN: dynamic_export_dir is null"`. 있으면 split마다
  - `cluster_dynamic_gate_mean` [C][K], `cluster_dynamic_gate_std` [C][K] (그 split에 표본이 없는 군집은 null, 분할표 행은 0. fork처럼 0으로 채우지 않음 19.5)
  - `contingency_cluster_vs_dynamic_argmax` [C][K] 개수 (argmax는 M12 규칙: 첫 index), `cluster_dynamic_argmax_distribution` 행 정규화 비율
  - `nmi_cluster_vs_dynamic_argmax`: `normalized_mutual_info_score(average_method="arithmetic")` 하나 `[결정]`: 번호에 의미가 없으므로 순열과 무관한 요약
  - 출처: export 경로, checkpoint epoch, NPZ sha256, `routing_mode = dynamic` 확인
- 그 밖에 `kmeans_params`, split별 numpy 대조(`numpy_check`: 불일치 수·id·두 거리·가장 작은 거리 차), `silhouette`(값·n·시간), `figures`, `reload_check`, `timings_s`, `created_utc`를 적음. M13 smoke와 M14 비교 도구가 `fit`, 크기, `silhouette_score_train`, NMI, `reload_check.passed`, `dynamic_export.z_max_abs_diff`를 읽음 `R ilora/cluster.py cluster_summary`
- 그림 (M12 함수) `[SPEC §8]`: `sasrec_pca_clusters` = fit 집합의 PCA 2차원(시각화 전용, sequence 단위는 66,126점 전부), 색 = 군집, 표기 `Cluster 0..3`. `cluster_dynamic_gate_heatmap` = train split 군집별 평균 dynamic gate, 범위 [0, 1] (dynamic export가 없으면 그리지 않고 NOT_RUN)

#### 19.6.7 cluster_hard 학습 연결 (M7, M10) `[SPEC §9, §11]`
- SPEC CLI ↔ 설정: `--routing_mode cluster_hard` = `router.routing_mode`(M7), `--cluster_model_path` = `cluster.artifact_dir`(joblib·배정·manifest가 있는 `cluster/` 폴더), `--cluster_expert_mapping` = `cluster.expert_mapping`
- `parse_expert_mapping`: `re.fullmatch(r"[0-9]+:[0-9]+(?:,[0-9]+:[0-9]+)*", text)` (`^…$`와 `\d`를 쓰지 않음: `$`는 끝 줄바꿈 앞에서도 맞고 `\d`는 비ASCII 숫자도 받음), 군집 id 0..C−1이 각각 정확히 한 번, expert id 0..K−1. 빈 항목·중복·누락·범위 밖·공백·끝 줄바꿈은 해당 토큰을 적은 오류. 주 실험은 항등 `0:0,1:1,2:2,3:3` `[SPEC §9]`. 두 군집을 한 expert에 주는 대응은 허용하되 쓰이지 않는 expert를 로그(`ClusterHardGate` 생성 때 경고)와 `run_summary.json`의 `train_summary.unused_experts`에 적음
- 학습 loop 전 (M10 14.6.4 0단계 2–3이 한 번 부름. M11 eval 명령 15.6.11과 M12 export 16.6.5도 같은 순서)
  1. `load_cluster_artifacts`: manifest sha256으로 파일 재확인, SASRec 가중치·데이터 sha256 일치, unit·n_clusters·`fit_split` 일치, 군집 id 범위, joblib 존재 `[SPEC §11]`. 표본 id: `expected_ids` = 현재 M1 세 split 표본 id 전체
     - manifest `data.subset = full`: 현재가 full이면 배정 키 집합 = `expected_ids` (66,126 + 94 + 95 = 66,315개), 현재가 smoke면 `expected_ids` ⊆ 배정 키 (M1 smoke는 row·sample_id를 다시 매기지 않음, 5.6.7)
     - manifest `data.subset = smoke`: 현재도 smoke이고 `smoke_users`·`train_rows`(M1 5.6.8)가 같아야 하며 배정 키 집합 = `expected_ids` (smoke 산출물을 full 실행에 쓰지 않음)
     - 현재 실행이 full인지는 `expected_ids`의 split별 수(66,126 / 94 / 95)로 정하고, smoke 산출물의 `smoke_users`는 [`expected_ids` train 표본의 서로 다른 사용자 수, val 수, test 수]와, `train_rows`는 `expected_ids`의 train 행과 대조함 `[결정]`: 19.6.1 signature에 현재 data 설정이 없음. `smoke_users`가 754/94/95인 smoke 설정은 full과 구별되지 않아 full로 다룸 `R ilora/cluster.py _check_sample_ids`
  2. (`build_model(hard_gate=ClusterHardGate(...))` 뒤) `verify_live_encoder`: 학습 장치에서 M3로 train 행 순서 첫 `live_check_n`개의 z를 계산해 저장 z(`representations_train.npz`)와 비교 (max |Δz| ≤ `live_check_atol`). 학습에 쓰는 SASRec이 군집에 쓴 SASRec과 같음(가중치·eval 모드·fp32·TF32 꺼짐)을 확인. KMeans 추론은 하지 않음. 함께 확인: encoder 장치 = 학습 장치, encoder `state_digest` = manifest의 SASRec digest, 표본이 train 행 순서. 결과 dict를 로그 한 줄(`cluster live check passed: n=.. max_abs_diff=.. atol=.. device=..`)로 남기고 M10은 `meta.json`, M11 eval은 `summary.json`, M12 export는 export `manifest.json`의 `cluster_live_check`에 적음 (M15-12가 읽음) `[결정]`. 학습 경로(읽기·gate·live 대조)는 scikit-learn을 import하지 않고 joblib을 열지 않음
- 학습·평가 중: M7이 `TrainBatch.sample_ids`·`EvalBatch.sample_ids`로 `ClusterHardGate`를 부름 → dict 조회 → mapping → `torch.zeros(B, K, dtype=float32, device).scatter_(1, idx, 1.0)` → M9가 `GateContext`에 넣음. loop 안에 KMeans 없음 `[SPEC §10]`
  - 호출마다 검사: shape [B, K], float32, 값이 0 또는 1, 행 합이 정확히 1.0, index 0..K−1, NaN·Inf 없음 (M12 `validate_gate` + one-hot 검사) `[SPEC §11]`
  - 모르는 sample_id는 `KeyError`(id와 artifact 경로 포함). 대체 값 없음. 빈 batch는 `ClusterError`. `lookups`는 호출이 끝까지 성공한 뒤에만 늘어남 (`KeyError`면 그대로)
- gate 미호출·gradient 없음·parameter 미갱신은 M7·M10이 보장하고 M12 기록(step마다 학습 창 증가분 `gate_calls` = 0, `cluster_lookups`)으로 확인 `[SPEC §11]`. E4(paper_spec과 같은 학습 설정) 학습 조회 수 = Σ `cluster_lookups` = 5 × 8,265 × 8 = 330,600 (drop_last, D3). 평가 조회(validation 5 × 94, test best·last 각 95)는 `lookups` 누적값에는 들어가지만 학습 합계에는 넣지 않음 (M12 16.6.2)
- E4는 E2와 같은 설정·seed로 처음부터 학습하고 routing만 다름 `[SPEC §9, §11]` (M14). 군집 번호에 의미가 없음을 문서에 적음 `[SPEC §9]`

#### 19.6.8 `variant_user_level_cluster` (`cluster.unit = user`, D25 변형)
- 사용자 = M1 pseudo `user_id`(D14): train 0–753, validation 754–847, test 848–942
- 사용자 표현 = 그 사용자 train 표본(len_seq ≥ 3) 중 행 번호가 가장 큰 window의 z `[결정]`: validation·test 표본도 사용자당 window 하나(가장 최근 이력)라 같은 종류의 벡터. 겹치는 window의 평균 z는 분포가 달라짐
- fit = 754행(user_id 순서). 사용자의 군집 = 그 fit window의 `predict`(다른 배정과 같은 함수, `labels_` 아님). train 표본의 군집 = 그 사용자의 군집이라 train CSV의 `cluster_id`는 그 window 자신의 최근접 중심이 아닐 수 있음 (`sq_dist_1`·`sq_dist_2`는 window 자신의 거리). validation·test = 각 window z의 predict (sequence 단위와 같음)
- 차이: sequence 단위는 사용자 가중이 window 수(3–506개, 중앙값 54)에 비례해 상위 100명(13%)이 train window의 38.9%를 차지. user 단위는 사용자마다 1표 (`design/tmp/windows_per_user.py`)
- 추가 산출물 `train_user_assignments.csv`, 나머지 형식·검사는 같음 (누수 검사의 fit 목록 = 사용자별 마지막 window)

#### 19.6.9 dtype·device·오류 처리
- z는 numpy float32 `[N, d]` C-contiguous로만 fit·predict. 중심 float32, 거리 대조만 float64
- one-hot `ω`는 float32로 batch 장치에서 직접 만듦 (생성 연산이라 autocast 영향 없음), fp16으로 바꾸지 않음
- 군집 명령은 CPU만 씀. 예상 시간: z 2–3초, fit 약 1초, silhouette 약 1분 (모두 합성 데이터 로컬 측정, 실제 z 시간은 TBD). 다만 M0 공통 준비의 환경 점검은 프로파일 기준이라 paper_spec의 `cluster` 명령도 GPU 1장·`LLM_PATH`를 요구함 (VESSL R11은 문제 없음). 로컬 M15-11은 `smoke_cpu` + `--set data.subset=full`로 같은 `cluster` 구역을 씀 `R tests/test_m15_cluster.py test_cluster_command_real_sasrec`
- 실패 조건 (모두 원인을 적은 예외): scikit-learn ≠ 1.2.2, 승격된 경고, 빈 군집, dtype·shape 위반, 누수 검사 실패, sample_id 누락·중복·추가, mapping 오류, manifest 해시 불일치, live 검사 실패, `verify_live_encoder`의 `n` > 현재 train 표본 수 또는 `n` < 1 (`ValueError`, M13 17.6.12 제안 7), dynamic export의 `routing_mode` ≠ dynamic·표본 집합 불일치·z 대조 실패
- 덮어쓰기 없음: 산출물은 새 실행 폴더에만 씀 (M0)

#### 19.6.10 설정 키 (`cluster` 구역)
| 키 | 타입 | paper_spec 값 | 라벨 | 설명 |
|---|---|---|---|---|
| `unit` | str {sequence, user} | sequence | `[결정]` D25 | 군집 단위. user는 `variant_user_level_cluster` |
| `n_clusters` | int ≥ 2 | 4 | `[SPEC §7]` | 군집 수 |
| `random_state` | int | 1234 | `[SPEC §7]` | KMeans 난수 (`derive_seed` 예외) |
| `n_init` | int ≥ 1 | 10 | `[SPEC §7]` 값 `[결정]` | 명시 요구는 SPEC, 값은 1.2.2 기본값 (1.4부터 'auto' = 1) |
| `init` | str {k-means++} | k-means++ | `[결정]` | 1.2.2 기본값 명시 |
| `algorithm` | str {lloyd} | lloyd | `[결정]` | 1.2.2 기본값 명시 (SPEC §7의 version 차이 회피 원칙) |
| `max_iter` | int ≥ 1 | 300 | `[결정]` | 1.2.2 기본값 명시 |
| `tol` | float > 0 | 1e-4 | `[결정]` | 1.2.2 기본값 명시 (차원별 분산 평균에 곱하는 상대값) |
| `fit_num_threads` | int ≥ 1 | 1 | `[결정]` | fit 중 OpenMP thread 수 (결정성) |
| `expert_mapping` | str | `0:0,1:1,2:2,3:3` | `[SPEC §9]` | 군집 → expert |
| `artifact_dir` | Optional[str] | null | `[결정]` | cluster_hard 학습이 읽을 `cluster/` 폴더. `cluster_hard.yaml`에서는 필수 |
| `dynamic_export_dir` | Optional[str] | null | `[결정]` | dynamic 비교용 E2 export 폴더 (`records/export/epoch{a}`) |
| `z_crosscheck_atol` | float > 0 | 1e-5 | `[결정]` | E2 export z와 군집 z의 최대 차 |
| `live_check_n` | int ≥ 1 | 256 | `[결정]` | 학습 전 live z 대조 표본 수 |
| `live_check_atol` | float > 0 | 1e-4 | `[결정]` | live(학습 장치) z와 저장(CPU) z의 최대 차 |

- paper_spec(dynamic)은 이 구역을 읽지 않지만 M0 규칙대로 모든 키를 적음. `routing_mode`는 `router` 구역(M7)

#### 19.6.11 계약 추가 제안
- 4.6.4 예외: KMeans `random_state`는 `cluster.random_state`(1234)를 그대로 쓰고 `derive_seed`를 거치지 않음 `[SPEC §7]`
- 3.1 `ω`의 cluster_hard 공급자: `ClusterHardGate(sample_ids, device) -> [B, K] float32` (M15). M7이 `TrainBatch`·`EvalBatch`의 `sample_ids`로 부르고 누적 `lookups`를 M10에 넘김
- 3.1 실행 폴더: 군집 실행의 `cluster/` 폴더 구조와 `manifest.json`(19.2). `cluster.artifact_dir`가 SPEC `--cluster_model_path` 역할
- M10(14.6.4 0단계 2–3)·M11 eval 명령(15.6.11)·M12 export(16.6.5): cluster_hard면 `load_cluster_artifacts` → `parse_expert_mapping` → `ClusterHardGate` → (`build_model`) → `verify_live_encoder`를 같은 순서로 한 번 부름. dynamic·none은 부르지 않음 `[SPEC §22]`
- `manifest.json`에 `data.subset`(·`smoke_users`)을 넣고 로드 때 현재 설정과 대조 (19.6.7): full 산출물은 smoke 실행이 부분집합으로 쓸 수 있고, smoke 산출물은 같은 smoke 설정에서만 씀
- M12의 `RepresentationTable` NPZ와 `compute_representations`를 z 계산에 공유 (16.6.11)

### 19.7 결정
- 관련 확정 결정: D12·D23(고정 z), D14(pseudo 사용자), D25(표본 단위, 사용자 단위 변형), D8(one-hot에서 활성 expert scaling 16), D2(seed 1234)
- 이 장의 새 `[결정]` (20장 등록 후보)
  - z는 M15가 M12 함수로 CPU 1 thread에서 직접 계산 (E2와 분리, E2 export와는 대조만)
  - KMeans 값: n_init 10, init k-means++, max_iter 300, tol 1e-4, algorithm lloyd (1.2.2 기본값 명시), scikit-learn 1.2.2 고정
  - fit은 OpenMP 1 thread (thread 수에 따라 결과가 달라짐을 합성 데이터로 확인)
  - 배정은 모든 split에 `predict`, 군집 재번호 없음, 빈 군집·수렴 경고는 실패
  - silhouette는 fit 집합 전체로 정확 계산 (추출 없음)
  - 요약 보조 항목: 군집별 gate 표준편차, NMI 하나, len_seq × 군집 표
  - 사용자 단위 표현 = 마지막 train window의 z
  - 학습 전 live z 대조(256개, 1e-4, 저장 z는 `representations_train.npz`), 산출물은 별도 군집 실행 폴더와 manifest 해시로 연결, manifest의 `data.subset`도 대조
  - mapping 파싱은 `re.fullmatch` + ASCII 숫자 `[0-9]` (끝 줄바꿈·비ASCII 숫자 거부)
  - E4 학습 조회 수는 M12 step 행의 학습 창 증가분 합으로 셈 (평가 조회 제외)
  - live 대조의 저장 z는 sample_id로 맞춤, 현재 subset은 `expected_ids`의 수로 판단, 사용자 단위 군집과 silhouette·PCA 색은 `predict` 배정, `z_scale`의 최대/최소 비는 표준편차에만, manifest의 데이터 sha256은 데이터 파일 4개

### 19.8 완료 기준
1. (CPU) `parse_expert_mapping`: `"0:0,1:1,2:2,3:3"` → (0, 1, 2, 3), `"0:3,1:2,2:1,3:0"` 허용. 군집 누락, 중복 군집, expert 4, 음수, 빈 항목, 공백, `"0-0"`, 끝 줄바꿈 `"0:0,1:1,2:2,3:3\n"`, 비ASCII 숫자 `"٠:0,1:1,2:2,3:3"`을 각각 거부
2. (CPU) `ClusterHardGate`: shape [B, K], float32, 값 {0, 1}, 행 합 정확히 1.0, argmax = mapping[군집], 요청한 장치에 생성, `lookups` 증가. 모르는 sample_id → `KeyError`
3. (CPU) 누수: `build_fit_set`(sequence)이 M1 train 목록을 순서까지 그대로 돌려줌. train 목록에 val 표본을 섞으면 실패. `KMeans.fit`을 감싼 spy가 호출 1회·행 수 = train 표본 수·열 수 = d를 확인. validation·test는 `predict`로만 지나감
4. (CPU) 결정성: 합성 z(66,126 × 64 float32)로 `fit_num_threads = 1` fit 두 번 → 중심·label bit 동일. joblib 저장·재로드 predict = CSV, `centroids.npz` = `cluster_centers_`
5. (CPU) float64·비연속 배열을 `predict`에 넣으면 우리 메시지로 거부. scikit-learn 버전을 1.2.1로 흉내 내면 실패
6. (CPU) 서로 다른 점이 3개뿐인 데이터로 n_clusters = 4 fit → 승격된 `ConvergenceWarning`으로 실패
7. (CPU) 실제 M1 구조로 `unit = user`: fit 754행 = 블록마다 행 번호가 가장 큰 train 표본, 모든 train 표본이 자기 사용자의 군집을 받음, validation·test는 predict
8. (CPU) 합성 입력으로 summary: 크기 합 = split 표본 수, ratio 합 1, 분할표 합 = n, 군집별 gate 평균 행 합 1(1e-6), NMI ∈ [0, 1], `dynamic_export_dir = null`이면 해당 항목 NOT_RUN이고 heatmap 없음
9. (CPU) `load_cluster_artifacts`: CSV 1 byte 변경(해시 불일치), SASRec 가중치 해시 불일치, 표본 id 누락·추가, n_clusters 불일치, smoke 산출물을 full 실행에 쓰는 경우를 각각 거부하고, full 산출물을 smoke 실행에 쓰는 경우(smoke id ⊆ 배정 키)는 통과. `verify_live_encoder`가 `representations_train.npz`의 앞 n행과 비교함 (그 파일 1 byte 변경 → 해시 불일치로 거부)
10. (CPU) E2 export 흉내 NPZ로 z 대조: 같은 값 통과, 1e-4 차이 → 실패, routing_mode cluster_hard인 export → 실패
11. (CPU) 변환된 실제 SASRec 가중치로 `cluster` 명령: fit 66,126행, 두 번 실행의 z 배열·중심·세 split 배정이 bit 단위로 같음 (시각이 든 `meta_json`·manifest, BLAS 끝자리가 다를 수 있는 silhouette, 그림은 비교 제외), silhouette 정확값과 시간 기록. 군집 크기·silhouette 값은 TBD (실행 전 기록 금지 `[SPEC §0.11]`)
12. (VESSL_REQUIRED) E4 시작 시 A100에서 `verify_live_encoder` 통과, max |Δz| 측정값(TBD) ≤ 1e-4
13. (VESSL_REQUIRED) E4 학습 기록: step 행 `gate_calls` 합 0, 학습 `cluster_lookups` 합 = 330,600 (M12 `optimizer_steps.csv`), 평가 `EvalRecord`의 gate가 배정 CSV와 mapping대로 one-hot
14. (VESSL_REQUIRED) E2 export를 준 군집 실행: dynamic 비교 항목과 `cluster_dynamic_gate_heatmap` 생성, z 대조 통과. 수치는 TBD

<!-- MODULE_SECTIONS_END -->

---

## 20. 결정 목록
모든 결정은 확정 상태다. `[결정]`은 근거가 없거나 모호해 우리가 정한 것이고, 이유를 함께 적는다. 변형은 같은 코드에서 설정만 바꿔 돌릴 수 있는 프로파일이다.

| 번호 | 항목 | 채택 | 근거 | 변형 | 모듈 |
|---|---|---|---|---|---|
| D1 | gate 구조 | ω = Softmax(W_g z), W_g ∈ R^{4×64}, bias·은닉층·정규화 없음. 공유 gate 1개가 sequence마다 ω 하나를 만들어 모든 MoE-LoRA 층·토큰·생성 단계에 씀 | `[논문 p.6 §3.2.2 식 11]`, K = 4 `[논문 p.9 §4.3.2]`, 공유 `[논문 p.4 그림 2, p.7 그림 3c (암시)]`. 논문-코드 충돌: 코드는 BatchNorm1d 19개 1D CNN + bias 있는 Linear(64→4) `U model/router/nlpr.py:39-84` | 없음 (CNN router는 재구현하지 않음, D26) | M7 |
| D2 | seed 수 | 1234 단일 실행 | LLM 방법 seed는 두 논문 모두 없음, `[SPEC §0.6]`, `U main.py:84`, `L main.py:84` | 추가 seed는 후속 선택 실험 (NOT_RUN) | M14 |
| D3 | 학습 prompt와 curriculum | LLaRA 식 9-11: micro-batch마다 확률 p(τ) = τ/T로 hybrid, 아니면 text-only. τ = 전체 학습의 micro-batch 번호(1부터), T = 전체 micro-batch 수(5 × 8,265 = 41,325), 마지막 micro-batch는 p = 1. 추첨은 `derive_seed(seed, "curriculum", τ)`. batch 안 모든 표본이 같은 형식 | `[LLaRA 간접 §4.3 p.6 식 9-11]` (p(τ)는 시각 τ의 하나의 추첨 I(τ)가 모든 표본에 공통). τ 단위·1부터 시작·batch 단위 추첨은 `[코드]` `U data/data_interface.py:28, :38-40, :67`, `L data/data_interface.py:40` (cur_step 1부터, collate마다 증가). 코드의 worker별 카운터 초기화는 결함 | `all_hybrid`(iLoRA 식 6-8 문자적), `reverse`(2024-08 방향: hybrid 확률 1 − τ/T) | M4, M10 |
| D4 | 평가 prompt | hybrid | 입력은 hybrid `[논문 p.4 식 6-7, p.5 식 8 (암시)]`, "fully integrated with the hybrid item representation" `[LLaRA §4.3 p.6]` | `text_only` (2024-08 코드 평가 방식, E3) | M4, M11 |
| D5 | 최대 lr | 1e-4 | `[논문 App E p.17]`. 논문-코드 충돌: 공개 스크립트 8e-4(LLaRA 2024-07 값 복사) | `variant_released_lr`: 8e-4 / 8e-6 / 8e-6 | M10 |
| D6 | warmup과 lr 최솟값 | warmup 시작 1e-6, 최솟값 1e-6, warmup 길이 = 전체 optimizer step // 20 = 129, 선형 증가 후 이어지는 cosine | 시작값 1/100 `[v1 App E p.16]` `[LLaRA §5.1.2 p.7]`, cosine `[논문 App E p.17]`. 최솟값·warmup 길이는 `[코드]` `U main.py:90-91`, `U model/model_interface.py:248`, `U optims.py:20-36` | D5 변형과 함께 | M10 |
| D7 | r, α, dropout, 대상 모듈 | 8, 32, 0.1, 32층 × q/k/v/o/gate/up/down_proj = 224 모듈 | 값은 두 논문 모두 없음. 대상은 "transformer layers" `[논문 p.6 §3.2.3]`, gate_proj 암시 `[논문 App C p.17]`. `[코드]` 계보 일치 `U main.py:128-130`, `L model/model_interface.py:215-220`, LLaRA checkpoint hparams | 없음 | M8 |
| D8 | scaling 규칙 | 각 expert에 α/r* = α/(r/K) = 32/2 = 16. h' = Wh + Σ_k ω_k (α/r*) B_k A_k h. K = 1이면 α/r = 4 | `[결정]` (아래 20.1). 논문 모호: 식 3은 표준 LoRA의 α/r만 정의하고 iLoRA 결합 순전파의 scaling은 쓰지 않음 `[논문 p.3 식 3, p.5 식 9, p.6 식 12]`. `[코드]` `U model/peft/tuners/moelora.py:618` | `variant_total_rank_scaling`: α/r = 4 (식 12를 식 3에 대입한 해석) | M8 |
| D9 | precision | 확정 (20.2) | | bf16 (불안정할 때만) | M9, M10 |
| D10 | 후보 | 정답 포함 20개(음성 19), 균등 비복원 추출 후 섞음. 평가 후보는 `(seed, split, row)`로 고정, 학습 후보는 `(seed, "train", row, epoch)`로 결정적 재추출 | 20개 `[논문 App B p.16]` `[LLaRA §5.1.3 p.7]`(정답 포함 여부 모호 → `[코드]` 정답 포함 20, `U data/movielens_data.py:44-47`, `U prompt/movie.txt:1-2` "20 movie titles"). 학습 prompt에도 후보 `[LLaRA 간접 §4.2 p.5]`. 고정 방식은 `[결정]`: 평가는 결정적이어야 하고(1.3), 학습은 코드처럼 매번 다른 후보를 보되 worker와 무관하게 | 없음 | M2, M11 |
| D11 | epoch와 checkpoint 선택 | 5 epoch, 매 epoch 끝 validation(94명)과 전체 상태 저장. test는 (a) validation CombinedMetric 최고 epoch(동률이면 늦은 epoch)와 (b) 마지막 epoch를 둘 다 평가·보고, 주 결과는 (a) | 5 epoch `[v1 App E p.16]` `[LLaRA §5.1.2 p.7]`. 선택 기준은 두 논문 모두 없음 → `[결정]`, `[코드]` 계보: `U main.py:23-32`(monitor `metric`, 모두 저장), LLaRA 2024-07 공개 checkpoint = validation 최고 epoch 2 | 없음 (선택 규칙은 학습 전에 고정, test로 고르지 않음) | M10, M14 |
| D12 | SASRec dropout | 항상 eval 모드 (dropout 끔) | SR-EMB 고정 `[논문 p.4 그림 2 (암시)]`, "well-trained and frozen" `[LLaRA 그림 2 caption p.4]`. `[결정]`: 식 10의 z가 이력만의 결정적 함수가 되고 KMeans 입력과 gate 입력이 같아짐 `[SPEC §10]`. 코드는 의도(eval)와 실제(Lightning이 train으로 바꿈)가 다름 | 없음 | M3 |
| D13 | 프레임워크와 버전 | 확정 (20.2) | | | M0, M9 |
| D14 | 사용자 ID와 smoke 데이터 | pseudo 사용자 ID: train은 연속 블록 번호 0–753, validation 754–847, test 848–942 (행 순서). smoke: train 80명(그 사용자의 모든 window), validation 10명, test 10명을 `("smoke_users", split)`로 선택 | 논문은 사용자 ID를 쓰지 않음. 8:1:1 `[LLaRA §5.1.1 p.7]`, 943 sequence `[LLaRA Table 1 p.7]` = 754 + 94 + 95. `[결정]`: 배포 파일만으로 복원 가능(감식: 754 블록 재구성과 0행 차이). 100명 smoke는 교수님 구두 제안 | 없음 | M1, M13 |
| D15 | 템플릿 | `U prompt/movie.txt` 3개(줄 끝 strip), 학습·평가 모두 표본마다 무작위. 학습 `("template", "train", row, epoch)`, 평가 `("template", split, row)` | "the instruction format for training and testing is randomly sampled from several prompts" `[LLaRA §5.1.2 p.7]`. 내용 `[코드]` `U prompt/movie.txt:1-3`, `U data/data_interface.py:182-190`. 표본 단위는 `[결정]`: batch 구성과 무관하게 결정적 (코드는 batch마다 1개) | `variant_fixed_template` (0번 고정) | M4 |
| D16 | 학습 결정성 수준 | 평가는 결정적, 학습은 GPU 비결정성 허용. 같은 seed 재실행 편차 측정은 후속 선택 실험 | 두 논문·코드 언급 없음 (`U main.py:40`). `[결정]`: `torch.use_deterministic_algorithms`는 느리고 일부 연산 미지원 | 없음 | M0, M10 |
| D17 | 지표 이름과 분모 | ValidRatio = 유효/전체, HR@1 = 정답/유효(구현 `hr`), CombinedMetric = HR@1 × ValidRatio = 정답/전체(구현 `metric`). 평가 N = split 전체(test 95, validation 94), drop_last 없음 | ValidRatio "over all sequences" `[논문 p.6, App B p.16]` `[LLaRA §5.1.3 p.7]`. HR@1 분모는 두 논문 모두 모호. 이름과 정의는 `[코드]` `U model/model_interface.py:180-184, :487-491` `[SPEC §12]`. 역산: iLoRA Table 1 0.5275 = 48/91(정답/유효, N=92), LLaRA Table 2 0.4421 = 42/95(정답/전체) | 보조 프로토콜 N=92(batch 4 + drop_last), N=88(batch 8 + drop_last) | M11, M14 |
| D18 | 학습 label | 입력 = `[BOS] + prompt + 응답`, 응답 = 정답 제목 + `"\n"`, label은 응답 토큰만. 두 번째 `[BOS]` 없음 | y = 다음 아이템의 텍스트 토큰 `[LLaRA 간접 §4.2 p.5]`, 식 2 `[논문 p.3]`. `[결정]`: 코드의 두 번째 `[BOS]`는 4.28.0 slow tokenizer 쌍 인코딩의 부산물(`U data/data_interface.py:73-82`, `U model/model_interface.py:38-42`)이고 4.29.0부터 shape 오류 | `variant_released_label` (두 번째 `[BOS]` 포함) | M5 |
| D19 | gate lr과 weight decay | gate lr = 다른 학습 parameter와 같은 lr 곡선(×1). weight decay 1e-5는 projector·gate에만, LoRA는 0 (Adam의 coupled L2) | projector는 "the same learning rate curve integrated with iLoRA" `[논문 App E p.17]`, gate는 iLoRA 구성요소 `[논문 p.5 §3.2 (2)]` → ×1 `(암시)`. weight decay 값 `[v1 App E p.16]`, 적용 범위는 논문 없음 → `[코드]` `U model/model_interface.py:227-232`. 코드의 router ×0.3은 CNN router용 | `variant_released_gate_lr` (×0.3) | M7, M10 |
| D20 | 제목 정규화 | 공개 코드 정규화 그대로 (끝 7자 제거, `", The"`·`", A"`·`", An"` 이동). 결함은 기록하고 평가 기록에 표본별 모호 표시 | `[코드]` `U data/movielens_data.py:61-66, :74` (`L` 같은 blob). `[결정]`: LLaRA 배포 데이터 처리의 일부라 p.6 §4 "exactly the same experimental settings"로 유지해야 논문 수치와 비교 가능 | `variant_fixed_titles` (관사 순서, item 266 빈 제목, 잘림·끝 공백 수정) | M1, M2, M11 |
| D21 | 음성 후보 풀 | 1,682개 중 보이는 이력 window(최대 10)와 정답을 뺀 아이템 | "non-interacted" `[논문 App B p.16]`의 기준 미기재. `[코드]` `U data/movielens_data.py:45`. validation·test 사용자의 전체 이력은 배포 파일에 없음 | 없음 | M2 |
| D22 | padding 방향 | 학습 오른쪽, 생성 왼쪽 + attention mask 기반 position_ids | `[결정]`: 코드의 오른쪽 padding 생성은 짧은 prompt의 첫 토큰을 `[PAD]` 뒤에서 예측 (`U model/model_interface.py:316`, `U debug/utils.py:2260`). transformers 4.28.0 `prepare_inputs_for_generation`이 mask로 position_ids를 만듦 | 없음 | M5, M11 |
| D23 | SASRec 계산 정밀도 | fp32, autocast 밖 | 근거 없음 → `[결정]`: gate 입력과 KMeans 입력을 같은 값으로 고정 | 없음 | M3 |
| D24 | epoch 끝 부분 accumulation | 모든 micro-batch 손실을 grad_accum(16)으로 나눔. 각 epoch의 마지막 9 micro-batch(8,265 = 16 × 516 + 9)도 16으로 나누고 optimizer step 1회. epoch당 517 step, 전체 2,585 step. step 기록의 `loss_mean`은 그 step `n_micro`개의 평균 그대로 | `[코드]`: `U main.py:57`(`Trainer.from_argparse_args`), `U train_movielens.sh:6`(`--accumulate_grad_batches 16`). Lightning 1.8.6은 micro-batch마다 손실을 `accumulate_grad_batches`로 나누고(`PL1.8.6 loops/optimization/optimizer_loop.py:88, :416`) epoch 마지막 batch에서도 step함(`PL1.8.6 loops/epoch/training_epoch_loop.py:333-337`). 두 논문은 부분 창을 말하지 않음(batch 128만, `[v1 App E p.16]` `[LLaRA §5.1.2 p.7]`). 16으로 나누는 것은 결함이 아님: 마지막 창도 micro-batch마다 무게 1/16. 공개 코드의 max_steps 2,583 절단은 결함(epoch 4가 8,240/8,265에서 멈춤, 14.5)이라 따르지 않음. 이전 `[결정]`(그 step의 실제 수로 평균)은 이유가 없어 2026-11-06에 이 `[코드]`로 바꿈 `R ilora/train.py accumulate_window` | 없음 | M10 |
| D25 | E4 군집 단위 | 표본(sequence) 단위: train 66,126 표본의 z로 KMeans fit, validation·test 표본은 predict | `[결정]`: dynamic gate가 sequence마다 ω를 만드는 것과 같은 단위(식 10-11), 공개 코드도 이 표현을 "user embedding"이라 부름(`U model/model_interface.py:416-421`). `[SPEC §7]` "user/sequence" | `variant_user_level_cluster` (pseudo 사용자 단위) | M15 |
| D26 | 공개 코드 동작 재현 범위 | 재구현에는 값 수준 변형만 둠(lr, gate lr, label, 평가 N, curriculum 방향, 평가 prompt, scaling). 구조 결함(CNN router BatchNorm, worker별 curriculum 카운터, 저장 필터, dropout 켬)은 재현하지 않음 | `[결정]`: 기존 fork baseline(6c7ebac)과 결과 보관 파일이 공개 코드 동작의 측정 기준 `[SPEC §0.7]` | 없음 | M14 |

### 20.1 D8 scaling을 16으로 정한 이유
- 논문은 표준 LoRA에 대해서만 h' = Wh + (α/r)BAh를 쓴다 `[논문 p.3 식 3]`. iLoRA는 B, A를 K개의 "up- and down-projection pairs for the k-th expert"(rank r* = r/K)로 나누고 `[논문 p.5 식 9]` ΔW(i<n) = Σ ω_k B_k A_k로 합친다 `[논문 p.6 식 12]`. 결합된 순전파의 scaling은 쓰지 않는다.
- 해석이 두 가지다. (i) 식 12를 식 3에 대입: α/r = 4. (ii) expert 쌍마다 식 3을 rank r*로 적용: α/r* = 16. 논문만으로는 고를 수 없다.
- ω는 softmax라 합이 1이다. (ii)에서는 균등 ω(1/K)일 때 Σ (1/K)(α/r*) B_k A_k = (α/r) BA로 표준 LoRA(K = 1, LLaRA)와 정확히 같아진다. (i)에서는 같은 상황의 갱신 크기가 표준 LoRA의 1/K(= 1/4)가 된다. Adam에서는 parameter 갱신 크기가 gradient 크기와 거의 무관하므로, (i)은 E2의 함수 공간 학습 속도를 E1의 약 1/4로 만들어 "routing의 효과"와 "갱신 크기 차이"가 섞인다.
- 공개 코드가 (ii)를 쓰고 `U model/peft/tuners/moelora.py:618`, 우리 fork baseline도 (ii)와 lr 1e-4로 validation HR@1 약 0.53까지 학습됐다 (측정값, `F` 6c7ebac).
- 따라서 (ii)를 주 설정으로, (i)을 `variant_total_rank_scaling`으로 둔다. 근거 종합(2026-11-06)은 (i)을 권고했으나, 위 비교 교란 때문에 바꿨다.

### 20.2 이전에 확정한 결정

**D13 프레임워크와 버전 (2026-11-02 확정)** `[결정]`
- **유지:** torch 2.0.0 (cu117), transformers 4.28.0
  - 이유: 공개 코드와 LLaRA가 같은 조합을 씀 (`U requirements.txt:132, :136`, `L requirements.txt:51, :54`). tokenizer와 수치 동작을 논문 시점과 같게 두어 비교 가능성을 지킴.
  - 참고: Llama-2-7b-hf `tokenizer_config.json`이 `"legacy": false`라 4.31 이상에서는 같은 prompt도 토큰이 달라짐. 별개로 4.29.0부터 `create_token_type_ids_from_sequences`가 바뀌어 공개 코드의 label mask가 shape 오류를 냄.
- **제거:** PyTorch Lightning, transformers 내부 파일 교체(`debug/utils.py`, `debug/modeling_llama.py`), 저장소에 넣은 수정 PEFT(`model/peft`), bitsandbytes
  - 학습 루프는 순수 PyTorch로 직접 작성 (M10)
  - gate는 transformers 내부를 고치지 않고 MoE-LoRA 층에 직접 전달 (`GateContext`, M8·M9)
  - MoE-LoRA 층은 PEFT 라이브러리 없이 직접 구현해 대상 Linear 층을 감쌈 (M8, M9)
  - Lightning 제거로 사라지는 동작(`model.train()` 덮어쓰기, max_steps 절단, epoch 끝 부분 accumulation, 저장 필터)은 M10에서 명시적으로 정함 (D12, D24)
- **Python:** 3.10. VESSL에는 같은 버전 고정으로 새 환경 `ilora-rebuild`를 만든다. 기존 `ilora` 환경은 `F setup_vessl.sh:87-108`이 transformers의 `generation/utils.py`와 `modeling_llama.py`를 `debug/` 사본으로 바꿔 두었으므로 재사용하지 않는다 (M14 R1, 2026-11-06 정정). 논문은 3.8이라고 했으나 공개 저장소의 바이트코드는 3.9와 3.10뿐이고, torch 2.0은 3.8–3.11을 지원함.
- **확인된 것:** 교체하지 않은 transformers 4.28.0에서 `inputs_embeds` 생성은 소스상 가능 (`modeling_llama.py:727-745`, `generation/utils.py:537-591`). 패치는 `user_embeds`·`gate_weights` kwargs 때문에만 필요했으므로 gate를 층에 직접 넘기면 stock `generate()`를 쓸 수 있음. GPU 실행 확인은 `VESSL_REQUIRED`
- torch 2.0 CUDA에 bf16 tril 연산이 없는 문제는 bf16 대안을 쓸 때만 해당 (D9)

**D9 precision (2026-11-02 확정)**
- **fp16 mixed precision:** fp16 autocast + GradScaler
  - 근거: "half-precision computation (mixed-precision training)" `[논문 App E p.17]`, `--precision 16` 기본값 = Lightning의 fp16 mixed precision `[코드 U main.py:79]` (LLaRA 2024-07 수정 전 기본값의 계승. LLaRA는 2024-07-12에 `bf16`으로 바꿨고, 논문 시기 LLaRA checkpoint(2024-03)는 fp16 AMP + GradScaler로 학습됨), 우리 baseline도 fp16
- **base LLaMA 가중치는 fp16 그대로 로드** `[결정]`
  - 공개 코드는 bf16으로 로드한 뒤 fp16 autocast로 계산함 `U model/model_interface.py:318`. 공식 checkpoint가 fp16이므로 정밀도를 한 번 깎는 셈이라 따르지 않음.
  - 결과: 공개 코드·우리 baseline과 base 가중치 값이 미세하게 다를 수 있음
- **학습되는 parameter** (LoRA, projector, gate)는 fp32
- **bf16은 대안:** fp16에서 overflow로 학습이 불안정할 때만 쓰고, 쓰면 편차로 기록
  - 참고: lr 1e-4 fp16 진단에서 최대 activation 22,400 (fp16 한계 65,504의 34%), GradScaler skip 4회
- **확인할 것:** VESSL의 Llama-2-7b-hf `config.json`에서 `torch_dtype`이 float16인지 (공개 mirror는 float16, VESSL shard 2개 12.6 GiB는 6.74e9 × 2 byte와 일치. VESSL 파일 자체는 `VESSL_REQUIRED`)

---

## 21. 진행 기록

| 날짜 | 내용 |
|---|---|
| 2026-11-02 | 뼈대 작성: 모듈 M0–M14 구분, 각 모듈 틀과 확인된 출처, 열린 결정 D1–D15 |
| 2026-11-02 | D13 확정: torch 2.0.0 + transformers 4.28.0 유지, Lightning·내부 패치·수정 PEFT·bitsandbytes 제거, Python 3.10 |
| 2026-11-02 | D9 확정: fp16 mixed precision, base 가중치 fp16 그대로 로드, 학습 parameter fp32, bf16은 불안정 시 대안 |
| 2026-11-02 | M0 작성 완료 (검토 대기). 열린 결정 D16(학습 결정성) 추가 |
| 2026-11-06 | 근거 종합 검토(iLoRA·LLaRA 논문, 코드 계보, 문서 주장 142건) 반영: 근거 계층 재정의(`[LLaRA]` 라벨), M0 정정, 결정 D1–D26 확정(D8은 종합 권고와 다르게 정함, 20.1), 모듈 간 계약(3.1)과 M15 추가 |
| 2026-11-06 | M1–M15 상세 작성(15장)과 반박 검토(3개 묶음), 모듈 간 일관성 점검(불일치 22건: 장 안에서 18건 수정, 4건은 이번 병합에서 반영). 설정 키 87개 확정(18.6.2), 계약 표(3.1)와 난수 key(4.6.4) 갱신, 출처 표기 `T4.28`·`torch2.0`·`sklearn 1.2.2` 추가(0.2), D13 문구 정정(새 VESSL 환경) |
| 2026-11-06 | 작성 중 확인한 사실: 배포 파일 평점은 3·4·5뿐(5.6.2), 제목 결함 3건 추가(176, 1004, 1018; 5.5), pseudo 사용자 754 블록이 배포 파일만으로 정확히 복원됨(5.6.5). fork OOM 원인의 근거는 148 MiB 요청 크기 일치(fp16 logits와도 맞아 결정적이지 않음)가 아니라 저장 텐서 측정(803 vs 139 KiB/token/layer)과 재작성 후 OOM 소멸로 정정(12.5) |
| 2026-11-06 | 구현: M0–M15 코드(`rebuild/ilora/`, `main.py`, `tools/`, `configs/`, `tests/`, `RUNBOOK.md`). 로컬 전체 시험(Windows CPU, slow 포함) `743 passed, 37 skipped in 976.64s (0:16:16)`, skip 37개는 모두 `VESSL_REQUIRED`. CPU smoke(`main.py smoke`, `smoke_cpu`) 통과: 통합 단계 `runs/rebuild/smoke_cpu_20261106T051752Z`, 검토 수정 뒤 `smoke_cpu_20261106T073253Z`, 검토 뒤 결정 반영 뒤 `smoke_cpu_20261106T085719Z` 모두 S0–S8 완료, 검사 84개 중 판정 78개 통과·실패 0·기록만 6개(값 TBD) |
| 2026-11-06 | 구현 반박 검토: 확인된 지적 18건(검토자 사이 중복 포함), 설계 질문 2건, 기각 1건. 수정: 살아 있는 실행의 재개를 `run.lock`으로 거부(4.6.3, 14.6.10), 강제 종료된 segment를 `interrupted`로 표시(14.6.10), MoE-LoRA ω 확장의 host 동기화 제거(12.6.2), epoch별 `frozen_checksum_at_save`(14.6.6), RUNBOOK 경로 기록·재개 조건·R1 `LLM_PATH`·R3 skip 문구·bf16 대안·R15 예시 변형(18.6.6), CPU 묶음 tokenizer 시험의 복사 축소. 검토 뒤 결정: D24를 `[코드]`(모든 micro-batch 손실 ÷ `grad_accum`)로 바꿈, ElapsedTime = 학습 계산 시간(18.6.7), 완료 기준 M10-19(CUDA 재개) 추가, export gate 대조가 실패해도 `crosscheck.json`을 먼저 씀(16.6.5). 넣지 않은 것: GPU 여유 메모리 점검(4.6.5), SIGTERM·SIGHUP 처리기(22.1) |
| 2026-11-06 | 설계 동기화(구현 반영): 구현·검토에서 정한 내용을 장 본문에 반영. 제안 수정 289건을 모두 적용(건너뜀 0): 머리 상태, 0.4, 3장(모듈 표, 3.1 계약, 3.2), 4–19장(본문과 완료 기준 문구. 기준 번호·절 제목은 그대로). 이어서 0.2에 `R` 출처 표기, 모듈 상태를 `테스트 통과(로컬)`로 바꾸고 22장 구현 기록을 새로 둠. 1·2·20장은 바꾸지 않음. VESSL 측정값은 TBD 그대로. 코드가 설계와 다를 수 있는 남은 질문은 설계-코드 대조 점검에서 확인 |
| 2026-11-06 | 설계-코드 대조 점검: 동기화로 바뀐 문단을 코드와 대조. 문서를 코드에 맞춤: 0.2(`R` 표기의 경로만·절 이름 꼴), 3.1 실행 폴더(R4의 `pip_freeze.txt`, smoke 단계 폴더 이름, `val_metrics.incomplete-<UTC>.json`에는 `-<n>` 없음), 4.6.3(`compare_inputs`는 `start_meta`가 씀), 4.6.6(`run.log` 둘째·셋째 줄), 15.6.1(M10 호출의 `ckpt_meta`), 15.6.11·16.6.5(`cluster_live_check`는 cluster_hard 밖에서 null), 17.6.9(M0-6·M8-10 행에 이미 기준 번호를 붙인 시험 추가, M9-5·M10-7 확인 문구), 18.6.9(경로 변수 = `ILORA_*` 기록값). 기준 번호·절 제목·표 열 수는 그대로. 코드에 남은 것(설계는 바꾸지 않음): `tools/compare_runs.py` `LIMITATIONS` 6번 문구가 아직 'D24 partial accumulation'(18.6.10 #6과 다름), `_Trainer.after_save`가 남은 `val_metrics.json`을 옮길 때 이름이 겹쳐도 번호를 붙이지 않음, CPU torch thread 수를 `meta.json`·`metrics.json`에 적지 않음(15.6.10) |
| 2026-11-06 | 대조 점검이 남긴 코드 3건 수정: `tools/compare_runs.py` `LIMITATIONS` 6번을 "no max_steps truncation D24"로 바꿈(18.6.10 #6과 같게), `_Trainer.after_save`가 남은 `val_metrics.json`을 옮길 때 이름이 겹치면 `-<n>`을 붙임(3.1, 14.6.10, `R ilora/train.py _keep_stale_val_metrics`), torch CPU thread 수를 `meta.json` 환경 블록과 `metrics.json` meta의 `torch_num_threads`에 적음(4.6.3, 15.6.9, 15.6.10). 회귀 시험 추가(M10-8 `::test_stale_val_metrics_kept_with_counter`, M14-7 `::test_limitation_e1_names_only_max_steps_for_d24`, M0-3·M11-10에 확인 추가). 0.2에 작업 공간 측정 스크립트 인용의 뜻을 적음. 17.6.9 M10-8 행에 이미 기준 번호를 붙인 시험 3개를 적음. 최종 로컬 확인: 전체 시험 `745 passed, 37 skipped in 1039.78s (0:17:19)`(skip 37개 모두 `VESSL_REQUIRED`), CPU smoke `runs/rebuild/smoke_cpu_20261106T142722Z` 통과(검사 84개: 통과 78, 실패 0, 기록 6). 직전 smoke `smoke_cpu_20261106T085719Z`와 `records.jsonl` 11개가 같은 byte, `representations.npz` 배열이 bit 단위로 같음(meta의 checkpoint 경로만 다름), 학습 CSV는 시간 열만 다름 |

---

## 22. 구현 기록

구현 중 정한 세부 사항 중 장 본문에 넣지 않은 것. 코드가 기준이다.

### 22.1 M0 설정과 실행 관리
- `derive_seed`는 bool key와 `|`가 든 문자열 key를 거부한다. 서로 다른 key 목록이 같은 문자열로 이어져 같은 seed가 되는 일을 막음 `R ilora/seeding.py derive_seed`
- `autocast_for`·`make_grad_scaler`의 `fp16_amp`는 장치가 CUDA이고 `torch.cuda.is_available()`일 때만 받음(14.6.1). `bf16_amp`는 CPU에서도 받음(13.6.9 CPU 대리 시험). 설정 검사(`llama2_7b` ⇒ `cuda:0`, `tiny_random` ⇒ fp32)를 지난 실행은 이 오류에 닿지 않음 `R ilora/precision.py autocast_for`
- 4.8에는 `resolve_path`·`run_command` 항목이 없어 그 시험은 M0-1·M0-3·M0-5·M0-6(경로, 명령 분기, check-env, 재개·`run.lock`)에 붙어 있음. precision 도구 시험은 M10-15에 붙음(17.6.9) `R tests/test_m0_config_run.py`
- `meta.json`의 패키지 판은 import하지 않고 설치 배포 정보(`importlib.metadata`)에서 읽음(matplotlib import 경고를 피함, 값은 같음). 4.6.3 목록 밖으로 matplotlib, scipy, joblib, threadpoolctl, safetensors, pytest 판과 `platform`, `gpu_count`, git branch도 적음 `R ilora/run.py environment_info`, `R ilora/run.py git_info`
- `meta.json` `code.rebuild_files`는 `rebuild/` 아래 `.py`·`.yaml`·`.yml`·`.md`·`.txt`만 CRLF→LF 정규화 sha256으로 적음(`__pycache__` 제외). Windows와 Linux 작업 트리가 같은 값을 냄 `R ilora/run.py rebuild_file_hashes`
- 판 고정 점검은 import한 모듈의 `__version__`을 읽음(시험에서 monkeypatch 가능). transformers 무결성 점검은 RECORD와 sha256이 다른 `.py`뿐 아니라 RECORD에 없는 `.py`, 디스크에 없는 RECORD 항목, RECORD가 가리키는 설치와 다른 위치에서 import된 transformers도 실패로 보고, 문제는 20개까지 적음 `R ilora/run.py record_problems`
- `run.lock`의 Windows 잠금은 `msvcrt.locking`으로 파일 2^30 위치의 1 byte를 잠가, 앞부분에 덧붙인 잡은 프로세스 줄은 다른 프로세스가 읽을 수 있음. Linux `fcntl.flock` 분기는 WSL Ubuntu에서만 확인함(살아 있는 동안 거부, SIGKILL 직후 해제). VESSL 컨테이너 파일 시스템에서의 동작은 첫 실행 때 확인(VESSL_REQUIRED) `R ilora/run.py hold_run_lock`
- SIGTERM·SIGHUP 처리기는 두지 않음(2026-11-06 검토 뒤 결정). fork된 DataLoader worker가 Python signal 처리기를 물려받고, 죽은 실행은 이미 다룸: 커널이 `run.lock`을 풀고 다음 `train --resume`이 끊긴 segment를 `interrupted`(`detected_utc`)로 표시함(14.6.10). 그래서 `kill`이나 `tmux kill-session`으로 끝난 실행은 `failed`가 아니라 `running`으로 남음 `R main.py main`
- 프로파일 이름은 `[A-Za-z0-9_.-]+`만 받음. 실행 폴더 이름의 UTC는 초 단위라 같은 프로파일을 같은 초에 두 번 시작하면 두 번째가 `FileExistsError`로 끝남. 시험은 `run_command(stamp=...)`로 이름을 고정함 `R ilora/run.py create_run_dir`
- `main.py`는 `run_command`에 `tokenizer_dir`를 넘기지 않음. 그래서 실제 실행에서는 check-env·smoke와 `meta.json` 기록이 쓰는 tokenizer 폴더와, `train`·`eval`이 `commands.tokenizer_dir_for`로 다시 정한 폴더(14.6.4)가 같음 `R ilora/commands.py run_command`
- `main.py smoke`에 smoke가 아닌 프로파일을 주어 `failed`로 남은 폴더(17.6.6)는 RUNBOOK `done_ok`가 명령과 상태를 확인하므로 다른 단계의 경로로 기록되지 않음. `command_requirements`에서 `smoke.smoke_plan`을 불러 폴더를 만들기 전에 막을 수도 있으나 지금은 하지 않음 `R ilora/commands.py command_requirements`
- 중첩 실행(smoke 단계)은 `run_command`가 바깥 실행의 로그 처리기·수준·propagate를 저장했다가 단계가 끝나면 되돌림. 닫힌 FileHandler는 다음 기록 때 다시 열림 `R ilora/commands.py _run_logging`
- `tracked_run`은 `BaseException`(Ctrl-C 포함)을 잡아 `failed`와 예외 종류·메시지를 적고 다시 올림. 디스크·환경 점검과 `self_check`도 이 안에서 돌아 실패하면 폴더가 `failed`로 남음 `R ilora/run.py tracked_run`, `R ilora/commands.py run_command`
- `update_meta`는 최상위 key 단위로 합치므로 모듈은 자기 블록(`train`, `smoke` 등)을 통째로 다시 씀. 원자적 쓰기는 임시 파일을 쓴 뒤 `os.replace`로 바꾸는 것이라(`check_env.json`도 같음) 중간에 끊겨도 반쯤 쓴 파일이 남지 않음 `R ilora/run.py update_meta`, `R ilora/run.py write_json_atomic`
- check-env는 데이터 적재가 실패하면 SASRec 점검을 '실행 안 함' 실패로 적음 `R ilora/commands.py check_env`
- `SasrecConfig.validate`는 `sasrec.source_sha256`이 `run.EXPECTED_INPUTS`의 `rec_model/movielens.pt` 값과 같은지도 설정 단계에서 검사함 `R ilora/sections.py SasrecConfig.validate`
- 설정 오류 메시지의 `§`는 cp949 한국어 Windows 콘솔에서 깨져 보일 수 있음. `run.log`는 UTF-8이고 Linux(VESSL)는 영향 없음

### 22.2 M1 데이터
- `load_dataset`은 `pd.read_pickle`과 `np_rng`를 부를 때마다 모듈 속성으로 찾음. 시험은 이것을 spy로 바꿔 unpickle 전 해시 확인(5.8 #1)과 `train_rows`일 때 `("smoke_users", "train")` key를 쓰지 않음(5.8 #13)을 확인 `R ilora/data.py load_dataset`
- `read_raw_titles`는 movie_id 필드가 숫자로만 된 문자열인지도 보고, `select_smoke`는 `train_rows`가 Python int의 엄격한 오름차순인지 설정 검사와 따로 다시 확인 `R ilora/data.py select_smoke`
- 검사·기록용 공개 상수 `ITEM_ENCODING`, `ITEM_FIELDS`(24), `RATINGS`(1–5), `TITLE_MODES`, `SUBSETS`, `FRAME_COLUMNS`, `CLUSTER_COLUMN`, `TITLE_REPORT_CATEGORIES`를 더 둠. 다른 모듈은 쓰지 않음 `R ilora/data.py`
- full 데이터 적재는 로컬 CPU에서 약 2.3초라 시험은 session fixture `full_data`를 함께 씀 `R tests/conftest.py full_data`

### 22.3 M2 후보 생성
- `candidate_flags`는 후보 20개·중복 없음·정답 1개·id 범위를 확인해 어기면 `CandidateError`, `match_norm`은 str이 아니면 `TypeError` `R ilora/candidates.py candidate_flags`
- `negative_pool`과 `candidate_flags`는 numpy 정수 id도 받음. Python int tuple만 받는 것은 계약 검사 `check_candidates`(6.6.3) `R ilora/candidates.py check_candidates`
- 균등성 시험(6.8 #8)은 p 값을 pytest `record_property`(`item_chi2_p`, `position_chi2_p`)로 남기고 p > 0.001만 판정. 구현 때 측정값 0.37969, 0.39844 `R tests/test_m2_candidates.py test_uniformity_chi_square`

### 22.4 M3 SASRec 인코더
- `SASRecEncoder` 생성자는 기본 초기화 난수를 `torch.random.fork_rng(devices=[])` 안에서 뽑아 전역 CPU 생성기를 바꾸지 않음. 값은 state로 덮어씀 `R ilora/sasrec.py SASRecEncoder.__init__`
- `from_config`는 `load_state_dict` 뒤 encoder의 `state_digest`를 다시 계산해 파일 digest와 비교함. `SASRecArch`는 정수 필드가 Python int ≥ 1, eps가 양의 유한 float, hidden이 head 수로 나눠지는지 확인 `R ilora/sasrec.py SASRecEncoder.from_config`
- `load_sasrec_state`는 `source`의 git blob(40자리 hex)·sha256(64자리 hex) 형식과 `dropped_keys`가 str 목록인지도 확인(유한값 검사는 7.6.6) `R ilora/sasrec.py load_sasrec_state`
- torch 2.0.0 `torch.load`가 내는 'TypedStorage is deprecated' 경고만 `torch.load` 주변과 두 M3 시험 모듈에서 끔 `R ilora/sasrec.py _load_weights_only`
- 변환 도구의 정적 검사는 protocol 4 `STACK_GLOBAL`의 두 문자열을 memo까지 따라가 풀고, 풀 수 없거나 copyreg 확장(`EXT1`·`EXT2`·`EXT4`)을 쓰면 거부 `R tools/convert_sasrec.py _globals_in_pickle`
- 변환 도구 11단계는 padding 칸에 실제 아이템 id 5를 넣어, padding이 있는 5,274행(train 5,272 + validation 2)에서도 pooled z가 bitwise 같음을 확인 `R tools/convert_sasrec.py self_check`
- 로컬 CPU 측정: 변환 도구 실행 약 28초, `verify_against_reference` 전체 대조 약 8초, batch 8의 최대 |Δz| 7.62939453125e-06, M12 규칙 export(1,024개씩, 스레드 1)의 최대 |Δz| 0 `R tests/test_m3_sasrec.py test_batch8_close`
- M3-1의 blob 사례(17.6.9)가 가리키는 bf16 수정본 blob은 `e1e33498`(fork 6c7ebac) `R tests/test_m3_convert.py test_tool_refuses`
- M3-11 시험은 check-env 전체 대신 그 SASRec 부분 `commands.check_sasrec`을 합성 state·기준 파일로 부름. 데이터·tokenizer 점검이 다른 이유로 실패하지 않게 하려는 것 `R tests/test_m3_sasrec.py test_check_env_sasrec_failures`

### 22.5 M4 prompt와 curriculum
- `parse_templates`는 bytes가 아니면 `PromptError`. `TemplateSet`은 다섯 조각이 템플릿을 그대로 다시 만드는지, `render_prompt`는 template index 범위와 `hybrid` bool, `choose_template`은 `n_templates`가 1 이상 int인지 확인. 오류 문구용 상수 `TEMPLATE_CHOICES`, `EVAL_PROMPTS`를 더 둠 `R ilora/prompts.py TemplateSet`

### 22.6 M5 tokenization과 batch 구성
- `load_tokenizer`는 판 고정값을 M0 `run.REQUIRED_TRANSFORMERS`와 `run.REQUIRED_LIBRARIES["sentencepiece"]`에서 읽어 M0 환경 점검과 같은 값을 씀(판 정의는 한 곳) `R ilora/batching.py load_tokenizer`
- `make_train_loader`는 첫 entry뿐 아니라 모든 entry의 micro-batch 크기를 확인함. `BatchPlan`이 같은 크기를 보장하므로 결과는 같음 `R ilora/batching.py make_train_loader`
- `TrainBatch`·`EvalBatch`는 tensor·list 필드를 가진 frozen dataclass라 `==`와 `hash()`가 동작하지 않음. 시험은 필드마다 `torch.equal`로 비교 `R tests/test_workers.py _assert_same_batch`
- loader 시험은 iterator 생성이 전역 CPU 생성기에서 base seed를 뽑으므로(9.6.7) 전역 random·numpy·torch 상태를 저장했다가 되돌림 `R tests/test_workers.py _read_restoring_global_rng`
- tokenizer 거부 시험은 tokenizer 파일 4개(`tokenizer.model`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`)만 복사하고 사본이 1 MiB 미만인지 확인. VESSL에서 `ILORA_TEST_TOKENIZER=$LLM_PATH`일 때 약 12.6 GiB를 복사하지 않게 하려는 것 `R tests/test_m5_batching.py copy_tokenizer_files`
- fast tokenizer 사례는 `LlamaTokenizer.is_fast`를 True로 바꿔 끼워 만들고(`from_pretrained`는 fast를 주지 않음), `sp_model_kwargs` 사례는 tmp 사본의 `tokenizer_config.json`에 sampling 설정을 넣어 만듦 `R tests/test_m5_batching.py test_load_tokenizer_rejects`
- 로컬 Windows spawn에서 66,126표본 train dataset을 worker 2개에 보내는 데 약 9.7초 걸림. Linux(fork)는 pickle하지 않음 `R ilora/batching.py PlannedTrainBatches`

### 22.7 M6 projector
- `Projector.forward`는 tensor가 아닌 입력에 `TypeError`를 냄. 10.6.3의 입력 검사는 그대로 `ValueError`, parameter dtype 검사는 `RuntimeError` `R ilora/projector.py Projector.forward`
- `build_projector`·`projector_param_count`는 `in_dim`·`out_dim`으로 양의 int만 받고 bool·float·문자열은 `ValueError` `R ilora/projector.py _check_dims`

### 22.8 M7 gate와 routing
- `Router`는 `hard_gate`를 `object.__setattr__`로 두어 nn.Module을 넘겨도 하위 모듈이 되지 않음. state_dict에 들어가지 않음 `R ilora/router.py Router.__init__`
- cluster_hard의 `hard_gate`는 호출할 수 있고 int `lookups`가 있어야 함(HardGate 프로토콜 확인). `num_experts`·`z_dim`은 양의 int, `main_seed`는 int만 받음. 어기면 `RouterError` `R ilora/router.py Router.__init__`
- `forward`는 z가 cpu·cuda가 아닌 장치에 있으면 `RouterError`를 냄 `R ilora/router.py Router._check_input`
- 기록 중 forward의 규칙(grad 꺼짐, `{split}:` 접두사, 중복 없음)은 gate를 부르기 전에 확인하므로 거부된 호출은 `gate_calls`를 늘리지 않음 `R ilora/router.py Router.forward`
- `strict_fp32`는 M3 `sasrec.py`에서 import함(정의 한 곳, 7.6.3). 그래서 `ilora.router`·`ilora.moe_lora` import가 sasrec → data → pandas를 불러오고, 순환 import는 없음 `R ilora/router.py`
- `ROW_SUM_ATOL`(1e-5)의 정의는 `router.py` 한 곳이고 M8 `moe_lora.py`가 import함 `R ilora/router.py ROW_SUM_ATOL`
- CPU fp32에서 batch 8과 batch 1의 ω 차이(15.6.10의 측정 max |Δω| 2.4e-7)는 11.8 항목 4의 허용 1e-6 안 `R ilora/evaluate.py compare_records`

### 22.9 M8 MoE-LoRA 층
- `CANONICAL_TARGETS`는 `sections.LORA_TARGETS`와 같은 tuple을 따로 둠(schema가 torch 코드를 import하지 않게). 같음은 시험이 확인 `R tests/test_m8_moe_lora.py test_canonical_targets_match_the_schema`
- `MoELoRALinear` 생성자가 base parameter의 `requires_grad`를 직접 끔. 호출자가 고정했다고 가정하지 않음 `R ilora/moe_lora.py MoELoRALinear.__init__`
- 생성자는 `gate_ctx`가 `GateContext`가 아니거나 K가 다를 때, `name`이 비었을 때도 `ValueError`. `attach_moe_lora`는 목록 밖·빈·중복 대상, K가 다른 `gate_ctx`, `llm.model.layers`가 없는 모델도 `ValueError` `R ilora/moe_lora.py attach_moe_lora`
- `attach_moe_lora`는 모든 래퍼를 먼저 만든 뒤 교체하므로, 거부되는 base가 하나라도 있으면 모델이 바뀌지 않음 `R ilora/moe_lora.py attach_moe_lora`
- `expert_slices(k)`는 autograd가 추적하는 view를 돌려줌(분석 때 detach). k가 0..K−1 밖이면 `IndexError` `R ilora/moe_lora.py MoELoRALinear.expert_slices`
- `lora_named_parameters`는 대상 위치(`…layers.<i>.<self_attn|mlp>.<대상>`) 밖의 `MoELoRALinear`가 있으면 `ValueError` `R ilora/moe_lora.py _canonical_position`
- `GateContext(num_experts)`는 양의 int만 받고(`ValueError`), `use`는 B ≥ 1도 확인함 `R ilora/moe_lora.py GateContext`
- 층 단위 M8-10도 M8-7처럼 M9 없이 13.6.9 모양의 작은 LLaMA를 `LlamaConfig`로 직접 만듦(seed scope `("test", "tiny_llama")`, 17.6.9). 모델 단위 M8-10은 `test_m9_model.py::test_gate_use_calls` `R tests/test_m8_moe_lora.py tiny_llama`
- VESSL M8-11은 M8-2의 seed·모양을 M8-2 폭(16→12)과 q_proj 폭(4096→4096)에서 돌리고 fp32 기준과의 차이는 기록만 함(값 TBD) `R tests/vessl/test_vessl_model.py test_moe_lora_layer_fp16`

### 22.10 M9 LLM 연결
- 공개 helper `torch_dtype_of`, `check_generation_config`, `tiny_llama_config`와 상수 `BACKBONES`, `TORCH_DTYPES`, `TINY_MAX_POSITIONS`도 둠. 13.6.2 단계를 시험에서 따로 부르려는 것 `R ilora/model.py`
- `check_token_ids`는 이름 붙은 다섯 id(`pad, ph, history_emb, cans_emb, item_emb`)가 공개 코드의 추가 순서대로 `new_token_ids`와 같은지도 확인함 `R ilora/model.py check_token_ids`
- `load_base_llm`은 `hf_device_map`이 없는지도 확인함(로드 정보·dtype·CPU 확인은 13.6.1) `R ilora/model.py load_base_llm`
- `resize_new_tokens`는 새 vocab 크기가 현재 행 수보다 커야 하고, resize 뒤 embedding·lm_head 행 수와 `config.vocab_size`가 모두 그 값인지 확인함 `R ilora/model.py resize_new_tokens`
- 사후 확인(13.6.2 8단계)은 그룹 키 순서, 그룹별 텐서 수(projector 4, lora 2 × 감싼 수, gate 1), 그룹 사이 중복, 학습 parameter fp32, SASRec fp32, 모든 parameter·buffer가 `model.device`에 있는지, 생성 설정 전체도 확인함 `R ilora/model.py ILoRAModel._post_build_check`
- batch 확인은 attention_mask 값이 0/1이고 1이 하나 이상인지, 토큰 id가 vocabulary 안인지도 봄 `R ilora/model.py ILoRAModel._row_checks`
- `generate`는 build 확인과 별도로 호출마다 `gen_config.use_cache is True`를 다시 확인함 `R ilora/model.py ILoRAModel.generate`
- `build_model`은 `ilora.model` logger로 LLaMA 로드 시간과 build 요약(routing, K, 감싼 모듈 수, 학습 parameter 수)을 남김 `R ilora/model.py build_model`
- `trainable_state()` 키 수는 작은 모델 dynamic 33, cluster_hard·none 32, Llama-2 453·452. `generated_frozen.*` 3개를 더한 checkpoint(`trainable.pt`) key는 36·35, 456·455 `R ilora/model.py ILoRAModel.trainable_state`, `R ilora/train.py _trainable_bytes`
- 시험이 쓰는 비공개 helper `_check_batch(batch, "train"|"eval")`, `_z_omega(batch)`, `_embed_inputs(batch)`는 13.6.3–13.6.5 단계를 그대로 나눈 것 `R ilora/model.py ILoRAModel`
- M9-9 시험은 LoRA B를 무작위로 두어 ω가 출력에 영향을 주게 한 뒤 smoke 20표본을 비교함(17.6.5). 94표본 비교는 M11-12 `test_batch_invariance_tiny` `R tests/test_m9_model.py test_batch_generation_equals_single`

### 22.11 M10 학습 루프
- `frozen_checksum_at_save`(14.6.6)를 더하기 전의 코드로 만든 checkpoint는 `load_checkpoint`가 'train_state misses'로 거부하므로 재개할 수 없음. VESSL 실행은 아직 없어 영향받는 실행은 없음 `R ilora/train.py load_checkpoint`
- 재개 순서는 준비(모델 생성까지) → `find_resume_point`(표지 없는 폴더 이름 바꿈) → `TrainRecorder.resume` → monitor → 시작 값 → `load_checkpoint`. 그래서 표지가 맞지 않는 폴더의 `TrainingError`는 모델을 만든 뒤에 남 `R ilora/train.py _Trainer.run`
- `train()`은 `evaluate_split`·`build_model`·`build_batch_plan`과 M15 함수를 부를 때마다 모듈 속성에서 찾으므로 시험의 monkeypatch가 그대로 먹음 `R ilora/train.py _Trainer.prepare`
- run.log 요약 줄은 `train step <s>/<T> epoch <e> tau …`, epoch 요약 줄은 `train epoch <e> done:`로 시작함 `R ilora/train.py SUMMARY_PREFIX`
- smoke_cpu(train 156표본)는 epoch당 micro 39, 창 [4]×9 + [3], step 10이고 2 epoch에 T 20·w 5·micro 78 (1 epoch이면 T 10·w 2). M13 smoke와 M14-5가 `StepCounts.window_sizes()`로 다시 계산함 `R ilora/train.py step_counts`
- M10-13 시험은 smoke_cpu에서 k step마다의 요약과 epoch 끝 요약이 겹치지 않게 `train.log_every_steps=3`으로 돌림(요약 s = 3, 6, 9, 10, 12, 15, 18, 20) `R tests/test_m10_train.py`
- fp16 GradScaler 경로, CUDA 난수 저장·복원, step별 최대 메모리는 CUDA에서만 실행됨. 로컬 시험은 `FakeScaler`로 skip 판정·scale 절반·skip 뒤 lr과 τ 진행·16회 연속 중단의 loop 논리만 확인하고, 실제 경로는 VESSL의 M10-15·M10-16·M10-19가 확인함 `R tests/test_m10_train.py`
- `build_optimizer`는 그룹 안 이름이 정렬·유일한지, 그룹 원소가 `nn.Parameter`이고 모델의 parameter인지도 확인함 `R ilora/train.py build_optimizer`

### 22.12 M11 생성과 평가
- `evaluate.py`에 `torch.load`가 없음을 정적 시험이 확인함(checkpoint 읽기는 M10 `load_model_state`로만, 3.1) `R ilora/evaluate.py run_eval_command`
- `ckpt_meta`는 `metrics.json`과 NPZ meta에만 들어가고 `records.jsonl`에는 없음. 그래서 학습 중 평가와 eval 명령의 `records.jsonl` sha256은 비교할 수 있고, NPZ sha256은 `tags`가 달라 다름 `R ilora/evaluate.py _npz_meta`
- `EvalRecord`는 검사 없는 frozen dataclass. 검사는 `generate_records`(기록 생성)와 M12 `write_eval`이 함(M12 시험이 일부러 어긋난 기록을 만듦) `R ilora/evaluate.py EvalRecord`
- `judge`는 정수가 아닌 후보 id와 str가 아닌 제목도 `EvalError`로 거부함 `R ilora/evaluate.py judge`
- eval 명령의 'test는 best_val·last로만' 규칙은 설정 검사(15.6.12)에 더해 명령 시작에서 한 번 더 확인함(같은 `ConfigError`) `R ilora/evaluate.py run_eval_command`
- smoke의 기대 표본 수는 `data.smoke_users`의 (train, val, test) 순서에서 읽음(full은 `FULL_SPLIT_SIZE` = M1 `data.EXPECTED_SAMPLES`, 5.6.1) `R ilora/evaluate.py _expected_size`
- 작은 모델 CPU fp32에서 split 하나 평가는 약 16.5초(8개씩 12 batch, 모든 행이 EOS 없이 64 토큰)이고 batch 1 점검이 validation에 약 29초를 더함. 시험 시간 참고값이고 실험 결과가 아님

### 22.13 M12 기록과 gate 분석
- `save_npz`는 zip을 직접 씀: member마다 `np.lib.format.write_array`, 압축 없음, member 시각 1980-01-01, `create_system` 3, 권한 0644 고정. numpy 1.24.2의 `np.savez`는 `create_system`이 OS를 따라(Windows 0) 같은 표도 OS마다 byte가 달라지므로, 같은 표면 Windows와 Linux에서 같은 byte를 만듦(측정 `design_sync/g4/savez_probe.py`) `R ilora/records.py save_npz`
- export NPZ의 meta는 `compute_representations`의 meta(`utc` 포함)에 `source: export`, `routing_mode`, `checkpoint`, `sasrec`, `source_run`을 더한 것. M15 dynamic 비교가 `routing_mode`와 `checkpoint.epoch`를 읽음 `R ilora/records.py _export_epoch`
- `act_max_module` 열은 모듈 이름(str)이고, 기록기는 장치 위 index tensor를 flush 때 `monitor.module_names`로 바꿈 `R ilora/records.py TrainRecorder._resolve_tensors`
- 실행끼리 학습 기록을 비교할 때는 `time_s`, `train_s`, `val_s`, `save_s`와 메모리 열을 뺌. `read_rows(path, fields)`가 열 타입대로 읽음 `R ilora/records.py read_rows`
- heatmap 표시 행은 정렬 순서에서 `np.linspace(0, n−1, 500).astype(int64)`로 고르고(n > 500이면 엄격히 증가), y축 이름에 'Samples (500 of n, evenly spaced)'를 적음 `R ilora/analysis.py plot_gate_heatmap`
- `gate_interpretation.txt`는 한국어 고정 틀이고 수치는 소수 4자리(아주 작거나 큰 값은 지수 표기). 값에 따라 바뀌는 문장은 없음 `R ilora/analysis.py write_interpretation`
- `crosscheck_gates`는 세 표본 집합을 위치가 아니라 sample_id로 맞추고, argmax가 다른 sample id는 처음 20개만 `crosscheck.json`에 적음 `R ilora/records.py crosscheck_gates`
- matplotlib 3.7.1은 첫 그림에서 pyparsing deprecation 경고를 냄. 그림을 그리는 시험은 이를 filterwarnings로 숨김 `R tests/test_m15_cluster.py pytestmark`
- 합성 측정(실험 결과 아님): paper_spec 크기의 기록기 입력(micro 41,325행, step 2,585행, epoch 5행)을 flush 140번으로 1.8초에 쓰고, 합성 원 실행(2 epoch)의 full data export와 3 split 분석은 27초가 걸림

### 22.14 M13 테스트
- `_RunFiles.train_state`는 epoch마다 `train_state.pt`를 한 번만 읽고 Adam moment는 들고 있지 않음: Llama-2 epoch 하나가 약 283 MiB라 25 epoch 과적합 계획의 메모리를 아낌 `R ilora/smoke.py _RunFiles.train_state`
- smoke 단계의 경로 `--set` 값은 POSIX 형태의 JSON 문자열로 넘겨 YAML 값으로 읽히고 따옴표·Windows 역슬래시에 안전함 `R ilora/smoke.py _path_value`
- 실패한 단계도 `steps/<id>_<이름>_*` 중 가장 늦은 폴더를 찾아 보고에 적음 `R ilora/smoke.py _SmokeRun._find_folder`
- CPU fp32 S1의 `batch_invariance.json`은 토큰 10/10이지만 batch 8과 1의 gate 값이 끝자리에서 달라 records·gates 일치는 0/10(13.6.3과 같은 GEMM 행 수 의존). S1 판정은 토큰만이고, one-hot(S5)·K = 1(S7)은 기록 일치도 10/10 `R ilora/smoke.py _batch_invariance`
- 로컬 CPU smoke 보고는 검사 84개(판정 78, 기록만 6: S1·S4·S5·S7 측정값, S5·S7 batch 무관성)이고 `main.py smoke`(smoke_cpu)는 약 2분 `R ilora/smoke.py run_smoke`
- `tests/conftest.py`와 `tests/vessl/conftest.py`는 모듈 이름이 같아 시험 모듈이 `SKIP_*` 상수를 import할 수 없음: 문서화된 skip은 conftest fixture 안에서만 냄 `R tests/conftest.py`
- conftest fixture는 모듈 코드를 본문 안에서 import하므로 그 모듈이 없어도 수집은 되고, 쓰는 시험은 skip이 아니라 오류가 됨 `R tests/conftest.py`

### 22.15 M14 실험 구성
- 시험과 로컬 확인은 `compare_runs.py`에 늘 `--root`를 줌(없으면 `runs/rebuild/compare_<UTC>/`에 씀, 18.6.8). `--help`는 아무것도 만들지 않음 `R tools/compare_runs.py main`
- 비교 폴더에도 M0 `create_run_dir`가 만드는 빈 `checkpoints/`·`eval/`·`records/`가 생김 `R tools/compare_runs.py run_compare`
- 비교 출력은 UTF-8·LF로 임시 이름에 쓴 뒤 바꾸고, 같은 이름이 있으면 `CompareError`, token 모양 문자열이 있으면 쓰지 않음 `R tools/compare_runs.py write_text`
- RUNBOOK을 고칠 때 `test_commands_are_valid`가 요구하는 형식을 지킴: `## R0.`–`## R16.` 제목, R0 도우미 `latest`·`done_ok`·`rec`·`need`·`outputs`, `V=<변형>   # <이름들>` 줄(text_only_eval·user_level_cluster·bf16과 합쳐 변형 11개), shlex로 나뉘는 줄, `compare_runs.py` 호출 2개·`convert_sasrec.py` 1개 `R tests/test_m14_runbook.py test_commands_are_valid`
- `test_output_selectors`는 `test_run_outputs.py`의 모든 시험이 RUNBOOK 선택어 하나에 걸리고 그 시험이 읽는 `ILORA_RUN_<NAME>`이 먼저 `rec`됐는지 봄: 새 실행 폴더 시험(예: E3)은 RUNBOOK에 `outputs` 단계를 더해야 통과 `R tests/test_m14_runbook.py test_output_selectors`
- 컨테이너에서는 `nvidia-smi`가 compute 프로세스를 보이지 않을 수 있어 R7은 `memory.used`와 `pgrep`을 함께 보고, 확실한 막음은 `run.lock` 거부임. `pgrep`은 VESSL 이미지에 procps가 있어야 함 `R RUNBOOK.md R7`
- R1의 LLaMA 내려받기용 env `hf-download`에는 `huggingface_hub>=0.20,<1.0`을 설치함(18.6.6 R1) `R RUNBOOK.md R1`

### 22.16 M15 SASRec 군집과 cluster-hard 배정
- `ClusterHardGate`의 `KeyError`는 `args[0]`에 id와 산출물 폴더를 담음. `str(KeyError)`는 Windows 역슬래시를 두 번 적으므로 메시지 대조는 `args[0]`로 `R ilora/cluster.py ClusterHardGate.__call__`
- `centroids.npz`는 zip 항목 시각을 1980-01-01로 고정해 같은 배열이면 같은 byte `R ilora/cluster.py _write_npz`
- `nearest_two`는 8,192행씩 float64로 차의 제곱합을 계산(노름 전개 없음)해 predict를 대조하고 동률은 첫 index. summary의 `numpy_check`는 불일치 id를 1,000개까지만 적음 `R ilora/cluster.py nearest_two`
- `test_fit_deterministic_one_thread`와 `test_full_artifacts_for_smoke`는 66k 정확 silhouette(약 1분)을 monkeypatch 대역으로 바꿈. 정확값은 `smoke_cluster_dir`와 M15-11이 계산 `R tests/test_m15_cluster.py stub_silhouette`
- `smoke_cluster_dir`는 session 공유라 산출물을 바꾸는 시험은 `copy_artifacts`로 복사본을 씀 `R tests/test_m15_cluster.py copy_artifacts`
- 로컬 M15-11에서 실제 SASRec z로 `cluster` 명령 한 번은 약 70–76초이고 그중 정확 silhouette이 약 61–65초, KMeans fit은 약 1.5–1.7초(군집 크기·silhouette 값은 TBD로 둠) `R tests/test_m15_cluster.py test_cluster_command_real_sasrec`
