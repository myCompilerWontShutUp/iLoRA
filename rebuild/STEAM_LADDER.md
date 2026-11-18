# Steam 단계별 재현 (사다리)

Steam에서 가장 쉬운 단계부터 하나씩 만들고, 각 단계의 결과가 논문 값과 비슷할 때만 다음 단계로 넘어가는 절차다. 근거와 결정은 `rebuild/DESIGN.md` 24장(D45–D47)에 있다. 서버 준비(셸, env, SASRec 변환, 환경 점검)는 `rebuild/RUNBOOK_datasets.md`의 DS0, DS1, DS3 (Steam)을 그대로 쓴다. `rebuild/tests/test_ladder.py`가 이 문서의 bash 블록을 실제 CLI와 프로파일로 검사한다.

## 규칙

- 한 번에 한 단계만 돌린다. 앞 단계가 통과하지 않으면 다음 단계를 시작하지 않는다.
- 통과 기준: 논문 값이 우리 test 결과의 Wilson 95% 구간 안에 있다. Steam test는 1,186개라 구간이 약 ±0.03이다.
- S1(학습 없는 LLM)만 예외다. 유효율은 prompt 형식에 따라 크게 달라지므로, ValidRatio가 0.5 미만이고 정답률이 0.05(후보 20개의 무작위 수준) 미만이면 통과로 보고 실제 출력을 함께 확인한다.
- 통과하지 못하면 그 단계에서 멈추고 원인을 기록한다. 기준을 바꾸거나 다음 단계로 넘어가지 않는다.
- 결과 수치는 측정한 뒤에만 적는다. 측정 전은 TBD다 `[SPEC §0.11]`.
- 모든 단계는 seed 1234, 같은 평가 후보(D10), validation 1,184개와 test 1,186개를 쓴다.
- 지표: ValidRatio = 유효/전체, 정답률 = 정답/전체. LLaRA 표의 Steam 값은 이 정의와 N = 1,186에서 정수로 정확히 맞는다. 정답/유효는 보조로 함께 적는다.
- A100은 1장만 쓰고 GPU 작업은 한 번에 하나만 돌린다 `[SPEC §0.12]`.

## 단계 한눈에 보기

| 단계 | 만드는 것 | 앞 단계에 더해지는 것 | 논문 기준 (ValidRatio / HitRatio@1) | 정수 역산 (N = 1,186) | 결과 |
|---|---|---|---|---|---|
| S0 | SASRec 단독 (LLM 없음) | 데이터, 후보, 고정 SASRec | 1.0000 / 0.4010 | 5 seed 평균이라 정수가 아님 | test 498/1186 = 0.4199, 통과 |
| S1 | Llama-2, 학습 없음 | LLM, prompt, 생성, 판정 | 0.1653 / 0.0135 | 유효 196, 정답 16 | TBD |
| S2 (선택) | LoRA 학습, text-only prompt | LoRA 학습 | 0.9840 / 0.4637 | 유효 1,167, 정답 550 | 미구현, TBD |
| S3 | LLaRA (expert 1개) | projector와 행동 토큰, curriculum | 0.9975 / 0.4949 | 유효 1,183, 정답 587 | TBD |
| S4 | expert 2개 | gate, expert 1개 추가 | 약 0.50 (그림에서 읽은 근사값) | 없음 | TBD |
| S5 | iLoRA (expert 4개) | expert 4개 | 0.9981 / 0.5264 | 공개 파일의 N으로는 정수해 없음 | TBD |

- 출처: S0–S3 `[LLaRA Table 2 p.8]`(iLoRA 표 1의 같은 행과 값이 같음), S4 `[논문 p.8 그림 4a, §4.3.2]`("4-expert가 2-expert보다 Steam에서 2.2% 높음"), S5 `[논문 p.7 Table 1]`.
- S2는 논문의 TALLRec 행(text-only로 LoRA를 학습한 LLM)에 해당한다. 넣을지는 S1 결과를 본 뒤 정한다.

## S0. SASRec 단독

LLM 없이, 고정 SASRec의 출력층 `s_fc(z)`로 평가 후보 20개의 순위를 매겨 1위가 정답인지 본다. CPU만 쓴다.

```bash
python -I rebuild/tools/sasrec_baseline.py --config rebuild/configs/datasets/steam_paper_spec.yaml
```

- 산출물: `runs/rebuild/steam_sasrec_baseline_<UTC>/`의 `meta.json`, `baseline.json`.
- 확인하는 것: 데이터와 후보 추출, 변환한 SASRec이 논문의 SASRec 행과 같은 수준인지.
- 결과 (2026-11-16, 로컬 Windows CPU):

| split | 정답 | 정답률 | Wilson 95% | 논문 |
|---|---|---|---|---|
| test | 498/1186 | 0.4199 | 0.3921–0.4482 | 0.4010 |
| validation | 485/1184 | 0.4096 | 0.3820–0.4379 | – |

- 판정: 통과 (논문 값이 구간 안). 논문 값은 저자가 따로 학습한 SASRec 5개의 평균이고, 여기서는 공개된 `rec_model/steam.pt` 하나를 썼다.
- 방법 확인: 같은 도구로 MovieLens를 계산하면 test 32/95, validation 41/94다.

## S1. Llama-2, 학습 없음

학습하지 않은 Llama-2에 text-only prompt를 넣고 생성 결과를 판정한다. checkpoint도 학습 실행도 쓰지 않는다. expert 1개이고 `lora_B`가 모두 0이라 LoRA의 기여가 정확히 0이므로, 출력은 기본 Llama-2의 출력이다.

```bash
python -I rebuild/main.py zeroshot --config rebuild/configs/steam_ladder/steam_s1_zero_shot.yaml
```

- 산출물: `runs/rebuild/steam_s1_zero_shot_<UTC>/eval/init/{val,test}/`의 `records.jsonl`, `metrics.json`, `representations.npz`, 그리고 `eval/summary.json`.
- 확인하는 것: prompt 구성, tokenization, 생성 설정, 첫 줄 규칙, 유효·정답 판정이 논문의 Llama2 행과 같은 수준을 내는지.
- 알려진 차이: text-only prompt는 공개 코드 형식대로 제목 뒤에 `[PH]` 토큰을 둔다. 이 토큰의 임베딩은 seed로 만든 값이고 학습되지 않는다. 논문은 Llama2 행에 쓴 prompt를 밝히지 않았다.
- 결과: TBD (VESSL A100 필요).

## S3. LLaRA (expert 1개)

S1에 LoRA 학습, projector와 행동 토큰, curriculum을 더한다. 학습 전에 `rebuild/RUNBOOK_datasets.md`의 DS4, DS5 (Steam)로 GPU smoke와 과적합 점검을 먼저 통과시킨다.

```bash
python -I rebuild/main.py train --config rebuild/configs/steam_ladder/steam_s3_llara.yaml
```

- 산출물: `runs/rebuild/steam_s3_llara_<UTC>/` (checkpoint 5개, `eval/`, `records/train/`).
- 결과: TBD.

## S4. expert 2개

S3에 gate(식 11)를 더하고 expert를 2개로 늘린다. r = 8을 rank 4인 expert 2개로 나눈다(식 9).

```bash
python -I rebuild/main.py train --config rebuild/configs/steam_ladder/steam_s4_two_experts.yaml
```

- 결과: TBD.

## S5. iLoRA (expert 4개)

```bash
python -I rebuild/main.py train --config rebuild/configs/steam_ladder/steam_s5_ilora.yaml
```

- 결과: TBD.

## 시간 (추정)

- S0: CPU 약 20초.
- S1: validation과 test를 모두 평가해 약 40–60분 (MovieLens 평가 속도에서 환산). 실제 값은 측정 뒤 적는다.
- GPU 메모리: MovieLens 평가의 최대 할당은 15.9 GiB였다 (2026-11-09 측정). S1의 Steam 값은 측정 뒤 적는다.
- S3, S4, S5: 학습 1회 약 17–19시간 (`rebuild/DESIGN.md` 23.9의 추정). 실제 값은 측정 뒤 적는다.
