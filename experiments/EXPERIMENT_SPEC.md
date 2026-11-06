이 repository는 다음 논문의 공식 구현을 fork한 것이다.

Paper:
Customizing Language Models with Instance-wise LoRA for Sequential Recommendation
https://arxiv.org/abs/2408.10159

원본 upstream:
https://github.com/AkaliKong/iLoRA

현재 repository 상태:

Fork:
https://github.com/myCompilerWontShutUp/iLoRA

Upstream:
https://github.com/AkaliKong/iLoRA

Local repository:
C:\Programming\Python\2026-2\ilora

origin:
https://github.com/myCompilerWontShutUp/iLoRA.git

upstream:
https://github.com/AkaliKong/iLoRA.git

current branch:
exp/gate-cluster

현재 working tree는 clean 상태다.

fork, clone, remote 설정, branch 생성은 이미 완료되어 있다.
따라서 이것들을 다시 수행하지 마라.

모든 변경은 현재 exp/gate-cluster branch에서 수행한다.

목표는 이 repository를 수정하여
VESSL의 단일 A100 80GB 환경에서

clone
→ 환경 설정
→ preflight
→ baseline 재현
→ gate 추출 및 분석
→ SASRec clustering
→ cluster-hard routing 학습
→ evaluation
→ 최종 비교

까지 최소 명령으로 전부 수행할 수 있는 상태로 만드는 것이다.

중요:

현재 local development 환경은 Windows다.

로컬에서는 GPU 대규모 학습을 실행하지 않는다.

코드 구현, repository 분석, 정적 검증,
dependency가 없어도 가능한 smoke test와 synthetic test만 수행한다.

실제 Llama-2-7B 학습 및 평가는 VESSL Linux + A100 80GB × 1 환경에서 진행한다.


==================================================
0. 절대 지켜야 할 원칙
==================================================

1. 원본 iLoRA의 dynamic routing baseline 동작을 절대 깨뜨리지 않는다.

2. 기존 파일을 불필요하게 대규모 refactor하지 않는다.
최소 수정 원칙을 지킨다.

3. 원본 코드의 동작과 수정된 부분을 명확히 구분한다.

4. baseline과 새로운 cluster-hard experiment는
CLI argument 또는 별도 shell script로 명확히 구분한다.

5. MovieLens만 대상으로 한다.

오늘 실험 범위에서는 Steam/LastFM을 수정하거나 실행할 필요가 없다.

공통 코드를 수정해야 하는 경우에도
Steam/LastFM 동작을 불필요하게 깨뜨리지 않도록 주의한다.

6. random seed는 우선 1234 하나만 사용한다.

7. 원본 MovieLens baseline 설정은 가능한 한 유지한다.

특히:

- num_moe = 4
- lora_r = 8
- 후보 수 = 원본 설정
- epoch = 원본 설정
- batch size = 원본 설정
- gradient accumulation = 원본 설정

단, 공개 shell script와 논문 본문의 hyperparameter가 다른 경우
임의로 논문 수치에 맞추지 않는다.

둘의 차이를:

experiments/IMPLEMENTATION_NOTES.md

에 기록한다.

"released code reproduction"에서는
공개 repository의 MovieLens shell script 값을 우선 사용한다.

8. test/validation 정보로 KMeans를 fit하면 안 된다.

KMeans는 반드시 train split에서 얻은 representation만 사용한다.

9. Hugging Face token, VESSL token, GitHub token,
API key, password, model weight를 repository에 저장하지 않는다.

10. checkpoint, Llama model weight 및 큰 결과 파일은
git에 commit하지 않는다.

11. 실제 실행 결과가 없는 수치는 절대 만들어내지 않는다.

결과가 없는 경우:

TBD

또는

NOT_RUN

으로 명확하게 표시한다.

12. VESSL 비용 사고를 막기 위해 multi-GPU를 사용하지 않는다.

모든 실제 GPU 실험은:

single GPU
CUDA:0

만 사용한다.

GPU 개수가 정확히 1개가 아니라면
실험 시작 전에 실패하도록 한다.

13. 현재 local development 환경은 Windows다.

로컬에서는 다음을 하지 않는다.

- Llama-2-7B 다운로드
- CUDA용 구버전 PyTorch 환경 구축
- bitsandbytes GPU 동작 검증
- 전체 MovieLens training
- 실제 GPU inference
- Linux 전용 dependency를 Windows에 억지로 설치
- VESSL 전용 CUDA 환경을 Windows에서 완전히 재현하려고 시도
- dependency 문제를 해결한다는 이유로 repository 전체를 최신 버전으로 포팅

로컬에서는 다음만 수행한다.

- repository 조사
- 코드 구현
- Python syntax check
- dependency가 없어도 가능한 unit test
- synthetic tensor 기반 routing test
- static validation
- shell script 내용 검토
- Git 작업

Linux/VESSL에서만 검증 가능한 사항은 명확히:

VESSL_REQUIRED

라고 문서에 기록한다.

Windows에서 테스트가 안 된다는 이유만으로
실험 코드나 dependency를 임의로 바꾸지 않는다.

14. 실제 GPU 실행 시간과 VESSL 비용은
실제 측정 전에는 임의로 결과에 기록하지 않는다.

실제 start/end timestamp가 확보되면
elapsed time을 기록할 수 있다.

비용 자체는 VESSL UI 또는 실제 billing 정보가 없는 한
추정값을 실험 결과처럼 기록하지 않는다.


==================================================
1. 먼저 코드베이스 조사
==================================================

코드를 수정하기 전에 전체 repository를 조사하고
다음을 실제 코드 기준으로 정확히 찾아라.

A. MovieLens training entry point

B. MovieLens test/evaluation entry point

C. dataset split 구조

D. batch 내부 데이터 구조

E. user_id 또는 sample을 식별할 수 있는 값

F. SASRec user/sequence representation을 생성하는 정확한 함수

G. SASRec representation dimension

H. learned router가 정의되는 위치

I. learned router가 실제 호출되는 위치

J. 4개 expert gate weights가 계산되는 위치

K. gate tensor의 실제 또는 예상 shape

L. gate_weights가 MoE-LoRA layer에 전달되는 과정

M. shared router가 모든 LoRA layer에서 어떻게 공유되는지

N. checkpoint save 구조

O. checkpoint load 구조

P. MovieLens 평가 metric 계산 구조

Q. ValidRatio / HR 또는 기타 metric의 정확한 정의

R. SASRec이 pretrained/frozen인지

S. SASRec이 iLoRA training 과정에서 update되는지

T. LoRA rank가 expert별로 실제 어떻게 나뉘는지

U. lora_r=8, num_moe=4일 때
각 expert의 실제 effective rank가 무엇인지

V. 공개 train_movielens.sh와
논문 Appendix의 hyperparameter 차이

W. 원본 repository README에 있는
transformers / modeling_llama / generation utils 관련
debug workaround

조사 결과는:

experiments/CODEBASE_ANALYSIS.md

에 기록한다.

추측하지 말고:

- 실제 파일 경로
- class 이름
- 함수 이름
- 중요한 argument
- tensor 흐름

을 기록한다.

가능하면 중요한 부분은 line number까지 적어도 된다.


==================================================
2. Git / 실험 구조
==================================================

현재 branch:

exp/gate-cluster

에서 계속 작업한다.

main branch에는 직접 수정하지 않는다.

최종적으로 repository를 대략 다음과 같이 정리한다.

단, 기존 repository 구조를 깨뜨리지 않고
실제 코드 구조에 맞게 경로는 조정 가능하다.

analysis/
    analyze_gates.py
    cluster_sasrec.py
    compare_results.py
    generate_summary.py

scripts/
    extract_representations.py
    fit_kmeans.py
    preflight_vessl.sh
    smoke_test_routing.py

experiments/
    CODEBASE_ANALYSIS.md
    IMPLEMENTATION_NOTES.md
    VESSL_GUIDE.md
    RESULT_TEMPLATE.md

artifacts/
    .gitkeep

results/
    .gitkeep

setup_vessl.sh

run_baseline_movielens.sh
run_cluster_hard_movielens.sh
run_all_vessl.sh

필요하면 파일명은 기존 프로젝트 구조와 충돌하지 않게
조금 조정해도 된다.


==================================================
3. Baseline 재현
==================================================

첫 실험은 원본 iLoRA MovieLens dynamic routing이다.

routing mode를 추가한다면:

--routing_mode dynamic

을 기본값으로 한다.

아무 argument를 추가하지 않고 기존 방식으로 실행해도
가능하면 기존 behavior가 그대로 유지되어야 한다.

dynamic에서는 기존 numerical behavior가
불필요하게 바뀌어서는 안 된다.

baseline output은 cluster-hard와 완전히 분리한다.

예:

outputs/baseline/

checkpoints/baseline/

results/baseline/

원본 평가 지표를 그대로 계산한다.

평가 결과는 사람이 읽는 로그뿐 아니라
machine-readable 형식으로도 저장한다.

예:

results/baseline/metrics.json

다음 metadata도 함께 기록한다.

- git commit hash
- seed
- dataset
- routing_mode
- num_moe
- lora_r
- epochs
- learning rate
- batch size
- gradient accumulation
- candidate count
- model path의 basename
- 실행 시작 시간
- 실행 종료 시간
- elapsed time
- GPU model
- torch version
- transformers version
- peft version


==================================================
4. Gate 값 및 SASRec representation 추출
==================================================

교수님이 확인하려는 핵심은:

"사용자/sequence마다 네 expert를 실제로 다르게 사용하고 있는가?"

"특정 expert 하나로 routing이 collapse되어 있는가?"

이다.

shared router를 사용하는 현재 구조에서
SASRec representation으로부터 계산되는
실제 4차원 gate probability를 저장한다.

MoE layer 내부 여러 곳에서 중복으로 gate를 수집하지 않는다.

가능하면 shared router가 실제 호출되는
한 지점에서 gate를 수집한다.

각 sample마다 최소 다음을 저장한다.

sample_id

user_id
(실제 dataset에 존재할 경우)

split

gate_0
gate_1
gate_2
gate_3

argmax_expert

sasrec_0
sasrec_1
...
sasrec_N

SASRec dimension은 코드에서 실제 값을 확인한다.

64라고 가정하지 않는다.

실제 dimension이 64이면 sasrec_0 ... sasrec_63으로 저장한다.

저장 예:

results/baseline/gates_validation.csv

results/baseline/gates_test.csv

train representation도 반드시 확보할 수 있는 구조를 만든다.

예:

results/baseline/representations_train.npz

또는 실제 구조상 CSV가 더 적절하면:

results/baseline/representations_train.csv

대용량인 경우 NPZ를 우선 고려한다.

중요:

training step마다 CSV I/O를 하지 않는다.

가능하면 inference/evaluation 과정에서
메모리에 accumulate하거나 적절한 buffer를 사용한 뒤
종료 시 한 번 저장한다.

tensor는 저장 전에:

detach()
float()
cpu()

처리한다.

gate tensor shape은 실제 코드 기준으로 처리한다.

예상 shape을 가정한 뒤 무조건 squeeze하지 말고
실제 runtime shape을 검증할 수 있게 한다.

각 sample의 gate probability 합이
floating point tolerance 내에서 약 1인지 검사한다.

NaN / Inf도 검사한다.


==================================================
5. Gate diversity 분석
==================================================

analysis/analyze_gates.py를 구현한다.

다음 지표를 반드시 계산한다.


1. Mean expert utilization

각 expert에 대해:

mean(gate_k)


2. Argmax expert utilization

각 sample에서 가장 높은 gate를 갖는 expert를 선택하고:

count
ratio

를 계산한다.


3. Per-sample normalized entropy

H(p) = -sum(p_i log p_i) / log(K)

K = 4

전체 sample 기준으로:

mean
std
median
min
max

를 계산한다.


4. Expert별 gate standard deviation

사용자별 gate distribution이
실제로 달라지는지 확인하기 위함이다.


5. Global mean gate와
각 sample gate 사이의 Jensen-Shannon divergence

전체:

mean
std
median
min
max

등을 적절히 계산한다.


6. Gate vector dispersion 보조 통계

필요하다면 한 가지 정도 추가해도 된다.

하지만 의미 없는 metric을 여러 개 남발하지 않는다.


7. Collapse diagnostic

다음 세 상황을 구분할 수 있게 분석한다.

Case A:

모든 사용자 gate가 거의

[0.25, 0.25, 0.25, 0.25]

인 경우

→ global utilization은 균등하지만
instance-wise personalization은 거의 없음.


Case B:

거의 모든 사용자 gate가

[0.90, 0.03, 0.03, 0.04]

와 비슷한 경우

→ 특정 expert로 collapse되었을 가능성.


Case C:

사용자마다 의미 있게 다른 distribution

→ instance-wise routing이 실제로 다르게 작동.


Collapse 여부를 하나의 arbitrary threshold로
무조건 TRUE/FALSE 판정하지 않는다.

수치를 제공하고
필요한 경우 보수적인 qualitative interpretation만
별도 text file에 작성한다.

결과:

results/baseline/gate_summary.json

results/baseline/gate_summary.csv

results/baseline/gate_interpretation.txt


==================================================
6. Gate 시각화
==================================================

다음 그림을 생성한다.

results/baseline/figures/gate_heatmap.png

results/baseline/figures/expert_mean_utilization.png

results/baseline/figures/argmax_expert_distribution.png

results/baseline/figures/entropy_distribution.png

필요하다면 PDF도 같이 저장해도 된다.


==================================================
6-1. 시각화 디자인 규칙
==================================================

매우 중요하다.

이 자료는 일반적인 대학 연구실에서
연구자가 matplotlib로 직접 만든 실험 결과 그래프처럼 보여야 한다.

AI가 만든 인포그래픽 같은 디자인을 사용하지 않는다.

목적은 "AI 탐지 회피"가 아니라,
불필요한 장식 없이 일반적인 학술 실험 그래프 스타일을 유지하는 것이다.

반드시 다음 규칙을 따른다.

- 기본적으로 matplotlib 사용
- 특별한 이유가 없으면 plain matplotlib research plot 스타일 사용
- 제목은 짧고 사실적으로 작성
- 축 이름 간결하게 작성
- 범례가 필요할 때만 범례 사용
- italic font 사용하지 않기
- decorative font 사용하지 않기
- 지나치게 굵은 글씨 사용하지 않기
- gradient 사용하지 않기
- 그림자 사용하지 않기
- rounded card 사용하지 않기
- dashboard 스타일 사용하지 않기
- 배경 장식 사용하지 않기
- 아이콘 사용하지 않기
- 이모지 사용하지 않기
- 가운데점(·)을 장식적인 separator로 사용하지 않기
- 특수 화살표나 장식 기호를 불필요하게 사용하지 않기
- 그래프 안에 긴 설명 문장 넣지 않기
- "high"
- "good"
- "bad"
- "collapse!"
- "best"
등의 해석 annotation을 그래프 내부에 넣지 않기
- 불필요한 callout box 사용하지 않기
- 그래프에 결론 문장을 직접 쓰지 않기
- subtitle 남발하지 않기
- 범례와 axis label 외의 부가 설명을 가능한 넣지 않기
- 지나치게 화려한 color palette 사용하지 않기
- 3D chart 사용하지 않기
- pie chart 사용하지 않기
- 막대그래프는 일반 rectangular bar 사용
- heatmap은 일반 matrix heatmap 사용
- 모든 bar 위에 숫자를 일일이 표시하지 않기
- 필요한 경우에만 최소한의 grid 사용
- figure background는 흰색
- 일반 sans-serif 기본 font 사용
- 학술 자료에서 흔히 볼 수 있는 figure ratio 사용
- 기본 DPI 300
- tight_layout 또는 constrained_layout 적절히 사용
- 과도한 여백이나 decorative spacing 금지
- 사람이 직접 만든 단순하고 읽기 쉬운 분석 그래프 스타일 사용

시각화 안에는 다음 문구를 넣지 않는다.

"AI generated"
"analysis result"
"AI analysis"
"interpretation"

등의 워터마크나 부가 설명.

데이터 해석은 별도의 markdown/text 문서에서 한다.


gate heatmap:

- rows = samples
- columns = Expert 1, Expert 2, Expert 3, Expert 4
- sample이 너무 많으면 시각화용으로만 sampling 가능
- 모든 통계 계산은 전체 sample을 사용
- 가능하면 argmax expert 기준으로 row 정렬
- y tick이 너무 많으면 생략
- heatmap 내부 모든 셀에 숫자를 쓰지 않는다


==================================================
7. SASRec representation clustering
==================================================

두 번째 실험:

learned dynamic routing 대신
SASRec user representation을 기반으로
사전에 user/sequence를 clustering한다.

각 cluster에 하나의 LoRA expert를 고정 할당한다.

우선 baseline의 SASRec representation 구조를 조사한다.

KMeans:

n_clusters = 4

random_state = 1234

을 기본으로 사용한다.

가능하면 sklearn KMeans에서
n_init을 명시적으로 설정해
version 변화에 따른 default 차이를 피한다.

중요:

fit은 반드시 TRAIN split representation만 사용한다.

validation/test는:

kmeans.predict()

만 사용한다.

PCA는 시각화 용도로만 사용한다.

PCA 2D representation으로 KMeans를 fit하지 않는다.

기본 실험은 raw SASRec representation에서 KMeans를 수행한다.

StandardScaler를 자동으로 삽입하지 않는다.

먼저 SASRec embedding scale을 확인한다.

특별한 이유가 없다면
raw representation으로 KMeans를 수행한다.

scaling이 필요하다고 판단되는 경우에도
오늘 primary experiment를 임의로 변경하지 말고
후속 variant로 분리하거나 문서에 기록한다.

결과:

artifacts/kmeans_k4_seed1234.joblib

results/clustering/train_assignments.csv

results/clustering/validation_assignments.csv

results/clustering/test_assignments.csv

results/clustering/cluster_summary.json


계산:

- train cluster size
- train cluster ratio
- validation cluster size
- test cluster size
- silhouette score
- cluster별 dynamic gate mean
- cluster별 dynamic argmax expert distribution
- cluster ID vs dynamic argmax expert contingency table


==================================================
8. Clustering 시각화
==================================================

다음 생성:

results/clustering/figures/sasrec_pca_clusters.png

results/clustering/figures/cluster_dynamic_gate_heatmap.png


sasrec_pca_clusters:

실제 SASRec dimension
→ PCA 2D
→ scatter plot

색상 = cluster id

PCA는 visualization only임을
코드와 문서에 명시한다.

PCA 결과를 clustering input으로 사용하지 않는다.

시각화 스타일은 앞의 규칙과 동일하게 적용한다.

특히:

- 긴 annotation 금지
- decorative ellipse 금지
- cluster 경계선을 임의로 예쁘게 그리지 않기
- convex hull을 장식 목적으로 추가하지 않기
- cluster 이름에 의미를 임의로 부여하지 않기

표시는:

Cluster 0
Cluster 1
Cluster 2
Cluster 3

정도로만 한다.


==================================================
9. Cluster-hard routing 구현
==================================================

새 routing mode:

--routing_mode cluster_hard

를 구현한다.


기존 dynamic:

SASRec representation
→ learned router
→ soft gate [p0, p1, p2, p3]
→ MoE-LoRA mixture


cluster_hard:

SASRec representation
→ fixed KMeans
→ cluster id
→ one-hot gate
→ 기존 MoE-LoRA 계산


예:

cluster = 2

gate = [0, 0, 1, 0]


기존 MoE-LoRA layer 자체는 가능한 한 수정하지 않는다.

현재 구조에서 gate_weights를 외부에서 전달할 수 있다면
gate input만 one-hot으로 바꾸는 방식으로 구현한다.

CLI:

--routing_mode dynamic

--routing_mode cluster_hard

--cluster_model_path PATH

--cluster_expert_mapping "0:0,1:1,2:2,3:3"

cluster_expert_mapping은 configurable하게 구현한다.

Primary experiment에서는:

0:0
1:1
2:2
3:3

identity mapping을 사용한다.

cluster 번호 자체에는 semantic meaning이 없다는 점을
문서에 기록한다.

cluster-hard 모델은 처음부터 retrain한다.

즉 기존 dynamic baseline에서 학습된
expert weight를 그대로 가져와
inference에서 router만 바꾸는 실험과 혼동하지 않는다.


==================================================
10. 매우 중요한 SASRec / KMeans 안정성 문제
==================================================

반드시 코드에서 실제 구조를 확인한다.

SASRec representation이 iLoRA training 과정에서
update되는 구조인지 확인한다.

다음 중 무엇인지 정확하게 판단한다.

A. SASRec이 pretrained + frozen

B. SASRec이 함께 train/update됨

C. 다른 구조


A이면:

고정 SASRec representation 기준으로
KMeans를 fit하고 사용하는 것이 자연스럽다.


B이면:

KMeans fit 시점 representation과
cluster-hard training 도중 representation이 달라질 수 있다.

이 경우 임의로 online clustering을 넣지 않는다.

cluster assignment가 epoch마다 바뀌는 구조로
실험을 변형하지 않는다.

재현 가능한 fixed reference를 사용한다.

우선순위:

1. 논문의 pretrained SASRec initialization representation

2. baseline initialization과 동일한 fixed SASRec snapshot

3. 코드 구조상 더 적절한 fixed reference가 발견되면
근거를 문서에 기록하고 사용


중요:

cluster_hard routing이 training 중
"현재 업데이트된 SASRec representation"으로
매 step 새로운 KMeans cluster를 예측하는 것인지,

아니면
"fixed SASRec representation 기반 assignment"를
사용해야 하는 것인지

실험 정의를 코드 조사 후 명확히 결정한다.

교수님의 요구는:

"미리 user를 clustering하고
cluster별로 expert를 할당"

하는 것이므로

가능하면 fixed precomputed cluster assignment가
우선되는 방향을 검토한다.

선택한 방식과 이유를:

experiments/IMPLEMENTATION_NOTES.md

에 정확히 기록한다.


==================================================
11. Cluster-hard training
==================================================

cluster-hard는 baseline과 가능한 동일 조건에서
처음부터 학습한다.

seed = 1234

다른 조건은 baseline과 최대한 동일하게 유지한다.

바뀌는 핵심 요소는 routing 방식이어야 한다.

cluster-hard에서는 neural router가
실제 forward에 사용되지 않아야 한다.

가능하면 router parameters를
optimizer 대상에서도 제외한다.

최소한:

- router forward 호출 여부
- router gradient 여부
- router parameter update 여부

를 확인할 수 있게 한다.

assert / validation:

- gate shape 정상
- one-hot property
- 각 row sum = 1
- expert index 0~3
- NaN 없음
- Inf 없음
- KMeans artifact 존재 확인
- assignment lookup 정상
- train/val/test split leakage 없음


output:

outputs/cluster_hard/

checkpoints/cluster_hard/

results/cluster_hard/


==================================================
12. 비교 결과
==================================================

analysis/compare_results.py를 구현한다.

최종적으로:

results/final_comparison.csv

results/final_comparison.md

를 생성한다.

columns:

Method

Routing

ValidRatio

HR@1

CombinedMetric
(원본 implementation에서 실제 사용하는 경우만)

Seed

Epoch

num_moe

lora_r

TrainableParams

ElapsedTime
(실제로 기록 가능할 경우)


baseline:

iLoRA Dynamic


new method:

Cluster Hard


metric 이름은 실제 repository implementation을 확인하고
정확히 사용한다.

존재하지 않는 metric을 새로 만들어
논문 metric처럼 사용하지 않는다.


==================================================
13. 해석 시 중요한 confounder
==================================================

iLoRA가 total LoRA rank를
여러 expert로 나누는 구조인지
실제 코드를 반드시 확인한다.

예를 들어:

lora_r = 8

num_moe = 4

이고

각 expert rank = 2

라면:


dynamic routing:

4개의 rank-2 expert output을
soft gate로 조합


cluster-hard:

하나의 rank-2 expert만 one-hot으로 활성화


가 될 수 있다.

이 경우 cluster-hard의 성능 저하가:

routing 전략 차이

때문인지

effective active capacity 감소

때문인지

완전히 분리하기 어렵다.

이 문제를 반드시 limitation으로 기록한다.

단, 오늘 primary experiment에서는
이 confounder를 없애겠다는 이유로
expert rank나 parameter budget을 임의로 변경하지 않는다.

원본과 동일한 total parameterization에서
cluster-hard routing 결과를 먼저 얻는다.

후속 실험 후보로만:

- expert당 full rank
- top-k routing
- hard routing with matched active rank

등을 기록할 수 있다.

오늘 primary experiment에는 넣지 않는다.


==================================================
14. VESSL에서 바로 실행 가능하게 만들기
==================================================

가장 중요하다.

VESSL:

Linux
A100 80GB × 1

환경에서 repository clone 후
최소 명령으로 전체 실험을 실행할 수 있어야 한다.

setup_vessl.sh를 만든다.

setup script 역할:

- set -euo pipefail
- OS 확인
- Python version 확인
- GPU/CUDA 확인
- dependency 설치
- 원본 requirements와 현재 환경 compatibility 확인
- 필요한 경우 최소한의 reproducible compatibility workaround 수행
- dependency version 출력
- 필요한 directory 생성

최신 package로 무작정 upgrade하지 않는다.

특히 원본 repository에서 사용하는:

torch

transformers

peft

pytorch-lightning

bitsandbytes

버전을 먼저 확인한다.

원본 README의 debug workaround를 확인한다.

환경 호환성 때문에 원본 버전을 그대로 설치할 수 없는 경우:

1. 왜 설치할 수 없는지 확인

2. 가장 작은 compatibility change 적용

3. experiments/IMPLEMENTATION_NOTES.md에 기록

4. 실제 변경된 dependency version 기록

setup script는 가능한 한 idempotent하게 만든다.

두 번 실행했다고 환경이 계속 망가지지 않아야 한다.


==================================================
15. VESSL preflight
==================================================

scripts/preflight_vessl.sh를 만든다.

학습 전에 반드시 다음을 검사한다.

- uname / OS
- nvidia-smi 성공 여부
- GPU 개수
- GPU model
- VRAM
- CUDA_VISIBLE_DEVICES
- Python version
- pip version
- torch version
- transformers version
- peft version
- pytorch-lightning version
- bitsandbytes version
- torch.cuda.is_available()
- torch.cuda.device_count()
- LLM_PATH 존재 여부
- dataset 존재 여부
- free disk space
- output directory writable 여부


비용 사고 방지를 위해:

GPU가 정확히 1개가 아니면
실험을 시작하지 않는다.

즉:

GPU count != 1

이면 error exit.


CUDA_VISIBLE_DEVICES=0을 명시한다.

LLM_PATH가 존재하지 않으면
training을 시작하지 않는다.

dataset path가 잘못되면
training을 시작하지 않는다.

preflight 자체는 실제 training을 시작하지 않는다.

preflight 마지막에:

"Preflight passed. No training has started."

와 같이 명확한 메시지를 출력한다.


==================================================
16. 환경변수
==================================================

민감한 값은 commit하지 않는다.

.env.example 또는 적절한 config example을 만든다.

예:

LLM_PATH=/path/to/Llama-2-7b-hf

HF_TOKEN=

CUDA_VISIBLE_DEVICES=0

SEED=1234


실제 token은 git에 저장하지 않는다.

shell script에서도 token을 hard-code하지 않는다.

.gitignore를 정리한다.

최소:

.env

checkpoints/

outputs/

*.ckpt

*.pt

*.pth

*.bin

*.safetensors

__pycache__/

wandb/

logs/

.cache/


단,

results/ 내부의 작은:

CSV
JSON
Markdown
PNG

는 연구 결과 보존을 위해
필요한 경우 commit할 수 있게 한다.

model weights는 절대 commit하지 않는다.


==================================================
17. VESSL 실행 스크립트
==================================================

다음을 만든다.


run_baseline_movielens.sh

하는 일:

1. preflight

2. baseline training

3. baseline evaluation

4. gate export

5. SASRec representation export

6. gate analysis

7. baseline 결과 검증


run_cluster_hard_movielens.sh

하는 일:

1. train representation 존재 확인

2. KMeans fit

3. val/test assignment 생성

4. clustering analysis

5. cluster-hard training

6. cluster-hard evaluation

7. final comparison


run_all_vessl.sh

하는 일:

baseline
→ gate export
→ gate analysis
→ KMeans
→ clustering analysis
→ cluster-hard
→ evaluation
→ final comparison
→ final summary

순서로 실행한다.


모든 shell script는:

set -euo pipefail

사용.


각 단계 시작/종료 시간을 log에 기록한다.

logs/ 아래에 stdout/stderr를 남긴다.

실험이 실패할 경우
어느 단계에서 실패했는지 알 수 있게 한다.

예를 들어:

[1/8] Baseline training
[2/8] Baseline evaluation
...

같은 평범한 step logging은 허용한다.

이미 성공적으로 생성된 artifact가 있는 경우
무조건 덮어쓰지 않는다.

가능하면:

--force

또는:

FORCE=1

같은 명시적인 옵션이 있을 때만
다시 생성하도록 한다.

checkpoint가 존재한다고 무조건 성공으로 간주하지 말고
완료 marker 또는 metric file 등으로
실제 완료 여부를 확인하는 구조를 검토한다.


==================================================
18. README / VESSL GUIDE
==================================================

experiments/VESSL_GUIDE.md를 만든다.

매우 간단하고 명확해야 한다.

VESSL에서 사용자가 해야 하는 것이
이상적으로 다음 정도가 되어야 한다.

git clone https://github.com/myCompilerWontShutUp/iLoRA.git

cd iLoRA

git checkout exp/gate-cluster

export LLM_PATH=/actual/path/to/Llama-2-7b-hf

export CUDA_VISIBLE_DEVICES=0

bash setup_vessl.sh

bash scripts/preflight_vessl.sh

bash run_all_vessl.sh


각 명령 뒤에 예상되는 결과를 짧게 적는다.

예:

preflight 성공:
GPU count 1
A100 확인
LLM_PATH 확인
No training started

등.

VESSL_GUIDE에 불필요하게 장황한 배경 설명을 넣지 않는다.

특히 실제 UI에서 GPU instance 생성 방법을
추측해서 문서화하지 않는다.

우리는 이미 VESSL에서 A100 ×1 Workspace를
사용자가 직접 생성할 예정이다.

코드는 생성된 workspace 내부에서 실행하면 된다.


==================================================
19. Final summary 자동 생성
==================================================

실험 종료 후 실제 결과 파일을 읽어서:

results/today_summary.md

를 생성한다.

가능하면:

analysis/generate_summary.py

또는 equivalent script를 만든다.

구조:

1. Experiment setup

2. Baseline reproduction

3. Gate diversity

4. SASRec clustering

5. Cluster-hard routing

6. Performance comparison

7. Limitations


반드시 실제 생성된 파일의 실제 값만 사용한다.

결과가 없으면:

NOT_RUN

으로 적는다.

어떤 결과도 임의 생성하지 않는다.

paper 숫자를 실제 실험 결과 대신 복사하지 않는다.


==================================================
20. 오늘 미팅용 최소 결과 파일
==================================================

VESSL 전체 실행이 끝났을 때
최소한 다음 파일들이 존재해야 한다.


results/baseline/metrics.json


results/baseline/gates_test.csv


results/baseline/gate_summary.json


results/baseline/figures/gate_heatmap.png


results/baseline/figures/expert_mean_utilization.png


results/baseline/figures/argmax_expert_distribution.png


results/baseline/figures/entropy_distribution.png


artifacts/kmeans_k4_seed1234.joblib


results/clustering/cluster_summary.json


results/clustering/figures/sasrec_pca_clusters.png


results/clustering/figures/cluster_dynamic_gate_heatmap.png


results/cluster_hard/metrics.json


results/final_comparison.csv


results/final_comparison.md


results/today_summary.md


==================================================
21. 코드 품질 및 검증
==================================================

전체 구현 후 로컬 Windows에서 가능한 검증을 수행한다.

1. Python syntax check

예:

python -m compileall

등.


2. import가 가능한 작은 standalone script는 import check


3. shell script static review

Windows에서 bash가 없다면
무리해서 실행하지 않는다.

WSL 또는 Git Bash가 이미 있고 안전하게 사용 가능하다면
bash -n 정도의 syntax check는 가능하다.

없으면:

VESSL_REQUIRED

로 기록한다.


4. CLI --help check

dependency 문제 없이 실행 가능한 범위만.


5. synthetic tensor 기반 routing unit test


6. mapping parser test


7. KMeans artifact save/load test
가능하면 작은 synthetic numpy array 사용


8. analysis script synthetic CSV test


routing unit test에서는 최소:

- dynamic gate shape
- cluster-hard gate shape
- one-hot property
- row sum = 1
- expert range
- cluster to expert mapping parser
- invalid mapping rejection
- NaN rejection 또는 validation


를 검증한다.

실제 Llama-2-7B weight가 없는 로컬에서는:

model download
full model initialization
GPU training

을 시도하지 않는다.


==================================================
22. Baseline 원본 보존 검증
==================================================

이번 수정에서 매우 중요하다.

dynamic routing baseline path가
수정 전과 논리적으로 동일한지 검토한다.

가능하면 다음을 확인한다.

- routing_mode argument default가 dynamic
- dynamic일 때 기존 router 사용
- 기존 gate 계산 방식 유지
- 기존 MoE layer 호출 유지
- cluster model을 load하지 않음
- cluster assignment를 참조하지 않음
- cluster-hard 전용 로직이 dynamic path에 개입하지 않음

원본 behavior를 바꿀 가능성이 있는 변경은
최종 diff에서 다시 검토한다.


==================================================
23. 작업 방식
==================================================

이 작업을 한꺼번에 무작정 수정하지 않는다.

다음 Phase 순서로 진행한다.


Phase 1
Repository 조사


Phase 2
구현 계획 작성


Phase 3
Baseline logging / gate / representation export


Phase 4
Gate analysis + clustering analysis scripts


Phase 5
Cluster-hard routing


Phase 6
VESSL automation


Phase 7
Tests / static validation


Phase 8
Documentation / final diff review / Git commit / push


각 Phase 종료 시:

- 변경 파일 확인
- git diff 확인
- syntax 검토
- 가능한 test 수행
- 다음 Phase 진행

한다.


매우 중요:

Phase가 끝날 때마다 사용자에게:

"계속할까요?"

"다음 단계로 갈까요?"

같이 승인 요청을 하지 않는다.

Phase 1부터 Phase 8까지
명백한 치명적 문제가 없다면
자율적으로 계속 진행한다.

중간 진행 상황을 출력하는 것은 괜찮지만
사용자 응답을 기다리며 작업을 멈추지 않는다.

다만 다음과 같은 경우에는 멈추고 보고한다.

- repository를 손상시킬 위험
- upstream 구조가 프롬프트 가정과 완전히 다름
- 연구 정의 자체를 바꿔야만 구현 가능
- Git credential 관련 사용자 조작 필요
- 파일 삭제나 force push가 필요
- 민감한 credential 입력 필요

그 밖의 일반적인 구현 문제는
스스로 분석하고 가능한 범위에서 해결한다.


==================================================
24. Git commit / push
==================================================

전체 구현과 local validation이 끝난 뒤:

git diff

를 최종 검토한다.

main branch에 있는지 반드시 확인한다.

현재 branch는 반드시:

exp/gate-cluster

여야 한다.

push 직전에 다음을 확인한다.

git status

git branch --show-current

git remote -v


적절한 단위로 commit을 만든다.

예:

feat: add gate logging and diversity analysis

feat: add SASRec clustering and hard routing

chore: add VESSL experiment automation


실제 변경 구조에 따라 commit 개수와 메시지는 조정 가능하다.

모든 commit이 끝난 뒤:

origin/exp/gate-cluster

로 push한다.

main에는 push하지 않는다.

force push는 절대 하지 않는다.

GitHub authentication이 이미 정상 설정되어 있으므로
일반 push를 시도해도 된다.

authentication token이나 password를
파일이나 로그에 저장하지 않는다.

push가 인증 문제로 실패하면
credential을 직접 요구하거나 기록하지 말고
상태를 보고한다.


==================================================
25. 구현 중 특히 조심할 것
==================================================

다음과 같은 shortcut을 사용하지 않는다.

1.
test representation으로 KMeans fit

금지.


2.
baseline에서 학습한 dynamic expert를
그대로 cluster-hard inference에 사용하고
그 결과를 cluster-hard training 결과라고 부르기

금지.


3.
논문 보고 수치와 결과가 다르다는 이유로
learning rate / seed / batch 등을 임의 조정

금지.


4.
VESSL에서 잘 돌게 하려고
repository 전체를 최신 transformers로 포팅

금지.


5.
결과가 나오기 전에
placeholder metric을 임의 숫자로 작성

금지.


6.
시각화를 보기 좋게 만들겠다는 이유로
데이터를 임의 smoothing

금지.


7.
PCA 2D 좌표를 KMeans의 primary input으로 사용

금지.


8.
cluster 의미를 임의로:

"Action users"
"Comedy users"
"Expert users"

등으로 이름 붙이기

금지.


9.
random cluster assignment를
KMeans 결과처럼 사용

금지.


10.
GPU 개수가 2개 이상인데도 그냥 실행

금지.


==================================================
26. 작업 완료 시 최종 보고
==================================================

모든 구현, local validation, commit, push를 마친 뒤에만
최종 사용자 보고를 한다.

최종 응답에는 다음을 간결하게 정리한다.


1. 변경한 파일 목록


2. 핵심 구현 내용


3. 원본 코드에서 발견한 중요한 구조

특히:

- SASRec frozen 여부
- SASRec dimension
- router 위치
- gate tensor shape
- expert별 rank
- lora_r=8 / num_moe=4의 실제 의미


4. local에서 성공한 test


5. local에서는 검증 불가능해서
VESSL_REQUIRED인 항목


6. VESSL에서 사용자가 실행해야 할 정확한 명령

복사해서 바로 실행할 수 있는 형태로 제공.


7. 예상되는 output 파일


8. 발견한 위험 요소 또는 아직 남은 문제


9. 생성한 commit 목록


10. push 성공 여부


==================================================
27. 지금 바로 시작
==================================================

이제 Phase 1부터 시작해라.

먼저 코드를 수정하지 말고
repository 전체를 실제로 조사한다.

조사 결과를:

experiments/CODEBASE_ANALYSIS.md

에 작성한다.

터미널에도 핵심 내용을 간단히 출력한다.

하지만 Phase 1 종료 후
사용자의 추가 승인을 기다리지 않는다.

치명적인 설계 모순이나 repository 손상 위험이 발견되지 않는 한:

Phase 2
→ Phase 3
→ Phase 4
→ Phase 5
→ Phase 6
→ Phase 7
→ Phase 8

까지 자율적으로 계속 진행한다.

각 Phase 종료 시 스스로 diff와 test 상태를 확인한다.

모든 구현과 local validation,
commit 및 origin/exp/gate-cluster push가 완료된 뒤에만
최종 보고한다.

이제 시작해라.