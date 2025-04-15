# 그래프 네트워크 기반 금융사기 탐지 및 피처 도출 및 분석 ( 팀 4mula )
### 1. 팀 구성원
* 이은정(팀장) / 박지연 / 여혜미 / 지은현

<br>
<br>

### 2. 프로젝트 배경 
#### 2.1 주제 선정 이유
 * 자금세탁 증가 : 국가 경제, 정부, 사회에 악영향을 미치고 국경을 초월하는 심각한 문제로 대두
 * 정교해 지는 금융 사기 : 기존 탐지 시스템을 보완하여 새로운 유형의 사기를 효과적으로 탐지해야 함
 * 금전적 손실과 고객 신뢰도 하락 : 정상 거래를 사기라고 탐지하는 경우, 정상거래 정지로 인한 자원손실 및 고객과의 신뢰도 하락 문제 발생

#### 2.2 프로젝트 목적 및 요구사항 정의
###### 2.2.1 프로젝트의 목적
 * GNN 기반의 금융 사기 탐지
 * 피쳐의 타당성과 유효성 설명
 * GNN모델과 Boosting 모델 비교
 * GAT모델의 성능 향상
###### 2.2.2 요구사항 정의
 * 금융사기 탐지 모델의 성능 개선 및 효율성 향상
 * 기존 모델에 GNN 기반 새로운 피쳐 추가
 * 모든 모델은 안정성. 보안성이 요구 됨
 * 금융결제원에서 제공하는 데이터의 보안 및 접근성 고려

<br>
<br>

### 3. 프로젝트 수행 절차 및 방법
#### 3.1 일정
![캡처2](https://github.com/user-attachments/assets/e0e6cf9a-25dd-425f-b643-2c94d2c8f097)

#### 3.2 Boostring Model ( CATBoost )
![캡처2](https://github.com/user-attachments/assets/38762259-f073-418a-8572-4cab02640332)

#### 3.3 GNN Model ( GAT )
![캡처2](https://github.com/user-attachments/assets/57fcd777-0cf8-45af-acfc-2629d43b132d)

#### 3.4 GAT 아키텍처 상세
![캡처2](https://github.com/user-attachments/assets/55b202f5-9101-4aba-96c7-57f3dfa974ad)

#### 3.5 성능지표
![캡처2](https://github.com/user-attachments/assets/55c07ef8-999e-47b6-a669-a02e509a0377)

<br>
<br>

### 4. 프로젝트 수행 과정
#### 4.1 데이터 정보
* 금융결제원 가상환경 내에 제공된 데이터 사용
    * 가명처리된 금융데이터 명세서 
    * 전자금융공동망 홈,펌뱅킹 이체정보
    * 개인용컴퓨터 및 전화(휴대폰 포함) 등을 이용하여 계좌이체를 한 내역
    * 전체 데이터의 0.1% 샘플링한 데이터
* 데이터 수집 기간
    * 2023년 1월 1일 ~ 2023년 12월 31일
* 총 거래 건수
    * 9,947,307

#### 4.2 컬럼 상세
![캡처2](https://github.com/user-attachments/assets/b76d6b89-eb6a-41fe-89e2-5bfb9a8b3a8c)


#### 4.3 데이터 전처리
* 날짜 데이터 반영
GAT와 CatBoost 모델에 datetime형태의 데이터를 직접 입력으로 사용할 수 없음 MinMaxScaler()로 정규화처리

* 금융기관이 다른 중복된 계좌정보 존재
금융기관정보와 계좌번호 정보를 하나의 컬럼으로 결합하여 금융기관_계좌번호 형식으로  변환
각각의 계좌가 개별성을 띠도록 함

* 범주형 데이터 변환
출금계좌, 입금계좌, 자금목적, 거래매체를 category형태로 변환

* 사기여부컬럼을 0과 1로 라벨링(ground truth)
01, 02로 라벨링 된 데이터에 대해 정상거래로 취급
정상거래를 0으로, 사기의심거래를 1로 처리

#### 4.4 데이터 특징
* 정상거래 99.98 % 인  상당히 불균형한 데이터

#### 4.5 GNN의 종류
![캡처2](https://github.com/user-attachments/assets/c56c5a47-2970-4b8a-8b0e-ac5e1dee8b13)

#### 4.6 GAT (Graph Attention Networks)의 구조 및 간략한 특징
* 전파 모듈 ( Propagation Modules )
    * 그래프 내에서 정보를 전달하는 역할을 수행하는 핵심 모듈
* Attention 메커니즘 활용
    * 각 노드가 이웃 노드들로부터 받을 정보를 다르게 가중치를 부여하여 학습하는 방식
    * 여러 개의 어텐션 메커니즘을 병렬 적용하여 안정성과 표현력을 향상 
    * 최종 출력은 여러 헤드의 평균 또는 연결(Concat)
* PyG
    * 메시지 전파(MessagePassing)를 자동으로 처리
    * 메시지 전달 그래프 신경망을 만드는데 도움이 되는, MessagePassing 기본 클래스 제공

#### 4.7 GNN 기반 사기 탐지를 위한 그래프 데이터 전처리 및 특성 엔지니어링
![캡처2](https://github.com/user-attachments/assets/7c4d0d7a-dd08-4c4a-b9ad-3dcf0d505afa)

##### 4.7.1 거래금액
![캡처2](https://github.com/user-attachments/assets/0bcb90b3-254c-4c59-9de1-7424f30464cb)

##### 4.7.1 피쳐
![캡처2](https://github.com/user-attachments/assets/ff383840-72c7-46a3-83fd-fb76ab5f3e0f)
##### 4.7.1 피쳐 선택
![캡처2](https://github.com/user-attachments/assets/08afbd9a-115c-4a08-9e08-3ef01e6524e9)

#### 4.8 파라미터조정 및 EDA 결과
![캡처2](https://github.com/user-attachments/assets/948541b3-5a5d-432d-a863-4818d35d86a4)
![캡처2](https://github.com/user-attachments/assets/56b0fd27-9473-4eb1-8f7f-cd3977c895f1)
![캡처2](https://github.com/user-attachments/assets/b38eb28b-b9a6-4e43-ab8a-c13f6391385e)
![캡처2](https://github.com/user-attachments/assets/a86ee310-35f3-4f5b-808b-1c2e022af9c4)


<br>
<br>



---------------------

📊 프로젝트 결과 분석
---
### GAT(Improved model) vs CatBoost
- 동일한 전처리와 피처 사용 후 모델링
- Sampling에서 GAT는 정상거래의 1% undersampling, 사기거래의 oversampling이 이루어졌고, CatBoost모델은 훈련데이터의 정상거래의 1%로 언더샘플링만 진행

### 성능비교
|Model|k|thresh|Pre@K|Rec@K|f1@K|
|---|---|---|---|---|---|
|GAT(Improved model)|3000|0.8800|0.6223|0.2418|0.3483|
|CatBoost|150|0.6297|1.0000|0.6329|0.7752|


### 성능비교 후 고찰
- GATImproved: 많은 탐지량 대비 정밀도가 양호. 높은 임계값(threshold)에서 비교적 높은 precision 유지
- Catboost: 정밀도(precision) 최적화에 강함. 적은 양을 탐지하더라도 핵심적인 사기 건을 놓치지 않을 가능성이 큼.

<br>

### ➡️ GAT(네트워크 기반 확장탐지) + CatBoost(정밀탐지) 앙상블 구조 

<br>
<br>


⚠️ 구현시 어려움 & 해결방안
---
### 제한된 환경에서의 진행
  - NVIDIA A40 GPU 4GB인 환경에서 가능한 다양한 실험들을 진행하였음.

### 시간적 제약
  - 어려움: SQL로 최근 시간대/거래일자별 거래 건수 및 금액 쿼리 시도 → 데이터 규모 과다로 장시간 소요, 결과 확보 실패
  - 향후 적용가능한 해결방안: 병렬처리 기법을 도입. 데이터 전처리 작업 최적화. 빅데이터 프레임워크(Spark 등) 활용

### GAT 및 GNN의 복잡성
  - 어려움: 복잡한 GNN (GAT 포함) 개념 이해 및 구현에 상당한 시간 소요, 모델 적용 과정에서 잦은 오류 발생
  - 해결 방안: 단계별 학습 및 실습, 팀/멘토 협력 통한 오류 해결, 코드 리뷰/테스트 케이스 활용

<br>
<br>


🚀 향후 발전 가능성
---
### 모델 확장 및 최적화
  - 더 깊은 아키텍처 탐색: 모델의 예측 성능 향상을 위해 다양한 모델의 아키텍처를 실험하고 적용
  - 하이퍼 파라미터 최적화: 모델의 성능을 극대화하기 위해 Bayesian Optimization, Grid Search 등 다양한 최적화 기법을 활용

### 데이터 활용 확대
  - 실시간 데이터 스트리밍: 실시간으로 유입되는 거래 데이터를 분석하고 모델에 적용하여 이상 징후를 즉각적으로 탐지
  - 추가 데이터 소스 통합: 외부 금융 정보 등 다양한 데이터 소스를 통합하여 모델의 예측 정확도를 향상

### 성능 평가 개선
  - 도메인 특화 지표 개발: 금융 거래의 특성을 더욱 잘 반영하는 맞춤형 성능 평가 지표를 개발하여 모델의 실제 효용성을 정확하게 평가
  - Top-K 성능평가 활용: 사기 발생 가능성이 높은 상위 K개 거래에 대한 모델의 예측 성능을 집중적으로 평가하여 실제 대응 전략 수립에 활용

<br>
<br>

## Appendix
[🔗최종발표 pdf](https://drive.google.com/drive/u/0/folders/1HKtCd3yKRaUH9TcDUyX4qIQE2lCwleaK)
