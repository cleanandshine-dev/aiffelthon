```
📂 etc
|
├── README.md    # 디렉토리 설명
├── [NetworkX]Credit_card_edges_classification.ipynb
├──
...
│
└──
```

📄 [NetworkX]Credit_card_edges_classification.ipynb
- 목적: 해당 코드를 참고하여 AML모델링과 금결원 데이터에 응용시도
- 신용카드 거래데이터를 이용한 사기탐지모델링
- networkx 라이브러리를 이용해 신용카드 거래 그래프 생성,
- 제공코드를 기반으로 함. 다만, 원본코드가 4년전에 작성된 코드라 에러나는 부분을 수정함.
- 특징
  1. 정상거래와 사기거래의 비율이 99.5%, 0.5%로 매우 불균형함. IBM AML 합성데이터에서 LI-Small_Trans에서의 비율과 비슷
  2. 총 129만 6천개의 데이터가 존재하지만, 전체를 사용하지 않고, 정상거래데이터 20%와 전체 사기거래 데이터를 사용하여 약 29만여개의 데이터 사용
  3. 23개의 컬럼이 존재하나 모든 컬럼을 사용하지 않고, `index`, `cc_num`, `merchant`, `amt`,`is_fraud`의 컬럼만 사용
  4. 이분(bipartite) 그래프 구축 및 시각화했으나 데이터가 많아 의미있는 시각화는 아님
  5. 네트워크 분석(Network Analysis)에서 노드와 엣지의 정보를 확인하고 degree, weight, degree_centrality, closeness_centrality 등을 히스토그램(Frequency)으로 시각화
  6. 커뮤니티 탐지(Community Detection)를 통해 엣지 특성을 집계해 추출 및 시각화
  7. 지도학습을 이용한 모델링
     - model: RandomForest
     - edge embedding: HadamardEmbedder, AverageEmbedder, WeightedL1Embedder, WeightedL2Embedder
     - evaluation: Precision, Recall, F1-Score
  8. 비지도학습을 이용한 모델링
     - model: KMeans clustering
     - edge embedding: HadamardEmbedder, AverageEmbedder, WeightedL1Embedder, WeightedL2Embedder
     - evaluation: NMI, Homogeneity, Completeness, V-Measure 
- 주요 Requirements(python 3.8)
  ```
  community==1.0.0b1
  gensim==3.8.3
  matplotlib==3.7.5
  matplotlib-inline==0.1.7
  networkx==3.1
  node2vec==0.3.3
  numpy==1.24.4
  pandas==2.0.3
  python-louvain==0.16
  scikit-learn==0.24.0
  scipy==1.9.3

  ```
- Reference
  - Data: [fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
  - Code: [Credit_card_edges_classification.ipynb](https://github.com/PacktPublishing/Graph-Machine-Learning/blob/main/Chapter08/01_Credit_card_edges_classification.ipynb)
