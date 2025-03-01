import os
import cudf
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import cugraph
import cupy as cp

# GPU를 위한 Dask CUDA 클러스터 초기화
cluster = LocalCUDACluster()
client = Client(cluster)

# Dask CUDF를 사용하여 데이터 로드
path = '../../../../../archive/raw/HI-Small_Trans.csv'
df = dask_cudf.read_csv(path)

#### 0. 데이터 전처리
def preprocess(df):
    df['Timestamp'] = cudf.to_datetime(df['Timestamp'])  # Timestamp를 날짜타입으로 변환
    df['Timestamp'] = df['Timestamp'].values.astype('datetime64[ms]').astype('int64')
    min_ts = df['Timestamp'].min()
    max_ts = df['Timestamp'].max()
    df['Timestamp'] = (df['Timestamp'] - min_ts) / (max_ts - min_ts)  # 0-1로 정규화

    df['Account'] = df['From Bank'].astype(str) + '_' + df['Account']
    df['Account.1'] = df['To Bank'].astype(str) + '_' + df['Account.1']
    
    df = df.sort_values(by=['Account'])
    receiving_df = df[['Account.1', 'Amount Received', 'Receiving Currency']]
    paying_df = df[['Account', 'Amount Paid', 'Payment Currency']]
    receiving_df = receiving_df.rename({'Account.1': 'Account'}, axis=1)
    currency_ls = sorted(df['Receiving Currency'].unique().compute())  # 통화 목록 생성

    return df, receiving_df, paying_df, currency_ls

df, receiving_df, paying_df, currency_ls = preprocess(df)
df.rename(columns={'Account': 'sender', 'Account.1': 'receiver'}, inplace=True)

#### 1. 정상거래 데이터만 필터링
normal_df = df[df['Is Laundering'] == 0]

# cuGraph를 사용하여 가중치 없는 그래프 생성
normal_df_pd = normal_df.to_pandas()  # Pandas DataFrame으로 변환
g = cugraph.Graph()
g.add_edge_df(normal_df_pd[['sender', 'receiver']])  # cuGraph에 엣지 추가

# 커뮤니티 탐지
communities = cugraph.label_propagation(g)  # 레이블 전파 알고리즘으로 커뮤니티 탐지
print("cuGraph를 사용한 커뮤니티 탐지 완료")

print("\n--- 커뮤니티 탐지 결과 ---")
print("\n1. 커뮤니티 결과:")
print(communities)

#### 2. 커뮤니티 크기 분포 확인
community_sizes = communities['labels'].value_counts().to_frame()  # 커뮤니티 라벨 수 계산
print(f"커뮤니티 통계:")
print(community_sizes.describe())