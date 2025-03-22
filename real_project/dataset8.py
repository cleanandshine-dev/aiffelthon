import datetime
import os
from typing import Optional, Callable
import pandas as pd
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# 🔹 GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# -------------------
# 1. 데이터 로딩 클래스
# -------------------
class AMLtoGraph(InMemoryDataset):
    def __init__(self, root: str, edge_window_size: int = 10,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.edge_window_size = edge_window_size
        super().__init__(root, transform, pre_transform)
        self.data_train, _, self.data_val, _, self.data_test, _ = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return 'HF_TRNS_TRAN_new.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def preprocess(self, df):
        df.fillna('00', inplace=True)
        df['ff_sp_ai'] = df['ff_sp_ai'].apply(lambda x: 1 if x == 'SP' else 0)
        df = df[df["tran_amt"] >= 10000]
        df["tran_dt"] = pd.to_datetime(df["tran_dt"], format="%Y%m%d")
        train_df = df[(df['tran_dt'].dt.year < 2023) | ((df['tran_dt'].dt.year == 2023) & (df['tran_dt'].dt.month <= 10))]
        val_df = df[(df['tran_dt'].dt.year == 2023) & (df['tran_dt'].dt.month == 11)]
        test_df = df[(df['tran_dt'].dt.year == 2023) & (df['tran_dt'].dt.month > 11)]
        scaler = MinMaxScaler()
        train_df['tran_dt'] = scaler.fit_transform(train_df[['tran_dt']])
        val_df['tran_dt'] = scaler.transform(val_df[['tran_dt']])
        test_df['tran_dt'] = scaler.transform(test_df[['tran_dt']])
        return train_df, val_df, test_df

    def sample_data(self, df, verbose: bool = False):
        """정상 거래 30%, 사기 거래를 동일 개수로 맞추는 샘플링 함수"""

        def print_graph_stats(df_part, label=""):
            """그래프 통계 출력 함수"""
            G = nx.Graph()
            for _, row in df_part.iterrows():
                G.add_edge(row['wd_fc_ac'], row['dps_fc_ac'], weight=row['tran_amt'])
            print(f"[{label}] 노드 수: {G.number_of_nodes()}, 엣지 수: {G.number_of_edges()}")

        # 샘플링 전 거래 개수 출력
        normal_data = df[df['ff_sp_ai'] == 0]
        fraud_data = df[df['ff_sp_ai'] == 1]
        print(f"원래 정상 거래 개수: {len(normal_data)}")
        print(f"원래 사기 거래 개수: {len(fraud_data)}")

        # 샘플링
        normal_sampled = normal_data.sample(frac=0.3, random_state=42)
        fraud_sampled = resample(fraud_data, replace=True if len(normal_sampled) > len(fraud_data) else False,
                                 n_samples=len(normal_sampled), random_state=42)

        df_sampled = pd.concat([normal_sampled, fraud_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

        # 샘플링 후 거래 개수 출력
        print(f"샘플링 후 정상 거래 개수: {len(normal_sampled)}")
        print(f"샘플링 후 사기 거래 개수: {len(fraud_sampled)}")

        # (옵션) 그래프 통계 출력
        if verbose:
            print_graph_stats(df, "샘플링 전 전체 데이터")
            print_graph_stats(df_sampled, "샘플링 후 데이터")

        return df_sampled

    def get_edge_df(self, df):
        df['wd_fc_ac'] = df['wd_fc_ac'].astype('category').cat.codes
        df['dps_fc_ac'] = df['dps_fc_ac'].astype('category').cat.codes
        edge_index = torch.tensor([df['wd_fc_ac'].values, df['dps_fc_ac'].values], dtype=torch.long)
        edge_attr = torch.tensor(df[['tran_amt', 'tran_dt', 'tran_tmrg']].values, dtype=torch.float)
        return edge_attr, edge_index

    def get_node_attr(self, df):
        node_attr = torch.tensor(df[["tran_dt", "tran_amt", "tran_tmrg"]].values, dtype=torch.float)
        node_label = torch.tensor(df['ff_sp_ai'].values, dtype=torch.float)
        return node_attr, node_label

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        train_df, val_df, test_df = self.preprocess(df)

        # train과 val만 샘플링 적용, test는 원본 유지
        datasets = {
            'train': self.sample_data(train_df, verbose=False),
            'val': self.sample_data(val_df, verbose=False),
            'test': test_df
        }

        processed_data = {key: self.create_graph_data(df) for key, df in datasets.items()}
        torch.save((processed_data['train'], torch.tensor([]),
                    processed_data['val'], torch.tensor([]),
                    processed_data['test'], torch.tensor([])), self.processed_paths[0])

    def create_graph_data(self, df):
        edge_attr, edge_index = self.get_edge_df(df)
        node_attr, node_label = self.get_node_attr(df)
        return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, y=node_label)


## 빠르게 실행하고 싶을 때	self.sample_data(train_df, verbose=False)
## 디버깅으로 그래프 통계도 보고 싶을 때	self.sample_data(train_df, verbose=True)