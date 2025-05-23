## train.py
import torch
import numpy as np
from model import GAT
from dataset import AMLtoGraph
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score

# GPU/CPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 로드
dataset = AMLtoGraph('./lej_dataset_002')
# 분할된 데이터셋을 각각 로드
train_data = dataset.data_train.to(device)
val_data = dataset.data_val.to(device)
test_data = dataset.data_test.to(device)

# 사기 계좌 비율 확인 (불균형 데이터 보정) - train 데이터 기준
fraud_ratio = (train_data.y == 1).sum().item() / train_data.y.shape[0]
pos_weight = torch.tensor([1.0 / fraud_ratio]).to(device)  # 불균형 보정

# 모델 초기화 (edge_attr 활용)
model = GAT(
    in_channels=train_data.num_features, # train data 기준으로 input channel 설정
    edge_dim=train_data.edge_attr.shape[1], # train data 기준으로 edge dimension 설정
    hidden_channels=16,
    out_channels=1,
    heads=8
).to(device)

# 손실 함수 및 옵티마이저 설정
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 불균형 보정 적용
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# 데이터 로더 설정
train_loader = NeighborLoader(train_data, num_neighbors=[30] * 2, batch_size=256, shuffle=True)
val_loader = NeighborLoader(val_data, num_neighbors=[30] * 2, batch_size=256, shuffle=False)  # 검증 로더
test_loader = NeighborLoader(test_data, num_neighbors=[30] * 2, batch_size=256, shuffle=False) # 테스트 로더

# Top-K 평가 함수
def top_k_indices(y_score, k):
    """ 예측 점수(y_score)에서 상위 K개 인덱스 가져오기 """
    return np.argsort(y_score)[-k:]

def precision_at_k(y_true, y_score, k=30):
    """ Precision@K 계산: 상위 K개 예측 중 실제 사기 비율 """
    top_k_idx = top_k_indices(y_score, k)
    y_top_k_pred = (y_score[top_k_idx] > 0.5).astype(int)
    y_top_k_true = y_true[top_k_idx]
    return precision_score(y_top_k_true, y_top_k_pred, average='binary')

def accuracy_at_k(y_true, y_score, k=30):
    """ Accuracy@K 계산: 상위 K개 예측 중 맞춘 개수 """
    top_k_idx = top_k_indices(y_score, k)
    y_top_k_pred = (y_score[top_k_idx] > 0.5).astype(int)
    y_top_k_true = y_true[top_k_idx]
    return accuracy_score(y_top_k_true, y_top_k_pred)

# 학습 루프 설정
num_epochs = 30
best_f1 = 0  # 최고 성능 저장

for epoch in range(num_epochs):
    total_loss = 0
    model.train()

    for batch in train_loader:
        optimizer.zero_grad()
        batch.to(device)

        # 모델 예측 (sigmoid 미적용)
        pred = model(batch.x, batch.edge_index, batch.edge_attr).view(-1)

        # 손실 계산 (이진 분류)
        loss = criterion(pred, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 🔹 10 epoch마다 평가 실행
    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d} | Loss: {total_loss:.4f}')
        model.eval()

        # validation loop
        with torch.no_grad():
            all_preds, all_labels = [], []
            for batch in val_loader:
                batch.to(device)
                pred = torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr).view(-1))
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch.y.cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='binary')
            precision = precision_score(all_labels, (all_preds > 0.5).astype(int), average='binary')
            roc_auc = roc_auc_score(all_labels, all_preds)

            top_k_acc = accuracy_at_k(all_labels, all_preds, k=30)
            top_k_prec = precision_at_k(all_labels, all_preds, k=30)

            print(f'[Validation] F1-score: {f1:.4f}, Precision: {precision:.4f}, ROC-AUC: {roc_auc:.4f}')
            print(f'[Validation] Top-30 Accuracy: {top_k_acc:.4f}, Top-30 Precision: {top_k_prec:.4f}')


        # Test loop (학습 과정에서 Test는 일반적으로 수행하지 않음.  최적 모델 선택 후 Test)
        with torch.no_grad():
            all_preds, all_labels = [], []
            for batch in test_loader:
                batch.to(device)
                pred = torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr).view(-1))
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch.y.cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='binary')
            precision = precision_score(all_labels, (all_preds > 0.5).astype(int), average='binary')
            roc_auc = roc_auc_score(all_labels, all_preds)

            top_k_acc = accuracy_at_k(all_labels, all_preds, k=30)
            top_k_prec = precision_at_k(all_labels, all_preds, k=30)

            print(f'[Test] F1-score: {f1:.4f}, Precision: {precision:.4f}, ROC-AUC: {roc_auc:.4f}')
            print(f'[Test] Top-30 Accuracy: {top_k_acc:.4f}, Top-30 Precision: {top_k_prec:.4f}')


# dataset.py
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
        """데이터 전처리 및 시간 정규화 (Train 기준으로 Scaling)"""

        # 🔹 결측값 처리
        df = df.fillna({'tran_amt': 0, 'md_type': '00', 'fnd_type': '00'}, inplace=False)

        # 🔹 사기 거래(SP) 레이블 변환
        df['ff_sp_ai'] = df['ff_sp_ai'].apply(lambda x: 1 if x == 'SP' else 0)

        # 🔹 거래 금액이 10,000 이상인 데이터만 유지
        df = df[df["tran_amt"] >= 10000]

        # 🔹 날짜 변환 (DateTime 형식으로 변환)
        df["tran_dt"] = pd.to_datetime(df["tran_dt"], format="%Y%m%d")

        # 🔹 데이터 분할 (정규화 전에 먼저!)
        train_df = df[(df['tran_dt'].dt.year < 2023) |
                      ((df['tran_dt'].dt.year == 2023) & (df['tran_dt'].dt.month <= 10))]
        val_df = df[(df['tran_dt'].dt.year == 2023) & (df['tran_dt'].dt.month == 11)]
        test_df = df[(df['tran_dt'].dt.year == 2023) & (df['tran_dt'].dt.month > 11)]

        # ✅ Train 데이터 기준으로 정규화 기준 학습 (fit)
        scaler = MinMaxScaler()
        train_df['tran_dt'] = scaler.fit_transform(train_df[['tran_dt']])

        # ✅ 동일한 정규화 기준을 Validation/Test에도 적용 (transform)
        val_df['tran_dt'] = scaler.transform(val_df[['tran_dt']])
        test_df['tran_dt'] = scaler.transform(test_df[['tran_dt']])

        return train_df, val_df, test_df



    def get_edge_df(self, df):
        """엣지 정보 생성"""
        df['wd_fc_ac'] = df['wd_fc_ac'].astype('category').cat.codes
        df['dps_fc_ac'] = df['dps_fc_ac'].astype('category').cat.codes
        edge_index = torch.tensor([df['wd_fc_ac'].values, df['dps_fc_ac'].values], dtype=torch.long)

        edge_features = ['tran_amt', 'md_type', 'fnd_type', 'tran_dt', 'tran_tmrg',
                         'dps_fc_ac_fnd_amt', 'dps_fc_ac_md_amt', 'dps_fc_ac_md_cnt',
                         'wd_fc_ac_fnd_amt', 'wd_fc_ac_md_amt', 'wd_fc_ac_md_cnt']

        edge_attr = torch.tensor(df[edge_features].values, dtype=torch.float)
        return edge_attr, edge_index
		
    def get_node_attr(self, df):
        """노드 속성 생성"""
        node_features = ["tran_dt", "tran_amt", "tran_tmrg"]
        node_attr = torch.tensor(df[node_features].values, dtype=torch.float)
        node_label = torch.tensor(df['ff_sp_ai'].values, dtype=torch.float)
        return node_attr, node_label
		
		
		
		
		

    def get_betweenness_tensor(self, nx_graph, accounts):
        betweenness = nx.betweenness_centrality(nx_graph)
        values = [betweenness.get(i, 0) for i in range(accounts.shape[0])]
        return torch.tensor(values, dtype=torch.float).unsqueeze(1)

    def get_degree_tensor(self, nx_graph, accounts):
        degree = nx.degree_centrality(nx_graph)
        values = [degree.get(i, 0) for i in range(accounts.shape[0])]
        return torch.tensor(values, dtype=torch.float).unsqueeze(1)

    def get_closeness_tensor(self, nx_graph, accounts):
        try:
            closeness = nx.closeness_centrality(nx_graph)
        except nx.NetworkXError:
            closeness = {}
        values = [closeness.get(i, 0) for i in range(accounts.shape[0])]
        return torch.tensor(values, dtype=torch.float).unsqueeze(1)

    def get_transaction_frequency_tensor(self, nx_graph, accounts):
        degree = nx.degree_centrality(nx_graph)
        values = [degree.get(i, 0) for i in range(accounts.shape[0])]
        return torch.tensor(values, dtype=torch.float).unsqueeze(1)

    def get_in_out_degree_ratio_tensor(self, nx_graph, accounts):
        in_degree = nx.in_degree_centrality(nx_graph)
        out_degree = nx.out_degree_centrality(nx_graph)
        in_values = [in_degree.get(i, 0) for i in range(accounts.shape[0])]
        out_values = [out_degree.get(i, 0) for i in range(accounts.shape[0])]
        in_tensor = torch.tensor(in_values, dtype=torch.float).unsqueeze(1)
        out_tensor = torch.tensor(out_values, dtype=torch.float).unsqueeze(1)
        return (out_tensor + 1e-6) / (in_tensor + 1e-6)

    def get_pagerank_tensor(self, nx_graph, accounts):
        pagerank = nx.pagerank(nx_graph)
        values = [pagerank.get(i, 0) for i in range(accounts.shape[0])]
        return torch.tensor(values, dtype=torch.float).unsqueeze(1)

    def get_clustering_coefficient_tensor(self, nx_graph, accounts):
        clustering = nx.clustering(nx_graph)
        values = [clustering.get(i, 0) for i in range(accounts.shape[0])]
        return torch.tensor(values, dtype=torch.float).unsqueeze(1)

    def get_avg_neighbor_degree_tensor(self, nx_graph, accounts):
        avg_neighbor_degree = nx.average_neighbor_degree(nx_graph)
        values = [avg_neighbor_degree.get(i, 0) for i in range(accounts.shape[0])]
        return torch.tensor(values, dtype=torch.float).unsqueeze(1)

    def get_shortest_path_related_tensor(self, nx_graph, accounts):
        try:
            avg_shortest_path = nx.average_shortest_path_length(nx_graph)
        except nx.NetworkXError:
            avg_shortest_path = 0
        try:
            diameter = nx.diameter(nx_graph)
        except nx.NetworkXError:
            diameter = 0
        return torch.tensor([[avg_shortest_path, diameter]] * accounts.shape[0], dtype=torch.float)
		
		
    def sample_data(self, df):
        """데이터 샘플링 (정상 거래 30%, 사기 거래 정상의 1:1 맞추기)"""
        normal_data = df[df['ff_sp_ai'] == 0]
        fraud_data = df[df['ff_sp_ai'] == 1]

        # ✅ 1. 정상 거래 언더샘플링 (30% 유지)
        normal_data_sampled = normal_data.sample(frac=0.30, random_state=42)

        # ✅ 2. 사기 거래 개수를 정상 거래의 1/3로 맞춤
        fraud_target_count = len(normal_data_sampled)

        # ✅ 3. 사기 거래 샘플링 (복원 샘플링 적용 가능)
        fraud_data_sampled = resample(
            fraud_data, 
            replace=True if fraud_target_count > len(fraud_data) else False, 
            n_samples=fraud_target_count, 
            random_state=42
        )

        # ✅ 4. 샘플링된 데이터 병합 후 섞기
        df_resampled = pd.concat([normal_data_sampled, fraud_data_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

        return df_resampled
		
		

    def process(self):
        """데이터 처리 및 학습/검증/테스트 세트 분할"""
        df = pd.read_csv(self.raw_paths[0])

        # 🔹 날짜 기준으로 train/val/test 분할 후 정규화 적용
        train_df, val_df, test_df = self.preprocess(df)

        # ✅ 데이터 샘플링 적용
        datasets = {'train': train_df, 'val': val_df, 'test': test_df}
        processed_data = {}
        for key, df in datasets.items():
            df = self.sample_data(df)  # 샘플링 적용
            processed_data[key] = self.create_graph_data(df)  # 그래프 데이터 생성

        # 🔹 저장할 데이터 튜플 생성 및 저장
        torch.save((processed_data['train'], None, 
                    processed_data['val'], None, 
                    processed_data['test'], None), self.processed_paths[0])

    def create_graph_data(self, df):
		
        """그래프 데이터 생성"""
        edge_attr, edge_index = self.get_edge_df(df)  
        node_attr, node_label = self.get_node_attr(df)

        # 노드 속성 추가 (중요!)
        # node_attr = torch.tensor(df[['tran_dt', 'tran_amt']].values, dtype=torch.float)

        # NetworkX 그래프 생성
        # nx_graph = nx.Graph()
        # edges = edge_index.T.tolist()
        # nx_graph.add_edges_from(edges)

        # 노드 속성 추가
        # node_attr = torch.cat([self.get_betweenness_tensor(nx_graph, df)], dim=1)
        # node_attr = torch.cat([node_attr, self.get_degree_tensor(nx_graph, df)], dim=1)
        # node_attr = torch.cat([node_attr, self.get_closeness_tensor(nx_graph, df)], dim=1)
        # node_attr = torch.cat([node_attr, self.get_transaction_frequency_tensor(nx_graph, df)], dim=1)
        # node_attr = torch.cat([node_attr, self.get_in_out_degree_ratio_tensor(nx_graph, df)], dim=1)
        # node_attr = torch.cat([node_attr, self.get_pagerank_tensor(nx_graph, df)], dim=1)
        # node_attr = torch.cat([node_attr, self.get_clustering_coefficient_tensor(nx_graph, df)], dim=1)
        # node_attr = torch.cat([node_attr, self.get_avg_neighbor_degree_tensor(nx_graph, df)], dim=1)
        # node_attr = torch.cat([node_attr, self.get_shortest_path_related_tensor(nx_graph, df)], dim=1)

        # PyTorch Geometric 데이터 변환
        data = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, y=node_label)
		
        return data
		
# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear

class GAT(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, out_channels, heads):
        super().__init__()

        # 첫 번째 GATv2 레이어 (Edge 특성 반영)
        self.conv1 = GATv2Conv(
            in_channels, hidden_channels, heads=heads, dropout=0.6, edge_dim=edge_dim
        )

        # 두 번째 GATv2 레이어 (Edge 특성 반영)
        self.conv2 = GATv2Conv(
            hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.6, edge_dim=edge_dim
        )

        # 출력 레이어
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # 첫 번째 GATv2 레이어 통과
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))

        # 두 번째 GATv2 레이어 통과
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr))

        # 최종 선형 레이어 (BCEWithLogitsLoss 사용 → sigmoid 미적용)
        return self.lin(x)



