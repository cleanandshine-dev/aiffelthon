## train.py
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, roc_auc_score
from model11 import GATImproved
from dataset11 import AMLtoGraph
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = AMLtoGraph("./lej_dataset_002")
train_loader = DataLoader([dataset.data_train], batch_size=1, shuffle=True)
val_loader = DataLoader([dataset.data_val], batch_size=1, shuffle=False)

model = GATImproved(
    in_channels=dataset.data_train.num_node_features,
    edge_dim=dataset.data_train.edge_attr.shape[1],
    out_channels=1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()

best_f1 = 0
for epoch in range(30):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr).view(-1)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    preds, labels, dates = [], [], []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr).view(-1)
            score = torch.sigmoid(out).cpu().numpy()
            label = data.y.cpu().numpy()
            date = data.x[:, 0].cpu().numpy()
            preds.extend(score)
            labels.extend(label)
            dates.extend(date)

    bin_preds = (np.array(preds) > 0.5).astype(int)
    f1 = f1_score(labels, bin_preds)
    precision = precision_score(labels, bin_preds)
    roc = roc_auc_score(labels, preds)

    print(f"Epoch {epoch:02d} | Loss: {total_loss:.4f}")
    print(f"[Val] F1: {f1:.4f}, Precision: {precision:.4f}, ROC-AUC: {roc:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_model.pth")
        np.save("val_preds.npy", preds)
        np.save("val_labels.npy", labels)
        np.save("val_tran_dt.npy", dates)
        print("Best model and predictions saved.")



# dataset.py
import os
import pandas as pd
import numpy as np
import torch
import networkx as nx
from typing import Optional, Callable
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from networkx.algorithms import community  # 추가된 부분


class AMLtoGraph(InMemoryDataset):
    def __init__(self, root: str, edge_window_size: int = 10,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.edge_window_size = edge_window_size
        super().__init__(root, transform, pre_transform)
        print("AMLtoGraph initialized!")
        self.data_train, _, self.data_val, _, self.data_test, _ = torch.load(self.processed_paths[0])
        print("Processed data loaded!")

    @property
    def raw_file_names(self): return 'hf_trns_tran_tmp_1_fraud_cnt_new.csv'
    
    @property
    def processed_file_names(self):
        return 'data.pt'



    def detect_communities(self, df):
        print("커뮤니티 탐지 시작")
        G = nx.Graph()

        # 그래프 생성: wd_fc_ac와 dps_fc_ac를 엣지로 추가
        for _, row in df.iterrows():
            G.add_edge(row['wd_fc_ac'], row['dps_fc_ac'], weight=row['tran_amt'])

        # Louvain 방법을 사용하여 커뮤니티 탐지
        communities = community.louvain_communities(G, weight='weight')
        
        community_map = {}
        for community_number, community_nodes in enumerate(communities):
            for node in community_nodes:
                community_map[node] = community_number

        # 커뮤니티 정보를 DataFrame에 추가 (노드 번호에 따라 매핑)
        df['community'] = df['dps_fc_ac'].map(community_map)

        return df
    # end of def detect_communities


    def preprocess(self, df):
        print("Preprocessing 시작")
        df.fillna(0, inplace=True)
        df['ff_sp_ai'] = df['ff_sp_ai'].apply(lambda x: 1 if x == 'SP' else 0)

        df = df[df["tran_amt"] >= 10000]
        df["tran_dt_raw"] = pd.to_datetime(df["tran_dt"], format="%Y%m%d")

        # 커뮤니티 탐지 추가
        df = self.detect_communities(df)

        train_df = df[(df['tran_dt_raw'].dt.year < 2023) |
                      ((df['tran_dt_raw'].dt.year == 2023) & (df['tran_dt_raw'].dt.month <= 10))].copy()
        val_df = df[(df['tran_dt_raw'].dt.year == 2023) & (df['tran_dt_raw'].dt.month == 11)].copy()
        test_df = df[(df['tran_dt_raw'].dt.year == 2023) & (df['tran_dt_raw'].dt.month > 11)].copy()

        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        for split_df in [train_df, val_df, test_df]:
            split_df.loc[:, 'tran_dt'] = split_df['tran_dt_raw'].values.astype(np.int64) // 10**9

        scaler = MinMaxScaler()
        train_df.loc[:, 'tran_dt'] = scaler.fit_transform(train_df[['tran_dt']])
        val_df.loc[:, 'tran_dt'] = scaler.transform(val_df[['tran_dt']])
        test_df.loc[:, 'tran_dt'] = scaler.transform(test_df[['tran_dt']])

        return train_df, val_df, test_df

    def sample_data(self, df, verbose: bool = False):
        print("샘플링 시작")
        def print_graph_stats(df_part, label=""):
            G = nx.Graph()
            for _, row in df_part.iterrows():
                G.add_edge(row['wd_fc_ac'], row['dps_fc_ac'], weight=row['tran_amt'])
            print(f"[{label}] 노드 수: {G.number_of_nodes()}, 엣지 수: {G.number_of_edges()}")

        normal_data = df[df['ff_sp_ai'] == 0]
        fraud_data = df[df['ff_sp_ai'] == 1]

        print(f"원래 정상 거래 수: {len(normal_data)}, 사기 거래 수: {len(fraud_data)}")

        normal_sampled = normal_data.sample(frac=0.01, random_state=42)
        fraud_sampled = resample(fraud_data, replace=True, n_samples=len(normal_sampled), random_state=42)

        df_sampled = pd.concat([normal_sampled, fraud_sampled], ignore_index=True).sample(frac=1, random_state=42)

        print(f"샘플링 후 정상 거래 수: {len(normal_sampled)}, 사기 거래 수: {len(fraud_sampled)}")

        if verbose:
            print_graph_stats(df, "샘플링 전 전체 데이터")
            print_graph_stats(df_sampled, "샘플링 후 데이터")

        return df_sampled

    def get_edge_df(self, df):
        print("엣지 생성 시작")
        df['wd_fc_ac'] = df['wd_fc_ac'].astype('category').cat.codes
        df['dps_fc_ac'] = df['dps_fc_ac'].astype('category').cat.codes

        edge_index = torch.tensor([df['wd_fc_ac'].values, df['dps_fc_ac'].values], dtype=torch.long)

        edge_feat_cols = [
            'tran_amt', 'tran_dt', 'tran_tmrg',
            'dps_fc_ac_fnd_amt', 'dps_fc_ac_fnd_cnt', 'dps_fc_ac_md_amt', 'dps_fc_ac_md_cnt',
            'wd_fc_ac_fnd_amt', 'wd_fc_ac_fnd_cnt', 'wd_fc_ac_md_amt', 'wd_fc_ac_md_cnt'
        ]

        edge_attr = torch.tensor(df[edge_feat_cols].values, dtype=torch.float)
        print(f"엣지 수: {edge_index.shape[1]}, 엣지 특성 차원: {edge_attr.shape[1]}")
        return edge_attr, edge_index

    def get_node_attr(self, df):
        print("노드 특성 생성 시작")
        df.fillna(0, inplace=True)
        node_cols = [
            'tran_dt', 'tran_amt', 'tran_tmrg',
            'md_type', 'fnd_type',
            'prev_dps_fraud_cnt', 'prev_wd_fraud_cnt',
            'dps_fc_ac_fnd_amt', 'dps_fc_ac_fnd_cnt', 'dps_fc_ac_md_amt', 'dps_fc_ac_md_cnt',
            'wd_fc_ac_fnd_amt', 'wd_fc_ac_fnd_cnt', 'wd_fc_ac_md_amt', 'wd_fc_ac_md_cnt', 'community'
        ]
        node_attr = torch.tensor(df[node_cols].values, dtype=torch.float)
        node_label = torch.tensor(df['ff_sp_ai'].values, dtype=torch.float)
        print(f"노드 수: {node_attr.shape[0]}, 노드 특성 차원: {node_attr.shape[1]}")
        return node_attr, node_label

    def process(self):
        print("데이터 처리 시작")
        print(f"CSV 로딩 경로: {self.raw_paths[0]}")
        df = pd.read_csv(self.raw_paths[0], low_memory=False)
        train_df, val_df, test_df = self.preprocess(df)

        np.save(os.path.join(self.processed_dir, "tran_dt_raw_val.npy"), val_df['tran_dt_raw'].values.astype(str))
        np.save(os.path.join(self.processed_dir, "tran_dt_raw_test.npy"), test_df['tran_dt_raw'].values.astype(str))

        datasets = {
            'train': self.sample_data(train_df),
            'val': self.sample_data(val_df),
            'test': test_df  # 테스트는 원본 유지
        }

        print("그래프 변환 시작")
        processed_data = {k: self.create_graph_data(d) for k, d in datasets.items()}
        torch.save((processed_data['train'], torch.tensor([]),
                    processed_data['val'], torch.tensor([]),
                    processed_data['test'], torch.tensor([])), self.processed_paths[0])
        print("데이터 저장 완료!")

    def create_graph_data(self, df):
        edge_attr, edge_index = self.get_edge_df(df)
        node_attr, node_label = self.get_node_attr(df)
        return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, y=node_label)

		
# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear

class GATImproved(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels=64, out_channels=1, heads=4):
        super().__init__()
        self.lin_in = Linear(in_channels, hidden_channels)

        # GAT layer 1
        self.gat1 = GATv2Conv(hidden_channels, hidden_channels, heads=heads,
                              dropout=0.5, edge_dim=edge_dim)
        self.res_lin1 = Linear(hidden_channels, hidden_channels * heads)  # residual for GAT1
        self.norm1 = nn.BatchNorm1d(hidden_channels * heads)

        # GAT layer 2
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False,
                              dropout=0.5, edge_dim=edge_dim)
        self.res_lin2 = Linear(hidden_channels * heads, hidden_channels)  # residual for GAT2
        self.norm2 = nn.BatchNorm1d(hidden_channels)

        self.lin_out = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.lin_in(x))
        x = F.dropout(x, p=0.5, training=self.training)

        # GAT layer 1 + Residual
        x1 = self.gat1(x, edge_index, edge_attr)
        x = F.relu(self.norm1(x1 + self.res_lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)

        # GAT layer 2 + Residual
        x2 = self.gat2(x, edge_index, edge_attr)
        x = F.relu(self.norm2(x2 + self.res_lin2(x)))

        return self.lin_out(x)




