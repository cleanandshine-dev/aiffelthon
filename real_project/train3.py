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
data = dataset[0].to(device)

# 모델 초기화 (이진 분류 설정)
model = GAT(
    in_channels=data.num_features,
    hidden_channels=16,
    out_channels=1,
    heads=8
).to(device)

# 손실 함수 및 옵티마이저
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# 학습/검증 데이터 분할
split = T.RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0)
data = split(data)

# 데이터 로더 설정
train_loader = NeighborLoader(
    data, num_neighbors=[30] * 2, batch_size=256, shuffle=True
)

test_loader = NeighborLoader(
    data, num_neighbors=[30] * 2, batch_size=256, shuffle=False
)

# 학습 루프 설정
epoch = 30  

for i in range(epoch):
    total_loss = 0
    model.train()

    for batch in train_loader:
        optimizer.zero_grad()
        batch.to(device)

        pred = model(batch.x, batch.edge_index, batch.edge_attr).view(-1)
        loss = criterion(pred, batch.y.float())  
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if i % 10 == 0:
        print(f'Epoch {i:03d} | Loss: {total_loss:.4f}')
        model.eval()

        with torch.no_grad():
            all_preds, all_labels = [], []

            for batch in test_loader:
                batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr).view(-1)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch.y.cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            all_labels = (all_labels > 0).astype(int)
            all_preds = (all_preds > 0.5).astype(int)

            f1 = f1_score(all_labels, all_preds, average='binary')
            precision = precision_score(all_labels, all_preds, average='binary')
            roc_auc = roc_auc_score(all_labels, all_preds)

            print(f'F1-score: {f1:.4f}, Precision: {precision:.4f}, ROC-AUC: {roc_auc:.4f}')
