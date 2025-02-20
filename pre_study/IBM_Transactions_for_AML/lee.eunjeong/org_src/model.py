# model.py

import torch
import torch.nn as nn # neural network 모듈 - 레이어, 손실 함수, 활성화 함수 등 신경망 구축에 필요한 도구 제공
import torch.nn.functional as F # 활성화 함수(예: ReLU, ELU), 드롭아웃, 손실 함수 포함 되어 있음
import torch_geometric.transforms as T # 그래프 데이터를 전처리하거나 변환 ( 고생했던 lib )
from torch_geometric.nn import GATConv, Linear 
# nn : PyTorch Geometric 라이브러리의 neural network 모듈 
# GATConv : Graph Attention Network(GAT)의 컨볼루션(합성곱) 레이어를 제공
# Linear : 완전 연결 레이어(fully connected layer)를 제공. 이 레이어는 입력 데이터를 선형 변환 함.

class GAT(torch.nn.Module):
    # GAT 괄호 안 torch.nn.Module 을 상속.
    # 모든 사용자 정의 모델은 이 클래스를 상속받아야 함. ( 아 그래? )
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        # python에서는 __init__ 이 생성자다!
        # in_channels: 입력 데이터의 차원(특성 수)
        # heads: GAT의 attention head 수를 지정. 멀티헤드 어텐션을 사용할 경우, 여러 개의 attention 메커니즘이 병렬로 작동

        super().__init__() # 부모 클래스 생성자 호출 해라.
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6) # 첫 번째 GAT 컨볼루션 레이어를 정의. 멀티헤드 어텐션 - concat=True(기본값)
        self.conv2 = GATConv(hidden_channels * heads, int(hidden_channels/4), heads=1, concat=False, dropout=0.6) 
        # 두 번째 GAT 컨볼루션 레이어를 정의
        # hidden_channels * heads: 첫 번째 레이어의 출력 차원. 멀티헤드 어텐션을 사용할 경우, 출력 차원은 hidden_channels * heads가 됨
        # int(hidden_channels/4): 두 번째 레이어의 은닉층 차원. 첫 번째 레이어의 차원보다 작게 설정
        # heads=1: 두 번째 레이어에서는 단일 헤드 어텐션 사용
        # concat=False: 멀티헤드 어텐션의 출력을 연결(concat)하지 않고 평균을 사용
        # 드롭아웃 비율을 0.6 : 과적합을 방지하기 위해 일부 뉴런을 무작위로 비활성화

        self.lin = Linear(int(hidden_channels/4), out_channels)
        # self.lin: 완전 연결 레이어(fully connected layer) 정의
        # int(hidden_channels/4): 입력 차원. 두 번째 GAT 레이어의 출력 차원과 동일
        # out_channels: 출력 차원. 모델의 최종 출력 차원을 지정
        
        self.sigmoid = nn.Sigmoid()
        # sigmoid : 출력 값을 0과 1 사이로 압축하며, 이진 분류 문제에서 사용

    def forward(self, x, edge_index, edge_attr):
        # x: 입력 데이터(노드 특성).
        # edge_index: 간선 정보(출발 노드와 도착 노드의 인덱스).
        # edge_attr: 간선 특성(옵션).

        x = F.dropout(x, p=0.6, training=self.training) # 모델이 학습 모드인지 여부를 확인. 학습 모드에서만 드롭아웃이 적용.
        x = F.elu(self.conv1(x, edge_index, edge_attr)) # 첫 번째 GAT 컨볼루션 레이어를 통과 / ELU(Exponential Linear Unit) 활성화 함수를 적용. ELU는 음수 입력에 대해 부드러운 곡선을 가지며, ReLU의 변형.
        x = F.dropout(x, p=0.6, training=self.training) # 두 번째 드롭아웃을 적용
        x = F.elu(self.conv2(x, edge_index, edge_attr)) # 두 번째 GAT 컨볼루션 레이어를 통과 / ELU 활성화 함수 적용.
        x = self.lin(x) # 완전 연결 레이어를 통과시켜 최종 출력을 계산.
        x = self.sigmoid(x) # 그모이드 활성화 함수를 적용하여 출력 값을 0과 1 사이로 압축.
        
        return x