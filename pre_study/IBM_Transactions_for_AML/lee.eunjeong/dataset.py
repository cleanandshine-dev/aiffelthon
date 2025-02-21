# dataset.py -  금융 거래 데이터를 그래프 데이터 구조로 변환하여, 그래프 신경망(GNN) 모델에 입력으로 사용할 수 있도록 처리

import datetime
import os
from typing import Callable, Optional
import pandas as pd
from sklearn import preprocessing # 레이블 인코딩과 같은 전처리 작업을 수행
import numpy as np
import torch
import networkx as nx  # NetworkX 라이브러리 import
from torch_geometric.data import (
    Data,
    InMemoryDataset
) # PyTorch Geometric의 데이터 구조를 정의

pd.set_option('display.max_columns', None)

################################################################ 2025.02.19 ################################################################
# networkX 중심성계좌 함수 추가 1개
# train.py 에서 GAT 모델을 생성할 때, 변경된 입력 채널 수를 반영해야 함.
# dataset.py를 수정한 후 train.py 를 처음 실행하면, dataset.process() 함수가 실행되어 data.pt 파일이 다시 생성되고, 
# data.num_features 값이 업데이트 됨. 이 업데이트된 data.num_features 값을 GAT 모델 생성 시 in_channels 로 사용해야 함.
############################################################################################################################################
# 다운샘플링 추가 - 중심성 계좌 함수 추가 후 끝나지 않음.
# dataset.py 파일의 process() 함수 내에 다운샘플링 로직을 추가
# 정상 거래 계좌 (label 0)를 다운샘플링하는 방법을 사용하겠음. 
# 자금 세탁 거래 계좌 (label 1)는 상대적으로 수가 적을 가능성이 높으므로, label 1 데이터는 그대로 유지하고 label 0 데이터의 수를 줄이는 방식.
############################################################################################################################################
# 새로 실행 할때, data.pt 삭제 하고 실행 할 것 : 수정된 dataset.py 의 process() 함수가 다시 실행되어 새로운 중심성 Feature가 포함된 data.pt 파일이 생성 됨.
############################################################################################################################################

class AMLtoGraph(InMemoryDataset):
# PyTorch Geometric의 InMemoryDataset을 상속받아 메모리 내에서 데이터셋을 처리

    def __init__(self, root: str, edge_window_size: int = 10,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.edge_window_size = edge_window_size
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # root: 데이터셋이 저장될 루트 디렉토리.
        # edge_window_size: 간선 데이터를 처리할 때 사용할 윈도우 크기 (현재 코드에서는 사용되지 않음).
        # transform, pre_transform: 데이터 변환 및 전처리를 위한 함수.
        # self.data, self.slices: 처리된 데이터를 로드.


    @property
    def raw_file_names(self) -> str:
        return 'HI-Small_Trans.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return self._data.edge_index.max().item() + 1
        # num_nodes: 그래프의 노드 수를 반환. edge_index에서 최대 노드 인덱스를 찾아 계산.

    def df_label_encoder(self, df, columns):
        le = preprocessing.LabelEncoder()
        for i in columns:
            df[i] = le.fit_transform(df[i].astype(str))
        return df
        # df_label_encoder: 지정된 열에 대해 레이블 인코딩을 수행. 문자열 값을 정수로 변환.


    def preprocess(self, df):
        df = self.df_label_encoder(df,['Payment Format', 'Payment Currency', 'Receiving Currency']) # 레이블 인코딩 : 해당 컬럼을 정수로 변환.
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Timestamp'] = df['Timestamp'].apply(lambda x: x.value)
        df['Timestamp'] = (df['Timestamp']-df['Timestamp'].min())/(df['Timestamp'].max()-df['Timestamp'].min()) # 1~0 사이 값으로 정규화

        df['Account'] = df['From Bank'].astype(str) + '_' + df['Account']
        df['Account.1'] = df['To Bank'].astype(str) + '_' + df['Account.1'] # 은행 정보와 결합하여 고유한 계정 식별자 생성
        df = df.sort_values(by=['Account'])
        receiving_df = df[['Account.1', 'Amount Received', 'Receiving Currency']] # 송금 및 수금 데이터를 분리.
        paying_df = df[['Account', 'Amount Paid', 'Payment Currency']]
        receiving_df = receiving_df.rename({'Account.1': 'Account'}, axis=1)
        currency_ls = sorted(df['Receiving Currency'].unique()) # 사용된 통화 종류 목록을 생성

        return df, receiving_df, paying_df, currency_ls

    def get_all_account(self, df):
        ldf = df[['Account', 'From Bank']]
        rdf = df[['Account.1', 'To Bank']]
        suspicious = df[df['Is Laundering']==1]
        s1 = suspicious[['Account', 'Is Laundering']]
        s2 = suspicious[['Account.1', 'Is Laundering']]
        s2 = s2.rename({'Account.1': 'Account'}, axis=1)
        suspicious = pd.concat([s1, s2], join='outer')
        suspicious = suspicious.drop_duplicates()

        ldf = ldf.rename({'From Bank': 'Bank'}, axis=1)
        rdf = rdf.rename({'Account.1': 'Account', 'To Bank': 'Bank'}, axis=1)
        df = pd.concat([ldf, rdf], join='outer')
        df = df.drop_duplicates()

        df['Is Laundering'] = 0
        df.set_index('Account', inplace=True)
        df.update(suspicious.set_index('Account'))
        df = df.reset_index()
        return df
        # 수신자, 송신자 모두 Is Laundering == 1 일때의 데이터 따로 떼어 row를 생성
    
    def paid_currency_aggregate(self, currency_ls, paying_df, accounts):
        for i in currency_ls:
            temp = paying_df[paying_df['Payment Currency'] == i]
            accounts['avg paid '+str(i)] = temp['Amount Paid'].groupby(temp['Account']).transform('mean')
        return accounts
        # 각 통화별로 송금 평균 금액을 계산.


    def received_currency_aggregate(self, currency_ls, receiving_df, accounts):
        for i in currency_ls:
            temp = receiving_df[receiving_df['Receiving Currency'] == i]
            accounts['avg received '+str(i)] = temp['Amount Received'].groupby(temp['Account']).transform('mean')
        accounts = accounts.fillna(0)
        return accounts
        # 각 통화별로 수금 평균 금액을 계산.

    def get_edge_df(self, accounts, df):
        accounts = accounts.reset_index(drop=True)
        accounts['ID'] = accounts.index
        mapping_dict = dict(zip(accounts['Account'], accounts['ID']))
        df['From'] = df['Account'].map(mapping_dict)
        df['To'] = df['Account.1'].map(mapping_dict)
        df = df.drop(['Account', 'Account.1', 'From Bank', 'To Bank'], axis=1)

        edge_index = torch.stack([torch.from_numpy(df['From'].values), torch.from_numpy(df['To'].values)], dim=0)

        df = df.drop(['Is Laundering', 'From', 'To'], axis=1)

        edge_attr = torch.from_numpy(df.values).to(torch.float)
        return edge_attr, edge_index
        # 간선 데이터를 생성.
        ## 계정을 정수 ID로 매핑.
        ## edge_index: 송금 계정과 수금 계정의 인덱스 쌍.
        ## edge_attr: 간선 특성 (예: 금액, 통화 등).

    def get_node_attr(self, currency_ls, paying_df,receiving_df, accounts):
        node_df = self.paid_currency_aggregate(currency_ls, paying_df, accounts)
        node_df = self.received_currency_aggregate(currency_ls, receiving_df, node_df)
        node_label = torch.from_numpy(node_df['Is Laundering'].values).to(torch.float)
        node_df = node_df.drop(['Account', 'Is Laundering'], axis=1)
        node_df = self.df_label_encoder(node_df,['Bank'])
        node_df = torch.from_numpy(node_df.values).to(torch.float)
        return node_df, node_label
        # 노드 특성과 레이블을 생성.
        ## node_df: 노드 특성 (예: 평균 송금/수금 금액).
        ## node_label: 불법 거래 여부.


####### networkX 의 계좌중심성 지표(3) / 자금세탁 패턴 지표(6) 함수 시작 ##############
## networkX 의 계좌중심성 지표(3) 시작 ###############################################
    def get_betweenness_tensor(nx_graph, accounts) :
        betweenness_centrality = nx.betweenness_centrality(nx_graph)
        # 중심성 Feature Node Feature에 추가

        betweenness_values = [betweenness_centrality.get(i, 0) for i in range(accounts.shape[0])] # 각 노드에 대한 betweenness 값 추출, 없는 경우 0으로 처리
        betweenness_tensor = torch.tensor(betweenness_values, dtype=torch.float).unsqueeze(1) # tensor로 변환 및 차원 조정

        return betweenness_tensor
    # end of def get_betweenness_tensor

    def get_degree_tensor(nx_graph, accounts) :
        degree_centrality = nx.degree_centrality(nx_graph)
        # 연결 중심성 (Degree Centrality)

        degree_values = [degree_centrality.get(i, 0) for i in range(accounts.shape[0])]
        degree_tensor = torch.tensor(degree_values, dtype=torch.float).unsqueeze(1)

        return degree_tensor
    # end of def get_degree_tensor

    def get_closeness_tensor(nx_graph, accounts) :
        closeness_centrality = {} # disconnected graph 에러 방지 위해 초기화
        # 근접 중심성 ( Closeness Centrality )
        try:
            closeness_centrality = nx.closeness_centrality(nx_graph)
        except nx.NetworkXError:
            print("Warning: Graph is disconnected. Closeness Centrality might not be calculated for all nodes.")
            # disconnected graph 경우, closeness_centrality는 빈 dict {} 로 남게 됨 (아래에서 get() 호출 시 기본값 0으로 처리됨)

        closeness_values = [closeness_centrality.get(i, 0) for i in range(accounts.shape[0])] # get(i, 0) : key i가 없으면 0 반환
        closeness_tensor = torch.tensor(closeness_values, dtype=torch.float).unsqueeze(1)
        
        return closeness_tensor
    # end of def get_closeness_tensor
## networkX 의 계좌중심성 지표(3) 끝 ###############################################
## networkX 의 자금세탁 패턴 지표(6) 함수 시작 #####################################
    def get_transaction_frequency_tensor(nx_graph, accounts) :
        # 1. 거래 빈도 (Transaction Frequency) / 거래량 (Transaction Volume) (Node Degree 활용)
        # - Degree Centrality 값 (이미 계산됨) 을 거래 빈도/량 지표로 재활용
        degree_centrality = nx.degree_centrality(nx_graph) # or use nx.degree(nx_graph) which returns degree directly
        degree_values = [degree_centrality.get(i, 0) for i in range(accounts.shape[0])]
        transaction_frequency_tensor = torch.tensor(degree_values, dtype=torch.float).unsqueeze(1) # tensor로 변환 및 차원 조정
        return transaction_frequency_tensor
    # end of def get_transaction_frequency_tensor


    def get_in_out_degree_ratio_tensor(nx_digraph, accounts) :
        # 2. In-Degree / Out-Degree 비율 (방향 그래프 고려)
        in_degree_centrality = nx.in_degree_centrality(nx_digraph)
        out_degree_centrality = nx.out_degree_centrality(nx_digraph)

        in_degree_values = [in_degree_centrality.get(i, 0) for i in range(accounts.shape[0])]
        out_degree_values = [out_degree_centrality.get(i, 0) for i in range(accounts.shape[0])]

        in_degree_tensor = torch.tensor(in_degree_values, dtype=torch.float).unsqueeze(1)
        out_degree_tensor = torch.tensor(out_degree_values, dtype=torch.float).unsqueeze(1)

        in_out_degree_ratio_tensor = (out_degree_tensor + 1e-6) / (in_degree_tensor + 1e-6) # 분모가 0이 되는 것 방지 위해 작은 값 더함
        return in_out_degree_ratio_tensor
    # end of def get_in_out_degree_ratio_tensor


    def get_pagerank_tensor(nx_digraph, accounts) :
        # 3. PageRank Centrality
        pagerank_centrality = nx.pagerank(nx_digraph) # 방향 그래프 사용
        pagerank_values = [pagerank_centrality.get(i, 0) for i in range(accounts.shape[0])]
        pagerank_tensor = torch.tensor(pagerank_values, dtype=torch.float).unsqueeze(1)
        return pagerank_tensor
    # end of def get_pagerank_tensor


    def get_clustering_coefficient_tensor(nx_graph, accounts) :
        # 4. Local Clustering Coefficient (클러스터링 계수)
        clustering_coefficient = nx.clustering(nx_graph)
        clustering_values = [clustering_coefficient.get(i, 0) for i in range(accounts.shape[0])]
        clustering_tensor = torch.tensor(clustering_values, dtype=torch.float).unsqueeze(1)
        return clustering_tensor
    # end of def get_clustering_coefficient_tensor


    def get_avg_neighbor_degree_tensor(nx_graph, accounts) :
        # 5. Average Neighbor Degree (평균 이웃 Degree)
        avg_neighbor_degree = nx.average_neighbor_degree(nx_graph)
        avg_neighbor_degree_values = [avg_neighbor_degree.get(i, 0) for i in range(accounts.shape[0])]
        avg_neighbor_degree_tensor = torch.tensor(avg_neighbor_degree_values, dtype=torch.float).unsqueeze(1)
        return avg_neighbor_degree_tensor
    # end of def get_avg_neighbor_degree_tensor


    def get_shortest_path_related_tensor(nx_graph, accounts) :
        # 6. Shortest Path Length 관련 지표 (평균 최단 경로 길이, Diameter 등)
        # 평균 최단 경로 길이 (disconnected graph일 경우 에러 발생 가능성 있음)
        try:
            avg_shortest_path = nx.average_shortest_path_length(nx_graph)
        except nx.NetworkXError: # disconnected graph인 경우 예외 처리 (0 또는 -1 등의 값으로 대체하거나, component별로 계산)
            avg_shortest_path = 0  # 또는 avg_shortest_path = -1

        # 네트워크 지름 (disconnected graph일 경우 에러 발생 가능성 있음)
        try:
            diameter = nx.diameter(nx_graph)
        except nx.NetworkXError: # disconnected graph인 경우 예외 처리
            diameter = 0 # 또는 diameter = -1

        avg_shortest_path_values = [avg_shortest_path] * accounts.shape[0] # 모든 노드에 동일 값 적용
        diameter_values = [diameter] * accounts.shape[0] # 모든 노드에 동일 값 적용

        avg_shortest_path_tensor = torch.tensor(avg_shortest_path_values, dtype=torch.float).unsqueeze(1)
        diameter_tensor = torch.tensor(diameter_values, dtype=torch.float).unsqueeze(1)

        shortest_path_related_tensor = torch.cat([avg_shortest_path_tensor, diameter_tensor], dim=1) # 평균 최단경로길이, 지름 함께 return
        return shortest_path_related_tensor
    # end of def get_shortest_path_related_tensor

## networkX 의 자금세탁 패턴 지표(6) 함수 끝 #######################################
####### networkX 의 계좌중심성 지표(3) / 자금세탁 패턴 지표(6) 함수 끝 ################


    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        df, receiving_df, paying_df, currency_ls = self.preprocess(df)
        accounts = self.get_all_account(df)
        node_attr, node_label = self.get_node_attr(currency_ls, paying_df,receiving_df, accounts)
        edge_attr, edge_index = self.get_edge_df(accounts, df)

# ############################################################## 2025.02.19 ################################################################
        # NetworkX 그래프 생성
        # nx_graph = nx.Graph() # or nx.DiGraph() depending on your graph directionality
        # edge_list = edge_index.t().tolist() # edge_index를 NetworkX edge 형식으로 변환
        # nx_graph.add_edges_from(edge_list)

        #### 계좌 중심성 지표 3가지 함수 효출 추가  - 기존 node_attr에 중심성 feature 연결 ###
        # node_attr = torch.cat([node_attr, get_betweenness_tensor(nx_graph, accounts)], dim=1) # 1. 중심성(Centrality) 계산 (Betweenness Centrality )
        # node_attr = torch.cat([node_attr, get_degree_tensor(nx_graph, accounts)], dim=1) # 2. 연결 중심성 (Degree Centrality)
        # node_attr = torch.cat([node_attr, get_closeness_tensor(nx_graph, accounts)], dim=1) #3. 근접 중심성 (Closeness Centrality)
        #### 계좌 중심성 지표 3가지 함수 효출 끝 ###

        #### 자금세탁 패텬 지표 6가지 함수 효출 추가 ###
        # node_attr = torch.cat([node_attr, get_transaction_frequency_tensor(nx_graph, accounts)], dim=1) # 1. 거래빈도
        # node_attr = torch.cat([node_attr, get_in_out_degree_ratio_tensor(nx_graph, accounts)], dim=1) # In-Degree / Out-Degree 비율 (방향 그래프 고려)
        # node_attr = torch.cat([node_attr, get_pagerank_tensor(nx_graph, accounts)], dim=1) # 3. Page Rank Centrality~
        # node_attr = torch.cat([node_attr, get_clustering_coefficient_tensor(nx_graph, accounts)], dim=1) # 4. 클러스터링 계수
        # node_attr = torch.cat([node_attr, get_avg_neighbor_degree_tensor(nx_graph, accounts)], dim=1) # 5. 평균 이웃 Degree
        # node_attr = torch.cat([node_attr, get_shortest_path_related_tensor(nx_graph, accounts)], dim=1) # 6. 평균 최단 경로 길이
        #### 자금세탁 패텬 지표 6가지 함수 효출 끝 ###
############################################################################################################################################


        data = Data(x=node_attr,
                    edge_index=edge_index,
                    y=node_label,
                    edge_attr=edge_attr
                    )


#############################################################[다운샘플링 코드 시작]###########################################################
        downsample_ratio = 0.2  # 다운샘플링 비율 (label 0 데이터 중 몇 %를 사용할지, 0.0 ~ 1.0)
        num_label_0 = (node_label == 0).sum().item() # label 0 개수
        num_label_1 = (node_label == 1).sum().item() # label 1 개수

        if num_label_0 > 0: # label 0 데이터가 있는 경우에만 다운샘플링 적용
            downsampled_label_0_count = int(num_label_0 * downsample_ratio) # 다운샘플링할 label 0 데이터 개수
            label_0_indices = (node_label == 0).nonzero(as_tuple=True)[0] # label 0 node index
            rand_perm = torch.randperm(num_label_0) # label 0 index 랜덤 섞기
            downsampled_indices_label_0 = label_0_indices[rand_perm[:downsampled_label_0_count]] # 다운샘플링된 label 0 index 선택
            label_1_indices = (node_label == 1).nonzero(as_tuple=True)[0] # label 1 node index
            train_indices = torch.cat([downsampled_indices_label_0, label_1_indices]) # 다운샘플링된 label 0 index + 전체 label 1 index
        else:
            train_indices = (node_label == 1).nonzero(as_tuple=True)[0] # label 0 데이터가 없으면 label 1 데이터만 사용 (혹은 에러 처리)

        train_mask = torch.zeros(node_label.size(0), dtype=torch.bool) # 전체 node mask 생성 (False로 초기화)
        train_mask[train_indices] = True # 다운샘플링된 index에 해당하는 mask만 True로 설정
        data.train_mask = train_mask # train_mask를 Data 객체에 할당
        data.val_mask = None # 다운샘플링 적용 : validation/test mask는 train.py에서 RandomNodeSplit으로 생성하므로 None으로 설정
        data.test_mask = None
##############################################################[다운샘플링 코드  끝]############################################################

        data_list = [data] 
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        # process: 전체 데이터 처리 파이프라인 - 위의 전처리 함수들을 호출 함
        ## 원본 데이터를 로드하고 전처리.
        ## 노드 및 간선 데이터를 생성.
        ## Data 객체를 생성하고 저장.