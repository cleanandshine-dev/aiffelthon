# dataset.py -  금융 거래 데이터를 그래프 데이터 구조로 변환하여, 그래프 신경망(GNN) 모델에 입력으로 사용할 수 있도록 처리

import datetime
import os
from typing import Callable, Optional
import pandas as pd
from sklearn import preprocessing # 레이블 인코딩과 같은 전처리 작업을 수행
import numpy as np

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset
) # PyTorch Geometric의 데이터 구조를 정의

pd.set_option('display.max_columns', None)


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

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        df, receiving_df, paying_df, currency_ls = self.preprocess(df)
        accounts = self.get_all_account(df)
        node_attr, node_label = self.get_node_attr(currency_ls, paying_df,receiving_df, accounts)
        edge_attr, edge_index = self.get_edge_df(accounts, df)

        data = Data(x=node_attr,
                    edge_index=edge_index,
                    y=node_label,
                    edge_attr=edge_attr
                    )
        
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