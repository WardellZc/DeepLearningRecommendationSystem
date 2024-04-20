# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 21:37
# @Author  : Lumen
# @File    : reader.py
import random
from typing import Tuple

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


class MovieLens100K:
    def __init__(self, dataset_path: str):
        # 读取数据集u.data，用户信息数据集u.user，物品信息数据集u.item
        cols = ['user_id', 'item_id', 'rating']
        self.data = pd.read_csv(f'{dataset_path}/u.data', sep='\t', header=None, names=cols, usecols=[0, 1, 2])
        cols_user = ['user_id', 'age', 'gender', 'occupation']
        self.user_data = pd.read_csv(f'{dataset_path}/u.user', sep='|', names=cols_user, usecols=[0, 1, 2, 3])
        cols_item = ['item_id', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                     'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film - Noir', 'Horror', 'Musical',
                     'Mystery', 'Romance', 'Sci - Fi', 'Thriller', 'War', 'Western']
        cols_read = [0] + list(range(5, 24))
        self.item_data = pd.read_csv(f'{dataset_path}/u.item', encoding='ISO-8859-1', sep='|', header=None,
                                     names=cols_item,
                                     usecols=cols_read)

        # 将用户ID和物品ID映射从0开始
        self.data['user_id'] = self.data['user_id'] - 1
        self.data['item_id'] = self.data['item_id'] - 1
        self.user_data['user_id'] = self.user_data['user_id'] - 1
        self.item_data['item_id'] = self.item_data['item_id'] - 1
        self.num_users = len(self.data['user_id'].unique())
        self.num_items = len(self.data['item_id'].unique())

        # 将用户信息中的age归一化，gender、occupation处理成onehot向量
        self.user_data = pd.get_dummies(self.user_data, columns=['gender', 'occupation']).astype(int)
        scaler = MinMaxScaler()
        ages = self.user_data['age'].values.reshape(-1, 1)
        self.user_data['age'] = scaler.fit_transform(ages)

        # 将评分数据集与用户信息和物品信息连接起来
        self.data = pd.merge(self.data, self.user_data, on='user_id')
        self.data = pd.merge(self.data, self.item_data, on='item_id')

        # 将数据集转化为隐式数据集
        self.data['rating'] = 1

        # 将数据集划分为训练集、验证集、测试集
        self.data = self.data.sample(frac=1).reset_index(drop=True)  # 打乱数据
        train_size = int(len(self.data) * 0.6)  # 计算分割点
        validation_size = int(len(self.data) * 0.2)
        self.train = self.data[:train_size]
        self.validation = self.data[train_size:train_size + validation_size]
        self.test = self.data[train_size + validation_size:]

        # 生成用户数量*物品数量的数据集,没有rating列，前两行为user_id、item_id
        user = pd.DataFrame({'user_id': range(self.num_users)})
        item = pd.DataFrame({'item_id': range(self.num_items)})
        user['key'] = 1
        item['key'] = 1
        self.user_item = pd.merge(user, item, on='key').drop('key', axis=1)
        self.user_item = pd.merge(self.user_item, self.user_data, on='user_id')
        self.user_item = pd.merge(self.user_item, self.item_data, on='item_id')

        # 生成评分矩阵式的数据集
        self.rating_matrix = self.data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

    @staticmethod
    def _interaction(data, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        users_tensor = torch.tensor(data['user_id'].values, dtype=torch.long)
        items_tensor = torch.tensor(data['item_id'].values, dtype=torch.long)
        ratings_tensor = torch.tensor(data['rating'].values, dtype=torch.long)
        return users_tensor.to(device), items_tensor.to(device), ratings_tensor.to(device)

    def train_interaction(self, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._interaction(self.train, device)

    def validation_interaction(self, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._interaction(self.validation, device)

    def test_interaction(self, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._interaction(self.test, device)
