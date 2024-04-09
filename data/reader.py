# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 21:37
# @Author  : Lumen
# @File    : reader.py
import random
from typing import Tuple

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class MovieLens100K:
    def __init__(self, dataset_path: str):
        # 下面第二种数据读取方法所需的用户和物品信息
        self.user = None
        self.item = None

        # 读取训练集和测试集数据
        cols = ['user_id', 'item_id', 'rating', 'timestamp']
        self.train = pd.read_csv(f'{dataset_path}/u1.base', sep='\t', header=None, names=cols)
        self.test = pd.read_csv(f'{dataset_path}/u1.test', sep='\t', header=None, names=cols)

        # 创建用户和物品的统一映射
        self.user_ids = pd.concat([self.train['user_id'], self.test['user_id']]).unique()
        self.item_ids = pd.concat([self.train['item_id'], self.test['item_id']]).unique()
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)

        self.user_to_new_id = {old: new for new, old in enumerate(self.user_ids)}
        self.item_to_new_id = {old: new for new, old in enumerate(self.item_ids)}

        # 应用映射更新ID
        self.train['user_id'] = self.train['user_id'].map(self.user_to_new_id)
        self.train['item_id'] = self.train['item_id'].map(self.item_to_new_id)
        self.test['user_id'] = self.test['user_id'].map(self.user_to_new_id)
        self.test['item_id'] = self.test['item_id'].map(self.item_to_new_id)

    @staticmethod
    def _interaction(data, implicit_feedback: bool, device: str = 'cpu') -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        users_tensor = torch.tensor(data['user_id'].values, dtype=torch.long)
        items_tensor = torch.tensor(data['item_id'].values, dtype=torch.long)

        if implicit_feedback:
            ratings_tensor = torch.tensor(data['rating'].values > 0, dtype=torch.float32)
        else:
            ratings_tensor = torch.tensor(data['rating'].values, dtype=torch.float32)

        return users_tensor.to(device), items_tensor.to(device), ratings_tensor.to(device)

    def train_interaction(self, device: str = 'cpu', implicit_feedback: bool = True) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._interaction(self.train, implicit_feedback, device)

    def test_interaction(self, device: str = 'cpu', implicit_feedback: bool = True) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._interaction(self.test, implicit_feedback, device)

    # 融合用户特征（年龄、性别、职业）以及物品特征（电影流派，19个流派）的数据集读取
    def feature_data(self, dataset_path: str):
        # 读取评分数据、用户数据、物品数据构成训练集和测试集
        cols = ['user_id', 'item_id', 'rating']
        train_rating = pd.read_csv(f'{dataset_path}/u1.base', sep='\t', header=None, names=cols, usecols=[0, 1, 2])
        test_rating = pd.read_csv(f'{dataset_path}/u1.test', sep='\t', header=None, names=cols, usecols=[0, 1, 2])
        # 转化为隐式评分
        train_rating['rating'] = 1
        test_rating['rating'] = 1

        # 读取物品信息
        cols_item = ['item_id', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                     'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film - Noir', 'Horror', 'Musical',
                     'Mystery', 'Romance', 'Sci - Fi', 'Thriller', 'War', 'Western']
        cols_read = [0] + list(range(5, 24))
        self.item = pd.read_csv(f'{dataset_path}/u.item', encoding='ISO-8859-1', sep='|', header=None, names=cols_item,
                                usecols=cols_read)
        # 读取用户信息
        cols_user = ['user_id', 'age', 'gender', 'occupation']
        self.user = pd.read_csv(f'{dataset_path}/u.user', sep='|', names=cols_user, usecols=[0, 1, 2, 3])

        # 对类别特征进行编码(性别、职业)
        gender_encoder = LabelEncoder()
        occupation_encoder = LabelEncoder()
        self.user['gender'] = gender_encoder.fit_transform(self.user['gender'])
        self.user['occupation'] = occupation_encoder.fit_transform(self.user['occupation'])
        # 对年龄进行归一化
        age_scaler = MinMaxScaler()
        self.user['age'] = age_scaler.fit_transform(self.user[['age']])

        # 合并数据
        train_data = pd.merge(train_rating, self.item, on='item_id')
        train_data = pd.merge(train_data, self.user, on='user_id')
        test_data = pd.merge(test_rating, self.item, on='item_id')
        test_data = pd.merge(test_data, self.user, on='user_id')

        # 创建融合了用户和物品特征的矩阵
        self.user['key'] = 1
        self.item['key'] = 1
        user_item = pd.merge(self.user, self.item, on='key').drop('key', axis=1)
        return torch.tensor(train_data.values, dtype=torch.float32), torch.tensor(test_data.values,
                                                                                  dtype=torch.float32), torch.tensor(
            user_item.values, dtype=torch.float32)

    @staticmethod
    def rating_matrix(dataset_path: str):
        cols = ['user_id', 'item_id', 'rating', 'timestamp']
        rating_matrix = pd.read_csv(f'{dataset_path}/u.data', sep='\t', header=None, names=cols)
        rating_matrix = rating_matrix.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        # 转化为隐式评分
        rating_matrix = rating_matrix.applymap(lambda x: 1 if x > 0 else 0)
        return torch.FloatTensor(rating_matrix.values)
