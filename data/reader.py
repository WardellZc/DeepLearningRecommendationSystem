# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 21:37
# @Author  : Lumen
# @File    : reader.py
import random
from typing import Tuple

import numpy as np
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

        # 将数据集转化为隐式数据集
        self.data['rating'] = 1

        # 将数据集划分为训练集、验证集、测试集
        # self.data = self.data.sample(frac=1).reset_index(drop=True)  # 打乱数据
        # train_size = int(len(self.data) * 0.6)  # 计算分割点
        # valid_size = int(len(self.data) * 0.2)
        # self.train = self.data[:train_size]
        # self.valid = self.data[train_size:train_size + valid_size]
        # self.test = self.data[train_size + valid_size:]
        def split_user_data(ratings, train_ratio=0.6, valid_ratio=0.2):
            # 按用户分组
            user_grouped = ratings.groupby('user_id')

            train_list = []
            valid_list = []
            test_list = []

            for user_id, group in user_grouped:
                # 打乱用户的记录
                group = group.sample(frac=1).reset_index(drop=True)
                n = len(group)
                train_end = int(n * train_ratio)
                valid_end = train_end + int(n * valid_ratio)

                # 划分数据集
                train_list.append(group.iloc[:train_end])
                valid_list.append(group.iloc[train_end:valid_end])
                test_list.append(group.iloc[valid_end:])

            train = pd.concat(train_list).reset_index(drop=True)
            valid = pd.concat(valid_list).reset_index(drop=True)
            test = pd.concat(test_list).reset_index(drop=True)

            return train, valid, test

        self.train, self.valid, self.test = split_user_data(self.data)

    @staticmethod
    def _interaction(data, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        users_tensor = torch.tensor(data['user_id'].values, dtype=torch.long)
        items_tensor = torch.tensor(data['item_id'].values, dtype=torch.long)
        ratings_tensor = torch.tensor(data['rating'].values, dtype=torch.long)
        return users_tensor.to(device), items_tensor.to(device), ratings_tensor.to(device)

    def train_interaction(self, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._interaction(self.train, device)

    def valid_interaction(self, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._interaction(self.valid, device)

    def test_interaction(self, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._interaction(self.test, device)

    # 将评分数据集与用户信息和物品连接起来
    def feature(self, data):
        data = pd.merge(data, self.user_data, on='user_id')
        data = pd.merge(data, self.item_data, on='item_id')
        return data

    # 融合用户信息和物品信息的特征数据集
    def user_item(self):
        user = pd.DataFrame({'user_id': range(self.num_users)})
        item = pd.DataFrame({'item_id': range(self.num_items)})
        user['key'] = 1
        item['key'] = 1
        user_item = pd.merge(user, item, on='key').drop('key', axis=1)
        user_item = pd.merge(user_item, self.user_data, on='user_id')
        user_item = pd.merge(user_item, self.item_data, on='item_id')
        return user_item

    # 得到每个用户评过分的物品id矩阵(要保证输入的数据包含所有用户，不然索引与用户id对应会出错)
    @staticmethod
    def itemid_matrix(data):
        # 1. 导入数据
        df = data

        # 2. 生成每个用户评过分的物品ID列表
        user_item_dict = df.groupby('user_id')['item_id'].apply(list).to_dict()

        # 3. 获取所有用户ID
        user_ids = sorted(user_item_dict.keys())

        # 4. 将结果转换成矩阵形式
        user_item_matrix = [user_item_dict[user_id] for user_id in user_ids]

        # 将矩阵numpy化加快运行速度
        max_len = max(len(lst) for lst in user_item_matrix)  # 找到最长的子列表的长度
        padded_matrix = np.array(
            [lst + [-1] * (max_len - len(lst)) for lst in user_item_matrix])  # 使用填充值 -1 填充每个子列表，使其具有相同的长度
        return padded_matrix

    # 从推荐列表中去除另外两个数据集中已经交互过的物品(保证两个矩阵具有相同行数`，即用户数)
    @staticmethod
    def remove_itemid(recommendation_matrix, other_matrix):
        filtered_recommendations = []
        num_users, num_recommendations = recommendation_matrix.shape

        for user_id in range(num_users):
            recommendations = recommendation_matrix[user_id]
            train_interactions = other_matrix[user_id]

            # 移除无效的交互项（假设用 -1 作为填充值）
            train_interactions = train_interactions[train_interactions >= 0]

            # 使用集合进行快速查找
            train_interactions_set = set(train_interactions)

            filtered_list = [item for item in recommendations if item not in train_interactions_set]
            filtered_recommendations.append(filtered_list)

        # 将矩阵numpy化加快运行速度
        max_len = max(len(lst) for lst in filtered_recommendations)  # 找到最长的子列表的长度
        padded_matrix = np.array(
            [lst + [-1] * (max_len - len(lst)) for lst in filtered_recommendations])  # 使用填充值 -1 填充每个子列表，使其具有相同的长度

        return padded_matrix


# data = MovieLens100K('../dataset_example/ml-100k')
# print("训练集用户个数：", len(data.train['user_id'].unique()))
# print("验证集用户个数：", len(data.valid['user_id'].unique()))
# print("测试集用户个数：", len(data.test['user_id'].unique()))
