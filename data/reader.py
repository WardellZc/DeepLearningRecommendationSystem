# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 21:37
# @Author  : Lumen
# @File    : reader.py
from typing import Tuple

import pandas as pd
import torch


class MovieLens100K:
    def __init__(self, dataset_path: str):
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
    def _interaction(data, implicit_feedback: bool, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        users_tensor = torch.tensor(data['user_id'].values, dtype=torch.long)
        items_tensor = torch.tensor(data['item_id'].values, dtype=torch.long)

        if implicit_feedback:
            ratings_tensor = torch.tensor(data['rating'].values > 0, dtype=torch.float32)
        else:
            ratings_tensor = torch.tensor(data['rating'].values, dtype=torch.float32)

        return users_tensor.to(device), items_tensor.to(device), ratings_tensor.to(device)

    def train_interaction(self, device: str = 'cpu', implicit_feedback: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._interaction(self.train, implicit_feedback, device)

    def test_interaction(self, device: str = 'cpu', implicit_feedback: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._interaction(self.test, implicit_feedback, device)


