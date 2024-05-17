# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 21:58
# @Author  : Lumen
# @File    : sampler.py
import random

import pandas as pd
import torch.nn


class Sampler:
    def __init__(self):
        self.negative_users = []
        self.negative_items = []

    def negative_sampling(self, num_user: int,  # 数据集存在的用户数量
                          num_item: int,  # 数据集存在的物品数量
                          excluded_pairs: set,  # 数据集存在的用户-物品对
                          num_negatives: int,  # 每个用户取多少个负样本
                          device: str = 'cpu'):  # 选择在cpu还是gup上运行
        for n_user in range(num_user):
            for _ in range(num_negatives):
                n_item = random.randint(0, num_item - 1)
                while (n_user, n_item) in excluded_pairs:
                    n_item = random.randint(0, num_item - 1)
                self.negative_users.append(n_user)
                self.negative_items.append(n_item)
        return torch.tensor(self.negative_users).to(device), torch.tensor(self.negative_items).to(device), torch.zeros(
            len(self.negative_users)).to(device)

    def negative_sampling2(self, num_user: int,  # 数据集存在的用户数量
                           num_item: int,  # 数据集存在的物品数量
                           excluded_pairs: set,  # 数据集存在的用户-物品对
                           num_negatives: int):  # 每个用户取多少个负样本
        for n_user in range(num_user):
            for _ in range(num_negatives):
                n_item = random.randint(0, num_item - 1)
                while (n_user, n_item) in excluded_pairs:
                    n_item = random.randint(0, num_item - 1)
                self.negative_users.append(n_user)
                self.negative_items.append(n_item)
        negative_ratings = [0 for _ in range(len(self.negative_users))]
        df = pd.DataFrame({
            'user_id': self.negative_users,
            'item_id': self.negative_items,
            'rating': negative_ratings
        })
        return df
