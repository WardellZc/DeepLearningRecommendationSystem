# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 22:00
# @Author  : Lumen
# @File    : mf.py
import torch
from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, num_feature: int):
        super().__init__()
        # 特征向量参数w和b
        self.linear = nn.Linear(num_feature, 1, True)

    def forward(self, feature_vector: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(feature_vector))

    def recommendation(self, num_users, user_item, k):
        array = []
        for i in range(num_users):
            user_vector = user_item[user_item['user_id'] == i]
            user_vector = torch.Tensor(user_vector.iloc[:, 2:].values)
            scores = self.forward(user_vector)
            values, indices = torch.topk(scores, k, dim=0)
            indices = indices.view(1, -1).tolist()[0]
            array.append(indices)
        return array
