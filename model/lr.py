# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 22:00
# @Author  : Lumen
# @File    : mf.py
import torch
import numpy as np
from torch import nn
from torch.nn.init import xavier_normal_


class LogisticRegression(nn.Module):
    def __init__(self, num_users, num_items, num_feature: int):
        super().__init__()
        # 加入用户id和物品id进行逻辑回归
        self.user = nn.Embedding(num_users, 1)
        self.item = nn.Embedding(num_items, 1)
        # 特征向量参数w和b
        self.linear = nn.Linear(num_feature, 1, True)

        # embedding初始化
        xavier_normal_(self.user.weight.data)
        xavier_normal_(self.item.weight.data)

    def forward(self, feature_vector: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.user(feature_vector[:, 0].long()) + self.item(feature_vector[:, 1].long()) + self.linear(feature_vector[:, 2:]))

    def recommendation(self, num_users, user_item, k):
        array = []
        device = next(self.parameters()).device
        for i in range(num_users):
            user_vector = user_item[user_item['user_id'] == i]
            user_vector = torch.Tensor(user_vector.values).to(device)
            scores = self.forward(user_vector)
            values, indices = torch.topk(scores, k, dim=0)
            indices = indices.view(1, -1).tolist()[0]
            array.append(indices)
        return np.array(array)
