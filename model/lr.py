# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 22:00
# @Author  : Lumen
# @File    : mf.py
import torch
from torch import nn
from torch.nn.init import xavier_normal_


class LogisticRegression(nn.Module):
    def __init__(self, num_feature: int, num_constant: int == 1):
        super().__init__()

        # 用户和物品的embedding
        self.W = nn.Embedding(num_feature, 1)
        self.B = nn.Embedding(num_constant, 1)

        # embedding初始化
        xavier_normal_(self.W.weight.data)
        xavier_normal_(self.B.weight.data)

    def forward(self, feature_vactor: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.matmul(feature_vactor, self.W.weight) + self.B.weight)
