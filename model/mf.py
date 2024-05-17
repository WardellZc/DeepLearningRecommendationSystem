# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 22:00
# @Author  : Lumen
# @File    : mf.py
import torch
from torch import nn
from torch.nn.init import xavier_normal_


class MatrixFactorization(nn.Module):

    def __init__(self, num_users: int, num_items: int, embedding_size: int):
        super().__init__()

        # 用户和物品的embedding
        self.user_embeddings = nn.Embedding(num_users, embedding_size)
        self.item_embeddings = nn.Embedding(num_items, embedding_size)

        # embedding初始化
        xavier_normal_(self.user_embeddings.weight.data)
        xavier_normal_(self.item_embeddings.weight.data)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        user_embedding = self.user_embeddings(user_indices)
        item_embedding = self.item_embeddings(item_indices)
        return torch.sigmoid(torch.sum(torch.multiply(user_embedding, item_embedding), dim=1))

    def recommendation(self, num_users, num_items, k):
        # 获取模型参数所在设备
        device = next(self.user_embeddings.parameters()).device
        user_vector = self.user_embeddings(torch.arange(num_users).to(device))
        item_vector = self.item_embeddings(torch.arange(num_items).to(device))
        ratings_matrix = torch.matmul(user_vector, item_vector.T)
        values, indices = torch.topk(ratings_matrix, k, dim=1)
        return indices.cpu().numpy()
