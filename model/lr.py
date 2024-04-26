# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 22:00
# @Author  : Lumen
# @File    : mf.py
import torch
from torch import nn
from torch.nn.init import xavier_normal_


class LogisticRegression(nn.Module):
    def __init__(self, num_feature: int):
        super().__init__()
        # 年龄、性别、职业和电影风格的embedding
        self.age_embeddings = nn.Embedding(1, num_feature)
        self.gender_embeddings = nn.Embedding(2, num_feature)
        self.occupations_embeddings = nn.Embedding(21, num_feature)
        self.movies_embeddings = nn.Embedding(19, num_feature)

        # embedding初始化
        xavier_normal_(self.age_embeddings.weight.data)
        xavier_normal_(self.gender_embeddings.weight.data)
        xavier_normal_(self.occupations_embeddings.weight.data)
        xavier_normal_(self.movies_embeddings.weight.data)

        # 特征向量参数w和b
        self.linear = nn.Linear(4 * num_feature, 1, True)

    def forward(self, feature_vector: torch.Tensor) -> torch.Tensor:
        # age:0,gender:1-2;occupation:3-23,movie:24-42
        age_vector = torch.matmul(feature_vector[:, 0].unsqueeze(1), self.age_embeddings.weight)
        gender_vector = torch.matmul(feature_vector[:, 1:3], self.gender_embeddings.weight)
        occupation_vector = torch.matmul(feature_vector[:, 3:24], self.occupations_embeddings.weight)
        # movie_vector为muti-hot向量,用矩阵乘法加和的方式
        movie_vector = torch.matmul(feature_vector[:, 24:43], self.movies_embeddings.weight)
        vector_combined = torch.cat((age_vector, gender_vector, occupation_vector, movie_vector), dim=1)
        return torch.sigmoid(self.linear(vector_combined))

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
