import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_normal_


# 对FM进行改进：特征交叉部分使用神经网络来进行——特征交叉池化层，两两特征元素积后对交叉向量取和，再输入多层全连接神经网络
class NFM(nn.Module):
    def __init__(self, num_users, num_items, hidden_units, embedding_dim):
        super().__init__()
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.age_embedding = nn.Embedding(1, embedding_dim)
        self.gender_embedding = nn.Embedding(2, embedding_dim)
        self.occupation_embedding = nn.Embedding(21, embedding_dim)
        self.movie_embedding = nn.Embedding(19, embedding_dim)

        # 特征交叉部分
        self.linear = nn.Linear(embedding_dim, hidden_units[0])  # 连接的隐向量转化为DNN输入的维度
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.relu = nn.ReLU()

        # 一阶部分
        self.user = nn.Embedding(num_users, 1)  # 用embedding的方式one-hot用户id和物品id
        self.item = nn.Embedding(num_items, 1)
        self.wide = nn.Linear(1 + 2 + 21 + 19, 1)  # 年龄、性别、职业、电影类型

        # Final layer
        self.output = nn.Linear(2, 1)

        # embedding初始化
        xavier_normal_(self.user_embedding.weight.data)
        xavier_normal_(self.item_embedding.weight.data)
        xavier_normal_(self.age_embedding.weight.data)
        xavier_normal_(self.gender_embedding.weight.data)
        xavier_normal_(self.occupation_embedding.weight.data)
        xavier_normal_(self.movie_embedding.weight.data)
        xavier_normal_(self.user.weight.data)
        xavier_normal_(self.item.weight.data)

    def forward(self, x):
        # Embeddings
        user_embed = self.user_embedding(x[:, 0].long())
        item_embed = self.item_embedding(x[:, 1].long())
        age_embed = torch.matmul(x[:, 2].unsqueeze(1), self.age_embedding.weight)
        # age = x[:, 2].unsqueeze(1)
        gender_embed = torch.matmul(x[:, 3:5], self.gender_embedding.weight)
        occupation_embed = torch.matmul(x[:, 5:26], self.occupation_embedding.weight)
        movie_embed = torch.matmul(x[:, 26:45], self.movie_embedding.weight)

        # 一阶部分
        wide_output = self.user(x[:, 0].long()) + self.item(x[:, 1].long()) + self.wide(x[:, 2:])

        # 特征交叉部分
        # age = age.expand(-1, user_embed.size(1))  # 将年龄进行广播，与embedding向量维度保持一致，以进行内积运算
        feature_list = [user_embed, item_embed, age_embed, gender_embed, occupation_embed, movie_embed]
        cross_sum = 0.0
        for i in range(len(feature_list)):
            for j in range(i + 1, len(feature_list)):
                cross_sum += feature_list[i] * feature_list[j]  # 特征元素积操作
        deep_input = self.linear(cross_sum)
        for linear in self.dnn_network:
            deep_input = linear(deep_input)
            deep_input = self.relu(deep_input)

        # 结合一阶部分和特征交叉部分
        combined_output = torch.cat((wide_output, deep_input), dim=1)
        final_output = torch.sigmoid(self.output(combined_output))

        return final_output

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
