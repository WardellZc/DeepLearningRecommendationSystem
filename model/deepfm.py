import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_normal_


# 用FM代替wide部分
class DeepFM(nn.Module):
    def __init__(self, num_users, num_items, hidden_units, embedding_dim):
        super().__init__()
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.age_embedding = nn.Embedding(1, embedding_dim)
        self.gender_embedding = nn.Embedding(2, embedding_dim)
        self.occupation_embedding = nn.Embedding(21, embedding_dim)
        self.movie_embedding = nn.Embedding(19, embedding_dim)

        # Deep part
        self.linear = nn.Linear(embedding_dim * 6, hidden_units[0])  # 连接的隐向量转化为DNN输入的维度
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.relu = nn.ReLU()

        # FM part
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

        # 连接类别特征和数值特征
        deep_input = torch.cat((user_embed, item_embed, age_embed, gender_embed, occupation_embed, movie_embed), dim=1)

        # Deep part
        deep_input = self.linear(deep_input)
        for linear in self.dnn_network:
            deep_input = linear(deep_input)
            deep_input = self.relu(deep_input)

        # FM part
        wide_output = self.user(x[:, 0].long()) + self.item(x[:, 1].long()) + self.wide(x[:, 2:])  # FM线性部分
        # age = age.expand(-1, user_embed.size(1))    # 将年龄进行广播，与embedding向量维度保持一致，以进行内积运算
        # feature_list = [user_embed, item_embed, age_embed, gender_embed, occupation_embed, movie_embed]  # FM特征交叉部分
        # cross_sum = 0.0
        # for i in range(len(feature_list)):
        #     for j in range(i + 1, len(feature_list)):
        #         cross_sum += torch.sum(feature_list[i] * feature_list[j], dim=1)
        # 特征交叉简化计算，降低时间复杂度：将所有特征嵌入堆叠成一个矩阵
        feature_list = [user_embed, item_embed, age_embed, gender_embed, occupation_embed, movie_embed]
        features = torch.stack(feature_list, dim=1)  # shape: (batch_size, num_features, embed_size)
        # 计算特征交叉项
        sum_of_squares = torch.sum(features, dim=1) ** 2  # shape: (batch_size, embed_size)
        square_of_sum = torch.sum(features ** 2, dim=1)  # shape: (batch_size, embed_size)
        cross_sum = 0.5 * torch.sum(sum_of_squares - square_of_sum, dim=1)  # shape: (batch_size,)
        wide_output += cross_sum.unsqueeze(1)  # 线性部分与特征交叉部分相加

        # 结合wide部分和deep部分
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