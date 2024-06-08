import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_normal_


# wide部分——逻辑回归、deep部分——DNN网络
class WideDeep(nn.Module):
    def __init__(self, num_users, num_items, hidden_units, embedding_dim):
        super().__init__()
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.gender_embedding = nn.Embedding(2, embedding_dim)
        self.occupation_embedding = nn.Embedding(21, embedding_dim)
        self.movie_embedding = nn.Embedding(19, embedding_dim)

        # Deep part
        self.linear = nn.Linear(embedding_dim * 5 + 1, hidden_units[0])  # 连接的隐向量转化为DNN输入的维度
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.relu = nn.ReLU()

        # Wide part
        self.user = nn.Embedding(num_users, 1)  # 用embedding的方式one-hot用户id和物品id
        self.item = nn.Embedding(num_items, 1)
        self.wide = nn.Linear(1 + 2 + 21 + 19, 1)  # 年龄、性别、职业、电影类型

        # Final layer
        self.output = nn.Linear(2, 1)

        # embedding初始化
        xavier_normal_(self.user_embedding.weight.data)
        xavier_normal_(self.item_embedding.weight.data)
        xavier_normal_(self.gender_embedding.weight.data)
        xavier_normal_(self.occupation_embedding.weight.data)
        xavier_normal_(self.movie_embedding.weight.data)
        xavier_normal_(self.user.weight.data)
        xavier_normal_(self.item.weight.data)

    def forward(self, x):
        # Embeddings
        user_embed = self.user_embedding(x[:, 0].long())
        item_embed = self.item_embedding(x[:, 1].long())
        age = x[:, 2].unsqueeze(1)
        gender_embed = torch.matmul(x[:, 3:5], self.gender_embedding.weight)
        occupation_embed = torch.matmul(x[:, 5:26], self.occupation_embedding.weight)
        movie_embed = torch.matmul(x[:, 26:45], self.movie_embedding.weight)

        # 连接类别特征和数值特征
        deep_input = torch.cat((user_embed, item_embed, age, gender_embed, occupation_embed, movie_embed), dim=1)

        # Deep part
        deep_input = self.linear(deep_input)
        for linear in self.dnn_network:
            deep_input = linear(deep_input)
            deep_input = self.relu(deep_input)

        # Wide part
        wide_output = self.user(x[:, 0].long()) + self.item(x[:, 1].long()) + self.wide(x[:, 2:])

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

