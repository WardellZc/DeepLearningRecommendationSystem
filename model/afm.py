import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_normal_


class AFM(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, attention_dim):
        super().__init__()
        """
        embedding_dim：类别特征向量化的维度
        attention_dim：注意力网络的映射维度
        """
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.gender_embedding = nn.Embedding(2, embedding_dim)
        self.occupation_embedding = nn.Embedding(21, embedding_dim)
        self.movie_embedding = nn.Embedding(19, embedding_dim)

        # 特征交叉部分
        self.attention_W = nn.Parameter(torch.randn(embedding_dim, attention_dim))
        self.attention_b = nn.Parameter(torch.randn(attention_dim))
        self.attention_h = nn.Parameter(torch.randn(attention_dim, 1))
        self.output_layer = nn.Linear(embedding_dim, 1)

        # 线性部分
        self.user = nn.Embedding(num_users, 1)  # 用embedding的方式one-hot用户id和物品id
        self.item = nn.Embedding(num_items, 1)
        self.linear = nn.Linear(1 + 2 + 21 + 19, 1)  # 年龄、性别、职业、电影类型

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

        # 线性部分
        linear_output = self.user(x[:, 0].long()) + self.item(x[:, 1].long()) + self.linear(x[:, 2:])

        # 特征交叉部分
        age = age.expand(-1, user_embed.size(1))  # 将年龄进行广播，与embedding向量维度保持一致，以进行内积运算
        feature_list = [user_embed, item_embed, age, gender_embed, occupation_embed, movie_embed]
        cross_product = []
        for i in range(len(feature_list)):
            for j in range(i + 1, len(feature_list)):
                cross_product.append(feature_list[i] * feature_list[j])  # 特征元素积操作
        cross_product = torch.stack(cross_product, dim=1)

        # 注意力机制
        attention_scores = torch.relu(torch.matmul(cross_product, self.attention_W) + self.attention_b)
        attention_weights = torch.softmax(torch.matmul(attention_scores, self.attention_h), dim=1)
        attention_output = torch.sum(attention_weights * cross_product, dim=1)
        cross_output = self.output_layer(attention_output)

        # 结合线性部分和特征交叉部分
        final_output = torch.sigmoid(linear_output + cross_output)

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
