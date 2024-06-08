import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_normal_


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.cross_weights = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_layers)])
        self.cross_biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            x = x0 * self.cross_weights[i](x) + self.cross_biases[i] + x
        return x


class DeepNetwork(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super().__init__()
        layers = []
        for in_dim, out_dim in zip([input_dim] + hidden_units[:-1], hidden_units):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepCross(nn.Module):
    def __init__(self, num_users, num_items, cross_layers, deep_hidden_units, embedding_dim):
        super().__init__()
        """
        cross_layers: cross部分的层数
        deep_hidden_units：deep部分的神经单元个数，列表形式
        """
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.gender_embedding = nn.Embedding(2, embedding_dim)
        self.occupation_embedding = nn.Embedding(21, embedding_dim)
        self.movie_embedding = nn.Embedding(19, embedding_dim)

        self.cross_network = CrossNetwork(embedding_dim * 5 + 1, cross_layers)
        self.deep_network = DeepNetwork(embedding_dim * 5 + 1, deep_hidden_units)
        self.output_layer = nn.Linear(embedding_dim * 5 + 1 + deep_hidden_units[-1], 1)

        # embedding初始化
        xavier_normal_(self.user_embedding.weight.data)
        xavier_normal_(self.item_embedding.weight.data)
        xavier_normal_(self.gender_embedding.weight.data)
        xavier_normal_(self.occupation_embedding.weight.data)
        xavier_normal_(self.movie_embedding.weight.data)

    def forward(self, x):
        # Embeddings
        user_embed = self.user_embedding(x[:, 0].long())
        item_embed = self.item_embedding(x[:, 1].long())
        age = x[:, 2].unsqueeze(1)
        gender_embed = torch.matmul(x[:, 3:5], self.gender_embedding.weight)
        occupation_embed = torch.matmul(x[:, 5:26], self.occupation_embedding.weight)
        movie_embed = torch.matmul(x[:, 26:45], self.movie_embedding.weight)

        # 连接类别特征和数值特征
        x = torch.cat((user_embed, item_embed, age, gender_embed, occupation_embed, movie_embed), dim=1)

        cross_output = self.cross_network(x)
        deep_output = self.deep_network(x)

        combined_output = torch.cat((cross_output, deep_output), dim=1)
        final_output = torch.sigmoid(self.output_layer(combined_output))

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
