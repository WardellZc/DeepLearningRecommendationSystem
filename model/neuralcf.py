import torch
from torch import nn


class NeuralCF(nn.Module):
    def __init__(self, num_user, num_item, mf_dim, layers):
        super().__init__()
        # Embedding隐向量
        self.GMF_Embedding_User = nn.Embedding(num_user, mf_dim)
        self.GMF_Embedding_Item = nn.Embedding(num_item, mf_dim)
        self.MLP_Embedding_User = nn.Embedding(num_user, int(layers[0] / 2))
        self.MLP_Embedding_Item = nn.Embedding(num_item, int(layers[0] / 2))

        # 全连接网络
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])
        self.relu = nn.ReLU()
        # 将mlp维度降与GMF一致
        self.linear = nn.Linear(layers[-1], mf_dim)

        # 连接GMF+MLP，二分类
        self.linear2 = nn.Linear(2 * mf_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        # GMF网络
        # 用户和物品的embedding
        gmf_embedding_user = self.GMF_Embedding_User(user_indices)
        gmf_embedding_item = self.GMF_Embedding_Item(item_indices)
        # 元素乘连接
        gmf_vec = torch.mul(gmf_embedding_user, gmf_embedding_item)

        # MLP网络
        # 用户和物品的embedding
        mlp_embedding_user = self.MLP_Embedding_User(user_indices)
        mlp_embedding_item = self.MLP_Embedding_Item(item_indices)
        # 连接两个隐向量
        x = torch.cat([mlp_embedding_user, mlp_embedding_item], dim=1)
        # 全连接网络
        for linear in self.dnn_network:
            x = linear(x)
            x = self.relu(x)
        mlp_vec = self.linear(x)

        # 连接gmf和mlp两个向量
        vector = torch.cat([gmf_vec, mlp_vec], dim=1)

        # 输出层预测评分
        linear = self.linear2(vector)
        output = self.sigmoid(linear)
        return output

    def recommendation(self, num_users, num_items, k):
        array = []
        # 获取模型参数所在的设备
        device = next(self.parameters()).device
        for i in range(num_users):
            user_vector = torch.full((num_items,), i).to(device)
            item_vector = torch.arange(num_items).to(device)
            scores = self.forward(user_vector, item_vector)
            values, indices = torch.topk(scores, k, dim=0)
            indices = indices.view(-1).tolist()
            array.append(indices)
        return array

