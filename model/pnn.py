import torch
import numpy as np
from torch import nn
from torch.nn.init import xavier_normal_


# 全连接层 DNN
class DNN(nn.Module):
    def __init__(self, hidden_units):
        """
        hidden_units: 列表，每个元素表示每一层的神经单元个数，也可用来控制神经网络层数；如[256,128,64],表示两层神经网络，形状分别为（256,128）、（128,64）
        """
        super().__init__()
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.relu = nn.ReLU()

    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = self.relu(x)

        return x


# 乘积层 ProductLayers
class ProductLayers(nn.Module):
    def __init__(self, num_feature, embed_dim, hidden_units, model="in"):
        """
        分为三个部分：线性操作、乘积操作、融合部分
        线性操作部分：将embedding化的特征线性拼接
        乘积操作部分：将特征两两进行内积或者外积操作
        融合部分：将线性操作和乘积操作融合，分别映射成全连接层输入维度的向量lz和lp，再将lz和lp叠加后输入DNN

        num_feature: 特征的个数
        embed_dim: 特征向量的维度
        hidden_units: DNN神经网络个数列表，同上
        model: 乘积操作的类型，默认是内积操作
        """
        super().__init__()
        self.model = model
        # 将乘线性操作部分的结果转化为DNN输入维度的向量
        self.linear1 = nn.Linear(num_feature * embed_dim, hidden_units[0])
        # 将乘积操作部分的结果转化为DNN输入维度的向量
        if self.model == "in":
            self.linear2 = nn.Linear(int(num_feature * (num_feature - 1) / 2), hidden_units[0])
        elif self.model == "out":
            self.linear2 = nn.Linear(embed_dim, hidden_units[0])

    def forward(self, feature_embed):
        """
        feature_embed: embedding化后的特征向量列表  [tensor1,tensor2,...]
        """
        # 线性操作部分 z
        z = torch.cat(feature_embed, dim=1).unsqueeze(0)

        # 乘积操作部分 p
        if self.model == "in":
            n = len(feature_embed)
            array = []
            for i in range(n):
                for j in range(i + 1, n):
                    in_result = (feature_embed[i] * feature_embed[j]).sum(dim=1, keepdim=True)
                    array.append(in_result)
            # 将两两内积的结果存于列表中，再转化为张量
            p = torch.cat(array, dim=1)
        elif self.model == "out":
            # 叠加tensor
            stacked_tensor = torch.stack(feature_embed)
            # 逐元素相加
            sum_tensor = torch.sum(stacked_tensor, dim=0)
            p = torch.matmul(sum_tensor.T, sum_tensor)

        # 融合部分
        lz = self.linear1(z)
        lp = self.linear2(p)
        result = lz + lp

        return result


# PNN 模型
class PNN(nn.Module):
    def __init__(self, embed_dim, hidden_units, model="in"):
        super().__init__()
        # embedding 层
        self.user_embed = nn.Embedding(943, embed_dim)
        self.item_embed = nn.Embedding(1682, embed_dim)
        self.age_embed = nn.Embedding(1, embed_dim)
        self.gender_embed = nn.Embedding(2, embed_dim)
        self.occupation_embed = nn.Embedding(21, embed_dim)
        self.movie_embed = nn.Embedding(19, embed_dim)

        # embedding初始化
        xavier_normal_(self.user_embed.weight.data)
        xavier_normal_(self.item_embed.weight.data)
        xavier_normal_(self.age_embed.weight.data)
        xavier_normal_(self.gender_embed.weight.data)
        xavier_normal_(self.occupation_embed.weight.data)
        xavier_normal_(self.movie_embed.weight.data)

        # 乘积层
        self.product = ProductLayers(6, embed_dim, hidden_units, model)

        # DNN层
        self.dnn = DNN(hidden_units)

        # 输出层
        self.output = nn.Linear(hidden_units[-1], 1)

    def forward(self, x):
        # embedding层
        user_embed = self.user_embed(x[:, 0].long())
        item_embed = self.item_embed(x[:, 1].long())
        age_embed = torch.matmul(x[:, 2].unsqueeze(1), self.age_embed.weight)
        gender_embed = torch.matmul(x[:, 3:5], self.gender_embed.weight)
        occupation_embed = torch.matmul(x[:, 5:26], self.occupation_embed.weight)
        movie_embed = torch.matmul(x[:, 26:45], self.movie_embed.weight)

        # 乘积层
        feature_embed = [user_embed, item_embed, age_embed, gender_embed, occupation_embed, movie_embed]
        product_result = self.product(feature_embed)

        # DNN层
        dnn_result = self.dnn(product_result)

        # 输出层,最后通过sigmoid函数转化为二分类问题
        output = self.output(dnn_result)
        result = torch.sigmoid(output).view(-1, 1)

        return result

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
