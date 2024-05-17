import torch
from torch import nn
from torch.nn.init import xavier_normal_


# 自定义一个残差块
class ResidualBlock(nn.Module):
    """
    Define Residual_block

    注意：残差块的输入输出需要一致
    """

    def __init__(self, hidden_unit, dim_stack):
        super().__init__()
        # 两个线性层   注意维度， 输出的时候和输入的那个维度一致， 这样才能保证后面的相加
        self.linear1 = nn.Linear(dim_stack, hidden_unit)
        self.linear2 = nn.Linear(hidden_unit, dim_stack)
        self.relu = nn.ReLU()

    def forward(self, x):
        orig_x = x.clone()
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        outputs = self.relu(x + orig_x)
        return outputs


class DeepCrossing(nn.Module):
    def __init__(self, num_user, num_item, num_feature, hidden_units):
        """
        num_feature:embedding层的维度
        hidden_units:残差层的维度，也可以用于设置残差层的数量
        """
        super().__init__()
        # 对类别特征的one-hot向量进行embedding层的转化，user_id、item_id、gender、occupation、movie
        self.user_embedding = nn.Embedding(num_user, num_feature)
        self.item_embedding = nn.Embedding(num_item, num_feature)
        self.gender_embedding = nn.Embedding(2, num_feature)
        self.occupation_embedding = nn.Embedding(21, num_feature)
        self.movie_embedding = nn.Embedding(19, num_feature)
        # 初始化
        xavier_normal_(self.user_embedding.weight.data)
        xavier_normal_(self.item_embedding.weight.data)
        xavier_normal_(self.gender_embedding.weight.data)
        xavier_normal_(self.occupation_embedding.weight.data)
        xavier_normal_(self.movie_embedding.weight.data)

        # 统计拼接后的特征维数,类别特征和数值特征的维度和
        dim_stack = num_feature * 5 + 1

        # 残差层
        self.res_layers = nn.ModuleList([
            ResidualBlock(unit, dim_stack) for unit in hidden_units
        ])

        # 线性层
        self.linear = nn.Linear(dim_stack, 1)

    def forward(self, feature_vector: torch.Tensor) -> torch.Tensor:
        # embedding层
        user_embed = self.user_embedding(feature_vector[:, 0].long())  # embedding要求是整数类型
        item_embed = self.item_embedding(feature_vector[:, 1].long())  # embedding要求是整数类型
        age = feature_vector[:, 2].unsqueeze(1)
        gender_embed = torch.matmul(feature_vector[:, 3:5], self.gender_embedding.weight)
        occupation_embed = torch.matmul(feature_vector[:, 5:26], self.occupation_embedding.weight)
        movie_embed = torch.matmul(feature_vector[:, 26:45], self.movie_embedding.weight)

        # stacking层
        r = torch.cat([user_embed, item_embed, age, gender_embed, occupation_embed, movie_embed], dim=1)

        # Multiple Residual Units层
        for res in self.res_layers:
            r = res(r)

        # scoring 层
        outputs = torch.sigmoid(self.linear(r))
        return outputs

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
        return array
