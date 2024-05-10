import torch
from torch import nn


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
        # 将用户id和物品id作为类别特征进行embedding
        self.user_embedding = nn.Embedding(num_user, num_feature)
        self.item_embedding = nn.Embedding(num_item, num_feature)

        # 统计拼接后的特征维数
        dim_stack = num_feature * 2

        # 残差层
        self.res_layers = nn.ModuleList([
            ResidualBlock(unit, dim_stack) for unit in hidden_units
        ])

        # 线性层
        self.linear = nn.Linear(dim_stack, 1)

    def forward(self, user_indices, item_indices) -> torch.Tensor:
        # embedding层
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)

        # stacking层
        r = torch.cat([user_embed, item_embed], dim=1)

        # Multiple Residual Units层
        for res in self.res_layers:
            r = res(r)

        # scoring 层
        outputs = torch.sigmoid(self.linear(r))
        return outputs

    def recommendation(self, num_users, num_items, k):
        array = []
        for i in range(num_users):
            user_vector = torch.full((num_items,), i)
            item_vector = torch.arange(num_items)
            scores = self.forward(user_vector, item_vector)
            values, indices = torch.topk(scores, k, dim=0)
            indices = indices.view(-1).tolist()
            array.append(indices)
        return array




