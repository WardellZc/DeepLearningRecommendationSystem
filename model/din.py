import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DIN(nn.Module):
    def __init__(self, num_items, embed_size):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embed_size)
        # 注意力激活单元
        self.attention = nn.Sequential(
            nn.Linear(embed_size * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # DNN神经网络
        self.fc = nn.Sequential(
            nn.Linear(embed_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # embedding初始化
        xavier_normal_(self.item_embedding.weight.data)

    def forward(self, hist, target_item):
        # 目标物品、历史行为物品列表embedding化
        target_item_embed = self.item_embedding(target_item)  # (batch_size, embed_size)
        hist_embed = self.item_embedding(hist)  # (batch_size, hist_len, embed_size)

        # 激活单元，原始向量与元素减后的结果连接，输入神经网络得到注意力得分
        target_item_embed_expanded = target_item_embed.unsqueeze(1).expand_as(
            hist_embed)  # (batch_size, hist_len, embed_size)
        concat_embed = torch.cat([hist_embed, hist_embed - target_item_embed_expanded, target_item_embed_expanded],
                                 dim=-1)  # (batch_size, hist_len, embed_size * 3)
        att_weights = self.attention(concat_embed).squeeze(-1)  # (batch_size, hist_len)
        att_weights = torch.softmax(att_weights, dim=-1)  # (batch_size, hist_len)

        # 用户embedding向量的加权和
        att_hist_embed = (hist_embed * att_weights.unsqueeze(-1)).sum(dim=1)  # (batch_size, embed_size)

        # 连接用户历史行为物品和目标物品的embedding向量
        x = torch.cat([att_hist_embed, target_item_embed], dim=1)  # (batch_size, embed_size * 2)
        output = self.fc(x)  # (batch_size,)

        return output

    def recommendation(self, num_users, num_items, hist_list, k):
        array = []
        device = next(self.parameters()).device
        for i in range(num_users):
            target_item = torch.arange(0, num_items).to(device)
            list = torch.tensor(hist_list[i])
            hist = list.repeat(num_items, 1).to(device)
            scores = self.forward(hist, target_item)
            values, indices = torch.topk(scores, k, dim=0)
            indices = indices.view(1, -1).tolist()[0]
            array.append(indices)
        return np.array(array)
