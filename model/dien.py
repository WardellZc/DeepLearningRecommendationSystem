import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
import numpy as np


# DIN 模型定义
class DIN(nn.Module):
    def __init__(self, num_items, embed_size):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embed_size)
        # 注意力激活单元
        self.attention = nn.Sequential(
            nn.Linear(embed_size * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        att_hist_embed = (hist_embed * att_weights.unsqueeze(-1))  # (batch_size, hist_len, embed_size)

        return att_hist_embed, target_item_embed


# DIEN 模型定义
class DIEN(nn.Module):
    def __init__(self, num_items, embed_size):
        super().__init__()
        self.din = DIN(num_items, embed_size)
        self.interest_evolution = nn.GRU(embed_size, embed_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embed_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, hist, target_item):
        att_hist_embed, target_item_embed = self.din(hist, target_item)

        # 进行兴趣进化
        output, hidden = self.interest_evolution(att_hist_embed)  # output: (batch_size, hist_len, embed_size), hidden: (1, batch_size, embed_size)

        # 获取最后一个时间步的隐状态作为兴趣向量
        final_interest = hidden[-1]  # (batch_size, embed_size)

        x = torch.cat([final_interest, target_item_embed], dim=-1)  # (batch_size, embed_size * 2)
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
