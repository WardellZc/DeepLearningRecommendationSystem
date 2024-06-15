import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DIN(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super().__init__()
        # 物品向量embedding
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # 注意力激活单元
        self.attention_fc = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # embedding初始化
        xavier_normal_(self.item_embedding.weight.data)

    # 输入用户历史有过行为的物品id序列，待推荐物品id:hist_item_ids,target_item_id
    def forward(self, x):
        # 处理物品ID
        item_indices = torch.tensor(x['item_id'].values, dtype=torch.long).to(device)
        if (item_indices >= self.item_embedding.num_embeddings).any() or (item_indices < 0).any():
            raise ValueError("Item index out of range in item_id")
        target_item_emb = self.item_embedding(item_indices)

        # 处理历史物品ID列表
        history_embs = []
        max_hist_length = max(x['history_list'].apply(len))  # 找到最长的历史列表长度
        for history in x['history_list']:
            history_tensor = torch.tensor(history, dtype=torch.long)
            if (history_tensor >= self.item_embedding.num_embeddings).any() or (history_tensor < 0).any():
                raise ValueError("Item index out of range in history_list")
            history_tensor = history_tensor.to(device)
            history_emb = self.item_embedding(history_tensor)
            if history_emb.size(0) < max_hist_length:
                padding = torch.zeros((max_hist_length - history_emb.size(0), history_emb.size(1))).to(device)
                history_emb = torch.cat((history_emb, padding), dim=0)
            history_embs.append(history_emb)

        # 注意力机制
        attention_weights = []
        for hist_emb in history_embs:
            repeat_target_emb = target_item_emb.unsqueeze(1).repeat(1, hist_emb.size(0), 1)
            concat_emb = torch.cat((hist_emb.unsqueeze(0), repeat_target_emb), dim=-1)
            attn_scores = self.attention_fc(concat_emb).squeeze(-1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            attention_weights.append(attn_weights.unsqueeze(-1) * hist_emb)

        # 计算加权历史嵌入
        weighted_history_emb = [torch.sum(attn, dim=0) for attn in attention_weights]
        weighted_history_emb = torch.stack(weighted_history_emb)

        # 进行元素减法
        diff_emb = weighted_history_emb - target_item_emb.unsqueeze(1)

        # 连接目标物品嵌入、加权历史嵌入和元素减法结果
        combined_emb = torch.cat((target_item_emb.unsqueeze(1), weighted_history_emb, diff_emb), dim=-1).squeeze(1)

        # 通过全连接层进行前向传播
        output = torch.sigmoid(self.fc(combined_emb))

        return output

        return output.squeeze()
