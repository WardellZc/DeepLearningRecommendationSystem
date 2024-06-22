import torch.nn
from torch import optim
import pandas as pd
import numpy as np

import sys

sys.path.append('../')

from data.reader import MovieLens100K
from model.dien import DIEN
from sampler.sampler import Sampler
from trainer.trainer import Trainer
from evaluator.ranking import Ranking

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = MovieLens100K('../dataset_example/ml-100k')

# 得到可训练的用户历史物品id列表
hist_len = 10  # 设置历史序列的长度


def pad_or_truncate(history, hist_len):
    # 确保 history 是一个列表
    if not isinstance(history, list):
        history = list(history)
    # 填充或截断列表
    if len(history) >= hist_len:
        return history[:hist_len]
    else:
        return [0] * (hist_len - len(history)) + history


# 负采样
# 将训练集、验证集、测试集中存在的用户-项目对得出
train_set = set(data.train.apply(lambda row: (row['user_id'], row['item_id']), axis=1))
valid_set = set(data.valid.apply(lambda row: (row['user_id'], row['item_id']), axis=1))
test_set = set(data.test.apply(lambda row: (row['user_id'], row['item_id']), axis=1))
excluded_pairs = train_set | valid_set | test_set
# 生成训练集负样本，并合并数据
train_sampler = Sampler()
train_negative = train_sampler.negative_sampling2(data.num_users, data.num_items, excluded_pairs, 30)  # 生成用户、项目、评分数据
train_combined = pd.concat([data.train, train_negative], axis=0).reset_index(drop=True)  # 合并正负样本
# 生成验证集负样本，并合并数据
valid_sampler = Sampler()
valid_negative = valid_sampler.negative_sampling2(data.num_users, data.num_items, excluded_pairs, 10)
valid_combined = pd.concat([data.valid, valid_negative], axis=0).reset_index(drop=True)  # 合并正负样本
# 生成测试集负样本，并合并数据
test_sampler = Sampler()
test_negative = test_sampler.negative_sampling2(data.num_users, data.num_items, excluded_pairs, 10)
test_combined = pd.concat([data.test, test_negative], axis=0).reset_index(drop=True)  # 合并正负样本

# 构建训练数据
# 生成训练集hist、target、rating
train_hist_list = data.itemid_matrix(data.train)
train_hist_list = np.array([row[row != -1] for row in train_hist_list],
                           dtype=object)  # 历史行为列表在生成时为了加快速度，转化为了numpy数组，用-1填补了空缺，现在要去掉
train_combined['history_list'] = train_combined['user_id'].apply(lambda x: train_hist_list[x])  # 将历史行为列表加入评分数据集
train_combined['history_list'] = train_combined['history_list'].apply(lambda x: pad_or_truncate(x, hist_len))
train_hist = torch.tensor(np.array(train_combined['history_list'].tolist()), dtype=torch.long).to(device)
train_target_item = torch.tensor(train_combined.iloc[:, 1].values, dtype=torch.long).to(device)
train_rating = torch.tensor(train_combined.iloc[:, 2].values, dtype=torch.float32).unsqueeze(1).to(device)  # 评分数据，用于计算损失
# print(train_hist)
# print(train_target_item)
# print(train_rating)
# 生成验证集hist、target、rating
valid_hist_list = data.itemid_matrix(data.valid)
valid_hist_list = np.array([row[row != -1] for row in valid_hist_list], dtype=object)
valid_combined['history_list'] = valid_combined['user_id'].apply(lambda x: valid_hist_list[x])  # 将历史行为列表加入评分数据集
valid_combined['history_list'] = valid_combined['history_list'].apply(lambda x: pad_or_truncate(x, hist_len))
valid_hist = torch.tensor(np.array(valid_combined['history_list'].tolist()), dtype=torch.long).to(device)
valid_target_item = torch.tensor(valid_combined.iloc[:, 1].values, dtype=torch.long).to(device)
valid_rating = torch.tensor(valid_combined.iloc[:, 2].values, dtype=torch.float32).unsqueeze(1).to(device)
# 生成测试集hist、target、rating
test_hist_list = data.itemid_matrix(data.test)
test_hist_list = np.array([row[row != -1] for row in test_hist_list], dtype=object)
test_combined['history_list'] = test_combined['user_id'].apply(lambda x: test_hist_list[x])  # 将历史行为列表加入评分数据集
test_combined['history_list'] = test_combined['history_list'].apply(lambda x: pad_or_truncate(x, hist_len))
test_hist = torch.tensor(np.array(test_combined['history_list'].tolist()), dtype=torch.long).to(device)
test_target_item = torch.tensor(test_combined.iloc[:, 1].values, dtype=torch.long).to(device)
test_rating = torch.tensor(test_combined.iloc[:, 2].values, dtype=torch.float32).unsqueeze(1).to(device)

# 定义模型
model = DIEN(data.num_items, 16).to(device)
loss_fn = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 模型训练
trainer = Trainer(model, loss_fn, optimizer)
epochs = 200
for epoch in range(epochs):
    trainer.train_loop(train_hist, train_target_item, train_rating=train_rating)
    trainer.valid_loop(valid_hist, valid_target_item, valid_rating=valid_rating)
    trainer.test_loop(test_hist, test_target_item, test_rating=test_rating)
    trainer.model_eval(epoch)

# 推荐部分
history_list = data.itemid_matrix(data.data)
history_list = np.array([row[row != -1] for row in history_list], dtype=object)
roc_list = model.recommendation(data.num_users, data.num_items, history_list, data.num_items)
train_real = data.itemid_matrix(data.train)
valid_real = data.itemid_matrix(data.valid)
test_real = data.itemid_matrix(data.test)
k = 50
# 验证集
valid_roc = data.remove_itemid(roc_list, train_real)  # 去除训练集中的物品
valid_roc = data.remove_itemid(valid_roc, test_real)  # 去除测试集中的物品
valid_rank = Ranking(valid_real, valid_roc, k)
print("验证集的指标：")
valid_rank.ranking_eval()
# 测试集
test_roc = data.remove_itemid(roc_list, train_real)
test_roc = data.remove_itemid(test_roc, valid_real)
test_rank = Ranking(test_real, test_roc, k)
print("测试集的指标：")
test_rank.ranking_eval()

# k = 100
# real_list = data.itemid_matrix()
# roc_list = model.recommendation(data.num_users, data.user_item(), k)
# rank = Ranking(real_list, roc_list, k)
# rank.ranking_eval()
