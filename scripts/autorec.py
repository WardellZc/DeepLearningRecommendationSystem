# 基于用户的AutoRec
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

import sys

sys.path.append('../')

from data.reader import MovieLens100K
from model.autorec import AutoRec
from sampler.sampler import Sampler
from trainer.trainer import Trainer
from evaluator.ranking import Ranking

# 加载数据
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = MovieLens100K('../dataset_example/ml-100k')

# 负采样
# 对整体评分进行负采样，这样在数据集划分后，训练集、验证集、测试集中都有负样本
excluded_pairs = set(data.data.apply(lambda row: (row['user_id'], row['item_id']), axis=1))
sampler = Sampler()
negative = sampler.negative_sampling2(data.num_users, data.num_items, excluded_pairs, 150)  # 生成用户、项目、评分数据
combined = pd.concat([data.data, negative], axis=0).reset_index(drop=True)  # 合并正负样本

# 构造矩阵数据
# 未评分的数据用0.5代替，以便后面生成掩码
rating_matrix = combined.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0.5)
# 将评分矩阵划分为训练集、验证集、训练集
# 先划分为测试集，再从剩下的数据划分出验证集
train_valid, test = train_test_split(rating_matrix, test_size=0.2, random_state=42)
train, valid = train_test_split(train_valid, test_size=0.25, random_state=42)

# 将数据转化为张量
train = torch.tensor(train.values, dtype=torch.float32).to(device)
valid = torch.tensor(valid.values, dtype=torch.float32).to(device)
test = torch.tensor(test.values, dtype=torch.float32).to(device)
# 生成掩码,未评分的地方为False，其余为True
train_mask = (train != 0.5)
valid_mask = (valid != 0.5)
test_mask = (test != 0.5)

# 定义模型
model = AutoRec(data.num_items, 256).to(device)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

# 模型训练
trainer = Trainer(model, loss_fn, optimizer)
epochs = 100
for epoch in range(epochs):
    trainer.train_loop2(train, train_mask)
    trainer.valid_loop2(valid, valid_mask)
    trainer.test_loop2(test, test_mask)
    trainer.model_eval(epoch)

# 推荐部分
k = 100
real_list = data.itemid_matrix()
rating_matrix = torch.tensor(rating_matrix.values, dtype=torch.float32).to(device)
roc_list = model.recommendation(rating_matrix, k)
rank = Ranking(real_list, roc_list, k)
rank.ranking_eval()





