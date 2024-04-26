# 基于物品的AutoRec
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

import sys

sys.path.append('../')

from data.reader import MovieLens100K
from model.autorec import AutoRec
from sampler.sampler import Sampler
from trainer.trainer import Trainer

# 加载数据
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = MovieLens100K('../dataset_example/ml-100k')

# 负采样
# 对整体评分进行负采样，这样在数据集划分后，训练集、验证集、测试集中都有负样本
excluded_pairs = set(data.data.apply(lambda row: (row['user_id'], row['item_id']), axis=1))
sampler = Sampler()
negative = sampler.negative_sampling2(data.num_users, data.num_items, excluded_pairs, 50)  # 生成用户、项目、评分数据
combined = pd.concat([data.data, negative], axis=0).reset_index(drop=True)  # 合并正负样本

# 构造矩阵数据
# 未评分的数据用0.5代替，以便后面生成掩码
rating_matrix = combined.pivot_table(index='item_id', columns='user_id', values='rating', fill_value=0.5)
# 将评分矩阵划分为训练集、验证集、训练集
# 先划分为测试集，再从剩下的数据划分出验证集
train_valid, test = train_test_split(rating_matrix, test_size=0.2, random_state=42)
train, valid = train_test_split(train_valid, test_size=0.25, random_state=42)
# 生成掩码,未评分的地方为0，其余为1
train_mask = train.applymap(lambda x: 0 if x == 0.5 else 1)
valid_mask = valid.applymap(lambda x: 0 if x == 0.5 else 1)
test_mask = test.applymap(lambda x: 0 if x == 0.5 else 1)

# 将数据转化为张量
train = torch.tensor(train.values, dtype=torch.float32).to(device)
valid = torch.tensor(valid.values, dtype=torch.float32).to(device)
test = torch.tensor(test.values, dtype=torch.float32).to(device)
train_mask = torch.tensor(train_mask.values, dtype=torch.float32).to(device)
valid_mask = torch.tensor(valid_mask.values, dtype=torch.float32).to(device)
test_mask = torch.tensor(test_mask.values, dtype=torch.float32).to(device)

# 定义模型
model = AutoRec(data.num_users, 64)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 模型训练
trainer = Trainer(model, loss_fn, optimizer)
epochs = 50
for epoch in range(epochs):
    trainer.train_loop3(train, train_mask)
    trainer.valid_loop3(valid, valid_mask)
    trainer.test_loop3(test, test_mask)
    trainer.model_eval2(epoch)


# 推荐部分
rating_matrix = torch.tensor(rating_matrix.values, dtype=torch.float32)
recommendation = model.i_recommendation(rating_matrix, 100)
same = 0.0
p = 0.0
r = 0.0
for i in range(data.num_users):
    real_list = data.data[data.data['user_id'] == i]['item_id'].values  # 用户真实评过分的物品id列表
    recommendation_list = recommendation[:, i]  # 用户的推荐列表
    same += len(set(real_list) & set(recommendation_list))
    p += len(recommendation_list)  # 推荐总数
    r += len(real_list)  # 用户真实评过分的物品数目
precision = same / p
recall = same / r
f1 = 2*precision*recall/(precision+recall)
print(f'Precision: {precision},Recall: {recall},F1: {f1}')

