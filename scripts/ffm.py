import random

import numpy as np
import torch.nn
from torch import optim
import pandas as pd

import sys

sys.path.append('../')

from data.reader import MovieLens100K
from model.ffm import FFM
from sampler.sampler import Sampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = MovieLens100K('../dataset_example/ml-100k')

# 负采样
# 将训练集、验证集、测试集中存在的用户-项目对得出
train_set = set(data.train.apply(lambda row: (row['user_id'], row['item_id']), axis=1))
validation_set = set(data.validation.apply(lambda row: (row['user_id'], row['item_id']), axis=1))
test_set = set(data.test.apply(lambda row: (row['user_id'], row['item_id']), axis=1))
excluded_pairs = train_set | validation_set | test_set
# 生成训练集负样本，并合并数据
train_sampler = Sampler()
train_negative = train_sampler.negative_sampling2(data.num_users, data.num_items, excluded_pairs, 10)  # 生成用户、项目、评分数据
train_negative = pd.merge(train_negative, data.user_data, on='user_id')  # 将负样本也连接上用户和物品信息
train_negative = pd.merge(train_negative, data.item_data, on='item_id')
train_combined = pd.concat([data.train, train_negative], axis=0).reset_index(drop=True)  # 合并正负样本
# 生成验证集负样本，并合并数据
validation_sampler = Sampler()
validation_negative = validation_sampler.negative_sampling2(data.num_users, data.num_items, excluded_pairs, 10)
validation_negative = pd.merge(validation_negative, data.user_data, on='user_id')
validation_negative = pd.merge(validation_negative, data.item_data, on='item_id')
validation_combined = pd.concat([data.validation, validation_negative], axis=0).reset_index(drop=True)
# 生成测试集负样本，并合并数据
test_sampler = Sampler()
test_negative = test_sampler.negative_sampling2(data.num_users, data.num_items, excluded_pairs, 10)
test_negative = pd.merge(test_negative, data.user_data, on='user_id')
test_negative = pd.merge(test_negative, data.item_data, on='item_id')
test_combined = pd.concat([data.test, test_negative], axis=0).reset_index(drop=True)

# 生成训练集、验证集、测试集的训练特征向量以及评分的torch
train_data = torch.tensor(train_combined.iloc[:, 3:].values, dtype=torch.float32)  # 去除user_di、item_id、rating
train_rating = torch.tensor(train_combined.iloc[:, 2].values, dtype=torch.float32).unsqueeze(1)  # 评分数据，用于计算损失
validation_data = torch.tensor(validation_combined.iloc[:, 3:].values, dtype=torch.float32)
validation_rating = torch.tensor(validation_combined.iloc[:, 2].values, dtype=torch.float32).unsqueeze(1)
test_data = torch.tensor(test_combined.iloc[:, 3:].values, dtype=torch.float32)
test_rating = torch.tensor(test_combined.iloc[:, 2].values, dtype=torch.float32).unsqueeze(1)

model = FFM(train_data.shape[1], 64)
loss_fn = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模型训练
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    train_predictions = model(train_data)
    train_loss = loss_fn(train_predictions, train_rating)
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        validation_predictions = model(validation_data)
        validation_loss = loss_fn(validation_predictions, validation_rating)
        if epoch < epochs - 1:
            print(f'Epoch {epoch + 1}: Training loss: {train_loss.item()} Validation Loss: {validation_loss.item()}')
        else:
            test_predictions = model(test_data)
            test_loss = loss_fn(test_predictions, test_rating)
            print(
                f'Epoch {epoch + 1}: Training loss: {train_loss.item()} Validation Loss: {validation_loss.item()} Test Loss: {test_loss.item()}')

# 推荐部分
recommendation = model.recommendation(data.num_users, data.user_item, 100)
same = 0.0
p = 0.0
r = 0.0
for i in range(data.num_users):
    real_list = data.data[data.data['user_id'] == i]['item_id'].values  # 用户真实评过分的物品id列表
    recommendation_list = recommendation[i]  # 用户的推荐列表
    same += len(set(real_list) & set(recommendation_list))
    p += len(recommendation_list)  # 推荐总数
    r += len(real_list)  # 用户真实评过分的物品数目
precision = same / p
recall = same / r
f1 = 2*precision*recall/(precision+recall)
print(f'Precision: {precision},Recall: {recall},F1: {f1}')
