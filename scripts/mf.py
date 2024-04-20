# -*- coding: utf-8 -*-
# @Time    : 2024/3/30 21:23
# @Author  : Lumen
# @File    : mf.py

import torch.nn
from torch import optim

import sys

sys.path.append('../')

from data.reader import MovieLens100K
from model.mf import MatrixFactorization
from sampler.sampler import Sampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = MovieLens100K('../dataset_example/ml-100k')

# 读取数据
train_user, train_item, train_rating = data.train_interaction(device=device)
validation_user, validation_item, validation_rating = data.validation_interaction(device=device)
test_user, test_item, test_rating = data.test_interaction(device=device)

model = MatrixFactorization(data.num_users, data.num_items, 64).to(device)
loss_fn = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 负采样
# 先得到训练集、验证集、测试集中已有的用户-项目对
excluded_pairs = set(zip(train_user.cpu().numpy(), train_item.cpu().numpy())) | set(
    zip(validation_user.cpu().numpy(), validation_item.cpu().numpy())) | set(
    zip(test_user.cpu().numpy(), test_item.cpu().numpy()))
# 生成训练集负样本，并合并数据
train_sampler = Sampler()
train_negative_users, train_negative_items, train_negative_ratings = train_sampler.negative_sampling(data.num_users,
                                                                                                     data.num_items,
                                                                                                     excluded_pairs,
                                                                                                     10)
train_users_combined = torch.cat([train_user, train_negative_users]).to(device)
train_items_combined = torch.cat([train_item, train_negative_items]).to(device)
train_ratings_combined = torch.cat([train_rating, train_negative_ratings]).to(device)
# 生成验证集负样本，并合并数据
validation_sampler = Sampler()
validation_negative_users, validation_negative_items, validation_negative_ratings = validation_sampler.negative_sampling(
    data.num_users,
    data.num_items,
    excluded_pairs,
    10)
validation_users_combined = torch.cat([validation_user, validation_negative_users]).to(device)
validation_items_combined = torch.cat([validation_item, validation_negative_items]).to(device)
validation_ratings_combined = torch.cat([validation_rating, validation_negative_ratings]).to(device)
# 生成测试集负样本，并合并数据
test_sampler = Sampler()
test_negative_users, test_negative_items, test_negative_ratings = test_sampler.negative_sampling(data.num_users,
                                                                                                 data.num_items,
                                                                                                 excluded_pairs,
                                                                                                 10)
test_users_combined = torch.cat([test_user, test_negative_users]).to(device)
test_items_combined = torch.cat([test_item, test_negative_items]).to(device)
test_ratings_combined = torch.cat([test_rating, test_negative_ratings]).to(device)

epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions_train = model(train_users_combined, train_items_combined)
    train_loss = loss_fn(predictions_train, train_ratings_combined)
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions_validation = model(validation_users_combined, validation_items_combined)
        validation_loss = loss_fn(predictions_validation, validation_ratings_combined)
        if epoch < epochs - 1:
            print(f'Epoch {epoch + 1}: Training loss: {train_loss.item()} Validation Loss: {validation_loss.item()}')
        else:
            predictions_test = model(test_users_combined, test_items_combined)
            test_loss = loss_fn(predictions_test, test_ratings_combined)
            print(f'Epoch {epoch + 1}: Training loss: {train_loss.item()} Validation Loss: {validation_loss.item()} Test'
                  f'Loss: {test_loss.item()}')

# 推荐部分
recommendation = model.recommendation(data.num_users, data.num_items, 100)  # 得到推荐列表矩阵，每一个行代表给每个用户推荐的物品id
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
