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
from trainer.trainer import Trainer
from evaluator.ranking import Ranking

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = MovieLens100K('../dataset_example/ml-100k')


# 读取数据
train_user, train_item, train_rating = data.train_interaction(device=device)
valid_user, valid_item, valid_rating = data.valid_interaction(device=device)
test_user, test_item, test_rating = data.test_interaction(device=device)

# 负采样
# 先得到训练集、验证集、测试集中已有的用户-项目对
excluded_pairs = set(zip(train_user.cpu().numpy(), train_item.cpu().numpy())) | set(
    zip(valid_user.cpu().numpy(), valid_item.cpu().numpy())) | set(
    zip(test_user.cpu().numpy(), test_item.cpu().numpy()))
# 生成训练集负样本，并合并数据
train_sampler = Sampler()
train_negative_users, train_negative_items, train_negative_ratings = train_sampler.negative_sampling(data.num_users,
                                                                                                     data.num_items,
                                                                                                     excluded_pairs,
                                                                                                     180,
                                                                                                     device=device)
train_users_combined = torch.cat([train_user, train_negative_users])
train_items_combined = torch.cat([train_item, train_negative_items])
train_ratings_combined = torch.cat([train_rating, train_negative_ratings])
# 生成验证集负样本，并合并数据
valid_sampler = Sampler()
valid_negative_users, valid_negative_items, valid_negative_ratings = valid_sampler.negative_sampling(data.num_users,
                                                                                                     data.num_items,
                                                                                                     excluded_pairs,
                                                                                                     60,
                                                                                                     device=device)
valid_users_combined = torch.cat([valid_user, valid_negative_users])
valid_items_combined = torch.cat([valid_item, valid_negative_items])
valid_ratings_combined = torch.cat([valid_rating, valid_negative_ratings])
# 生成测试集负样本，并合并数据
test_sampler = Sampler()
test_negative_users, test_negative_items, test_negative_ratings = test_sampler.negative_sampling(data.num_users,
                                                                                                 data.num_items,
                                                                                                 excluded_pairs,
                                                                                                 60,
                                                                                                 device=device)
test_users_combined = torch.cat([test_user, test_negative_users])
test_items_combined = torch.cat([test_item, test_negative_items])
test_ratings_combined = torch.cat([test_rating, test_negative_ratings])
# 定义模型
model = MatrixFactorization(data.num_users, data.num_items, 64).to(device)
loss_fn = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

# 训练模型
trainer = Trainer(model, loss_fn, optimizer)
epochs = 100
for epoch in range(epochs):
    trainer.train_loop(train_users_combined, train_items_combined, train_rating=train_ratings_combined)
    trainer.valid_loop(valid_users_combined, valid_items_combined, valid_rating=valid_ratings_combined)
    trainer.test_loop(test_users_combined, test_items_combined, test_rating=test_ratings_combined)
    trainer.model_eval(epoch)

# 推荐部分
k = 100
real_list = data.itemid_matrix()
roc_list = model.recommendation(data.num_users, data.num_items, k)
rank = Ranking(real_list, roc_list, k)
rank.ranking_eval()


