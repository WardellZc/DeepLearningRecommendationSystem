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
train_data, test_data, user_item = data.feature_data('../dataset_example/ml-100k')

train_userid_itemid = train_data[:, :2]
test_userid_itemid = test_data[:, :2]
train_excluded_pairs = set(tuple(pair.tolist()) for pair in train_userid_itemid)
test_excluded_pairs = set(tuple(pair.tolist()) for pair in test_userid_itemid)

# 训练集负样本采集
train_sampler = Sampler()
train_negative_users, train_negative_items = train_sampler.rns(data.num_users, data.num_items, train_excluded_pairs, 10)
train_negative_rating = [0 for _ in range(len(train_negative_users))]
train_negative_data = {'user_id': train_negative_users, 'item_id': train_negative_items,
                       'rating': train_negative_rating}
train_negative_data = pd.DataFrame(train_negative_data)
train_negative_data = pd.merge(train_negative_data, data.item, on='item_id').drop('key', axis=1)
train_negative_data = pd.merge(train_negative_data, data.user, on='user_id').drop('key', axis=1)
train_negative_data = torch.tensor(train_negative_data.values, dtype=torch.float32)
train_data = torch.cat((train_data, train_negative_data), dim=0)

# 测试集负样本采集
test_sampler = Sampler()
test_negative_users, test_negative_items = test_sampler.rns(data.num_users, data.num_items, test_excluded_pairs, 2)
test_negative_rating = [0 for _ in range(len(test_negative_users))]
test_negative_data = {'user_id': test_negative_users, 'item_id': test_negative_items, 'rating': test_negative_rating}
test_negative_data = pd.DataFrame(test_negative_data)
test_negative_data = pd.merge(test_negative_data, data.item, on='item_id').drop('key', axis=1)
test_negative_data = pd.merge(test_negative_data, data.user, on='user_id').drop('key', axis=1)
test_negative_data = torch.tensor(test_negative_data.values, dtype=torch.float32)
test_data = torch.cat((test_data, test_negative_data), dim=0)

# 筛选训练特征（电影特征、年龄、性别、职业）
train_movie_age = train_data[:, 3:23]
list1 = list(range(3, 22)) + [23]
train_movie_gender = train_data[:, list1]
list2 = list(range(3, 22)) + [24]
train_movie_occupation = train_data[:, list2]
train_age_gender = train_data[:, 22:24]
list3 = [22, 24]
train_age_occupation = train_data[:, list3]
train_gender_occupation = train_data[:, 23:25]
train_real = train_data[:, 2]
train_real = train_real.view(-1, 1)  # 转化为二维张量

# 筛选测试特征
test_movie_age = test_data[:, 3:23]
list1 = list(range(3, 22)) + [23]
test_movie_gender = test_data[:, list1]
list2 = list(range(3, 22)) + [24]
test_movie_occupation = test_data[:, list2]
test_age_gender = test_data[:, 22:24]
list3 = [22, 24]
test_age_occupation = test_data[:, list3]
test_gender_occupation = test_data[:, 23:25]
test_real = test_data[:, 2]
test_real = test_real.view(-1, 1)  # 转化为二维张量

model = FFM(10)
loss_fn = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(train_movie_age, train_movie_gender, train_movie_occupation, train_age_gender,
                        train_age_occupation, train_gender_occupation)
    loss = loss_fn(predictions, train_real)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions_test = model(test_movie_age, test_movie_gender, test_movie_occupation, test_age_gender,
                                 test_age_occupation, test_gender_occupation)
        test_loss = loss_fn(predictions_test, test_real)
    print(f'Epoch {epoch + 1}: Training loss: {loss.item()} Test Loss: {test_loss.item()}')


