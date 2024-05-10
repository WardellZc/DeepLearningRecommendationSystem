import torch.nn
from torch import optim
import pandas as pd

import sys

sys.path.append('../')

from data.reader import MovieLens100K
from model.deepcrossing import DeepCrossing
from sampler.sampler import Sampler
from trainer.trainer import Trainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = MovieLens100K('../dataset_example/ml-100k')

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
train_feature = data.feature(train_combined)  # 将合并后的样本连接上用户和物品信息构成特征数据集
# 生成验证集负样本，并合并数据
valid_sampler = Sampler()
valid_negative = valid_sampler.negative_sampling2(data.num_users, data.num_items, excluded_pairs, 10)
valid_combined = pd.concat([data.valid, valid_negative], axis=0).reset_index(drop=True)  # 合并正负样本
valid_feature = data.feature(valid_combined)  # 将合并后的样本连接上用户和物品信息构成特征数据集
# 生成测试集负样本，并合并数据
test_sampler = Sampler()
test_negative = test_sampler.negative_sampling2(data.num_users, data.num_items, excluded_pairs, 10)
test_combined = pd.concat([data.test, test_negative], axis=0).reset_index(drop=True)  # 合并正负样本
test_feature = data.feature(test_combined)  # 将合并后的样本连接上用户和物品信息构成特征数据集

# 生成训练集、验证集、测试集的训练特征向量以及评分的torch
train_data = torch.tensor(train_feature.iloc[:, 3:].values, dtype=torch.float32).to(device)  # 去除user_di、item_id、rating
train_rating = torch.tensor(train_feature.iloc[:, 2].values, dtype=torch.float32).unsqueeze(1).to(device)   # 评分数据，用于计算损失
valid_data = torch.tensor(valid_feature.iloc[:, 3:].values, dtype=torch.float32).to(device)
valid_rating = torch.tensor(valid_feature.iloc[:, 2].values, dtype=torch.float32).unsqueeze(1).to(device)
test_data = torch.tensor(test_feature.iloc[:, 3:].values, dtype=torch.float32).to(device)
test_rating = torch.tensor(test_feature.iloc[:, 2].values, dtype=torch.float32).unsqueeze(1).to(device)

# 定义模型
hidden_units = [256, 128, 64, 32]
model = DeepCrossing(32, hidden_units)
loss_fn = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

# 模型训练
trainer = Trainer(model, loss_fn, optimizer)
epochs = 100
for epoch in range(epochs):
    trainer.train_loop(train_data, train_rating=train_rating)
    trainer.valid_loop(valid_data, valid_rating=valid_rating)
    trainer.test_loop(test_data, test_rating=test_rating)
    trainer.model_eval(epoch)

# 推荐部分
recommendation = model.recommendation(data.num_users, data.user_item(), 100)
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
f1 = 2 * precision * recall / (precision + recall)
print(f'Precision: {precision},Recall: {recall},F1: {f1}')
