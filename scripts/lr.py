import pandas as pd
import torch.nn
from torch import optim

import sys

sys.path.append('../')

from data.reader import MovieLens100K
from model.lr import LogisticRegression
from sampler.sampler import Sampler
from trainer.trainer import Trainer
from evaluator.ranking import Ranking

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
train_negative = train_sampler.negative_sampling2(data.num_users, data.num_items, excluded_pairs, 10)  # 生成用户、项目、评分数据
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

# 生成用户id和物品id的one-hot向量
# 先合并生成one-hot向量再拆分
all_data = pd.concat([train_feature, valid_feature, test_feature], axis=0)
all_data.drop('rating', axis=1, inplace=True)  # 去除评分数据
all_data = pd.get_dummies(all_data, columns=['user_id', 'item_id']).astype(int)
train_data = all_data.iloc[:train_feature.shape[0], :]
valid_data = all_data.iloc[train_feature.shape[0]:train_feature.shape[0] + valid_feature.shape[0], :]
test_data = all_data.iloc[
            train_feature.shape[0] + valid_feature.shape[0]:train_feature.shape[0] + valid_feature.shape[0] +
                                                            test_feature.shape[0], :]

# 生成训练集、验证集、测试集的训练特征向量以及评分的torch
train_rating = torch.tensor(train_feature.iloc[:, 2].values, dtype=torch.float32).unsqueeze(1).to(device)  # 评分数据，用于计算损失
train_data = torch.tensor(train_data.values, dtype=torch.float32).to(device)
valid_rating = torch.tensor(valid_feature.iloc[:, 2].values, dtype=torch.float32).unsqueeze(1).to(device)
valid_data = torch.tensor(valid_data.values, dtype=torch.float32).to(device)
test_rating = torch.tensor(test_feature.iloc[:, 2].values, dtype=torch.float32).unsqueeze(1).to(device)
test_data = torch.tensor(test_data.values, dtype=torch.float32).to(device)

# 定义模型
model = LogisticRegression(train_data.size(1)).to(device)
loss_fn = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# 模型训练
trainer = Trainer(model, loss_fn, optimizer)
epochs = 100
for epoch in range(epochs):
    trainer.train_loop(train_data, train_rating=train_rating)
    trainer.valid_loop(valid_data, valid_rating=valid_rating)
    trainer.test_loop(test_data, test_rating=test_rating)
    trainer.model_eval(epoch)

# 推荐部分
k = 100
real_list = data.itemid_matrix()
user_item = data.user_item()
user_item = pd.get_dummies(user_item, columns=['user_id', 'item_id']).astype(int)  # 将用户id和物品id one-hot化   运行慢
roc_list = model.recommendation(data.num_users, user_item, k)  # 运行慢
rank = Ranking(real_list, roc_list, k)
rank.ranking_eval()


