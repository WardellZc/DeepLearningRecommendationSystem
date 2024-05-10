import torch.nn
from torch import optim

import sys

sys.path.append('../')

from data.reader import MovieLens100K
from model.neuralcf import NeuralCF
from sampler.sampler import Sampler
from trainer.trainer import Trainer

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
                                                                                                     60)
train_users_combined = torch.cat([train_user, train_negative_users]).to(device)
train_items_combined = torch.cat([train_item, train_negative_items]).to(device)
train_ratings_combined = torch.cat([train_rating, train_negative_ratings]).unsqueeze(1).to(device)
# 生成验证集负样本，并合并数据
valid_sampler = Sampler()
valid_negative_users, valid_negative_items, valid_negative_ratings = valid_sampler.negative_sampling(data.num_users,
                                                                                                     data.num_items,
                                                                                                     excluded_pairs,
                                                                                                     20)
valid_users_combined = torch.cat([valid_user, valid_negative_users]).to(device)
valid_items_combined = torch.cat([valid_item, valid_negative_items]).to(device)
valid_ratings_combined = torch.cat([valid_rating, valid_negative_ratings]).unsqueeze(1).to(device)
# 生成测试集负样本，并合并数据
test_sampler = Sampler()
test_negative_users, test_negative_items, test_negative_ratings = test_sampler.negative_sampling(data.num_users,
                                                                                                 data.num_items,
                                                                                                 excluded_pairs,
                                                                                                 20)
test_users_combined = torch.cat([test_user, test_negative_users]).to(device)
test_items_combined = torch.cat([test_item, test_negative_items]).to(device)
test_ratings_combined = torch.cat([test_rating, test_negative_ratings]).unsqueeze(1).to(device)

# 定义模型
model = NeuralCF(data.num_users, data.num_items, 256, [512, 256, 128, 64, 32]).to(device)
loss_fn = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 训练模型
trainer = Trainer(model, loss_fn, optimizer)
epochs = 50
for epoch in range(epochs):
    trainer.train_loop(train_users_combined, train_items_combined, train_rating=train_ratings_combined)
    trainer.valid_loop(valid_users_combined, valid_items_combined, valid_rating=valid_ratings_combined)
    trainer.test_loop(test_users_combined, test_items_combined, test_rating=test_ratings_combined)
    trainer.model_eval(epoch)

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
f1 = 2 * precision * recall / (precision + recall)
print(f'Precision: {precision},Recall: {recall},F1: {f1}')
