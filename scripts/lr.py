import numpy as np
import torch.nn
from torch import optim

import sys

sys.path.append('../')

from data.reader import MovieLens100K
from model.lr import LogisticRegression

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = MovieLens100K('../dataset_example/ml-100k')
train_data, test_data, user_item = data.feature_data('../dataset_example/ml-100k')

train_list = train_data  # 提取训练集物品id
test_list = test_data  # 提取测试集物品id
train_data = train_data[:, 3:]  # 筛选训练数据
train_real = train_data[:, 2].view(-1, 1)  # 训练集真实评分数据，用于计算损失
test_data = test_data[:, 3:]  # 筛选测试数据
test_real = test_data[:, 2].view(-1, 1)  # 测试集真实评分数据，用于计算损失
model = LogisticRegression(train_data.shape[1], 1)
loss_fn = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(train_data)
    loss = loss_fn(predictions, train_real)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions_test = model(test_data)
        test_loss = loss_fn(predictions_test, test_real)
    print(f'Epoch {epoch + 1}: Training loss: {loss.item()} Test Loss: {test_loss.item()}')


def recommendation(user_id, user_item, k):
    user_vector = user_item[user_item[:, 0] == user_id]
    list_indices = [1, 2, 3] + list(range(5, 24))
    user_vector = user_vector[:, list_indices]
    pre = torch.matmul(user_vector, model.W.weight) + model.B.weight

    # 推荐列表（去除训练样本中的正负样本）
    # 得到预测推荐列表
    pre = pre.detach().numpy()
    indices = np.argsort(-pre, axis=0).tolist()
    indices = [x + 1 for sublist in indices for x in sublist]  # 将索引+1变为物品id

    # 得到训练集中的物品id，去除它得到推荐列表
    train_indices = train_list[train_list[:, 0] == user_id][:, 1].tolist()
    recommendation_list = [x for x in indices if x not in train_indices][:k]

    # 得到测试集中过的物品id
    test_indices = test_list[test_list[:, 0] == user_id][:, 1].tolist()
    test_indices = [int(x) for x in test_indices]

    return recommendation_list, test_indices


recommendation_list, test_indices = recommendation(1, user_item, 20)

k = 20
pr = 0.0
re = 0.0
f1 = 0.0
for n_user in range(1, data.num_users + 1):
    recommendation_list, test_indices = recommendation(n_user, user_item, k)
    same = set(recommendation_list) & set(test_indices)
    pr += len(same) / (len(recommendation_list) * 1.0)
    if len(test_indices) > 0:
        re += len(same) / (len(test_indices) * 1.0)
    else:
        re += 0
pr = pr / data.num_users
re = re / data.num_users
f1 = 2 * pr * re / (pr + re)
print(f' Precision: {pr} Recall: {re} F1:{f1}')
