# 基于用户的AutoRec
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

import sys

sys.path.append('../')

from data.reader import MovieLens100K
from model.AutoRec import AutoRec

# 加载数据
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = MovieLens100K('../dataset_example/ml-100k')

# 将评分矩阵划分为训练集、验证集、训练集
# 先划分为测试集，再从剩下的数据划分出验证集
train_valid, test = train_test_split(data.rating_matrix, test_size=0.2, random_state=42)
train, valid = train_test_split(train_valid, test_size=0.25, random_state=42)

# 将数据转化为张量
train = torch.tensor(train.values, dtype=torch.float32)
valid = torch.tensor(valid.values, dtype=torch.float32)
test = torch.tensor(test.values, dtype=torch.float32)

model = AutoRec(data.num_items, 64)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 50
for epoch in range(epochs):
    # 训练模型
    model.train()
    optimizer.zero_grad()
    output = model(train)
    loss = loss_fn(output, train)
    loss.backward()
    optimizer.step()

    model.train()
    with torch.no_grad():
        valid_output = model(valid)
        valid_loss = loss_fn(valid_output, valid)
        if epoch < epochs - 1:
            print(f'Epoch {epoch + 1}/{epochs},Train Loss: {loss.item()},Valid Loss: {valid_loss.item()}')
        else:
            test_output = model(test)
            test_loss = loss_fn(test_output, test)
            print(
                f'Epoch {epoch + 1}/{epochs},Train Loss: {loss.item()},Valid Loss: {valid_loss.item()},Test Loss: {test_loss.item()}')

# 推荐部分
data.rating_matrix = torch.tensor(data.rating_matrix.values, dtype=torch.float32)
recommendation = model.recommendation(data.rating_matrix, 100)
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
