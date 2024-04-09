import torch
import torch.nn as nn
import torch.optim as optim

import sys

sys.path.append('../')

from data.reader import MovieLens100K
from model.AutoRec import AutoRec

# 加载数据
data = MovieLens100K('../dataset_example/ml-100k')
rating_matrix = data.rating_matrix('../dataset_example/ml-100k')
print(rating_matrix.shape)

# 实例化模型、损失函数和优化器
model1 = AutoRec(data.num_items, 50)  # 基于用户的AutoRec
model2 = AutoRec(data.num_users, 50)  # 基于物品的AutoRec
criterion = nn.BCELoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.01)
optimizer2 = optim.Adam(model2.parameters(), lr=0.01)

# 训练模型
epochs = 100
for epoch in range(epochs):
    # 训练模型
    model1.train()
    optimizer1.zero_grad()
    output1 = model1(rating_matrix)
    loss1 = criterion(output1, rating_matrix)
    loss1.backward()
    optimizer1.step()

    model2.train()
    optimizer2.zero_grad()
    output2 = model2(rating_matrix.T)  # 转置矩阵将输入变为物品向量
    loss2 = criterion(output2, rating_matrix.T)
    loss2.backward()
    optimizer2.step()

    print(f'Epoch {epoch + 1}/{epochs},U-AutoRec Loss: {loss1.item()},I-AutoRec Loss: {loss2.item()}')
