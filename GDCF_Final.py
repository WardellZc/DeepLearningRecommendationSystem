# 基于矩阵分解的协同过滤算法
# 构建用户-物品隐式反馈矩阵
# 定义超参数：嵌入向量的维度大小、迭代次数、学习率
# 初始化模型以及优化器
# 训练模型：前向传播、反向传播
# 进行推荐
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# 加载数据集
data_path = 'E:/Python Project/Demo/Recommended System Practice/ml-100k'
ratings = pd.read_csv(data_path + '/u1.base', sep='\t', names=['userId', 'itemId', 'rating', 'timestamp'])
# 加载测试集，计算召回率和准确率
test = pd.read_csv('E:/Python Project/Demo/Recommended System Practice/ml-100k/u1.test', sep='\t',
                   names=['user_id', 'item_id'], usecols=[0, 1])
# 构建用户-电影隐式反馈矩阵
implicit_ratings = ratings.pivot(index='userId', columns='itemId', values='rating').fillna(0)
implicit_ratings = implicit_ratings.applymap(lambda x: 1 if x > 0 else 0)
# 转化成了numpy数组
data = implicit_ratings.values

# 定义超参数
embedding_size = 100  # 隐向量维度
lr = 0.01  # 学习速率
n = 10  # 迭代次数

# 初始化模型以及优化器
num_users = data.shape[0]
num_items = data.shape[1]
P = np.random.rand(num_users, embedding_size)  # 随机浮点数取值为[0,1）
Q = np.random.rand(num_items, embedding_size)
# 转化为张量在torch中训练
P = torch.tensor(P, requires_grad=True)
Q = torch.tensor(Q.T, requires_grad=True)
# 二分类问题的损失函数选择交叉熵损失函数
loss_fn = torch.nn.BCEWithLogitsLoss()  # 结合了 Sigmoid 层和二进制交叉熵损失
optimizer = torch.optim.Adam([P, Q], lr=lr)

# 绘制图像
# 记录损失值、召回率、准确率、F1值
losses = []
Recalls = []
Precisions = []
F1s = []

# Adam梯度下降
for n_iter in range(n):
    print("开始第", n_iter + 1, "次迭代")
    loss = 0.0
    # 预测得分
    pre = torch.matmul(P, Q)
    # 计算损失
    loss = loss_fn(pre, torch.tensor(data, dtype=torch.double))
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()
    # 清空梯度
    optimizer.zero_grad()
    # 记录损失值
    losses.append(loss.item())

    # 将推荐的物品id和测试集中的物品id转化为集合进行指标计算
    Recall = 0.0
    Precision = 0.0
    for user_id in range(1, num_users + 1):
        # 获取用户对所有物品的评分预测
        user_predicted_ratings = pre[user_id - 1, :]
        # 找到用户对所有物品的评分预测中排名前50的物品
        top_indices = torch.argsort(user_predicted_ratings, descending=True)[:50]
        # 获取排名前50的物品的ID
        top_item_ids = top_indices + 1  # 将数组索引转换为物品ID（从1开始）
        # 返回给用户的推荐结果
        print("用户ID为", user_id, "的用户的top 50推荐物品ID：", top_item_ids)
        # 将tensor转化为列表，先转化为numpy数组再转化为列表
        top_item_ids = top_item_ids.numpy()
        top_item_ids = set(top_item_ids)
        # 获得用户测试集中的物品id
        user_items = test[test['user_id'] == user_id]['item_id'].tolist()
        user_items = set(user_items)
        same = top_item_ids.intersection(user_items)
        # print(same)
        if len(user_items) > 0:
            Recall += len(same) / (1.0 * len(user_items))
        else:
            Recall += 0.0
        Precision += len(same) / (1.0 * len(top_item_ids))
    Recall /= num_users
    Precision /= num_users
    F1 = 2 * Recall * Precision / (Recall + Precision)
    Recalls.append(Recall)
    Precisions.append(Precision)
    F1s.append(F1)

# 将指标绘制在图像上
# 绘制准确率、召回率、F1 值的图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(Precisions) + 1), Precisions, label='Precision', color='b')
plt.plot(range(1, len(Recalls) + 1), Recalls, label='Recall', color='g')
plt.plot(range(1, len(F1s) + 1), F1s, label='F1', color='r')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Precision, Recall, and F1 vs. Epoch')
plt.legend()
# 绘制损失值的图像
plt.subplot(1, 2, 2)
plt.plot(range(1, len(losses) + 1), losses, label='Loss', color='r')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.title('Loss vs. Epoch')
plt.legend()

plt.tight_layout()
plt.show()
