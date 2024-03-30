# -*- coding: utf-8 -*-
# @Time    : 2024/3/30 21:23
# @Author  : Lumen
# @File    : mf.py

import torch.nn
from torch import optim

from data.reader import MovieLens100K
from model.mf import MatrixFactorization


def negative_sampling(num_items: int,
                      potential_negatives: torch.Tensor,
                      excluded_pairs: set,
                      num_negatives: int = 1,
                      device: str = 'cpu'):
    negative_items = []
    negative_users = []

    for user in potential_negatives:
        user = user.item()
        for _ in range(num_negatives):
            negative_item = torch.randint(0, num_items, (1,)).item()
            while (user, negative_item) in excluded_pairs:
                negative_item = torch.randint(0, num_items, (1,)).item()
            negative_items.append(negative_item)
            negative_users.append(user)

    return (torch.tensor(negative_users).to(device),
            torch.tensor(negative_items).to(device),
            torch.zeros(len(negative_users)).to(device))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = MovieLens100K('../dataset_example/ml-100k')
train_user, train_item, train_rating = data.train_interaction(device=device)
test_user, test_item, test_rating = data.test_interaction(device=device)
model = MatrixFactorization(data.num_users, data.num_items, 64).to(device)
loss_fn = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_excluded_pairs = set(zip(test_user.cpu().numpy(), test_item.cpu().numpy()))
test_excluded_pairs = set(zip(train_user.cpu().numpy(), train_item.cpu().numpy()))

epochs = 50
for epoch in range(epochs):
    model.train()
    # 生成训练集负样本
    train_negative_users, train_negative_items, train_negative_ratings = negative_sampling(
        data.num_items,
        train_user.unique(),
        train_excluded_pairs,
        num_negatives=10,
        device=device
    )
    train_excluded_pairs.update(zip(train_negative_users.cpu().numpy(), train_negative_items.cpu().numpy()))
    # 合并正样本和负样本
    users_combined = torch.cat([train_user, train_negative_users]).to(device)
    items_combined = torch.cat([train_item, train_negative_items]).to(device)
    ratings_combined = torch.cat([train_rating, train_negative_ratings]).to(device)

    optimizer.zero_grad()
    predictions = model(users_combined, items_combined)
    loss = loss_fn(predictions, ratings_combined)
    loss.backward()
    optimizer.step()

    model.eval()
    # 生成测试集负样本，对整个测试集进行评估
    test_negative_users, test_negative_items, test_negative_ratings = negative_sampling(
        data.num_items,
        test_user.unique(),
        test_excluded_pairs,
        num_negatives=10,
        device=device
    )
    users_combined_test = torch.cat([test_user, test_negative_users]).to(device)
    items_combined_test = torch.cat([test_item, test_negative_items]).to(device)
    ratings_combined_test = torch.cat([test_rating, test_negative_ratings]).to(device)

    with torch.no_grad():
        predictions_test = model(users_combined_test, items_combined_test)
        test_loss = loss_fn(predictions_test, ratings_combined_test)
        print(f'Epoch {epoch + 1}: Training loss: {loss.item()} Test Loss: {test_loss.item()}')
