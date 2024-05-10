import torch
from torch import nn
from torch.nn.init import xavier_normal_


class FFM(nn.Module):
    def __init__(self, num_feature: int, num_vector: int):  # 特征维度、特征交叉隐向量维度
        super().__init__()
        # 特征交叉，分为用户和物品两个域，用户域包括年龄、性别、职业，物品域包括电影风格
        self.age_user = nn.Embedding(1, num_vector)  # 特征年龄对应用户域的影响
        self.age_item = nn.Embedding(1, num_vector)  # 特征年龄对应物品域的影响
        self.gender_user = nn.Embedding(2, num_vector)
        self.gender_item = nn.Embedding(2, num_vector)
        self.occupation_user = nn.Embedding(21, num_vector)
        self.occupation_item = nn.Embedding(21, num_vector)
        self.movie_user = nn.Embedding(19, num_vector)
        # self.movie_item = nn.Embedding(19, num_vector)

        # embedding初始化
        xavier_normal_(self.age_user.weight.data)
        xavier_normal_(self.age_item.weight.data)
        xavier_normal_(self.gender_user.weight.data)
        xavier_normal_(self.gender_item.weight.data)
        xavier_normal_(self.occupation_user.weight.data)
        xavier_normal_(self.occupation_item.weight.data)
        xavier_normal_(self.movie_user.weight.data)
        # xavier_normal_(self.movie_item.weight.data)

        # 特征向量参数w和b
        self.linear = nn.Linear(num_feature, 1, True)

    def forward(self, feature_vector: torch.Tensor) -> torch.Tensor:
        # age:0,gender:1-2;occupation:3-23,movie:24-42
        # 特征交叉
        age_user = torch.matmul(feature_vector[:, 0].unsqueeze(1), self.age_user.weight)
        age_item = torch.matmul(feature_vector[:, 0].unsqueeze(1), self.age_item.weight)
        gender_user = torch.matmul(feature_vector[:, 1:3], self.gender_user.weight)
        gender_item = torch.matmul(feature_vector[:, 1:3], self.gender_item.weight)
        occupation_user = torch.matmul(feature_vector[:, 3:24], self.occupation_user.weight)
        occupation_item = torch.matmul(feature_vector[:, 3:24], self.occupation_item.weight)
        movie_user = torch.matmul(feature_vector[:, 24:43], self.movie_user.weight)
        # movie_item = torch.matmul(feature_vector[:, 24:43], self.movie_item.weight)

        # 计算点积
        age_gender = torch.sum(age_user * gender_user, dim=1)
        age_occupation = torch.sum(age_user * occupation_user, dim=1)
        age_movie = torch.sum(age_item * movie_user, dim=1)
        gender_occupation = torch.sum(gender_user * occupation_user, dim=1)
        gender_movie = torch.sum(gender_item * movie_user, dim=1)
        occupation_movie = torch.sum(occupation_item * movie_user, dim=1)
        feature_cross = age_gender + age_occupation + age_movie + gender_occupation + gender_movie + occupation_movie

        return torch.sigmoid(self.linear(feature_vector) + feature_cross.unsqueeze(1))

    def recommendation(self, num_users, user_item, k):
        array = []
        for i in range(num_users):
            user_vector = user_item[user_item['user_id'] == i]
            user_vector = torch.Tensor(user_vector.iloc[:, 2:].values)
            scores = self.forward(user_vector)
            values, indices = torch.topk(scores, k, dim=0)
            indices = indices.view(1, -1).tolist()[0]
            array.append(indices)
        return array
