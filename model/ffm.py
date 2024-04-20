import torch
from torch import nn


class FFM(nn.Module):
    def __init__(self, num_feature: int, num_vector: int):  # 特征维度、隐向量维度
        super().__init__()

        # 特征向量参数w和b
        self.linear = nn.Linear(num_feature, 1, True)

        # 特征特征域的隐向量
        self.movie_age = nn.Parameter(torch.randn(1, num_vector))
        self.movie_gender = nn.Parameter(torch.randn(1, num_vector))
        self.movie_occupation = nn.Parameter(torch.randn(1, num_vector))
        self.age_movie = nn.Parameter(torch.randn(1, num_vector))
        self.age_gender = nn.Parameter(torch.randn(1, num_vector))
        self.age_occupation = nn.Parameter(torch.randn(1, num_vector))
        self.gendr_movie = nn.Parameter(torch.randn(1, num_vector))
        self.gendr_age = nn.Parameter(torch.randn(1, num_vector))
        self.gendr_occupation = nn.Parameter(torch.randn(1, num_vector))
        self.occupation_movie = nn.Parameter(torch.randn(1, num_vector))
        self.occupation_age = nn.Parameter(torch.randn(1, num_vector))
        self.occupation_gender = nn.Parameter(torch.randn(1, num_vector))

    def forward(self, feature_vector: torch.Tensor) -> torch.Tensor:
        # 特征交叉的权重
        movie_age_weight = torch.matmul(self.movie_age, self.age_movie.t())
        movie_gender_weight = torch.matmul(self.movie_gender, self.gendr_movie.t())
        movie_occupation_weight = torch.matmul(self.movie_occupation, self.occupation_movie.t())
        age_gender_weight = torch.matmul(self.age_gender, self.gendr_age.t())
        age_occupation_weight = torch.matmul(self.age_occupation, self.occupation_age.t())
        gendr_occupation_weight = torch.matmul(self.gendr_occupation, self.occupation_age.t())

        # 特征交叉
        list1 = [0] + list(range(24, 43))
        movie_age = feature_vector[:, list1]
        list2 = [1, 2] + list(range(24, 43))
        movie_gender = feature_vector[:, list2]
        list3 = list(range(3, 43))
        movie_occupation = feature_vector[:, list3]
        list4 = [0, 1, 2]
        age_gender = feature_vector[:, list4]
        list5 = [0] + list(range(3, 24))
        age_occupation = feature_vector[:, list5]
        list6 = list(range(1, 24))
        gendr_occupation = feature_vector[:, list6]

        return torch.sigmoid(
            self.linear(feature_vector) + torch.matmul(movie_age, movie_age_weight.expand(20, 1)) + torch.matmul(
                movie_gender, movie_gender_weight.expand(21, 1)) + torch.matmul(movie_occupation,
                                                                                movie_occupation_weight.expand(40,
                                                                                                               1)) + torch.matmul(
                age_gender, age_gender_weight.expand(3, 1)) + torch.matmul(age_occupation,
                                                                           age_occupation_weight.expand(22,
                                                                                                        1)) + torch.matmul(
                gendr_occupation, gendr_occupation_weight.expand(23, 1)))

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
