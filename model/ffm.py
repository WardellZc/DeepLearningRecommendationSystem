import torch
from torch import nn
from torch.nn.init import xavier_normal_


class FFM(nn.Module):
    def __init__(self, num_feature: int):  # 隐向量维度
        super().__init__()

        # 特征的特征域的embedding
        self.movie_age = nn.Embedding(1, num_feature)
        self.movie_gender = nn.Embedding(1, num_feature)
        self.movie_occupation = nn.Embedding(1, num_feature)
        self.age_movie = nn.Embedding(1, num_feature)
        self.age_gender = nn.Embedding(1, num_feature)
        self.age_occupation = nn.Embedding(1, num_feature)
        self.gendr_movie = nn.Embedding(1, num_feature)
        self.gendr_age = nn.Embedding(1, num_feature)
        self.gendr_occupation = nn.Embedding(1, num_feature)
        self.occupation_movie = nn.Embedding(1, num_feature)
        self.occupation_age = nn.Embedding(1, num_feature)
        self.occupation_gender = nn.Embedding(1, num_feature)

        # embedding初始化
        xavier_normal_(self.movie_age.weight.data)
        xavier_normal_(self.movie_gender.weight.data)
        xavier_normal_(self.movie_occupation.weight.data)
        xavier_normal_(self.age_movie.weight.data)
        xavier_normal_(self.age_gender.weight.data)
        xavier_normal_(self.age_occupation.weight.data)
        xavier_normal_(self.gendr_movie.weight.data)
        xavier_normal_(self.gendr_age.weight.data)
        xavier_normal_(self.gendr_occupation.weight.data)
        xavier_normal_(self.occupation_movie.weight.data)
        xavier_normal_(self.occupation_age.weight.data)
        xavier_normal_(self.occupation_gender.weight.data)

    def forward(self, movie_age: torch.Tensor, movie_gender: torch.Tensor, movie_occupation: torch.Tensor,
                age_gender: torch.Tensor, age_occupation: torch.Tensor, gendr_occupation: torch.Tensor):
        movie_age_weight = torch.matmul(self.movie_age.weight, self.age_movie.weight.t())
        movie_gender_weight = torch.matmul(self.movie_gender.weight, self.gendr_movie.weight.t())
        movie_occupation_weight = torch.matmul(self.movie_occupation.weight, self.occupation_movie.weight.t())
        age_gender_weight = torch.matmul(self.age_gender.weight, self.gendr_age.weight.t())
        age_occupation_weight = torch.matmul(self.age_occupation.weight, self.occupation_age.weight.t())
        gendr_occupation_weight = torch.matmul(self.gendr_occupation.weight, self.occupation_age.weight.t())
        return torch.sigmoid(
            torch.matmul(movie_age, movie_age_weight.expand(20, 1)) + torch.matmul(movie_gender,
                                                                                   movie_gender_weight.expand(20,
                                                                                                              1)) + torch.matmul(
                movie_occupation, movie_occupation_weight.expand(20, 1)) + torch.matmul(age_gender,
                                                                                        age_gender_weight.expand(2,
                                                                                                                 1)) + torch.matmul(
                age_occupation, age_occupation_weight.expand(2, 1)) + torch.matmul(gendr_occupation,
                                                                                   gendr_occupation_weight.expand(2,
                                                                                                                  1)))
