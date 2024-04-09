# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 21:58
# @Author  : Lumen
# @File    : sampler.py
import random


class Sampler:
    def __init__(self):
        self.negative_users = []
        self.negative_items = []

    def rns(self, num_user: int, num_item: int, excluded_pairs: set, num_negatives: int):
        for n_user in range(1, num_user + 1):
            for _ in range(num_negatives):
                n_item = random.randint(1, num_item)
                while (n_user, n_item) in excluded_pairs:
                    n_item = random.randint(1, num_item)
                self.negative_users.append(n_user)
                self.negative_items.append(n_item)
        return self.negative_users, self.negative_items
