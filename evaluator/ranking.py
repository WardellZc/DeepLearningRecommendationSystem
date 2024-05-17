import numpy as np


class Ranking:
    def __init__(self, real_list, rec_list, k):
        self.actual = real_list
        self.predicted = rec_list
        self.k = k

    # 准确率、召回率、F1值
    def precision_recall_f1(self):
        """
        计算推荐系统的准确率、召回率和F1值
        actual: 实际相关的物品id矩阵，每行代表一个用户
        predicted: 推荐的物品id矩阵，每行代表一个用户
        k: 截至k个推荐物品
        """
        same = 0  # 统计推荐物品在实际物品中的个数
        rec = 0  # 统计推荐的总个数
        real = 0  # 统计实际物品个数

        for a, p in zip(self.actual, self.predicted):
            if len(p) > self.k:
                p = p[:self.k]

            # 计算相关物品数
            relevant_items = set(a)
            recommended_items = set(p)
            relevant_recommended_items = relevant_items.intersection(recommended_items)

            # 统计个数
            same += len(relevant_recommended_items)
            rec += len(recommended_items)
            real += len(relevant_items)

        # 计算准确率、召回率、F1值
        precision = same / (rec * 1.0)
        recall = same / (real * 1.0)
        f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1

    # MAP
    @staticmethod
    def apk(actual, predicted, k):
        """
        计算一个用户的平均精度（AP）
        actual: 实际相关的物品id列表
        predicted: 推荐的物品id列表
        k: 截至k个推荐物品
        """
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):  # 返回索引和元素值
            if p in actual:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        return score / len(actual)

    def mapk(self):
        """
        计算均值平均精度（MAP）
        actual: 实际相关的物品id矩阵，每行代表一个用户
        predicted: 推荐的物品id矩阵，每行代表一个用户
        k: 截至k个推荐物品
        """
        return np.mean([self.apk(a, p, self.k) for a, p in zip(self.actual, self.predicted)])

    # NDCG
    @staticmethod
    def dcg(relevance_scores, k):
        """
        计算折损累计增益（DCG）
        relevance_scores: 相关性评分列表
        k: 截至k个推荐物品
        """
        relevance_scores = np.asarray(relevance_scores)[:k]
        dcg_score = np.sum((2 ** relevance_scores - 1) / np.log2(np.arange(1, len(relevance_scores) + 1) + 1))
        return dcg_score

    def ndcg(self, actual, predicted, k):
        """
        计算归一化折损累计增益（NDCG）
        actual: 实际相关的物品id列表
        predicted: 推荐的物品id列表
        k: 截至k个推荐物品
        """
        # 生成实际相关的物品的相关性评分
        relevance_scores = [1 if item in actual else 0 for item in predicted]

        # 计算DCG
        dcg_score = self.dcg(relevance_scores, k)

        # 计算理想情况下的相关性评分（相关性排序）
        ideal_relevance_scores = sorted(relevance_scores, reverse=True)

        # 计算IDCG
        idcg_score = self.dcg(ideal_relevance_scores, k)

        # 计算NDCG
        return dcg_score / idcg_score if idcg_score > 0 else 0

    def mean_ndcg(self):
        """
        计算所有用户的平均NDCG
        actual: 实际相关的物品id矩阵，每行代表一个用户
        predicted: 推荐的物品id矩阵，每行代表一个用户
        k: 截至k个推荐物品
        """
        return np.mean([self.ndcg(a, p, self.k) for a, p in zip(self.actual, self.predicted)])

    # MRR
    @staticmethod
    def rr(actual, predicted):
        """
        计算单个用户的倒数排名（Reciprocal Rank, RR）
        actual: 实际相关的物品id列表
        predicted: 推荐的物品id列表
        """
        for i, p in enumerate(predicted):
            if p in actual:
                return 1.0 / (i + 1)
        return 0.0

    def mrr(self):
        """
        计算平均倒数排名（Mean Reciprocal Rank, MRR）
        actual: 实际相关的物品id矩阵，每行代表一个用户
        predicted: 推荐的物品id矩阵，每行代表一个用户
        """
        return np.mean([self.rr(a, p) for a, p in zip(self.actual, self.predicted)])

    def ranking_eval(self):
        precision, recall, f1 = self.precision_recall_f1()
        map_score = self.mapk()
        mean_ndcg_score = self.mean_ndcg()
        mrr_score = self.mrr()
        print(f"""
                - Precision@{self.k}:  {precision}
                - Recall@{self.k}:  {recall}
                - F1 Score@{self.k}:  {f1}
                - MAP@{self.k}: {map_score}
                - Mean NDCG@{self.k}: {mean_ndcg_score}
                - MRR: {mrr_score}
                """)
