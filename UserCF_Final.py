# 基于用户的协同过滤算法
# 构建用户-物品隐式反馈矩阵
# 计算用户之间的相似度
# 预测评分函数
# 推荐列表函数
# 召回率、准确率、F1值计算
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 加载数据集
data_path = 'E:/Python Project/Demo/Recommended System Practice/ml-100k'
ratings = pd.read_csv(data_path + '/ua.base', sep='\t', names=['userId', 'itemId', 'rating', 'timestamp'])
# 加载测试集，计算召回率和准确率
test = pd.read_csv('E:/Python Project/Demo/Recommended System Practice/ml-100k/ua.test', sep='\t',
                   names=['user_id', 'item_id'], usecols=[0, 1])
# 构建用户-物品隐式反馈矩阵
implicit_ratings = ratings.pivot(index='userId', columns='itemId', values='rating').fillna(0)
implicit_ratings = implicit_ratings.applymap(lambda x: 1 if x > 0 else 0)
# 转化成了numpy数组
data = implicit_ratings.values
# print(data)

# 计算用户之间的相似度(余弦相似度)
user_similarity = cosine_similarity(data)  # 每个元素对应相应行的相似度


# print(user_similarity)

# 预测评分函数
def prediction_dating(data, neighbors, user_similarity, user_id, item_id):
    numerator = 0  # 分子
    denominator = 0  # 分母
    for neighbor in neighbors:
        similarity = user_similarity[user_id][neighbor]
        rating = data[neighbor, item_id]
        numerator += similarity * rating
        denominator += similarity
    if denominator != 0:
        predicition = numerator / (1.0 * denominator)
    else:
        predicition = 0
    return predicition


# 推荐列表函数
def recommendations_list(data, user_similarity, user_id, k, n):
    # 找到目标用户已经评分的物品
    rated_items = data[user_id]

    # 找到与目标用户相似度最高的k个邻居用户
    similarities = list(enumerate(user_similarity[user_id]))  # 获取与目标用户的相似度列表
    similarities.sort(key=lambda x: x[1], reverse=True)  # 按照列表的第二个元素进行降序排列
    neighbors = [x[0] for x in similarities[1:k + 1]]  # 找到除自身外的K个最近邻，得到其索引
    # print("邻居用户在矩阵中的索引为：", neighbors)
    # 根据邻居用户的评分记录和预测评分公式生成推荐列表
    recommendations = []
    for item_id in range(data.shape[1]):
        if rated_items[item_id] == 0:  # 只考虑目标用户未评分过的物品
            prediction = prediction_dating(data, neighbors, user_similarity, user_id, item_id)
            recommendations.append((item_id, prediction))

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)  # 按照预测评分降序排序
    return recommendations[:n]  # 返回前n个结果


# 召回率、准确率、F1值计算
Recall = 0.0
Precision = 0.0
F1 = 0.0
for user_id in range(1,data.shape[0]+1):
    # 获得用户的推荐列表，邻居数为10，推荐列表长度为20
    recommendations = recommendations_list(data, user_similarity, user_id-1, 10, 20)  # user_id-1是因为第一个用户的索引从0开始
    print(user_id)
    # 因为得到的是物品在矩阵中的索引，所以转化为列表后，物品id的值要+1
    recommendations = [item[0] for item in recommendations]
    recommendations = [x + 1 for x in recommendations]
    # 得到测试集中用户产生过行为的物品id
    item_ids = test.loc[test['user_id'] == user_id, 'item_id'].tolist()
    # 转化为集合
    recommendations = set(recommendations)
    item_ids = set(item_ids)
    # print(recommendations)
    # print(item_ids)
    same = recommendations.intersection(item_ids)
    if len(item_ids) > 0:
        Recall += len(same) / (1.0 * len(item_ids))
    else:
        Recall += 0.0
    Precision += len(same) / (1.0 * len(recommendations))
Recall /= data.shape[0]
Precision /= data.shape[0]
F1 = 2 * Recall * Precision / (Recall + Precision)
print("召回率、准确率、F1值分别为：",Recall,Precision,F1)