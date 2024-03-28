# 基于物品的协同过滤算法
# 运行速度很慢，因为对于每一个未评分的物品都要计算K个邻居，如何解决呢？事先存储每个物品的k个邻居？
# 构建用户-物品隐式反馈矩阵
# 计算物品之间的相似度
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

# 计算物品之间的相似度(余弦相似度)
item_similarity = cosine_similarity(data.T)


# 预测评分函数
def prediction_item_based(item_similarity, user_ratings, item_id, k):
    numerator = 0
    denominator = 0
    similarities = list(enumerate(item_similarity[item_id]))
    similarities.sort(key=lambda x: x[1], reverse=True)
    neighbors = [x[0] for x in similarities[1:k + 1]]
    for neighbor in neighbors:
        similarity = item_similarity[item_id][neighbor]
        rating = user_ratings[neighbor]
        numerator += similarity * rating
        denominator += similarity
    return numerator / denominator if denominator != 0 else 0


# 推荐列表函数
def recommendations_list_item_based(data, item_similarity, user_id, k, n):
    user_ratings = data[user_id]
    recommendations = []
    for item_id in range(data.shape[1]):
        if user_ratings[item_id] == 0:  # 只考虑用户未评分的物品
            prediction = prediction_item_based(item_similarity, user_ratings, item_id, k)
            recommendations.append((item_id, prediction))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n]


# 召回率、准确率、F1值计算
Recall = 0.0
Precision = 0.0
F1 = 0.0
for user_id in range(1, data.shape[0]):
    # 获得用户的推荐列表，邻居数为10，推荐列表长度为10
    recommendations = recommendations_list_item_based(data, item_similarity, user_id - 1, 10,
                                                      20)  # user_id-1是因为第一个用户的索引从0开始
    print(user_id)
    # 因为得到的是物品在矩阵中的索引，所以转化为列表后，物品id的值要+1
    recommendations = [item[0] for item in recommendations]
    recommendations = [x + 1 for x in recommendations]
    # 得到测试集中用户产生过行为的物品id
    item_ids = test.loc[test['user_id'] == user_id, 'item_id'].tolist()
    # 转化为集合
    recommendations = set(recommendations)
    item_ids = set(item_ids)
    # print(recitemid)
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
print("召回率、准确率、F1值分别为：", Recall, Precision, F1)
