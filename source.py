# Copyright 2019 Gang Wei wg0502@bu.edu
from sklearn import preprocessing
from copy import deepcopy
import pandas as pd
import numpy
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
one_hot = preprocessing.OneHotEncoder(sparse=False)



train = pd.read_csv('./bu-cs565-project-2-summer-2019/train_data.csv')
test = pd.read_csv('./bu-cs565-project-2-summer-2019/test_data.csv')

train_id = train['video_id']

source = test['source']
target = test['target']

uploader = train['uploader'].fillna('none')
uploader = np.array(uploader).reshape(-1, 1)
fixed_uploader = one_hot.fit_transform(uploader)

age = train['age'].fillna(605.7)
age = np.array(age)
a_range = np.max(age) - np.min(age)
age = age/a_range

cate = train['category'].fillna('none')
cate = np.array(cate).reshape(-1, 1)
fixed_cate = one_hot.fit_transform(cate)

length = train['length'].fillna(235)
length = np.array(length)
length_range = np.max(length) - np.min(length)
length = length/length_range

views = train['views'].fillna(43994)
views = np.array(views)
v_range = np.max(views) - np.min(views)
views = views/v_range

rate = train['rate'].fillna(3.77)
rate = np.array(rate)
rate_range = np.max(rate) - np.min(rate)
rate = rate/rate_range

ratings = train['ratings'].fillna(168)
ratings = np.array(ratings)
ratings_range = np.max(ratings) - np.min(ratings)
ratings = ratings/ratings_range

comments = train['comments'].fillna(74)
comments = np.array(comments)
comments_range = np.max(comments) - np.min(comments)
comments = comments/comments_range

num_features = np.stack((age,length,views,rate,ratings,comments),axis=1)
# X = X.reshape(-1,1)
scores = []
_max = 0
# for i in range(2,100):
#     y_pred = KMeans(n_clusters=i).fit_predict(X)
#     score = metrics.calinski_harabaz_score(X, y_pred)
#     scores.append(score)
#     if score > _max:
#         _max = score
#         print(i,score)
# print(_max)
# y_pred = KMeans(n_clusters=100).fit_predict(X)
# score = metrics.calinski_harabaz_score(X, y_pred)
# print(score)
# plt.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred)
# plt.show()

reco = [str(i) for i in range(1, 21)]
rec = train[reco]
all_reco = deepcopy(rec.values)
# all_reco = all_reco.tolist()


def get_distance(vec1,vec2):
    # print(vec1,vec2)
    dist = numpy.linalg.norm(vec1 - vec2)
    return dist


def find_shortest_path(graph, start, end, path=[]):
    path = path + [start]
    if len(path) > 4:
        return None
    if start == end:
        return path
    if start not in graph:
        return None
    shortest = None
    for node in graph[start]:
        if node not in path:
            newpath = find_shortest_path(graph, node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest


print('start make graph')
g = {}
for id in range(len(train_id)):
    g[train_id[id]] = []
    # top_20 = rec.loc[id]
    # r = top_20.values.tolist()
    # for i in range(len(r)):
    #     g[r[i]] = []
print('graph init')
for id in range(len(train_id)):
    top_20 = rec.loc[id]
    for i in range(len(top_20.values.tolist())):
        r = top_20.values.tolist()
        if r[i] not in g[train_id[id]]:
            g[train_id[id]].append(r[i])
        # if train_id[id] not in g[r[i]]:
        #     g[r[i]].append(train_id[id])
print('graph done')
# print(len(g))
# print(get_distance(num_features[717], num_features[722]))
#
# print(get_distance(num_features[722], num_features[724]) +
#       get_distance(fixed_uploader[722], fixed_uploader[724]) +
#       get_distance(fixed_cate[722], fixed_cate[724]))

exist = [0]*len(target)
times = 0
count = 0
# diss = []
for t in range(len(target)):
    times += 1
    # print(times)
    if target[t] in train_id.values:
        # print('found in id')
        target_index = train[train.video_id == target[t]].index.values[0]
        source_index = train[train.video_id == source[t]].index.values[0]
        num_dis = get_distance(num_features[target_index],num_features[source_index])
        vec_dis = get_distance(fixed_cate[target_index], fixed_cate[source_index]) + \
            get_distance(fixed_uploader[target_index], fixed_uploader[source_index])

        vec_dis = vec_dis / 1.414
        # total_dis = num_dis + vec_dis
        if vec_dis < 1.1:
            exist[t] = 1
        else:
            # diss.append(num_dis)
            if num_dis < 0.3:
                exist[t] = 1
    else:

        one = find_shortest_path(graph=g, start=source[t], end=target[t])
        two = find_shortest_path(graph=g, start=target[t], end=source[t])
        if one:
            l1 = len(one)
        else:
            l1 = 99
        if two:
            l2 = len(two)
        else:
            l2 = 99
        if min(l1, l2) < 5:
            exist[t] = 1

    if target[t] not in all_reco:
        exist[t] = 1

    if times % 100 == 0:
        print(times)


exist = pd.DataFrame(exist)
sol = pd.read_csv('./bu-cs565-project-2-summer-2019/random_solution.csv')
sol['edge_present'] = exist
sol.to_csv(path_or_buf='./solution.csv',index=False)
