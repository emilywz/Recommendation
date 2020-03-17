# !/usr/bin/evn python3
# -*- coding: UTF-8 -*-
import collections
from copy import deepcopy, copy
from django.db.models import F
from math import sqrt
import operator
import numpy as np
import pandas as dp
# 特征向量化
from sklearn.feature_extraction import DictVectorizer
# 余弦相似性
from sklearn.metrics.pairwise import cosine_similarity

# user_act = {
#     'A': {'跑步': 1, '游泳': 8, '音乐': 1},
#     'B': {'游泳': 2, '登山': 1, '学习': 1},
#     'C': {'登山': 5, '音乐': 5},
#     'D': {'游泳': 1, '登山': 1, '音乐': 7, 'IT': 9, '二次元': 1, '动漫': 7},
#     'E': {'跑步': 1, '音乐': 1}
# }
#
# inst_list = [
#     {'跑步': 1, '游泳': 8, '音乐': 1},
#     {'游泳': 2, '登山': 1, '学习': 1},
#     {'登山': 5, '音乐': 5},
#     {'游泳': 1, '登山': 1, '音乐': 7, 'IT': 9, '二次元': 1, '动漫': 7},
#     {'跑步': 1, '音乐': 1}
# ]
from .models import Activity, UserInfo, InterestTag


# 1.构建用户-->兴趣标签的数据结构
# 组织数据结构'A': {'跑步': 1, '游泳': 8, '音乐': 1},
def getALLDataStruct():
    # 用户自己添加的兴趣
    """
    从数据库中提取数据组织数据
    :return: 
    """
    pass
    """组织数据格式：
    {用户：[兴趣1,兴趣2,兴趣1,兴趣1]}
    """
    user_tmp={'用户':['兴趣1','兴趣2','兴趣1','兴趣1']}

    #TODO 增加用户的兴趣权重的参考字段 比如关于兴趣标签的点击量
    print("----1.{用户：[兴趣1,兴趣2,兴趣1,兴趣1]} ----")
    print(user_tmp)

    user_data = getUserInstData(user_tmp)

    #print('----------------user_data-------------------------')
    #print(user_data)
    #print('@@@@@@@@@@@@@@@@@user_data@@@@@@@@@@@@@@@@@@@@@@@@')
    return user_data

def getUserInstData(user_tmp):
    # 创建用户与兴趣权重的数据结构
    # {用户ID：{兴趣:权重}}
    user_data = collections.OrderedDict()
    for uid, items in user_tmp.items():
        user_data.setdefault(uid, {})
        if items is None:
            continue
        else:
            unique = set(items)
            for term in unique:
                user_data[uid][term] = items.count(term)
    print("----1.{用户：{兴趣:权重}} ----")
    print(user_data)
    print("####1.{用户：{兴趣:权重}} ####")
    return user_data


def getInterestList(user_data):
    """
    用户兴趣标签
    """
    inst_list = []
    user_index=[]
    # for uid, item in enumerate(user_data):
    for i in user_data:
        if user_data[i]!={}:
            inst_list.append(user_data[i])
            user_index.append(i)
    return inst_list,user_index


def getUserIndex(user_data):
    user_index = []
    for uid in user_data:
        user_index.append(uid)
    return user_index


# def getUserSimilaryData(user_tmp, result):
#     user_list = []
#     for user in user_tmp:
#         user_list.append(user)
#     for i in range(len(result)):
#         user_tmp[user_list[i]] = result[i]
#     return user_tmp


# 2.计算
# 2.1 构造兴趣-->兴趣矩阵
def similarity(inst_list):
    # 将用户兴趣权重特征向量化
    #TODO 将用户的搜索兴趣标签 词频向量化
    dict_vectorizer = DictVectorizer(dtype=np.int64, sparse=False)
    # 返回用户的兴趣矩阵
    result = dict_vectorizer.fit_transform(inst_list)
    # 所有的兴趣
    inst = dict_vectorizer.get_feature_names()
    #print('@@@@@@@@@@inst@@@@@@@@@@@')
    #print(inst)
    #print('@@@@@@@@@@inst@@@@@@@@@@@')
    # print(result)
    # print(inst)

    # 余弦相似度矩阵
    user_similar = cosine_similarity(result)
    #print('############user_similar#############')
    #print(user_similar)
    #print('############user_similar#############')

    # print('aaaaaaaaa',user_similar.mean)
    # for i in user_similar.std:
    #     print('iiiiiiiiiiii',i)
    return user_similar


# #获取相似度最接近的用户,根据用户推荐兴趣
# def interestWeight(user_similar,N=1):
#     dic_similar={}
#     S=np.array(user_similar,dtype=np.float64)
#     for i in range(S.shape[0]):
#         if i == S.shape[0] - 1:
#             break
#         cur = S[i][i + 1]
#         for j in range(i + 1, S.shape[1]):
#             if cur < S[i][j]:
#                 dic_similar.update({i: j})
#             else:
#                 dic_similar.update({i: i + 1})
#     print(dic_similar)
#     #user_index={0: 1, 1: 2, 2: 3, 3: 4}
#     return dic_similar


# 获取相似度最接近的用户,根据相似用户推荐兴趣
def interestWeight(user_similar, N=1):
    dic_similar = {}
    S = np.array(user_similar, dtype=np.float64)
    for i in range(S.shape[0]):
        list = S[i]
        line = i
        cur = list[0]
        index = 0
        if line == 0:
            cur = list[1]
            index = 1
        for j in range(S.shape[1]):
            if j == line:
                continue
            if S[line][j] > cur:
                cur = S[line][j]
                index = j
        dic_similar[i] = index
    #TODO 提取相似的N个用户
    #print('$$$$$$$$dic_similar$$$$$$$$')
    #print(dic_similar)
    #print('$$$$$$$$dic_similar$$$$$$$$')
    return dic_similar


# 给用户推荐
def recommendList(user_data, uid, n=1, N=8):
    # 待推荐
    original_user = user_data[uid]
    if original_user=={}:
        return []

    original = copy(original_user)
    #print('*******用户兴趣标签*********')
    #print("用户兴趣标签", original)
    #print('*******用户兴趣标签*********')


    all_need=getInterestList(user_data)
    #print('%%%%%%%%all_need%%%%%%%%')
    #print(all_need)
    #print('!!!!!!!!all_need[0]!!!!!!!!')
    #print(all_need[0])
    #print('!!!!!!!!all_need[0]!!!!!!!!')
    #print(all_need[1])
    #print('%%%%%%%%all_need%%%%%%%%')

    inst_list = all_need[0]

    user_similar = similarity(inst_list)

    dic_similar = interestWeight(user_similar)

    # user_index = getUserIndex(user_data)
    user_index=all_need[1]

    # 获取用户对应索引值
    uindex = user_index.index(uid)

    # 获取相似用户对应索引值
    similar_uindex = dic_similar[uindex]
    similar_user = user_index[similar_uindex]
    # 获取相似用户兴趣
    similar_inst = user_data[similar_user]

    for key in similar_inst:
        if key not in original:
            original[key] = user_similar[uindex][dic_similar[uindex]]

    # 推荐的用户的兴趣列表
    # 当前username没有看过
    #print("%s为该用户的推荐" % (uid))
    # 添加到推荐列表中
    # recommend = sorted(original.items(), key=operator.itemgetter(1), reverse=True)[0:N]
    return original

# if __name__ == '__main__':
#     user_act = {
#         'A': {'跑步': 1, '游泳': 8, '音乐': 1},
#         'B': {'游泳': 2, '登山': 1, '学习': 1},
#         'C': {'登山': 5, '音乐': 5},
#         'D': {'游泳': 1, '登山': 1, '音乐': 7, 'IT': 9, '二次元': 1, '动漫': 7},
#         'E': {'跑步': 1, '音乐': 1}
#     }
#
#     # print(inst_list)
#     # data = getDataStruct()  # 获得数据
#     print('''''''''''''')
#     # similar = similarity(inst_list)  # 计算兴趣相似矩阵
#     # s = interestWeight(similar)
#     print('''''''''''''')
#     recommendList(user_act, 'A')  # 推荐
