#电影推荐
import json
import  numpy as np
'''
数据格式：
用户:{
    电影1：评分;
    电影2：评分
}
'''

with open('./ratings.json','r') as f:
    ratings=json.loads(f.read())

users=list(ratings.keys())
print(users)
cmat=[]
for user1 in users:
    crow=[]
    for user2 in users:
        #计算users与user2之间的相似度，保存到相似度矩阵中
        #找到都看过的电影
        movies=set()
        for movie in ratings[user2].keys():
            if movie in ratings[user1].keys():
                movies.add(movie)
        if len(movies)==0:
            score=0
        else:
            A,B=[],[]
            for movie in movies:
                A.append(ratings[user1][movie])
                B.append(ratings[user1][movie])
            #计算两人的欧式距离得分
            A,B=np.array(A),np.array(B)
            #score=1/(1+np.sqrt(np.mean(A-B)**2))
            #皮尔逊系数
            score=np.corrcoef(A,B)[0,1]
        crow.append(score)
    cmat.append(crow)

cmat=np.array(cmat)
print(np.round(cmat,2))

users=np.array(users)

#召回业务 针对每个用户的相似用户排序
for i,user in enumerate(users):
    sorted_inds=cmat[i].argsort()[::-1]
    print(sorted_inds)
    sorted_inds=sorted_inds[sorted_inds!=i]
    sim_users=users[sorted_inds]#相似用户排名
    sim_scores=cmat[i][sorted_inds]#相似得分排名
    #构建推荐清单内
    #找到所有相似度正相关的相似用户
    pos_mask=sim_scores>0
    sim_users=sim_users[pos_mask]
    #遍历相似用户，形成推荐列表
    rec_list={}
    for sim_user in sim_users:
        for movie,score  in  ratings[sim_user].items():
            if movie not in ratings [user].key():
                #把movie存入推荐列表
                if movie not in rec_list.keys():
                    rec_list[movie]=[score]
                else:
                    rec_list[movie].append(score)

    rec_list=sorted(rec_list.items(),
                    key=lambda x:np.mean(x[1]),
                    reverse=True)
    final_list=[]
    for rm in rec_list:
        final_list.append(rm[0])
    print(user)
    print(final_list)

