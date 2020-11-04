#coding=utf-8
import numpy as np
import pandas as pd

# 提取特征值的列名
feature_columns = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'score']

# 从dataframe中获取特征值数组
def get_features(df):
	return df[feature_columns].values
# 读取csv文件, 得到dataframe
res_df = pd.read_csv("./res/res.csv")
# 用户事务循环
while True:
    # 读取用户输入的电影id
    id = int(input("请输入电影id(-1退出):"))
    if id == -1:
        break
    # 根据电影id查找电影，找不到则继续循环
    find_res = res_df[res_df['id'] == id]
    if(find_res.empty):
        continue
    # 打印用户输入id对应的电影信息
    print(find_res[['name', 'type', 'score']].squeeze())
    # 获取输入电影所属的簇号
    cluster_num = find_res['prediction'].squeeze()
    # 获取输入电影的特征值
    feature = get_features(find_res)
    # 获取相同簇号的所有电影信息并重置行索引
    cluster = res_df[res_df['prediction'] == cluster_num].reset_index(drop = True)
    # 特征值间距离列表(欧式距离)
    dis_list = []
    # 计算相同簇的所有电影的特征值与输入电影特征值的距离
    for index in cluster.index:
        # 跳过输入电影
        if cluster.iloc[index, 0] == id:
            continue
        # 获取电影特征值
        feature_other = get_features(cluster.loc[index])
        # 计算欧式距离并加入列表
        dis = np.linalg.norm(feature - feature_other)
        dis_list.append(dis)
    # 将列表转换为numpy数组
    dis_narray = np.array(dis_list)
    # 打印排序前10名的电影信息
    print(cluster.iloc[dis_narray.argsort()[:10]][['name', 'type', 'score']].reset_index(drop = True))
    
