#coding=utf-8
from pyspark import SparkContext, SparkConf
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, lit
from pyspark.sql import functions as F
import pyspark.sql.types as T

#电影类型列表
movie_type = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

conf = SparkConf().setAppName("minProject").setMaster("local")
sc = SparkContext(conf = conf)
spark = SparkSession(sc)

# 读取评分文件
ratings = sc.textFile("file:///home/yuan/lab2/ratings.dat")
# 获得电影id与其平均评分信息
ratings_rdd = ratings.map(lambda x:(x.split("::")[1],(int(x.split("::")[2]),1))).reduceByKey(lambda x,y : (x[0] + y[0], x[1] + y[1])).map(lambda a : (int(a[0]) , a[1][0] / a[1][1]))
# 将RDD转换为dataframe
ratings_df = ratings_rdd.toDF(['id', 'score'])
#print(ratings_df.count())

# 读取电影信息
movies = spark.read.text("file:///home/yuan/lab2/movies.dat")
# 分割电影信息，分别获取电影id、电影name、电影类型
split_col = split(movies['value'], "::")
movies_df = movies.withColumn('id', split_col.getItem(0)).withColumn('name', split_col.getItem(1)).withColumn('type', split_col.getItem(2))
movies_df = movies_df.drop('value')
# 将电影类型分割为列表
movies_df = movies_df.withColumn('type_list', split(movies_df['type'], "\|"))

'''
向电影表中加入电影类型对应的列
对于每一条电影，若该电影属于该类型，则该电影的该类型的格子值为1，否则为0
'''
for s in movie_type:
	movies_df = movies_df.withColumn(s, lit(0))
for s in movie_type:
	movies_df = movies_df.withColumn(s, F.when(F.array_contains(F.col('type_list'), s), 1).otherwise(0))
#print(movies_df.count())

'''
合并电影评分与电影信息
'''
df = movies_df.join(ratings_df, ["id"], "inner")
df = df.drop('type_list')
#print(df.count())

# 获取特征值向量(电影类型及评分)
vectorAssembler = VectorAssembler(inputCols = movie_type + ["score"], outputCol = "vectorFeature")
vdf = vectorAssembler.transform(df)
#vdf.select("vectorFeature").show(3, False)

'''
#使用手肘法确定合适的K值(K取值范围[10, 80])
cost = list(range(10,80))
for k in range(10, 80):
	kmeans = KMeans(k = k, seed = 1, featuresCol = 'vectorFeature')
	model = kmeans.fit(vdf)
	cost[k - 10] = model.computeCost(vdf)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.plot(range(10,80), cost)
ax.set_xlabel('k')
ax.set_ylabel('cost')
plt.show()
'''
'''
根据最终选择的特征值(K = 20)进行聚类
'''

kmeans = KMeans(k = 20, seed = 1, featuresCol = 'vectorFeature')
model = kmeans.fit(vdf)

#打印各类簇中心点
#print(model.clusterCenters())

res_df = model.transform(vdf)

res_df = res_df.drop('vectorFeature')

'''
输出为csv文件
'''
file="file:///home/yuan/lab2/res.csv"
res_df.write.csv(path = file, header=True, sep=",", mode='overwrite')

sc.stop()
spark.stop()
