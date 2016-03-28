# --*-- coding:utf8 --*--

## Spark Application - execute with spark-submit
 
## Imports
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating



import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt

 
## Module Constants
APP_NAME = "movieLines-c04"
 
## Closure Functions



#ALS

def als_data(sc):
    raw_data = sc.textFile("./data/ml-100k/u.data")
    fields_data=raw_data.map(lambda line:line.split('\t'))
    print fields_data.first()

    #取每行的前三个元素,即时间戳不需要
    fields_data=fields_data.map(lambda fields:fields[:-1])

    #load rating data. Each row consists of a user, a product and a rating
    ratings=fields_data.map(lambda fields: Rating(int(fields[0]),int(fields[1]),float(fields[2])))
    print ratings.first()

    # Build the recommendation model using Alternating Least Squares
    rank = 10 #ALS模型中的因子个数,一般取10-200
    numIterations = 10 #运行时候的迭代次数
    noridx=0.18  #正则化参数
    model = ALS.train(ratings, rank, numIterations,noridx) 
    #model = ALS.train(ratings, rank, numIterations)

    #print model.userFeatures().take(2)


    # Evaluate the model on training data
    testdata = ratings.map(lambda p: (p[0], p[1]))
    predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
    ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    print("Mean Squared Error = " + str(MSE))


    #test
    predictedRating =model.predict(455,629) #3
    print "uid 455,item 629 will ratign=%f" % predictedRating
   
    #94  1032    2   891723807 
    predictedRating =model.predict(94,1032) #5
    print "uid 94 item 1032 will ratign=%f" % predictedRating


    #topKRecs =model.recommendProducts(94,10)
    #print "will recommend item to userid 789:"
    #print topKRecs


    # Save and load model
    #各个worker存在同步问题,该model有点worker没有完整保存,why?
    print "model save...c04.m"
    model.save(sc, "./Model/c04.m")

#根据电影id转化为电影名称
def covert_to_title_by_itemid(sc):
    
    raw_data=sc.textFile("./data/ml-100k/u.item")
    fields_data=raw_data.map(lambda line:line.split('|'))
    #collectAsMap直接返回一个字典,collect()返回array. 如果用collect,自己转为字典
    id_titles=fields_data.map(lambda fields:(int(fields[0]),fields[1])).collectAsMap()#collect()
    id_titles_dict=id_titles#dict(id_titles)
    return id_titles_dict

    #return id_titles_dict[itemid]
    

def get_all_item_by_uid(sc,userid):
    raw_data = sc.textFile("./data/ml-100k/u.data")
    fields_data=raw_data.map(lambda line:line.split('\t'))

    #取每行的前三个元素,即时间戳不需要
    fields_data=fields_data.map(lambda fields:fields[:-1])

    #load rating data. Each row consists of a user, a product and a rating
    ratings=fields_data.map(lambda fields: Rating(int(fields[0]),int(fields[1]),float(fields[2])))

    #用kyBy函数从ratings创建一个键值对,主键为user,然后用lookup返回给定用户id的数据
    moviesForUser=ratings.keyBy(lambda rating:rating.user).lookup(789) #返回list
    print len(moviesForUser)

    moviesForUser=sorted(moviesForUser,key=lambda r:r.rating, reverse=True)

    return moviesForUser
    #return moviesForUser[0:10] #只返回10条

    

def recommendProducts_by_uid(sc,uid):
    model= MatrixFactorizationModel.load(sc, "./Model/c04.m")
    
    #向该用户推荐前10个电影
    #Recommends the top “num” number of products for a given user and 
    #returns a list of Rating objects sorted by the predicted rating in descending order.
    #recommendProducts(user, num)
    topKRecs = model.recommendProducts(uid,10)

    print topKRecs


def recommand_star_for_user(sc,uid,item_id):

    model= MatrixFactorizationModel.load(sc, "./Model/c04.m")
   
    predictedRating =model.predict(uid,item_id)
    titles=covert_to_title_by_itemid(sc)

    #向该用户推荐前10个电影
    topKRecs = model.recommendProducts(uid,10)
    print [(titles[r.product], r.rating) for r in topKRecs]
   # for record in topKRecs:
   #     (record.user,titles[record.product],record.rating)
        #print "%d \t %s \t %f " % (record.user,titles[record.product],record.rating)
   
    print "-------------------" 

    #验证推荐的数据靠谱否
    #获得该用户已经评分最高的10个电影
    actual_all_item_array=get_all_item_by_uid(sc,uid)
    top10item_array=actual_all_item_array[0:10] 

    print [(titles[r.product], r.rating) for r in top10item_array]
    #for record in top10item_array:
    #    print "%d \t %s \t %f " % (record.user,titles[record.product],record.rating)

    #用APK评估推荐质量
    predictedMovies = [tkr.product for tkr in topKRecs]
    actualMovies=[r.product for r in actual_all_item_array]
    apk10 = avgPrecisionK(actualMovies, predictedMovies, 10) 
    print "predicteMovies = %s" % predictedMovies
    print "actual movies :%s" % actualMovies 
    print "APK Top 10 =%f" % apk10
    
   

#输入item id,返回相似的item 
#比较相似,可以用item因子向量 做余弦相似度来实现

# cosine=(v1.v2)/(||v1|| ||V2||)

def cosine_similarity(vec1,vec2):
    from scipy.spatial.distance import cosine
    import math

    #scipy 有现成的函数可以用,结果一样
    #cosine函数实际上是是求1-cosine,所以我们返回1-...
    #return 1-cosine(vec1,vec2)



    vlen=len(vec1)

    #考虑把vec中的元素正则化:每个元素减去平均值
    vec1_mean =np.mean(vec1)
    vec1_mean_array=vec1-vec1_mean #向量相减
    vec1=vec1_mean_array


    vec2_mean =np.mean(vec2)
    vec2_mean_array=vec2-vec2_mean
    vec2=vec2_mean_array


    #v1,v2的点乘
    dot_res=0
    for index in range(0,vlen):
        dot=vec1[index] * vec2[index]
        dot_res +=dot

    #计算 ||vec||
    vec1_length= 0
    for index in range(0,vlen):
        dot=vec1[index]*vec1[index]
        vec1_length +=dot
    vec1_length=math.sqrt(vec1_length)

    vec2_length=0
    for index in range(0,vlen):
        dot=vec2[index]*vec2[index]
        vec2_length +=dot
    vec2_length=math.sqrt(vec2_length)

    #cosine=(v1.v2)/(||v1|| ||V2||)

    cosine=dot_res/((vec1_length)*(vec2_length))
    return cosine
        



def item_like_item(sc,item_id): 

    model= MatrixFactorizationModel.load(sc,"./Model/c04.m")
    titles=covert_to_title_by_itemid(sc)

    #获得该item的产品因子,返回一个array
    itemFactor=model.productFeatures().lookup(item_id)[0] #返回第一个array

    #和所有的item比较相似性,并返回(id,sim)
    sims=model.productFeatures().map(lambda (id,vec):(id,cosine_similarity(vec,itemFactor)))

    #需要根据sim的大小排序
    sorteSims=sims.sortBy(lambda s:s[1],False) #False 表示从高到底排序,默认从低到高

    #增加item名
    sorteSims=sorteSims.map(lambda (id,sim):(id,titles[id],sim))

    sorteSims =sorteSims.collect()[0:11]
    print sorteSims



# 计算APK(Average Precision at K metric)K值平均准确率
def avgPrecisionK(actual, predicted, k):

    predictedTopK=predicted[0:k]
    score=0.0
    numHits=0.0

    #enumerate 函数用于遍历序列中的元素以及它们的下标 
    for index,item_id in enumerate(predictedTopK):
        if item_id in actual:
            numHits +=1
            score +=numHits/float(index+1)
    
    if(len(actual)==0):
        return 1.0

    else:
        return score/float(min(k,len(actual)))
    
    
#计算所有用户的平均APK    
def getAllAPK(sc):
    model= MatrixFactorizationModel.load(sc,"./Model/c04.m")
  
    #电影矩阵  moviesid和10个因子组成的矩阵
    moviesFactors = model.productFeatures().map(lambda (id,factors):factors).collect()
    moviesMatrix=np.array(moviesFactors)
    print moviesMatrix.shape #(841, 10)

    #广播给所有的worker
    imBroadcast=sc.broadcast(moviesMatrix) 


    userFactors=model.userFeatures().map(lambda (id,factors):factors).collect()
    userMatrix=np.array(userFactors)
    print userMatrix.shape

    scoresForUser = model.userFeatures().map(lambda (userId, array): (userId, np.dot(imBroadcast.value, array)))

    allRecs = scoresForUser.map(lambda (userId, scores): 
                            (userId, sorted(zip(np.arange(1, scores.size), scores), key=lambda x: x[1], reverse=True))
                           ).map(lambda (userId, sortedScores): (userId, np.array(sortedScores, dtype=int)[:,0]))
    print allRecs.first()[0]
    print allRecs.first()[1]

    # groupByKey返回(int, ResultIterable), 其中ResultIterable.data才是数据

    raw_data = sc.textFile("./data/ml-100k/u.data")
    fields_data=raw_data.map(lambda line:line.split('\t'))
    fields_data=fields_data.map(lambda fields:fields[:-1])
    ratings=fields_data.map(lambda fields: Rating(int(fields[0]),int(fields[1]),float(fields[2])))

    userMovies = ratings.map(lambda r: (r.user, r.product)).groupByKey()
    print userMovies.first()[0]
    print userMovies.first()[1].data

    K = 10
    MAPK = allRecs.join(userMovies).map(lambda (userId, (predicted, actual)):
                                    avgPrecisionK(actual.data, predicted, K)
                                   ).sum() / allRecs.count()
    print "Mean Average Precision at K =", MAPK


    
    



    
    
    
    




    
    



 
## Main functionality
 
def main(sc):
    #创建模型
    #als_data(sc)

    #向用户推荐可能感兴趣的电影
    #recommand_star_for_user(sc,455,629)
    #recommendProducts_by_uid(sc,455)

    #计算APK
    #getAllAPK(sc)

    #找到指定电影的相似电影集合
    item_like_item(sc,567)




 
if __name__ == "__main__":
    
# Configure Spark
    conf = SparkConf().setAppName(APP_NAME)
    sc   = SparkContext(conf=conf)
 
    
    # Execute Main functionality
    main(sc)
