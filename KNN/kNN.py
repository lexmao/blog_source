# -*- coding:utf8 -*-

'''
Created on August 18,2015
Machine Learning on kNN: k Nearest Neighbors

Input: inX
Output: the most pooular class label

@author: Stone <stone@bzline.cn>
'''

import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt


#归一化数据公式
#newvalue=(oldvalue-min)/(max-min)
def autoNorm(dataSet):

    min = dataSet.min(0)
    max = dataSet.max(0)

    ranges=max-min
    normDataset=np.zeros(np.shape(dataSet))
    m =dataSet.shape[0]

    normDataset=dataSet - np.tile(min,(m,1))
    normDataset=normDataset/np.tile(ranges,(m,1))

    return normDataset,ranges,min



#第一件事,导入已知数据
#filename 数据源文件名
#feature_len 特征变量的个数,本例子中为4个
#返回包含特征的矩阵和目标变量向量

def loadDataFromFile(filename,feature_len):
    fr = open(filename)

    dataLines =fr.readlines()
    lineNumber = len(dataLines)
    classLabelVector=[]

    for line in dataLines:
        if line.find('#') == 0 : #find '\#'line
            dataLines.remove(line)
   
    lineNumber = len(dataLines)
    classLabelVector=[] 
    dataMat=np.zeros((lineNumber,feature_len))

    row_index=0
    for line in dataLines:
        line = line.strip()
        listFromLine =line.split(' ')
        dataMat[row_index,:]=listFromLine[0:feature_len]
        classLabelVector.append((listFromLine[-1]))
        row_index +=1    
    
    fr.close()
    return dataMat,classLabelVector 
        
        

#开始计算距离
#首先要传入要求解的目标向量inX
#dataSet  是上导入的竞争和目标向量,k是进行比较的条数
def classify0(inX, dataSet, labels, k):
    row_size = dataSet.shape[0] #获得矩阵中得行数
    #构造一个矩阵,每一行的向量是inX,一共有row_size行
    #目的是要实施两个矩阵运算行列个数一致 

    tempMat =np.tile(inX,(row_size,1))


    #欧氏距离公式 d(A，B) =√ [ ∑( a[i] - b[i] )^2 ] (i = 1，2，…，n)

    diffMat = tempMat-dataSet #矩阵运算,相减
    sqDiffMat = diffMat**2
    #求和 axis=1,计算每一行的和,axis=0按列计算
    sqDistances = sqDiffMat.sum(axis=1)

    #开方运算
    distances = sqDistances**0.5 

    #排序
    sortedDistIndicies = distances.argsort()
    classCount={}


    for i in range(k):
        votellabel = labels[sortedDistIndicies[i]]
        classCount[votellabel] = classCount.get(votellabel,0) + 1

    sortedClassCount = sorted(classCount.iteritems(),
        key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]
    

    
    



if __name__ == '__main__':

    level_list=['未知','一般','良好','优秀']
    dataMat,labels=loadDataFromFile('developer.txt',4)
    normDataMat,ranges,min =autoNorm(dataMat)    


    #对inX实现归一化
    inX=np.array([2,3,9,8])
    inX =(inX-min)/ranges

    k=3

    who =classify0(inX, normDataMat, labels, k)

    outs='该程序员属于*%s*类别'%level_list[int(who)]
    print(outs)

