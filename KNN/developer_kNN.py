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
import kNN


if __name__ == '__main__':

    level_list=['未知','一般','良好','优秀']
    dataMat,labels=kNN.loadDataFromFile('developer.txt',4)
    normDataMat,ranges,min =kNN.autoNorm(dataMat)    

    #对inX实现归一化
    inX=np.array([2,3,9,8])
    inX =(inX-min)/ranges

    k=3

    who =kNN.classify0(inX, normDataMat, labels, k)

    outs='该程序员属于*%s*类别'%level_list[int(who)]
    print(outs)

