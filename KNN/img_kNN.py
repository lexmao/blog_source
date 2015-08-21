# -*- coding:utf8 -*-

'''
Created on August 18,2015

This is example

Input: 
Output: 

@author: Stone <stone@bzline.cn>
'''

import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
import kNN

import os,os.path

#把每一个32*32像素的图片转化为一个1*1024的向量
#方便 KNN处理(向量)
def imageToVector(filename):

    imageVector =np.zeros((1,1024))
    fr =open(filename)
    
    for i in range(32): #读取32行
        line =fr.readline()
        for j in range(32):#读取每行的32个字节
            #把获得的每个字节转化为数字存储在向量中
            imageVector[0,i*32+j]=int(line[j])

    fr.close()
    
    return imageVector


#根据文件规则名,获得目标特征
#文件格式类似:0_48.txt , 第一'0'表示该图片的目标变量为0
def fromFilenameGetFeature(filename):
    return filename[0]


#从一个目录下获取每一文件进行学习
#文件格式类似:0_48.txt , 第一'0'表示该图片的目标变量为0



def scanDirForlearning(dir):
    #os.path.walk(dir,callbackForScanDir,(dir))

    if os.path.exists(dir) == False:
        print('dir do not exists')
        return False
    if os.path.isdir(dir) ==False:
        print('is not dir')
        return False

    for dirname,subpaths,names in os.walk(dir):
        if dirname ==dir:
            lineNumber =len(names)
            classLabelVector=[]
            dataMat=np.zeros((lineNumber,1024))#1024列,1024个特征
    
            m=0 #行数指示
            for name in names:
                filename ='/'.join((dirname,name))
                dataMat[m,:] = imageToVector(filename)
                classLabelVector.append(int(fromFilenameGetFeature(name)))
                m+=1
            
            return True,dataMat,classLabelVector 
     

    
    return False
    


if __name__ == '__main__':

    #test function
    #filename ='trainingDigits/2_178.txt'
    #v=imageToVector(filename)
    #print(v)

    
    isOk,dataMat,labels= scanDirForlearning('./trainingDigits')
    if isOk ==False:
        print('data error')
        exit(0)

    #不需要归一化,数值范围一致
    #normDataMat,ranges,min =kNN.autoNorm(dataMat)
    normDataMat=dataMat

    #求解的矩阵 
    inX =imageToVector('./testDigits/3_45.txt') 

    k=3   
    who =kNN.classify0(inX, normDataMat, labels, k)

    print('该图像内容为文字:%d'%int(who))
    
