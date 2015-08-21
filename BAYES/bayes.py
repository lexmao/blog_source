# -*- coding:utf8 -*-
'''
Created on August 20,2015

@author: Stone <stone@bzline.cn>
'''

import numpy as np
import operator

'''
P(h+) 垃圾邮件的概率
P(h-) 普通邮件的概率
P(h+|D) 新邮件是垃圾邮件的概率
P(h+|D) =P(D|h+)P(h+)/P(D) -> P(d1,d2,..|h+)P(h+)
        =p(d1|h+)P(d2|h+)....

P(h-|D) 新邮件是正常邮件的概率
P(h-|D) =P(D|h-)P(h-)/P(D) -> P(d1,d2,..|h-)P(h-)  
'''


#获得每个样本文件中的的行数(每一行表示一封邮件的词串)
def getLinesNumber(filename):
    fr =open(filename)
    lines = fr.readlines()
    return len(lines)

#返回P(h+),P(h-)的概率
def getEmailPr(email_file,spam_email_file):
    email_l =  getLinesNumber(email_file)
    spam_email_l = getLinesNumber(spam_email_file)
   
    return float(email_l)/float((email_l+spam_email_l)),\
            float(spam_email_l)/float((email_l+spam_email_l))


'''
统计每一个词在所有垃圾邮件和正常邮件中出现的概率
P(d1|h+) 表示d1这个词在垃圾邮件中出现的概率,等于
该词在垃圾邮件中出现的次数/垃圾邮件中所有词的数量
这个问题转化为:在一个元素的集合中,计算该元素出现的概率

Output:
一个向量表示在某一类邮件中不重合的词串
['a','b','c','d','e']
一个向量表示这个词串出现概率
[0.01,0.02,0.03,0.1,0.04]
返回上面2个向量

Input:
requestWordList是待求解的邮件的词串列表
'''

def createVocabList(filename,requestWordList):
    fr=open(filename)
    dataLines=fr.readlines()
    lineNumber=len(dataLines)

    vocabSet=set([])

    #该类邮件中所有词的数量
    totalWord=0
   
    for line in dataLines:
        line =line.strip()
        listFromLine=line.split(' ')
        vocabSet =vocabSet|set(listFromLine)

    '''
    把requestWordList加入到词池中,主要是避免万一邮件池中不存在
    待求解的词,这导致p(d|h)概率为0,那么p(d1|h)*p(d2||h)...全部为0
    为了避免这个情况,直接把新邮件的词加入到池中
    '''
    vocabSet =vocabSet|set(requestWordList)
        
    returnVocabList=list(vocabSet) 
    returnVecList =[0]*len(returnVocabList)

    for word in requestWordList:
        if word in returnVocabList:
            returnVecList[returnVocabList.index(word)] +=1
        totalWord +=1

    for line in dataLines:
        line =line.strip()
        listFromLine=line.split(' ')
        for word in listFromLine:
            if word in returnVocabList:
                returnVecList[returnVocabList.index(word)] +=1
            totalWord +=1

    


    tempMat=np.array(returnVecList)

    #为了避免小数*小数可能为0的情况,采用对数来处理
    #原来returnVecList中每个元素记录的是p(d|h)的概率,现在变为ln(p(d|n))
    tempMat =np.log(tempMat/float(totalWord))
    returnVecList=list(tempMat)

    #returnVocabList 不重复的词串数组
    #returnVecList 对应returnVocabList中每个词的概率p(d|h)
    return returnVocabList,returnVecList


'''
Input:
    returnVocabList 不重复的词串数组
    returnVecList 对应returnVocabList中每个词的概率p(d|h)
    email_word_list 求解邮件的单词数组

Output:
    返回该类别的判断概率->转化为log(ln)
'''
def decideSpamEmail(returnVocabList,returnVecList,email_word_list):

    returnPr=0.0

    for word in email_word_list:
        if word  in returnVocabList:
            '''
            returnVecList[]中保存的是log对数的值,所以这里用+,二不是*
            p(d1|h)*p(d2|h) =log(p(d1|h))+log(p(d2|h))
            '''
            returnPr =returnPr+returnVecList[returnVocabList.index(word)]
            #print(returnPr)

    return returnPr


#test main

if __name__ == '__main__':

    pL,pS= getEmailPr('./emails.txt','./spam_emails.txt')
    #print('正常邮件和垃圾邮件的概率分别为:%f-%f\n'%(pL,pS))

    email_word_list=['退订','订阅','转发','取消','消息','显示','复制','DM']
    #email_word_list=['退订','DM']

    returnVocabList,returnVecList=createVocabList('./emails.txt',email_word_list)
    email_p=decideSpamEmail(returnVocabList,returnVecList,email_word_list)

    ph0=(email_p*pL)
    #print('求解的邮件为正常邮件的概率(ln):%f'%(ph0))

    returnVocabList,returnVecList=createVocabList('./spam_emails.txt',email_word_list)
    spam_email_p=decideSpamEmail(returnVocabList,returnVecList,email_word_list)
    ph1=(spam_email_p*pS)
    #print('求解的邮件为垃圾邮件的概率(ln):%f'%(ph1))
  
    print(email_word_list)
    print('\r\n') 
    if(ph0 >= ph1):
        print('这是一封正常邮件')
    else:
        print('这是一封垃圾邮件')
    
