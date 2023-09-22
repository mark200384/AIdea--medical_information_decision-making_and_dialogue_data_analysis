import pandas as pd
import numpy as np
import os

risk_file = os.path.join("data", "Train_risk_classification_ans.csv")
risk_pd = pd.read_csv(risk_file, usecols=['article_id', 'text', 'label'])
print(risk_pd)

import csv
import pandas as pd
import numpy as np

sentences = []
for i in range(len(risk_pd)):
    sentences.append(risk_pd['text'][i])

temp_text = ""

all_doctor_text = []
all_Manager_text = []
all_people_text = []
label = []
pos_d = 0
pos_m = 0
pos_p = 0
pos_e = 0

for i in range(346):
    doctor_text = []
    Manager_text = []
    people_text = []
    label.append(risk_pd['label'][i])
    sentence = sentences[i]
    pos_m = sentence.find('個管師：')
    pos_d = sentence.find('醫師：')
    while pos_m != -1:
        pos_m = sentence.find('個管師：')
        pos_e = sentence.find('民眾：')  # 找到下一個對話者
        temp_text = sentence[pos_m + 4:pos_e]
        if temp_text == "":
            t = 0
        else:
            Manager_text.append(temp_text)
        sentence = sentence[pos_e:]  # 重新裁剪句子
        pos_e = sentence.find('民眾：')
        pos_m = sentence.find('個管師：')
        temp_text = sentence[pos_e + 3:pos_m]

        if temp_text == "":
            t = 0
        else:
            people_text.append(temp_text)
        sentence = sentence[pos_m:]

    while pos_d != -1:  # 主角是醫生
        pos_d = sentence.find('醫師：')
        pos_e = sentence.find('民眾：')  # 找到下一個對話者
        temp_text = sentence[pos_m + 3:pos_e]
        if temp_text == "":
            t = 0
        else:
            doctor_text.append(temp_text)
        sentence = sentence[pos_e:]  # 重新裁剪句子
        pos_e = sentence.find('民眾：')
        pos_m = sentence.find('醫師：')
        temp_text = sentence[pos_e + 3:pos_m]

        if temp_text == "":
            t = 0
        else:
            people_text.append(temp_text)
        sentence = sentence[pos_d:]
    all_doctor_text.append(doctor_text)
    all_Manager_text.append(Manager_text)
    all_people_text.append(people_text)

import jieba.analyse
import re

# for i in range(346):
#     if(len(all_Manager_text[i])!=0):
#         print("個管師:", all_Manager_text[i])
#     if (len(all_doctor_text[i]) != 0):
#         print("醫師:", all_doctor_text[i])
#     if (len(all_people_text[i]) != 0):
#         print("民眾:", all_people_text[i])
#     print("Label:",label[i])

jieba.set_dictionary('dict.txt')
jieba.analyse.set_stop_words('stopword.txt')  # 自設停用詞表

stream = pd.read_excel('word.xls')  # 教育部常用字
word = list(stream['詞彙'])  # 詞彙表

counter = []
for i in range(0, len(risk_pd)):
    counter.append([0] * 13374)
testers = []
print(counter)
for i in range(0, len(all_Manager_text)):
    if(len(all_Manager_text[i])):
        s = " ".join(all_Manager_text[i])
        testers.append(s)

for i in range(0, len(all_doctor_text)):
    if (len(all_doctor_text[i])):
        s = " ".join(all_doctor_text[i])
        testers.append(s)
print(len(testers))
print(len(counter))
# print(word)
# print(testers)
for i in range(0, len(testers)):
    for k in range(13374):
        if word[k] in testers[i]:
            a = word.index(word[k])
            counter[i][a] = 1

from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(counter, label, train_size=0.8)
crossvalueX = []
crossvalueY = []
for i in range(0, 346):
    crossvalueX.append(counter[i])
    crossvalueY.append(label[i])

# BernoulliNB預測
from sklearn.naive_bayes import BernoulliNB  # import BernoulliNB

BNB_cross = BernoulliNB()
BNB = BNB_cross.fit(trainX, trainY)
Berpredict = BNB.predict(testX)  # BernoulliNB的預測結果

# KNeighbors
from sklearn.neighbors import KNeighborsClassifier  # import KNeighborsClassifier

neigh_cross = KNeighborsClassifier(n_neighbors=5)
neigh = neigh_cross.fit(trainX, trainY)
KNN = neigh.predict(testX)  # KNeighborsClassifier預測結果

# Support Vendor Machine
from sklearn import svm

svmclf = svm.SVC(kernel='linear', C=1).fit(trainX, trainY)
SVM_predict = svmclf.predict(testX)

# SVM預測結果

# Random Forest
from sklearn.ensemble import RandomForestClassifier

Random_cross = RandomForestClassifier(n_estimators=100, random_state=50)
Randomclf = Random_cross.fit(trainX, trainY)
RandomForest_predict = Randomclf.predict(testX)  # random forest預測結果

from sklearn import datasets
from sklearn.model_selection import cross_val_score, train_test_split

print("crossvalidation:")

print("Berpredic:")
print(cross_val_score(BNB, crossvalueX, crossvalueY, cv=5))  # 使用Berpredict的cross結果
print("Berpredic average accuracy:", cross_val_score(BNB, crossvalueX, crossvalueY, cv=5).mean())

print("KNN:")
print(cross_val_score(neigh, crossvalueX, crossvalueY, cv=5))  # 使用KNN的cross結果
print("KNN average accuracy:", cross_val_score(neigh, crossvalueX, crossvalueY, cv=5).mean())

print("SVM:")
print(cross_val_score(svmclf, crossvalueX, crossvalueY, cv=5))  # 使用svm的cross結果
print("SVM average accuracy:", cross_val_score(svmclf, crossvalueX, crossvalueY, cv=5).mean())

print("Random Forest:")
print(cross_val_score(Randomclf, crossvalueX, crossvalueY, cv=5))  # random forest的cross結果
print("Random Forest average accuracy", cross_val_score(Randomclf, crossvalueX, crossvalueY, cv=5).mean())
