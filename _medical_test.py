import pandas as pd
import numpy as np
import jieba
import os
from sklearn.metrics import classification_report

jieba.set_dictionary('dict.txt')

stream = pd.read_excel('word.xls')
word = list(stream['詞彙'])  # 詞彙表

risk_file = os.path.join("data", "Train_risk_classification_ans.csv")
source_data = pd.read_csv(risk_file, usecols=['article_id', 'text', 'label'])
print(source_data)

corpus = []

for i in range(0, len(source_data)):
    corpus.append(" ".join(jieba.cut(source_data.loc[i]['text'])))
print("corpus",corpus)

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
word = vectorizer.get_feature_names()
weight = tfidf.toarray()

reference = []
for i in range(len(weight)):
    for j in range(len(word)):
        if weight[i][j] > 0.15 and word[j] not in reference:
            reference.append(word[j])
print(len(reference))

counter = []
label = []
for i in range(0, len(source_data)):
    counter.append([0] * len(reference))
    label.append(source_data.loc[i]['label'])

testers = []

for i in range(0, len(source_data)):
    testers.append(source_data.loc[i]['text'])
for i in range(0, len(testers)):
    for k in range(len(reference)):
        if word[k] in testers[i]:
            a = word.index(word[k])
            counter[i][a] = 1

from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(counter, label, train_size=0.8)
print("trainX", trainX)
print("trainX length:", len(trainX))
print("trainY", trainY)
crossvalueX = []
crossvalueY = []
for i in range(0, 300):
    crossvalueX.append(counter[i])
    crossvalueY.append(source_data.loc[i]['label'])

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

target_names = ['娛樂', '產經', '健康']

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
