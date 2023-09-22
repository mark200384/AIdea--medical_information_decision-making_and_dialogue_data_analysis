import json
import pandas as pd
import numpy as np
from pandas import json_normalize


def read_data():
    file = 'data/Train_qa_ans.json'
    with open(file, 'r', encoding='utf-8') as obj:
        data_json = json.load(obj)
    # print(data_json)
    data = toDataFrame(data_json)
    return data


id = []
article_id = []
temp_text = []
QA = []
questions = []
choices = []
answer = []


def toDataFrame(data_json):
    data = pd.DataFrame(data_json)
    print(data)
    # for i in range(len(data)):
    #     id.append(data_json[i]['id'])
    #     article_id.append(data_json[i]['article_id'])
    #     temp_text.append(data_json[i]['text'])
    #     QA.append(data_json[i]['question'])
    #     questions.append(QA[i]["stem"])
    #     choices.append(QA[i]["choices"])
    #     answer.append(data_json[i]["answer"])
    for i in range(len(data)):
        id.append(data_json[i]['id'])
        article_id.append(data_json[i]['article_id'])
        temp_text.append(data_json[i]['text'])
        QA.append(data_json[i]['question'])
        questions.append(QA[i]["stem"])
        choices.append(QA[i]["choices"])
        answer.append(data_json[i]["answer"])
    print(choices[0][0]['text'])  # 第1篇第1個選項
    print(len(choices))
    return data


import jieba
import jieba.analyse

jieba.set_dictionary('dict.txt')
jieba.analyse.set_stop_words('stopword.txt')  # 自設停用詞表


def list_of_groups(init_list, childern_list_len):
    list_of_group = zip(*(iter(init_list),) * childern_list_len)
    end_list = [list(i) for i in list_of_group]
    count = len(init_list) % childern_list_len
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


import re


def preprocess(data):
    text = []  # article文章(不重複)
    temp_corpus = []
    for i in temp_text:
        if i not in text:
            text.append(i)
    for i in range(len(text)):
        text[i] = re.sub(r'[^\u4e00-\u9fa5]', '', text[i])  # 去除數字標點符號
        corpus.append(" ".join(jieba.cut(text[i], cut_all=True)))

    for i in range(len(QA)):
        for j in range(3):
            temp_QA = QA[i]['stem'] + choices[i][j]['text']
            temp_QA = re.sub(r'[^\u4e00-\u9fa5]', '', temp_QA)
            temp_corpus.append(" ".join(jieba.cut(temp_QA, cut_all=True)))
    QA_corpus = list_of_groups(temp_corpus, 3)
    print(QA_corpus)

    return corpus, QA_corpus


import gensim
import word2vec
from gensim.models import Word2Vec

averages_of_article = []
averages_of_qa = []


def WordtoVec():
    fileSegWordDonePath = 'corpus_text\\article.txt'
    fileSegWordDonePath2 = 'corpus_text\\answers.txt'

    # for i in range(346):
    #     fileSegWordDonePath = 'corpus_text\\article_' + str(i + 1) + '.txt'
    #     with open(fileSegWordDonePath, 'wb') as fW:
    #         fW.write(corpus[i].encode('utf-8'))
    #         fW.write('\n'.encode('utf-8'))
    #     word2vec.word2vec('corpus_text\\article_' + str(i + 1) + '.txt',
    #                       'corpus_text\\w2v_' + str(i + 1) + '.bin', size=300,
    #                       verbose=True)
    #     model = word2vec.load('corpus_text\\w2v_' + str(i + 1) + '.bin')
    #     # print(model.vectors)
    #     print(model.vocab)
    #     mean = np.mean(model.vectors)
    #     averages_of_article.append(mean)  # 第0個位置第1篇文章的平均
    # print(averages_of_article)
    Test()

    # for i in range(len(choices)):
    #     consistency = []
    #     for j in range(3):
    #         # temp = corpus[i]+QA_corpus[i][j]
    #         # print(temp)
    #         fileSegWordDonePath2 = 'corpus_text\\id' + str(i + 1) + '_choices_' + str(j + 1) + '.txt'
    #         with open(fileSegWordDonePath2, 'wb')as fW:
    #             fW.write(QA_corpus[i][j].encode('utf-8'))
    #             fW.write('\n'.encode('utf-8'))
    #         word2vec.word2vec('corpus_text\\id' + str(i + 1) + '_choices_' + str(j + 1) + '.txt',
    #                           'corpus_text\\w2v_id' + str(i + 1) + '_choices_' + str(j + 1) + '.bin',
    #                           size=10,
    #                           verbose=True)
    #         print(QA_corpus[i][j])
    #         model2 = word2vec.load('corpus_text\\w2v_id' + str(i + 1) + '_choices_' + str(j + 1) + '.bin')
    #         mean2 = np.mean(model2.vectors)
            # print(model2.vectors)
            # averages_of_qa.append(mean2)  # 三個選項的向量
        # print(averages_of_qa)

        # if(averages_of_qa[0]==averages_of_qa[1] and averages_of_qa[1]==averages_of_qa[2]):
        #     my_ans.append('N')
        # else:
        #     temp = consistency.index(max(averages_of_qa))
        #     if (temp == 0):
        #         my_ans.append('A')
        #     elif (temp == 1):
        #         my_ans.append('B')
        #     elif (temp == 2):
        #         my_ans.append('C')
        # print(my_ans)

def Test():
    for i in range(len(QA)):
        key_word=""
        for j in range(3):
            key_word = jieba.cut(choices[i][j])
            key_word = [word for word in key_word if word !='']
            key_word = ' '.join(data)
        print(key_word)


my_ans = []  # 猜測的答案
window_size = 2
embedding_size = 5
corpus = []
choice_corpus = []
QA_corpus = []
if __name__ == '__main__':
    context_pair = []
    word_set = set()
    data = read_data()
    corpus, QA_corpus = preprocess(data)  # corpus是346篇合起來的

    # print(corpus)
    WordtoVec()

    # Doc2Vec()
