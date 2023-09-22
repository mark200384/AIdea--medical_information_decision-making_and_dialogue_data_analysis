import pandas as pd
import jieba
import jieba.analyse
import re
import json

text_a=[]
text_b=[]
train_text = []
train_QA = []
train_questions = []
train_choices = []
train_choices_A = []
train_choices_B = []
train_choices_C = []
jieba.set_dictionary('dict.txt')
jieba.analyse.set_stop_words('stopword.txt')  # 自設停用詞表
stopwords = [line.strip() for line in open('stopword.txt', encoding='UTF-8').readlines()]
stopwords.append('何者')
stopwords.append('請')
stopwords.append('問')
stopwords.append('下列')
stopwords.append('?')
stopwords.append('？')



print(stopwords)
train_file = 'data/Train_qa_ans.json'

with open(train_file, 'r', encoding='utf-8') as obj:
    data_json = json.load(obj)
df_train = pd.DataFrame(data_json)
for i in range(len(df_train)):
    train_text.append(data_json[i]['text'])
    train_QA.append(data_json[i]['question'])
    train_questions.append(train_QA[i]["stem"])
    train_choices.append(train_QA[i]["choices"])

def Remove_word(text):
    output=""
    for word in text:
        if word not in stopwords:
            output+=word
        elif word =='不':
            output+=word
    return output
MAX_LENGTH = 150
for i in range(695):
    train_choices_A.append(train_choices[i][0]['text'])
    train_choices_B.append(train_choices[i][1]['text'])
    train_choices_C.append(train_choices[i][2]['text'])

for i in range(695):
    temp = [w for w in jieba.cut(train_questions[i])]
    train_questions[i] =  Remove_word(temp)

    temp = [w for w in jieba.cut(train_choices_A[i])]
    train_choices_A[i]=Remove_word(temp)

    temp = [w for w in jieba.cut(train_choices_B[i])]
    train_choices_B[i] = Remove_word(temp)

    temp = [w for w in jieba.cut(train_choices_C[i])]
    train_choices_C[i] = Remove_word(temp)


print(train_questions)
print(train_choices_A)
print(train_choices_B)
print(train_choices_C)
def split_text(text):
    output=text.split('。')
    return output
for i in range(695):
    keywords=[]
    temp = train_questions[i]+" "+train_choices_A[i]+" "+train_choices_B[i]+" "+train_choices_C[i]
    # print(temp)
    temp2 = [t for t in jieba.cut(temp)]
    for w in temp2:
        if w !=' ' and w not in keywords:
            keywords.append(w)
    for i in range
print(keywords)
# train_data = toDataFrame(data_json, 0)
