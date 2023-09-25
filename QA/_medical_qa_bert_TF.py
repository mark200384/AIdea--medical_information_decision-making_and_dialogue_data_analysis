import json
import jieba.analyse
import pandas as pd

import re
import jieba
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from IPython.display import clear_output
import torch.nn.functional as F

jieba.set_dictionary('dict.txt')
PRETRAINED_MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
print("PyTorch 版本：", torch.__version__)
stopwords = [line.strip() for line in open('stopword.txt', encoding='UTF-8').readlines()]
train_text = []
temp_train_text = []
test_text = []
train_QA = []
test_QA = []
train_questions = []
test_questions = []
train_choices = []
test_choices = []
train_choices_A = []
test_choices_A = []
train_choices_B = []
test_choices_B = []
train_choices_C = []
test_choices_C = []
contradict_question = []
answer = []
label_A = []
label_B = []
label_C = []
accuracy_A = []
accuracy_B = []
accuracy_C = []


def text_select(text, choice_text, i):
    # print("-------------" + str(i) + "------------")
    question_and_choice = train_questions[i] + choice_text
    select_train_text = ""
    temp_train_text = text.split('。')
    question_and_choice = re.sub(r'[^\u4e00-\u9fa5]', '', question_and_choice)
    temp = " ".join(jieba.cut_for_search(question_and_choice))
    question_choice_keywords = temp.split(" ")
    question_choice_keywords2 = []
    for m in range(len(question_choice_keywords)):
        for n in range(len(temp_train_text)):
            if question_choice_keywords[m] in temp_train_text[n]:
                select_train_text = "".join(temp_train_text[n])
    if len(select_train_text) < 30:
        select_train_text = ""
        tags = jieba.analyse.extract_tags(question_and_choice, 20)  # 分析字串裡是否有停用詞
        temp = ''.join(tags)
        for word in temp:
            question_choice_keywords2.append(word)
        for m in range(len(question_choice_keywords2)):
            for n in range(len(temp_train_text)):
                if question_choice_keywords2[m] in temp_train_text[n]:
                    if temp_train_text[n] not in select_train_text:
                        select_train_text = "".join(temp_train_text[n])

    garbage_words=["個管師：","醫師：","民眾："]
    for g in range(3):
        select_train_text= select_train_text.replace(garbage_words[g],"")

    if len(select_train_text) > 150:
        select_train_text = select_train_text[0:150]
    return select_train_text


def read_data():
    train_file = 'data/Train_qa_ans.json'
    test_file = 'data\\Develop_QA.json'
    with open(train_file, 'r', encoding='utf-8') as obj:
        data_json = json.load(obj)
    train_data = toDataFrame(data_json, 0)
    with open(test_file, 'r', encoding='utf-8') as obj:
        data_json = json.load(obj)
    test_data = toDataFrame(data_json, 1)

    return train_data, test_data


def toDataFrame(data_json, flag):
    data = pd.DataFrame(data_json)
    if flag == 0:
        for i in range(len(data)):
            train_text.append(data_json[i]['text'])
            train_QA.append(data_json[i]['question'])
            train_questions.append(train_QA[i]["stem"])
            train_choices.append(train_QA[i]["choices"])
            answer.append(data_json[i]["answer"])
    elif flag == 1:
        for i in range(len(data)):
            test_text.append(data_json[i]['text'])
            test_QA.append(data_json[i]['question'])
            test_questions.append(test_QA[i]["stem"])
            test_choices.append(test_QA[i]["choices"])
    # print("第1篇第1個選項", choices[0][0]['text'])  # 第1篇第1個選項
    return data


def list_of_groups(init_list, childern_list_len):
    list_of_group = zip(*(iter(init_list),) * childern_list_len)
    end_list = [list(i) for i in list_of_group]
    count = len(init_list) % childern_list_len
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


class FakeNewsDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, mode, tokenizer):
        assert mode in ["trainA", "trainB", "trainC", "testA", "testB", "testC"]  # 一般訓練你會需要 dev set
        self.mode = mode
        # 大數據你會需要用 iterator=True
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.label_map = {'T': 0, 'F': 1}
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
        print("Fake function init")

    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        if self.mode == "testA" or self.mode == "testB" or self.mode == "testC":
            text_a, text_b = self.df.iloc[idx, :2].values
            label_tensor = None
        else:
            text_a, text_b, label = self.df.iloc[idx, :].values
            # 將 label 文字也轉換成索引方便轉換成 tensor
            label_id = self.label_map[label]
            label_tensor = torch.tensor(label_id)

        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)

        # 第二個句子的 BERT tokens
        tokens_b = self.tokenizer.tokenize(text_b)
        word_pieces += tokens_b + ["[SEP]"]
        len_b = len(word_pieces) - len_a

        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b,
                                       dtype=torch.long)

        return (tokens_tensor, segments_tensor, label_tensor)

    def __len__(self):
        return self.len


def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]

    # 測試集有 labels
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None

    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors,
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors,
                                    batch_first=True)

    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape,
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)

    return tokens_tensors, segments_tensors, masks_tensors, label_ids


def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0

    with torch.no_grad():
        # 遍巡整個資料集
        for data in dataloader:
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]

            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=segments_tensors,
                            attention_mask=masks_tensors)

            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            accuracy = F.softmax(logits.data, dim=0)
            # print(accuracy)
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()

            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))

    if compute_acc:
        acc = correct / total
        return predictions, acc

    return predictions, accuracy


def get_learnable_params(module):
    return [p for p in module.parameters() if p.requires_grad]


if __name__ == '__main__':
    corpus = []
    train_data, test_data = read_data()
    # print(train_data, test_data)
    vocab = tokenizer.vocab
    print("tokenizer 裡頭的字典大小：", len(vocab))

    for i in range(192):
        if ('不是' or '不會' or '非' or '錯誤' or '有誤') in test_questions[i]:
            contradict_question.append(1)
        else:
            contradict_question.append(0)
    for i in range(len(train_data)):
        if answer[i] == 'A':
            label_A.append('T')
            label_B.append('F')
            label_C.append('T')
        elif answer[i] == 'B':
            label_A.append('T')
            label_B.append('F')
            label_C.append('F')
        elif answer[i] == 'C':
            label_A.append('F')
            label_B.append('F')
            label_C.append('T')
    for i in range(len(train_choices)):
        train_choices_A.append(train_choices[i][0]['text'])
        train_choices_B.append(train_choices[i][1]['text'])
        train_choices_C.append(train_choices[i][2]['text'])
    for i in range(len(test_choices)):
        test_choices_A.append(test_choices[i][0]['text'])
        test_choices_B.append(test_choices[i][1]['text'])
        test_choices_C.append(test_choices[i][2]['text'])
    for t in range(3):
        temp_train_text = train_text
        new_train_text = []
        if t == 0:
            for i in range(len(train_choices)):
                temp = text_select(temp_train_text[i], train_choices_A[i], i)
                new_train_text.append(temp)
            df_train = pd.DataFrame({'Train_text': new_train_text, 'choice_A': train_choices_A, 'answer': label_A})

        elif t == 1:
            for i in range(len(train_choices)):
                temp = text_select(temp_train_text[i], train_choices_B[i], i)
                new_train_text.append(temp)
            df_train = pd.DataFrame({'Train_text': new_train_text, 'choice_B': train_choices_B, 'answer': label_B})
        else:
            for i in range(len(train_choices)):
                temp = text_select(temp_train_text[i], train_choices_C[i], i)
                new_train_text.append(temp)
            df_train = pd.DataFrame({'Train_text': new_train_text, 'choice_C': train_choices_C, 'answer': label_C})
        MAX_LENGTH = 120
        print("訓練樣本數：", len(df_train))
        if t == 0:
            df_train.to_csv("trainA.tsv", sep='\t', index=False)
        elif t == 1:
            df_train.to_csv("trainB.tsv", sep='\t', index=False)
        else:
            df_train.to_csv("trainC.tsv", sep='\t', index=False)
        # df_train.to_csv("train.tsv", sep='\t', index=False)

        temp_test_text = test_text
        new_test_text = []
        if t == 0:
            for i in range(len(test_choices)):
                temp_test_text = text_select(test_text[i], test_choices_A[i], i)
                new_test_text.append(temp_test_text)
            df_test = pd.DataFrame({'Test_text': new_test_text, 'choice_A': test_choices_A})
        elif t == 1:
            for i in range(len(test_choices)):
                temp_test_text = text_select(test_text[i], test_choices_B[i], i)
                new_test_text.append(temp_test_text)
            df_test = pd.DataFrame({'Test_text': new_test_text, 'choice_B': test_choices_B})
        elif t == 2:
            for i in range(len(test_choices)):
                temp_test_text = text_select(test_text[i], test_choices_C[i], i)
                new_test_text.append(temp_test_text)
            df_test = pd.DataFrame({'Test_text': new_test_text, 'choice_C': test_choices_C})
        if t == 0:
            df_test.to_csv("testA.tsv", sep="\t", index=False)
        elif t == 1:
            df_test.to_csv("testB.tsv", sep="\t", index=False)
        else:
            df_test.to_csv("testC.tsv", sep="\t", index=False)
        # df_test.to_csv("test.tsv", sep="\t", index=False)
        # print("預測樣本數:", len(df_test))
        # print(df_test.head())
        # ratio = len(df_test) / len(df_train)
        # print("測試集樣本數 / 訓練集樣本數 = {:.1f} 倍".format(ratio))
        if t == 0:
            trainset = FakeNewsDataset("trainA", tokenizer=tokenizer)
        elif t == 1:
            trainset = FakeNewsDataset("trainB", tokenizer=tokenizer)
        else:
            trainset = FakeNewsDataset("trainC", tokenizer=tokenizer)
        # trainset = FakeNewsDataset("trainA", tokenizer=tokenizer)
        # print(trainset)

        # 查看function轉換過程，型態
        # print("[原始文本]")
        # print(trainset.df.iloc[0].values)
        # print(trainset.df.iloc[121].values)

        BATCH_SIZE = 32
        trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
        data = next(iter(trainloader))
        tokens_tensors, segments_tensors, masks_tensors, label_ids = data

        from transformers import BertForSequenceClassification

        PRETRAINED_MODEL_NAME = "bert-base-chinese"
        NUM_LABELS = 2

        model = BertForSequenceClassification.from_pretrained(
            PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
        #
        clear_output()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device:", device)
        model = model.to(device)
        # if t == 0:
        #     model = torch.load('TF_qa_A.pt')
        #     print("load ok")
        # elif t == 1:
        #     model = torch.load('TF_qa_B.pt')
        # else:
        #     model = torch.load('TF_qa_C.pt')
        # _, acc = get_predictions(model, trainloader, compute_acc=True)
        # print("classification acc:", acc)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        EPOCHS = 3  # 幸運數字
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for data in trainloader:
                tokens_tensors, segments_tensors, \
                masks_tensors, labels = [t.to(device) for t in data]
                # 將參數梯度歸零
                optimizer.zero_grad()
                # forward pass
                outputs = model(input_ids=tokens_tensors,
                                token_type_ids=segments_tensors,
                                attention_mask=masks_tensors,
                                labels=labels)
                loss = outputs[0]
                # backward
                loss.backward()
                optimizer.step()

                # 紀錄當前 batch loss
                running_loss += loss.item()
        if t == 0:
            testset = FakeNewsDataset("testA", tokenizer=tokenizer)
        elif t == 1:
            testset = FakeNewsDataset("testB", tokenizer=tokenizer)
        else:
            testset = FakeNewsDataset("testC", tokenizer=tokenizer)
        testloader = DataLoader(testset, batch_size=256,
                                collate_fn=create_mini_batch)

        # 用分類模型預測測試集
        predictions, accuracy = get_predictions(model, testloader)
        if t == 0:
            accuracy_A = accuracy.tolist()
            print("accuracy_A", accuracy_A)
        elif t == 1:
            accuracy_B = accuracy.tolist()
            print("accuracy_B", accuracy_B)
        elif t == 2:
            accuracy_C = accuracy.tolist()
            print("accuracy_C", accuracy_C)

        # 用來將預測的 label id 轉回 label 文字
        index_map = {v: k for k, v in testset.label_map.items()}
        if t == 0:
            torch.save(model, "TF_qa_A.pt")
        elif t == 1:
            torch.save(model, "TF_qa_B.pt")
        elif t == 2:
            torch.save(model, "TF_qa_C.pt")

    predictions = []
    print(len(accuracy_A), accuracy_A)
    print(len(accuracy_B), accuracy_B)
    print(len(accuracy_C), accuracy_C)
    print(contradict_question)
    for i in range(192):
        if contradict_question[i] == 1:
            if accuracy_A[i][1] >= accuracy_B[i][1] and accuracy_A[i][1] >= accuracy_C[i][1]:
                predictions.append('A')
            elif accuracy_B[i][1] >= accuracy_A[i][1] and accuracy_B[i][1] >= accuracy_C[i][1]:
                predictions.append('B')
            else:
                predictions.append('C')
        else:
            if accuracy_A[i][0] >= accuracy_B[i][0] and accuracy_A[i][0] >= accuracy_C[i][0]:
                predictions.append('A')
            elif accuracy_B[i][0] >= accuracy_A[i][0] and accuracy_B[i][0] >= accuracy_C[i][0]:
                predictions.append('B')
            else:
                predictions.append('C')
    print(predictions)
    print(len(predictions))
    import csv

    with open('TF_qa2.csv', 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)

        # 寫入一列資料
        writer.writerow(['id', 'answer'])

        for k in range(len(predictions)):
            writer.writerow([str(k + 1), predictions[k]], )
