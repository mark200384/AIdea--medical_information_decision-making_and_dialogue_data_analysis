import json
import jieba.analyse
import pandas as pd
import numpy as np
import re
import jieba
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from transformers import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from IPython.display import clear_output
import torch.nn.functional as F

# jieba.set_dictionary('dict.txt')
PRETRAINED_MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
print("PyTorch 版本：", torch.__version__)
stopwords = [line.strip() for line in open('stopword.txt', encoding='UTF-8').readlines()]
train_text = []

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
train_contradict_question = []
answer = []
label_A = []
label_B = []
label_C = []
accuracy_A = []
accuracy_B = []
accuracy_C = []
train_file = 'data/Train_qa_ans.json'
# test_file = '/content/drive/MyDrive/data/Develop_QA.json'
test_file = 'data/Test_QA.json'


def load_json(file_path):
    data = []
    with open(file_path, newline='',encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
    return data


def read_data():
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


"""Q2B"""


def strQ2B(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全形空格直接轉換
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全形字元（除空格）根據關係轉化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def get_data_list(data, mode="train"):
    text_list = []
    question_list = []
    choices_list = []
    answer_list = []
    answer_map = {
        "A": 0,
        "B": 1,
        "C": 2,
    }
    for i in range(0, len(data)):
        text_list.append(data[i]['text'])
        question_list.append(data[i]['question']['stem'])
        choices_list.append([data[i]['question']['choices'][y]['text'] for y in range(3)])
        if mode == "train":
            answer_list.append(answer_map[strQ2B(data[i]['answer'])[0]])
        else:
            answer_list.append(0)

    return text_list, question_list, choices_list, answer_list


def get_similarity(sen1, sen2):
    sim = 0
    if len(sen2) == 0:
        return 0

    for i in range(0, len(sen2)):
        if sen2[i] in sen1:
            sim += 1

    return sim / len(sen2)


def get_similarity_list(text, choices, question):
    similarity_list = []
    """
    key_words_1 = jieba.lcut("".join(re.split("會|有|的|是|，|都|再",choices[0])))
    key_words_2 = jieba.lcut("".join(re.split("會|有|的|是|，|都|再",choices[1])))
    key_words_3 = jieba.lcut("".join(re.split("會|有|的|是|，|都|再",choices[2])))
    key_words_q = jieba.lcut("".join(re.split("會|有|的|是|民眾|何者|錯誤|敘述|，|？|下列|不|什麼",question)))
    """
    key_words = jieba.lcut("".join(re.split("會|有|的|是|，|都|再", choices)))
    key_words_q = jieba.lcut("".join(re.split("會|有|的|是|民眾|何者|錯誤|敘述|，|？|下列|不|什麼", question)))

    for i in range(0, len(text)):
        similarity_list.append(get_similarity(text[i], key_words) + get_similarity(text[i], key_words_q))

    return similarity_list


def count_current_number(text, need_pos_list):
    num = 0
    for item in need_pos_list:
        num += len(text[int(item)])
    return num


def preprocess_data(text_list, question_list, choices_list, max_length=140):
    new_text_list = []
    delete_char = "恩|嗯|喔|哦|哼|…|啊|，|。|哈|阿|诶|蛤|⋯⋯|嘿|？"
    other_nonneed_char = ["好", "好的", "了解"]

    for i in range(0, len(text_list)):
        text = text_list[i]
        choices = choices_list[i]
        question = question_list[i]

        per_dialog_similarity = []
        per_sentence_similarity = []

        split_t = re.split(r'([\u4E00-\u9FFFa-zA-Z0-9]+[：])', text)

        buffer_text = []
        for j in range(0, len(split_t)):
            if ("護理師：" in split_t[j]) or ("個管師：" in split_t[j]) or ("醫師：" in split_t[j]):
                current_speak = 1
                continue

            elif ("民眾：" in split_t[j]) or ("家屬：" in split_t[j]):
                current_speak = 0
                continue

            else:
                buffer_text.append(split_t[j])
        text = buffer_text

        text = "".join(text)
        text = re.split(delete_char, text)
        while ("" in text): text.remove("")
        ###########################
        # 把"對"字與他的前一句加上"?"後合併,若前一句>=2個字
        buffer_text = []
        for j in range(0, len(text)):
            if j != 0:
                if text[j] == "對" and len(text[j - 1]) >= 2:
                    previous = buffer_text.pop(-1)
                    buffer_text.append(previous + "?對")
                else:
                    buffer_text.append(text[j])
            else:
                buffer_text.append(text[j])
        text = buffer_text

        # 刪除結巴(後)
        previous_char = ""
        buffer_text = []
        for item in text:
            if item in previous_char or item in other_nonneed_char:
                continue
            else:
                buffer_text.append(item)
                previous_char = item
        text = buffer_text
        # 刪除結巴(前)
        buffer_text = []
        for j in range(0, len(text)):
            if (j + 1) != len(text):
                if text[j] in text[j + 1] or text[j] in other_nonneed_char:
                    continue
                else:
                    buffer_text.append(text[j])
        text = buffer_text

        # 刪除<=2個字的句子
        buffer_text = []
        for item in text:
            if len(item) > 2:
                buffer_text.append(item)
        text = buffer_text

        # 平均分攤各個choices,question與text的similarity
        # choices_text_sim_list的每個list,其長度會與text長度相同
        choices_text_sim_list = [get_similarity_list(text, choices[0], question),
                                 get_similarity_list(text, choices[1], question),
                                 get_similarity_list(text, choices[2], question)]
        """
        #測試是否有文章找不到答案,此題無解
        all_sum = sum(choices1_text_sim_list)+sum(choices2_text_sim_list)+sum(choices3_text_sim_list)
        if all_sum==0:
            print(i+1,"找不到關聯度句子")
        """
        count = 0
        need_pos_list = []
        while True:
            count %= 3
            if (sum(choices_text_sim_list[0]) + sum(choices_text_sim_list[1]) + sum(choices_text_sim_list[2])) == 0 or (
                    count_current_number(text, need_pos_list) > max_length):
                break
            current_ctsl = choices_text_sim_list[count]
            bigger_sim = max(current_ctsl)
            if bigger_sim != 0:
                # 每輪只抓27個字,若current+下一個已經超過,則不取前一句
                per_round_char_count = 0
                bigger_sim_pos = current_ctsl.index(bigger_sim)
                if bigger_sim_pos not in need_pos_list:
                    need_pos_list.append(bigger_sim_pos)
                    per_round_char_count += len(text[int(bigger_sim_pos)])
                if (bigger_sim_pos + 1) not in need_pos_list and (bigger_sim_pos + 1) != len(current_ctsl):
                    need_pos_list.append(bigger_sim_pos + 1)
                    per_round_char_count += len(text[int(bigger_sim_pos) + 1])
                if (bigger_sim_pos - 1) not in need_pos_list and (bigger_sim_pos - 1) != -1:
                    if (per_round_char_count + len(text[int(bigger_sim_pos) - 1]) <= (max_length / 3)):
                        need_pos_list.append(bigger_sim_pos - 1)
                choices_text_sim_list[count][bigger_sim_pos] = 0
            count += 1

        # 取出需要的位置
        need_pos_list.sort()
        buffer_text = []
        for num in need_pos_list:
            buffer_text.append(text[int(num)])
        text = buffer_text

        if len(",".join(text)) > max_length:
            while True:
                choices_text_sim_list = np.array(get_similarity_list(text, choices[0], question)) + np.array(
                    get_similarity_list(text, choices[1], question)) + np.array(
                    get_similarity_list(text, choices[2], question))
                choices_text_sim_list = choices_text_sim_list.tolist()
                cut = False

                if len(",".join(text)) <= max_length:
                    break
                else:
                    head_sim = choices_text_sim_list[0]
                    tail_sim = choices_text_sim_list[len(text) - 1]
                    min_sim_pos = choices_text_sim_list.index(min(choices_text_sim_list))
                    if head_sim == 0:
                        text.pop(0)
                        cut = True

                    if tail_sim == 0 and cut == False:
                        text.pop(-1)
                        cut = True

                    if cut == False:
                        text.pop(min_sim_pos)

        text = ",".join(text)
        """
        #print(len(text))
        print(text)
        print(question)
        print(choices)
        print("")
        """
        new_text_list.append(text)

    return new_text_list


class FakeNewsDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "test"]  # 一般訓練你會需要 dev set
        self.mode = mode
        # 大數據你會需要用 iterator=True
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.label_map = {'T': 0, 'F': 1}
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
        print("Fake function init")

    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        if self.mode == "test":
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
            # print(logits.data)
            accuracy = F.softmax(logits.data, dim=1)
            # print(accuracy)
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()

            # 將當前 batch 記錄下來
            if predictions is None:
                test_predictions = accuracy
                predictions = pred
            else:
                test_predictions = torch.cat((test_predictions, accuracy))
                predictions = torch.cat((predictions, pred))

    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions, test_predictions


def get_learnable_params(module):
    return [p for p in module.parameters() if p.requires_grad]


if __name__ == '__main__':
    corpus = []
    train_data, test_data = read_data()
    # print(train_data, test_data)
    vocab = tokenizer.vocab
    print("tokenizer 裡頭的字典大小：", len(vocab))

    for i in range(len(train_data)):
        if ('不是' or '不會' or '非' or '錯誤' or '有誤' or '誤' or '不') in train_questions[i]:
            train_contradict_question.append(1)
        else:
            train_contradict_question.append(0)
    for i in range(len(test_data)):
        if ('不是' or '不會' or '非' or '錯誤' or '有誤' or '誤' or '不') in test_questions[i]:
            contradict_question.append(1)
        else:
            contradict_question.append(0)

    print("answer:", answer)
    print(len(train_contradict_question))
    for i in range(len(train_data)):
        if (train_contradict_question[i] == 1):
            if answer[i] == 'A':
                label_A.append('F')
                label_B.append('T')
                label_C.append('T')
            elif answer[i] == 'B':
                label_A.append('T')
                label_B.append('F')
                label_C.append('T')
            elif answer[i] == 'C':
                label_A.append('T')
                label_B.append('T')
                label_C.append('F')
        else:
            if answer[i] == 'A':
                label_A.append('T')
                label_B.append('F')
                label_C.append('F')
            elif answer[i] == 'B':
                label_A.append('F')
                label_B.append('T')
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

    train_data = load_json(train_file)
    print("len(train_data)", len(train_data))
    text_list, question_list, choices_list, answer_list = get_data_list(train_data, "train")
    text_list = preprocess_data(text_list, question_list, choices_list)
    # for i in range(len(train_data)):
    #   train_choices_A[i] = train_questions[i] +train_choices_A[i]
    #   train_choices_B[i] = train_questions[i] +traina_choices_B[i]
    #   train_choices_C[i] = train_questions[i] +train_choices_C[i]
    df_train1 = pd.DataFrame({'Train_text': text_list, 'choice': train_choices_A, 'answer': label_A})
    df_train2 = pd.DataFrame({'Train_text': text_list, 'choice': train_choices_B, 'answer': label_B})
    df_train3 = pd.DataFrame({'Train_text': text_list, 'choice': train_choices_C, 'answer': label_C})
    frames = [df_train1, df_train2, df_train3]
    df_train = pd.concat([df_train1, df_train2, df_train3], axis=0, ignore_index=True)
    print("df_train\n", df_train)
    print("訓練樣本數：", len(df_train))
    from sklearn.utils import shuffle

    shuffle_df = shuffle(df_train)
    print("shuffle train",df_train)
    df_train = shuffle_df[:int(2085*0.6)]
    print("dftrain", df_train)
    df_test = shuffle_df[int(2085*0.6):]
    print("dftest",df_test)

    df_train.to_csv("train.tsv", sep='\t', index=False)
    df_test.to_csv("test.tsv", sep='\t', index=False)
    print("測試樣本總數:", len(df_test))
    print("df_test\n", df_test)
    BATCH_SIZE = 16
    trainset = FakeNewsDataset("train", tokenizer=tokenizer)
    print("trainset",trainset)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
    data = next(iter(trainloader))
    tokens_tensors, segments_tensors, masks_tensors, label_ids = data

    from transformers import BertForSequenceClassification

    PRETRAINED_MODEL_NAME = "bert-base-chinese"
    NUM_LABELS = 2

    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
    #
    # clear_output()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    PATH = 'qa_text=140_epochs=7.pt'
    model = model.to(device)
    model = torch.load(PATH)
    model.eval()


    testset = FakeNewsDataset("test", tokenizer=tokenizer)
    testloader = DataLoader(testset, batch_size=256,
                            collate_fn=create_mini_batch)
    # 用分類模型預測測試集
    predictions, test_predict = get_predictions(model, testloader)
    accuracy = test_predict.tolist()
    print("len(test_predict)", len(test_predict))
    print("accuracy", accuracy)
    for i in range(3):
        for j in range(len(test_data)):
            if i == 0:
                accuracy_A.append(accuracy[j])
            elif i == 1:
                accuracy_B.append(accuracy[i * len(test_data) + j])
            elif i == 2:
                accuracy_C.append(accuracy[i * len(test_data) + j])
    print("accuracy_A", accuracy_A)
    print("accuracy_B", accuracy_B)
    print("accuracy_C", accuracy_C)
    predictions = []
    for i in range(len(test_data)):
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
    print("predictions", predictions)
    print("len(predictions)", len(predictions))

    


#