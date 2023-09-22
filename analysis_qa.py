import csv
import pandas as pd

df = pd.read_csv("output\\qa2.csv")
print(df)
print(len(df))
print(df.answer.value_counts(),'\n',df.answer2.value_counts(),'\n',df.answer3.value_counts())
new_ans = []
for i in range(192):
    A_probabilyty = 0
    B_probabilyty = 0
    C_probabilyty = 0
    if df['answer'][i] == 'A':
        A_probabilyty += 0.35
    elif df['answer'][i] == 'B':
        B_probabilyty += 0.33
    elif df['answer'][i] == 'C':
        C_probabilyty += 0.33

    if df['answer2'][i] == 'A':
        A_probabilyty += 0.33
    elif df['answer2'][i] == 'B':
        B_probabilyty += 0.35
    elif df['answer2'][i] == 'C':
        C_probabilyty += 0.33

    if df['answer3'][i] == 'A':
        A_probabilyty += 0.33
    elif df['answer3'][i] == 'B':
        B_probabilyty += 0.33
    elif df['answer3'][i] == 'C':
        C_probabilyty += 0.35
    if A_probabilyty > B_probabilyty and A_probabilyty > C_probabilyty:
        new_ans.append('A')
    elif B_probabilyty > A_probabilyty and B_probabilyty > C_probabilyty:
        new_ans.append('B')
    elif C_probabilyty > A_probabilyty and C_probabilyty > B_probabilyty:
        new_ans.append('C')
    else:
        new_ans.append('no')
        print(i)

print(new_ans)
print(len(new_ans))
d = pd.DataFrame(new_ans)
print(d.value_counts())
with open('output\\qa.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    # 寫入一列資料
    writer.writerow(['id', 'answer'])

    for k in range(192):
        writer.writerow([str(k + 1),new_ans[k]], )