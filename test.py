import pandas as pd
import json


##############################################################################################
# boolq_data = pd.read_csv('data/BoolQ_data/BoolQ_test_labeling.tsv', delimiter="\t")
# print(boolq_data.isnull().sum())

# boolq_test_label = []

# for idx in range(len(boolq_data)):

#     # label = boolq_data['Answer(FALSE = 0, TRUE = 1)']

#     boolq_test_label.append({"idx": idx+1,"label" : int(boolq_data['Answer(FALSE = 0, TRUE = 1)'][idx])})

# boolq_dic = {"boolq" : boolq_test_label}

# with open("data/BoolQ_data/BoolQ_test_label.json", 'w') as f:
#     json.dump(boolq_dic, f, indent= 4)

###########################################################################################
# cola_data = pd.read_csv('data/CoLA_data/NIKL_CoLA_test_labeling.tsv', delimiter="\t")
# print(cola_data.isnull().sum())

# cola_test_label = []

# for idx in range(len(cola_data)):

#     cola_test_label.append({"idx": idx,"label" : int(cola_data['acceptability_label'][idx])})

# cola_dic = {"cola" : cola_test_label}

# with open("data/CoLA_data/cola_test_label.json", 'w') as f:
#     json.dump(cola_dic, f, indent= 4)

###########################################################################################
# copa_data = pd.read_csv('submission/submission/COPA_biBERT_rs6_epoch20.18by1_bs20.pt_koelectraExTrain_epc0by1_bs20.json', delimiter="\t")
# print(copa_data.isnull().sum())

# copa_test_label = []

# for idx in range(len(copa_data)):

#     copa_test_label.append({"idx": idx+1,"label" : int(copa_data['Answer'][idx])})

# copa_dic = {"copa" : copa_test_label}

# with open("submission/submission/COPA_biBERT_rs6_epoch20.18by1_bs20.pt_koelectraExTrain_epc0by1_bs20_1.json", 'w') as f:
#     json.dump(copa_dic, f, indent= 4)

###########################################################################################
# wic_data = pd.read_csv('data/WiC_data/NIKL_SKT_WiC_Test_labeling_v2.tsv', delimiter="\t")
# print(wic_data.isnull().sum())
# wic_data["ANSWER"] = wic_data['ANSWER'].map(lambda x : 1 if x else 0)

# wic_test_label = []
# true = True
# false = False

# for idx in range(len(wic_data)):

#     wic_test_label.append({"idx": idx+1,"label" : wic_data['ANSWER'][idx]})

# wic_dic = {"wic" : wic_test_label}

# for i in range(len(wic_data)):
#     if wic_dic["wic"][i]["label"] == 1:
#         wic_dic["wic"][i]["label"] = true

#     elif wic_dic["wic"][i]["label"] == 0:
#         wic_dic["wic"][i]["label"] = false

# with open("data/WiC_data/WiC_test_label_v2.json", 'w') as f:
#     json.dump(wic_dic, f, indent= 4)

###########################################################################################

import pandas as pd
import json
import jsonlines
from pandas import json_normalize

# data = pd.read_csv("data/BoolQ_data/boolq_train_eng.jsonl", delimiter= '\t')
# data= []
# with open('data/BoolQ_data/boolq_train_eng.jsonl', 'r') as json_file:
#     for line in json_file:
#         data.append(line) #라인을 기준으로 일단 자름

# print(data)


text_col = []
question_col = []
answer_col = []
with jsonlines.open('data/BoolQ_data/boolq_dev_eng.jsonl') as f:

    for line in f.iter():

        text_col.append(line['passage']) # or whatever else you'd like to do
        question_col.append(line['question'])
        answer_col.append(line['answer'])

text_df = pd.DataFrame(text_col)
q_df = pd.DataFrame(question_col)
a_df = pd.DataFrame(answer_col)

all_df = pd.concat([text_df, q_df, a_df], axis=1)
all_df.columns = ['Text', 'Question', 'Answer(FALSE = 0, TRUE = 1)']

all_df['Answer(FALSE = 0, TRUE = 1)'].replace(True, 1, inplace=True)
all_df['Answer(FALSE = 0, TRUE = 1)'].replace(False, 0, inplace=True)

print(all_df)

all_df.to_csv('data/BoolQ_data/boolq_dev_eng.tsv', index=False, sep="\t")



# Text	Question	Answer(FALSE = 0, TRUE = 1)
# wic_label_df = json_normalize(wic_label_data['wic']) #Results contain the required data

# wic_label_df = wic_label_df['label']


# with open('submission/tunib-electra-ko-base-WiC-25-30-64-1234-v9.json') as f:          ###### 내가 원하는 file
#     wic_target_data = json.load(f)

# wic_target_df = json_normalize(wic_target_data['wic']) #Results contain the required data

# wic_target_df = wic_target_df['label']

# all_wic_df = pd.concat([wic_label_df, wic_target_df], axis=1)      ####
# all_wic_df.columns = ['label', 'target']

# count = 0

# for idx in range(len(all_wic_df)):

#     if all_wic_df['label'][idx] == all_wic_df['target'][idx]:
#         count += 1

# # print(count)

# print("WiC_acc : " , count/len(all_wic_df))