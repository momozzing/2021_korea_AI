import pandas as pd
import json
from pandas import json_normalize
from sklearn.metrics import matthews_corrcoef


with open('data/CoLA_data/cola_test_label.json') as f:
    cola_label_data = json.load(f)

cola_label_df = json_normalize(cola_label_data['cola']) #Results contain the required data

cola_label_df = cola_label_df['label']


with open('submission/submission/ensemble-last_cola1.json') as f:          ###### 내가 원하는 file
    cola_target_data = json.load(f)

cola_target_df = json_normalize(cola_target_data['cola']) #Results contain the required data

cola_target_df = cola_target_df['label']

all_cola_df = pd.concat([cola_label_df, cola_target_df], axis=1)      ####
all_cola_df.columns = ['label', 'target']

count = 0
label_list = []
target_list = []

for idx in range(len(all_cola_df)):
    
    label_list.append(all_cola_df['label'][idx])
    target_list.append(all_cola_df['target'][idx])


    if all_cola_df['label'][idx] == all_cola_df['target'][idx]:
        count += 1

# print(count)

print("CoLA_acc : " , count/len(all_cola_df))


def MCC(preds, labels):
    assert len(preds) == len(labels)
    return matthews_corrcoef(labels, preds)

print("CoLA_mcc : ", MCC(target_list, label_list))