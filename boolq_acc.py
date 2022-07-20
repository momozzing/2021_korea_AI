import pandas as pd
import json
from pandas import json_normalize

with open('data/BoolQ_data/BoolQ_test_label.json') as f:
    boolq_label_data = json.load(f)

boolq_label_df = json_normalize(boolq_label_data['boolq']) #Results contain the required data

boolq_label_df = boolq_label_df['label']


with open('submission/submission/ensemble-v4_boolq_test8.json') as f:          ###### 내가 원하는 file
    boolq_target_data = json.load(f)

boolq_target_df = json_normalize(boolq_target_data['boolq']) #Results contain the required data

boolq_target_df = boolq_target_df['label']

all_boolq_df = pd.concat([boolq_label_df, boolq_target_df], axis=1)      ####
all_boolq_df.columns = ['label', 'target']

count = 0

for idx in range(len(all_boolq_df)):

    if all_boolq_df['label'][idx] == all_boolq_df['target'][idx]:
        count += 1

# print(count)

print("BoolQ_acc : " , count/len(all_boolq_df))