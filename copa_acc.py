import pandas as pd
import json
from pandas import json_normalize

with open('data/COPA_data/copa_test_label.json') as f:
    copa_label_data = json.load(f)

copa_label_df = json_normalize(copa_label_data['copa']) #Results contain the required data

copa_label_df = copa_label_df['label']


with open('submission/submission/ensemble-copa-last7.json') as f:          ###### 내가 원하는 file
    copa_target_data = json.load(f)

copa_target_df = json_normalize(copa_target_data['copa']) #Results contain the required data

copa_target_df = copa_target_df['label']

all_copa_df = pd.concat([copa_label_df, copa_target_df], axis=1)      ####
all_copa_df.columns = ['label', 'target']

count = 0

for idx in range(len(all_copa_df)):

    if all_copa_df['label'][idx] == all_copa_df['target'][idx]:
        count += 1

# print(count)

print("COPA_acc : " , count/len(all_copa_df))