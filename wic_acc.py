import pandas as pd
import json
from pandas import json_normalize

with open('data/WiC_data/WiC_test_label_v2.json') as f:
    wic_label_data = json.load(f)

wic_label_df = json_normalize(wic_label_data['wic']) #Results contain the required data

wic_label_df = wic_label_df['label']


with open('submission/submission/ensemble-v4_wic.json') as f:          ###### 내가 원하는 file
    wic_target_data = json.load(f)

wic_target_df = json_normalize(wic_target_data['wic']) #Results contain the required data

wic_target_df = wic_target_df['label']

all_wic_df = pd.concat([wic_label_df, wic_target_df], axis=1)      ####
all_wic_df.columns = ['label', 'target']

count = 0

for idx in range(len(all_wic_df)):

    if all_wic_df['label'][idx] == all_wic_df['target'][idx]:
        count += 1

# print(count)

print("WiC_acc : " , count/len(all_wic_df))