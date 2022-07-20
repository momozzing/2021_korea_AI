import pandas as pd
import json
from pandas import json_normalize

with open('submission/submission/monologg-koelectra-base-v3-discriminator-BoolQ-8-5760-1234-v30_post_eval.json') as f:
    v1 = json.load(f)

with open('submission/submission/monologg-koelectra-base-v3-discriminator-BoolQ-2-5760-1234-v35_post_eval.json') as f:
    v2 = json.load(f)

with open('submission/tunib-electra-ko-base-BoolQ-4-2560-1234-v33_post_eval.json') as f:
    v3 = json.load(f)

with open('submission/tunib-electra-ko-base-BoolQ-15-2560-1234-v15_post_eval.json') as f:
    v4 = json.load(f)

with open('submission/tunib-electra-ko-base-BoolQ-3-3840-1234-v16_post_eval.json') as f:
    v5 = json.load(f)

with open('submission/submission/ensemble-v4_boolq_test9.json') as f:
    ens = json.load(f)

with open('data/BoolQ_data/BoolQ_test_label.json') as f:
    test = json.load(f)

data = pd.read_csv('data/BoolQ_data/BoolQ_test_labeling.tsv', delimiter='\t')

# ##################################################
# v1_df = json_normalize(v1['cola']) #Results contain the required data
# v2_df = json_normalize(v2['cola']) #Results contain the required data
# v3_df = json_normalize(v3['cola']) #Results contain the required data
# v4_df = json_normalize(v4['cola']) #Results contain the required data
# v5_df = json_normalize(v5['cola']) #Results contain the required data
# ens_df = json_normalize(ens['cola']) #Results contain the required data
# test_df = json_normalize(test['cola']) #Results contain the required data


# v1_df = v1_df['label']
# v2_df = v2_df['label']
# v3_df = v3_df['label']
# v4_df = v4_df['label']
# v5_df = v5_df['label']
# ens_df = ens_df['label']
# test_df = test_df['label']
# sen_df = data['sentence']

# all_df = pd.concat([v1_df, v2_df, v3_df, v4_df, v5_df, ens_df, test_df, sen_df], axis=1)      ####
# all_df.columns = ['v1', 'v2', 'v3', 'v4', 'v5', 'ens', 'test', 'sen']                  


# all_df = all_df[(all_df['ens'] != all_df['test'])]

# all_df.to_csv('data/error_check_data/CoLA_error_check.tsv', index=False, sep="\t")

# ##########################################################################################
# v1_df = json_normalize(v1['wic']) #Results contain the required data
# v2_df = json_normalize(v2['wic']) #Results contain the required data
# v3_df = json_normalize(v3['wic']) #Results contain the required data
# v4_df = json_normalize(v4['wic']) #Results contain the required data
# v5_df = json_normalize(v5['wic']) #Results contain the required data
# ens_df = json_normalize(ens['wic']) #Results contain the required data
# test_df = json_normalize(test['wic']) #Results contain the required data


# v1_df = v1_df['label']
# v2_df = v2_df['label']
# v3_df = v3_df['label']
# v4_df = v4_df['label']
# v5_df = v5_df['label']
# ens_df = ens_df['label']
# test_df = test_df['label']
# sen_df = data[['Target','SENTENCE1','SENTENCE2','ANSWER','start_s1','end_s1','start_s2','end_s2']]

# # print(sen_df)

# all_df = pd.concat([v1_df, v2_df, v3_df, v4_df, v5_df, ens_df, test_df, sen_df], axis=1)      ####
# # all_df = pd.concat([v1_df, v2_df, v3_df, v4_df, v5_df, ens_df, test_df], axis=1)      ####
# all_df.columns = ['v1', 'v2', 'v3', 'v4', 'v5', 'ens', 'test', 'Target','SENTENCE1','SENTENCE2','ANSWER','start_s1','end_s1','start_s2','end_s2']                  
# # all_df.columns = ['v1', 'v2', 'v3', 'v4', 'v5', 'ens', 'test']                  


# all_df = all_df[(all_df['ens'] != all_df['test'])]

# all_df.index=all_df.index+1
# print(all_df)
# all_df.to_csv('data/error_check_data/WiC_error_check.tsv', index=False, sep="\t")


################################################################################################################

# v1_df = json_normalize(v1['copa']) #Results contain the required data
# v2_df = json_normalize(v2['copa']) #Results contain the required data
# v3_df = json_normalize(v3['copa']) #Results contain the required data
# v4_df = json_normalize(v4['copa']) #Results contain the required data
# v5_df = json_normalize(v5['copa']) #Results contain the required data
# ens_df = json_normalize(ens['copa']) #Results contain the required data
# test_df = json_normalize(test['copa']) #Results contain the required data


# v1_df = v1_df['label']
# v2_df = v2_df['label']
# v3_df = v3_df['label']
# v4_df = v4_df['label']
# v5_df = v5_df['label']
# ens_df = ens_df['label']
# test_df = test_df['label']
# sen_df = data[['sentence','question','1','2']]

# # print(sen_df)

# all_df = pd.concat([v1_df, v2_df, v3_df, v4_df, v5_df, ens_df, test_df, sen_df], axis=1)      ####
# # all_df = pd.concat([v1_df, v2_df, v3_df, v4_df, v5_df, ens_df, test_df], axis=1)      ####
# all_df.columns = ['v1', 'v2', 'v3', 'v4', 'v5', 'ens', 'test', 'sentence','question','1','2']                  
# # all_df.columns = ['v1', 'v2', 'v3', 'v4', 'v5', 'ens', 'test']                  


# all_df = all_df[(all_df['ens'] != all_df['test'])]

# all_df.index=all_df.index+1
# print(all_df)
# all_df.to_csv('data/error_check_data/COPA_error_check.tsv', index=False, sep="\t")

####################################################################################


v1_df = json_normalize(v1['boolq']) #Results contain the required data
v2_df = json_normalize(v2['boolq']) #Results contain the required data
v3_df = json_normalize(v3['boolq']) #Results contain the required data
v4_df = json_normalize(v4['boolq']) #Results contain the required data
v5_df = json_normalize(v5['boolq']) #Results contain the required data
ens_df = json_normalize(ens['boolq']) #Results contain the required data
test_df = json_normalize(test['boolq']) #Results contain the required data


v1_df = v1_df['label']
v2_df = v2_df['label']
v3_df = v3_df['label']
v4_df = v4_df['label']
v5_df = v5_df['label']
ens_df = ens_df['label']
test_df = test_df['label']
sen_df = data[['ID','Text','Question']]

# print(sen_df)

all_df = pd.concat([v1_df, v2_df, v3_df, v4_df, v5_df, ens_df, test_df, sen_df], axis=1)      ####
# all_df = pd.concat([v1_df, v2_df, v3_df, v4_df, v5_df, ens_df, test_df], axis=1)      ####
all_df.columns = ['v1', 'v2', 'v3', 'v4', 'v5', 'ens', 'test', 'Text','Question','ID']                  
# all_df.columns = ['v1', 'v2', 'v3', 'v4', 'v5', 'ens', 'test']                  


all_df = all_df[(all_df['ens'] != all_df['test'])]

all_df.index=all_df.index+1
print(all_df)
all_df.to_csv('data/error_check_data/boolq_error_check.tsv', index=False, sep="\t")