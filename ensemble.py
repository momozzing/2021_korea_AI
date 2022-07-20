import pandas as pd
import json
from pandas import json_normalize

with open('submission/submission/copa_last/COPA_biBERT_rs6_epoch20.18by1_bs20.pt_koelectraExTrain_epc0by1_bs20.json') as f:
    v1 = json.load(f)

with open('submission/submission/COPA_koelectra_rs12_epoch14step4000_bs20.pt_koelectraExTrain_epc0by1_bs20dddd.json') as f:
    v2 = json.load(f)

with open('submission/submission/copa_last/COPA_koelectra_rs13_epoch16step3000_bs20.pt_koelectraExTrain_epc0by1_bs20.json') as f:
    v3 = json.load(f)

with open('submission/submission/copa_last/COPA_koelectra_rs103_epoch14step3500_bs20.pt_koelectraExTrain_epc0by1_bs20.json') as f:
    v4 = json.load(f)

with open('submission/submission/copa_last/COPA_koelectra_rs131_epoch12step3500_bs20.pt_koelectraExTrain_epc0by1_bs20.json') as f:
    v5 = json.load(f)

# with open('submission/tunib-electra-ko-base-BoolQ-3-3840-1234-v16_post_eval.json') as f:
#     v5 = json.load(f)


##################################################
# v1_boolq_df = json_normalize(v1['boolq']) #Results contain the required data
# v2_boolq_df = json_normalize(v2['boolq']) #Results contain the required data
# v3_boolq_df = json_normalize(v3['boolq']) #Results contain the required data
# v4_boolq_df = json_normalize(v4['boolq']) #Results contain the required data
# v5_boolq_df = json_normalize(v5['boolq']) #Results contain the required data

# v1_boolq_df = v1_boolq_df['label']
# v2_boolq_df = v2_boolq_df['label']
# v3_boolq_df = v3_boolq_df['label']
# v4_boolq_df = v4_boolq_df['label']
# v5_boolq_df = v5_boolq_df['label']

# all_boolq_df = pd.concat([v1_boolq_df, v2_boolq_df, v3_boolq_df, v4_boolq_df, v5_boolq_df], axis=1)      ####
# all_boolq_df.columns = ['v1', 'v2', 'v3', 'v4', 'v5']                                      ####

# all_boolq_df['tmp'] = all_boolq_df.sum(axis=1) / 5
# all_boolq_df['ensemble'] = int(2)

# boolq_submission = []

# for idx in range(len(all_boolq_df)):
#     if all_boolq_df['tmp'][idx] >= 0.5:
#         all_boolq_df['ensemble'][idx] = 1
#     else:
#         all_boolq_df['ensemble'][idx] = 0

#     boolq_submission.append({"idx": idx+1,"label" : int(all_boolq_df['ensemble'][idx])})

# boolq_dic = {"boolq" : boolq_submission}
# with open(f"submission/submission/ensemble-v4_boolq_test11.json", 'w') as f:
#     json.dump(boolq_dic, f, indent= 4)

#################################################
# v1_wic_df = json_normalize(v1['wic']) #Results contain the required data
# v2_wic_df = json_normalize(v2['wic']) #Results contain the required data
# v3_wic_df = json_normalize(v3['wic']) #Results contain the required data
# v4_wic_df = json_normalize(v4['wic']) #Results contain the required data
# v5_wic_df = json_normalize(v5['wic']) #Results contain the required data

# v1_wic_df = v1_wic_df['label']
# v2_wic_df = v2_wic_df['label']
# v3_wic_df = v3_wic_df['label']
# v4_wic_df = v4_wic_df['label']
# v5_wic_df = v5_wic_df['label']



# all_wic_df = pd.concat([v1_wic_df, v2_wic_df, v3_wic_df, v4_wic_df, v5_wic_df], axis=1)      ####
# all_wic_df.columns = ['v1', 'v2', 'v3', 'v4', 'v5']                                     ####

# all_wic_df['tmp'] = all_wic_df.sum(axis=1) / 5
# all_wic_df['ensemble'] = int(2)

# wic_submission = []
# true = True
# false = False

# for idx in range(len(all_wic_df)):
#     if all_wic_df['tmp'][idx] >= 0.5:
#         all_wic_df['ensemble'][idx] = 1
#     else:
#         all_wic_df['ensemble'][idx] = 0

#     wic_submission.append({"idx": idx+1,"label" : int(all_wic_df['ensemble'][idx])})

# wic_dic = {"wic" : wic_submission}

# for i in range(len(wic_submission)):
#     if wic_dic["wic"][i]["label"] == 1:
#         wic_dic["wic"][i]["label"] = true

#     elif wic_dic["wic"][i]["label"] == 0:
#         wic_dic["wic"][i]["label"] = false

# with open(f"submission/submission/ensemble-v4_wic.json", 'w') as f:
#     json.dump(wic_dic, f, indent= 4)

# print(wic_dic)

#################################################
# v1_cola_df = json_normalize(v1['cola']) #Results contain the required data
# v2_cola_df = json_normalize(v2['cola']) #Results contain the required data
# v3_cola_df = json_normalize(v3['cola']) #Results contain the required data
# v4_cola_df = json_normalize(v4['cola']) #Results contain the required data
# v5_cola_df = json_normalize(v5['cola']) #Results contain the required data

# v1_cola_df = v1_cola_df['label']
# v2_cola_df = v2_cola_df['label']
# v3_cola_df = v3_cola_df['label']
# v4_cola_df = v4_cola_df['label']
# v5_cola_df = v5_cola_df['label']


# all_cola_df = pd.concat([v1_cola_df, v2_cola_df, v3_cola_df, v4_cola_df, v5_cola_df], axis=1)      ####
# all_cola_df.columns = ['v1', 'v2', 'v3', 'v4', 'v5']                                     ####

# all_cola_df['tmp'] = all_cola_df.sum(axis=1) / 5
# all_cola_df['ensemble'] = int(2)

# cola_submission = []

# for idx in range(len(all_cola_df)):
#     if all_cola_df['tmp'][idx] >= 0.5:
#         all_cola_df['ensemble'][idx] = 1
#     else:
#         all_cola_df['ensemble'][idx] = 0

#     cola_submission.append({"idx": idx,"label" : int(all_cola_df['ensemble'][idx])})

# cola_dic = {"cola" : cola_submission}

# with open(f"submission/submission/ensemble-last_cola1.json", 'w') as f:
#     json.dump(cola_dic, f, indent= 4)

# print(cola_dic)

# #################################################
v1_copa_df = json_normalize(v1['copa']) #Results contain the required data
v2_copa_df = json_normalize(v2['copa']) #Results contain the required data
v3_copa_df = json_normalize(v3['copa']) #Results contain the required data
v4_copa_df = json_normalize(v4['copa']) #Results contain the required data
v5_copa_df = json_normalize(v5['copa']) #Results contain the required data

v1_copa_df = v1_copa_df['label']
v2_copa_df = v2_copa_df['label']
v3_copa_df = v3_copa_df['label']
v4_copa_df = v4_copa_df['label']
v5_copa_df = v5_copa_df['label']


all_copa_df = pd.concat([v1_copa_df, v2_copa_df, v3_copa_df, v4_copa_df, v5_copa_df], axis=1)      ####
all_copa_df.columns = ['v1', 'v2', 'v3', 'v4', 'v5']   

all_copa_df['tmp'] = all_copa_df.sum(axis=1) / 5
all_copa_df['ensemble'] = int(2)

copa_submission = []

for idx in range(len(all_copa_df)):
    if all_copa_df['tmp'][idx] >= 1.5:
        all_copa_df['ensemble'][idx] = 1
    else:
        all_copa_df['ensemble'][idx] = 0

    copa_submission.append({"idx": idx+1,"label" : int(all_copa_df['ensemble'][idx]+1)})

copa_dic = {"copa" : copa_submission}

with open(f"submission/submission/ensemble-copa-last1214.json", 'w') as f:
    json.dump(copa_dic, f, indent= 4)

# print(copa_dic)
# #################################################

# all_data = {"boolq" : boolq_submission, "wic" : wic_submission, "cola" : cola_submission, "copa" : copa_submission}

# with open(f"submission/submission/ensemble-v1.json", 'w') as f:
#     json.dump(all_data, f, indent= 4)