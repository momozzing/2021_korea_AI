"""
python submission.py
"""

from argparse import ArgumentParser
import json
from re import S
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pandas import json_normalize

model_name = "tunib/electra-ko-base"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

parser = ArgumentParser()
parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
parser.add_argument("--target_epochs", default=0, type=int)
parser.add_argument("--random_seed", default=18, type=int)
parser.add_argument("--target_step", default=2560, type=int)


args = parser.parse_args()

model_path_name = f'tunib-electra-ko-base-WiC-{args.target_epochs}-{args.target_step}-{args.random_seed}-v10_eval'


# model_name = "tunib/electra-ko-base"
ckpt_name = f"model_save/{model_path_name}.pt"                                    ##


model.eval()
model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()
eval_data = pd.read_csv("data/WiC_data/NIKL_SKT_WiC_Test.tsv", delimiter="\t")
eval_text, eval_sent1, eval_sent2, eval_start1, eval_end1, eval_start2, eval_end2 = (
    eval_data["Target"].values,
    eval_data["SENTENCE1"].values,
    eval_data["SENTENCE2"].values,
    eval_data["start_s1"].values,
    eval_data["end_s1"].values,
    eval_data["start_s2"].values,
    eval_data["end_s2"].values,
)

dataset = [
    {"data": t + args.sep_token + sen1 + args.sep_token + sen2 + args.sep_token + str(s1) + args.sep_token + str(e1) + args.sep_token + str(s2) + args.sep_token + str(e2)}
    for t, sen1, sen2, s1, e1, s2, e2 in zip(eval_text, eval_sent1, eval_sent2, eval_start1, eval_end1, eval_start2, eval_end2)
]

# boolq_dic = {"boolq" : []}
submission = []

true = True
false = False

for idx, data in tqdm(enumerate(dataset)):
    text = data["data"]
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    input_ids = tokens.input_ids.cuda()
    attention_mask = tokens.attention_mask.cuda()

    output = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    classification_results = output.logits.argmax(-1)

    submission.append({"idx": idx+1,"label" : classification_results.item()})

dic = {"wic" : submission}

for i in range(len(submission)):
    if dic["wic"][i]["label"] == 1:
        dic["wic"][i]["label"] = true

    elif dic["wic"][i]["label"] == 0:
        dic["wic"][i]["label"] = false


with open(f"submission/{model_path_name}.json", 'w') as f:
    json.dump(dic, f, indent= 4)

with open('data/WiC_data/WiC_test_label_v2.json') as f:
    wic_label_data = json.load(f)

wic_label_df = json_normalize(wic_label_data['wic']) #Results contain the required data

wic_label_df = wic_label_df['label']


with open(f'submission/{model_path_name}.json') as f:          ###### 내가 원하는 file
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
wic_acc = count/len(all_wic_df)
save_result = []
save_result.append(f"{model_path_name}: WiC_acc : {wic_acc} \n")
print(f"{model_path_name}: WiC_acc : " , count/len(all_wic_df))

with open('wic_test_acc.txt', 'a') as file:    # hello.txt 파일을 쓰기 모드(w)로 열기
    file.writelines(save_result)
file.close()