"""
python submission.py
"""

from argparse import ArgumentParser
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pandas import json_normalize
from sklearn.metrics import matthews_corrcoef


model_pt = 'tunib-electra-ko-base-CoLA-5-30-512-64-v11'


model_name = "tunib/electra-ko-base"
ckpt_name = f"model_save/{model_pt}.pt"                                    ##
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

parser = ArgumentParser()
parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
args = parser.parse_args()

model.eval()
model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()
eval_data = pd.read_csv("data/CoLA_data/NIKL_CoLA_test.tsv", delimiter="\t")
eval_text = (
    eval_data["sentence"].values
    # eval_data["Question"].values,
)

dataset = [
    {"data": t}
    for t in eval_text
]

# boolq_dic = {"boolq" : []}
submission = []

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

    # submission.write(str(classification_results.item()) + "\n")

    submission.append({"idx": idx,"label" : classification_results.item()})

submission_dic = {"cola" : submission}

with open(f"submission/{model_pt}.json", 'w') as f:
    json.dump(submission_dic, f, indent= 4)

with open('data/CoLA_data/cola_test_label.json') as f:
    cola_label_data = json.load(f)

cola_label_df = json_normalize(cola_label_data['cola']) #Results contain the required data

cola_label_df = cola_label_df['label']


with open(f'submission/{model_pt}.json') as f:          ###### 내가 원하는 file
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