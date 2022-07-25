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

task = "WiC"
model_name = "tunib/electra-ko-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

parser = ArgumentParser()
parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
parser.add_argument("--target_epochs", default=1, type=int)
args = parser.parse_args()

model_path_name = f'tunib-electra-ko-base-{task}-{args.target_epochs}'

ckpt_name = f"model_save/{model_path_name}.pt"

model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()

eval_data = pd.read_csv("data/WiC/NIKL_SKT_WiC_Test.tsv", delimiter="\t")
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

submission = []

true = True
false = False

with torch.no_grad():
    model.eval() 
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

    wic_dic = {"wic" : submission}

    for i in range(len(submission)):
        if wic_dic["wic"][i]["label"] == 1:
            wic_dic["wic"][i]["label"] = true

        elif wic_dic["wic"][i]["label"] == 0:
            wic_dic["wic"][i]["label"] = false

    with open(f"submission/{model_path_name}.json", 'w') as f:
        json.dump(wic_dic, f, indent= 4)