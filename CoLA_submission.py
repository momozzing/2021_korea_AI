"""
python CoLA_submission.py
"""

from argparse import ArgumentParser
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

task = "CoLA"
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

eval_data = pd.read_csv("data/CoLA/NIKL_CoLA_test.tsv", delimiter="\t")
eval_text = (
    eval_data["sentence"].values
)

dataset = [
    {"data": t}
    for t in eval_text
]

submission = []

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

        submission.append({"idx": idx,"label" : classification_results.item()})

    cola_dic = {"cola" : submission}

    with open(f"submission/{model_path_name}.json", 'w') as f:
        json.dump(cola_dic, f, indent= 4)