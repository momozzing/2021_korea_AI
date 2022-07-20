"""
python boolq_submission.py
"""
from argparse import ArgumentParser
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer



model_name = "tunib/electra-ko-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

parser = ArgumentParser()
parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
parser.add_argument("--target_epochs", default=24, type=int)
parser.add_argument("--random_seed", default=1234, type=int)
parser.add_argument("--target_step", default=7680, type=int)
args = parser.parse_args()


model_path_name = f'tunib-electra-ko-base-BoolQ-{args.target_epochs}-{args.target_step}-{args.random_seed}-v33_post_eval'

ckpt_name = f"model_save/{model_path_name}.pt"


model.eval()
model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()
eval_data = pd.read_csv("data/BoolQ_data/SKT_BoolQ_Test.tsv", delimiter="\t")
eval_text, eval_question = (
    eval_data["Text"].values,
    eval_data["Question"].values,
)

dataset = [
    {"data": t + args.sep_token + q}
    for t, q in zip(eval_text, eval_question)
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

    submission.append({"idx": idx+1,"label" : classification_results.item()})

boolq_dic = {"boolq" : submission}

with open(f"submission/{model_path_name}.json", 'w') as f:
    json.dump(boolq_dic, f, indent= 4)