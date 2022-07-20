"""
python interactive.py
"""

from argparse import ArgumentParser
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "tunib/electra-ko-en-base"
ckpt_name = "model_save/tunib-electra-ko-en-base-3.pt"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

parser = ArgumentParser()
parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
args = parser.parse_args()

model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()

while True:
    t = input("\nText: ")
    q = input("Question: ")
    tokens = tokenizer(
        t + args.sep_token + q,
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
    print(f"Result: {'True' if classification_results.item() else 'False'}")
