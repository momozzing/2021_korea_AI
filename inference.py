"""
python inference.py
"""

from argparse import ArgumentParser
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "tunib/electra-ko-en-base"
ckpt_name = "tunib-electra-ko-en-base-3.pt"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

parser = ArgumentParser()
parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
args = parser.parse_args()

model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()
eval_data = pd.read_csv("SKT_BoolQ_Dev.tsv", delimiter="\t")
eval_text, eval_question, eval_labels = (
    eval_data["Text"].values,
    eval_data["Question"].values,
    eval_data["Answer(FALSE = 0, TRUE = 1)"].values,
)

dataset = [
    {"data": t + args.sep_token + q, "label": l}
    for t, q, l in zip(eval_text, eval_question, eval_labels)
]


acc = 0
for data in tqdm(dataset):
    text, label = data["data"], data["label"]
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    input_ids = tokens.input_ids.cuda()
    attention_mask = tokens.attention_mask.cuda()
    label = torch.tensor(label).cuda()

    output = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    classification_results = output.logits.argmax(-1)
    if classification_results == label:
        acc += 1


print(f"acc: {acc / len(dataset)}")
