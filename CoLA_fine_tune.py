'''
deepspeed --num_gpus=1 CoLA_fine_tune.py
'''

from argparse import ArgumentParser
import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import wandb

#############################################    -> 실험결과 FIX
random_seed = 64
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
##################################


task = "CoLA"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
model_name = "monologg/koelectra-base-v3-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# SPECIAL_TOKENS = {
#     "bos_token": "<bos>",
#     "eos_token": "<eos>",
#     "pad_token": "<pad>",
#     "sep_token": "<seq>"
#     }
# SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<seq>"]
# tokenizer.add_special_tokens(SPECIAL_TOKENS)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
).cuda()

# model.resize_token_embeddings(len(tokenizer)) 

parser = ArgumentParser()
parser.add_argument("--deepspeed_config", type=str, default="CoLA_ds_config.json")
parser.add_argument("--local_rank", type=int)
parser.add_argument("--epoch", default=30, type=int)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
args = parser.parse_args()

wandb.init(project="GPT-finetune", name=f"CoLA-{model_name}-30-512-{random_seed}-v11")
train_data = pd.read_csv("data/CoLA_data/cola_train_dev_aug_shuffle.tsv", delimiter="\t")
train_text, train_labels = (
    train_data["sentence"].values,
    train_data["acceptability_label"].values,
)

dataset = [
    {"data": t , "label": l}
    for t, l in zip(train_text, train_labels)
]
train_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

eval_data = pd.read_csv("data/CoLA_data/NIKL_CoLA_test_labeling.tsv", delimiter="\t")
eval_text, eval_labels = (
    eval_data["sentence"].values,
    eval_data["acceptability_label"].values,
)

dataset = [
    {"data": t, "label": l}
    for t, l in zip(eval_text, eval_labels)
]
eval_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

optimizer = DeepSpeedCPUAdam(
    lr=3e-5, weight_decay=3e-7, model_params=model.parameters()
)

engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, optimizer=optimizer
)
epochs = 0
for epoch in range(args.epoch):
    epochs += 1
    model.train()
    for train in tqdm(train_loader):
        optimizer.zero_grad()
        text, label = train["data"], train["label"].cuda()
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        input_ids = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()
        output = engine.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label,
        )
        loss = output.loss
        wandb.log({"loss": loss})
        engine.backward(loss)
        optimizer.step()
        classification_results = output.logits.argmax(-1)

        acc = 0
        for res, lab in zip(classification_results, label):
            if res == lab:
                acc += 1

        wandb.log({"acc": acc / len(classification_results)})
    wandb.log({"epoch": epochs})


    model.eval()
    for eval in tqdm(eval_loader):

        eval_text, eval_label = eval["data"], eval["label"].cuda()
        eval_tokens = tokenizer(
            eval_text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        input_ids = eval_tokens.input_ids.cuda()
        attention_mask = eval_tokens.attention_mask.cuda()
        eval_out = engine.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=eval_label
        )
        classification_results = eval_out.logits.argmax(-1)

        acc = 0
        for res, lab in zip(classification_results, eval_label):
            if res == lab:
                acc += 1

    wandb.log({"eval_acc": acc / len(classification_results)})
    wandb.log({"eval_loss": eval_out.loss})

    torch.save(model.state_dict(), f"model_save/{model_name.replace('/', '-')}-{task}-{epoch}-30-512-{random_seed}-v11.pt")
