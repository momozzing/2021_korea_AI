'''
deepspeed --num_gpus=1 BoolQ_fine_tune.py
'''

from argparse import ArgumentParser
import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import wandb

#############################################    -> 실험결과 FIX
random_seed = 1234
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
##################################

task = "BoolQ"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
model_name = "monologg/koelectra-base-v3-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
).cuda()

parser = ArgumentParser()
parser.add_argument("--epoch", default=5, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
args = parser.parse_args()

wandb.init(project="2021_Korea_AI", name=f"BoolQ-{model_name}-{random_seed}")
train_data = pd.read_csv("data/BoolQ/SKT_BoolQ_Train.tsv", delimiter="\t")
# train_data = train_data[:50000]
train_text, train_question, train_labels = (
    train_data["Text"].values,
    train_data["Question"].values,
    train_data["Answer(FALSE = 0, TRUE = 1)"].values
)

dataset = [
    {"data": t + str(args.sep_token) + q, "label": l}
    for t, q, l in zip(train_text, train_question, train_labels)
]
train_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

eval_data = pd.read_csv("data/BoolQ/SKT_BoolQ_Dev.tsv", delimiter="\t")
# eval_data = eval_data[:1000]
eval_text, eval_question, eval_labels = (
    eval_data["Text"].values,
    eval_data["Question"].values,
    eval_data["Answer(FALSE = 0, TRUE = 1)"].values
)

dataset = [
    {"data": t + str(args.sep_token) + q, "label": l}
    for t, q, l in zip(eval_text, eval_question, eval_labels)
]
eval_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

optimizer = AdamW(params=model.parameters(),
    lr=3e-5, weight_decay=3e-7
)


epochs = 0
step = 0
for epoch in range(args.epoch):
    epochs += 1
    model.train()
    for train in tqdm(train_loader):
        model.train()
        optimizer.zero_grad()
        text, label = train["data"], train["label"].cuda()
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        input_ids = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()
        output = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label
        )
        loss = output.loss
        loss.backward()
        optimizer.step()

        classification_results = output.logits.argmax(-1)

        acc = 0
        for res, lab in zip(classification_results, label):
            if res == lab:
                acc += 1

        wandb.log({"loss": loss})
        wandb.log({"acc": acc / len(classification_results)})

    model.eval()
    tmp=[]
    for idx, eval in tqdm(enumerate(eval_loader)):
        # step+=1
        # if idx % 1 == 0:
        eval_text, eval_label = eval["data"], eval["label"].cuda()
        eval_tokens = tokenizer(
            eval_text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        input_ids = eval_tokens.input_ids.cuda()
        attention_mask = eval_tokens.attention_mask.cuda()
        
        eval_out = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=eval_label
        )
        
        # eval_out_loss += engine(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     labels=eval_label
        # ).loss
            
        eval_classification_results = eval_out.logits.argmax(-1)
        eval_loss = eval_out.loss
        # eval_out_loss = sum(eval_loss) /len(eval_loader)

        acc = 0
        for res, lab in zip(eval_classification_results, eval_label):
            if res == lab:
                acc += 1
        
    # tmp.append(eval_loss)
    # eval_losses = sum(tmp)/len(eval_loader)
    # print(eval_losses)
    
    # lossss = (sum(tmp)/len(tmp)).item()
    # print(lossss)
        # wandb.log({"step": step})
        wandb.log({"eval_loss": eval_loss})   ## 이미 다 적용된 상태인듯..
        wandb.log({"eval_acc": acc / len(eval_classification_results)})
        wandb.log({"epoch": epochs})
        # torch.save(model.state_dict(), f"model_save/{model_name.replace('/', '-')}-{task}-{step}-{epochs}-{random_seed}-v12.pt")
        torch.save(model.state_dict(), f"model_save/{model_name.replace('/', '-')}-{task}-{epoch}-{random_seed}.pt")



    # torch.save(model.state_dict(), f"model_save/{model_name.replace('/', '-')}-{task}-{step}-{epoch}-{random_seed}-v12.pt")
    # torch.save(model.state_dict(), f"model_save/{model_name.replace('/', '-')}-{task}-{epoch}-30-64-{random_seed}-base.pt")

