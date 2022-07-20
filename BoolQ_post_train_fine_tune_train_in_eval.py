'''
deepspeed --num_gpus=1 BoolQ_post_train_fine_tune_train_in_eval.py
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
# SPECIAL_TOKENS = {
#     "bos_token": "<bos>",
#     "eos_token": "<eos>",
#     "pad_token": "<pad>",
#     "sep_token": "<seq>"
#     }
# SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<seq>"]
# tokenizer.add_special_tokens(SPECIAL_TOKENS)



model = AutoModelForSequenceClassification.from_pretrained(model_name)                                                  #####

# model = AutoModelForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=2,
# ).cuda()

# model.resize_token_embeddings(len(tokenizer)) 

parser = ArgumentParser()
parser.add_argument("--deepspeed_config", type=str, default="BoolQ_ds_config.json")
parser.add_argument("--local_rank", type=int)
parser.add_argument("--epoch", default=30, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
parser.add_argument("--exam_name", type=str, default="v30_post_eval")
parser.add_argument("--post_name", type=str, default="last_post")

args = parser.parse_args()



# ckpt_name = f"model_save/boolq_post_pt/tunib-electra-ko-base-BoolQ-3-1234-{args.post_name}.pt"                                    ###

ckpt_name = f"model_save/boolq_post_pt/monologg-koelectra-base-v3-discriminator-BoolQ-0-1234-mono_post.pt"       


model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))                                              ####
model.cuda()                                                                                                #####


wandb.init(project="GPT-finetune", name=f"BoolQ-{model_name}-{random_seed}-{args.exam_name}")
train_data = pd.read_csv("data/BoolQ_data/BoolQ_train_pt_dev_pt.tsv", delimiter="\t")
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

eval_data = pd.read_csv("data/BoolQ_data/BoolQ_test_labeling.tsv", delimiter="\t")
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

optimizer = DeepSpeedCPUAdam(
    lr=3e-5, weight_decay=3e-7, model_params=model.parameters()
)

engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, optimizer=optimizer
)




def evalModel():
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
                    
        eval_classification_results = eval_out.logits.argmax(-1)
        eval_loss = eval_out.loss

        acc = 0
        for res, lab in zip(eval_classification_results, eval_label):
            if res == lab:
                acc += 1
        
    wandb.log({"eval_loss": eval_loss})   ## 이미 다 적용된 상태인듯..
    wandb.log({"eval_acc": acc / len(eval_classification_results)})
    wandb.log({"mini_batch": mini_batch})
    torch.save(model.state_dict(), f"model_save/{model_name.replace('/', '-')}-{task}-{epochs}-{mini_batch}-{random_seed}-{args.exam_name}.pt")
    # torch.save(model.state_dict(), f"model_save/{model_name.replace('/', '-')}-{task}-{epoch}-{random_seed}-v14.pt")

    

epochs = 0
step = 0
for epoch in range(args.epoch):
    epochs += 1

    mini_batch=0
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
        output = engine.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label
        )
        loss = output.loss
        engine.backward(loss)
        optimizer.step()
        classification_results = output.logits.argmax(-1)
        # print(classification_results.size(), label.size())   ### size 동일 
        # print(output.logits)
        #test_batch = 1280
        mini_batch+=args.batch_size

        if mini_batch%(args.batch_size*10) == 0: #1000
            evalModel()
            model.train()

        acc = 0
        for res, lab in zip(classification_results, label):
            if res == lab:
                acc += 1

        wandb.log({"loss": loss})
        wandb.log({"acc": acc / len(classification_results)})
        
        ##eval
    evalModel()

    wandb.log({"epoch": epochs})

        # torch.save(model.state_dict(), f"model_save/{model_name.replace('/', '-')}-{task}-{step}-{epoch}-{random_seed}-v12.pt")
        # torch.save(model.state_dict(), f"model_save/{model_name.replace('/', '-')}-{task}-{epoch}-30-64-{random_seed}-base.pt")

