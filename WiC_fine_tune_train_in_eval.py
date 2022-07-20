'''
deepspeed --num_gpus=1 WiC_fine_tune_train_in_eval.py
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


task = "WiC"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
model_name = "tunib/electra-ko-base"
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
parser.add_argument("--random_seed", default=1234, type=int)
parser.add_argument("--deepspeed_config", type=str, default="WiC_ds_config.json")
parser.add_argument("--local_rank", type=int)
parser.add_argument("--epoch", default=30, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
args = parser.parse_args()


#############################################    -> 실험결과 FIX
# random_seed = 256
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.random_seed)
random.seed(args.random_seed)
##################################


wandb.init(project="GPT-finetune", name=f"WiC-{model_name}-{args.random_seed}-v10_eval")
train_data = pd.read_csv("data/WiC_data/WiC_train_dev_aug.tsv", delimiter="\t")
# train_data = train_data[:2000]
# train_data["ANSWER"] = train_data["ANSWER"].astype(int)    ### True/False 1, 0으로 변경 
train_data["ANSWER"] = train_data['ANSWER'].map(lambda x : 1 if x else 0)

train_text, train_sent1, train_sent2, train_start1, train_end1, train_start2, train_end2, train_labels = (
    train_data["Target"].values,
    train_data["SENTENCE1"].values,
    train_data["SENTENCE2"].values,
    train_data["start_s1"].values,
    train_data["end_s1"].values,
    train_data["start_s2"].values,
    train_data["end_s2"].values,
    train_data["ANSWER"].values,
)


dataset = [
    {"data": t + args.sep_token + sen1 + args.sep_token + sen2 + args.sep_token + str(s1) + args.sep_token + str(e1) + args.sep_token + str(s2) + args.sep_token + str(e2),"label": l}
    for t, sen1, sen2, s1, e1, s2, e2, l in zip(train_text, train_sent1, train_sent2, train_start1, train_end1, train_start2, train_end2, train_labels)
]
train_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)



eval_data = pd.read_csv("data/WiC_data/NIKL_SKT_WiC_Test_labeling_v2.tsv", delimiter="\t")

eval_data["ANSWER"] = eval_data['ANSWER'].map(lambda x : 1 if x else 0)

eval_text, eval_sent1, eval_sent2, eval_start1, eval_end1, eval_start2, eval_end2, eval_labels = (
    eval_data["Target"].values,
    eval_data["SENTENCE1"].values,
    eval_data["SENTENCE2"].values,
    eval_data["start_s1"].values,
    eval_data["end_s1"].values,
    eval_data["start_s2"].values,
    eval_data["end_s2"].values,
    eval_data["ANSWER"].values,
)

dataset = [
    {"data": t + args.sep_token + sen1 + args.sep_token + sen2 + args.sep_token + str(s1) + args.sep_token + str(e1) + args.sep_token + str(s2) + args.sep_token + str(e2),"label": l}
    for t, sen1, sen2, s1, e1, s2, e2, l in zip(eval_text, eval_sent1, eval_sent2, eval_start1, eval_end1, eval_start2, eval_end2, eval_labels)
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
    bestAcc = 0
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
    wandb.log({"mini_batch": mini_batch})
    torch.save(model.state_dict(), f"model_save/{model_name.replace('/', '-')}-{task}-{epochs}-{mini_batch}-{args.random_seed}-v10_eval.pt")

    resultEvalAcc = acc / len(classification_results)
    return resultEvalAcc


bestAcc, bestEpoch, bestMB = 0, 0, 0
epochs = 0
for epoch in range(args.epoch):
    epochs += 1

    mini_batch=0
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

        mini_batch+=args.batch_size
        if mini_batch%(args.batch_size*20) == 0: #1000
            result = evalModel()
            if result>bestAcc:
                bestAcc = max(bestAcc,result)
                bestEpoch = epoch
                bestMB = mini_batch
            model.train()

        acc = 0
        for res, lab in zip(classification_results, label):
            if res == lab:
                acc += 1

        wandb.log({"acc": acc / len(classification_results)})
    wandb.log({"epoch": epochs})

    result = evalModel()
    if result>bestAcc:
        bestAcc = max(bestAcc,result)
        bestEpoch = epoch
        bestMB = mini_batch


save_result = []
save_result.append(f'bestEvalAcc:{bestAcc} at epoch{bestEpoch} minibatch{bestMB}, ramdom_seed{args.random_seed} \n') 
# save_result.to_csv(f'test/wic_best_{args.random_seed}.tsv', index=False, sep="\t")

with open('wic_test.txt', 'a') as file:    # hello.txt 파일을 쓰기 모드(w)로 열기
    file.writelines(save_result)
file.close()

print(f'bestEvalAcc:{bestAcc} at epoch{bestEpoch} minibatch{bestMB}')
