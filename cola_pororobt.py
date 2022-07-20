from pororo import Pororo
import pandas as pd
from tqdm import tqdm


train_data = pd.read_csv("data/CoLA_data/cola_data_aug.tsv", delimiter=",")

sample_data = train_data

mt = Pororo(task="translation", lang="multi")

eng_text_list = []
# question_list = []
eng_all_list = []

for idx in tqdm(range(len(sample_data))):
    labels, text = sample_data["acceptability_label"][idx], sample_data["sentence"][idx]

    text_result = mt(text, src="ko", tgt="en")
   
    eng_all_list.append([labels] + [text_result])

eng_df = pd.DataFrame(eng_all_list, columns=['acceptability_label', 'sentence'])

# print(eng_all_list)
# eng_df = eng_df['sentence'].astype(str)

ko_all_list = []

for idx in tqdm(range(len(eng_df))):
    labels, text = eng_df["acceptability_label"][idx], eng_df["sentence"][idx]

    text_result = mt(text, src="en", tgt="ko")
   
    ko_all_list.append([labels] + [text_result])
    
df = pd.DataFrame(ko_all_list, columns=['acceptability_label', 'sentence'])

# aug_df = pd.concat([train_data, df])

df.to_csv('data/CoLA_data/cola_data_aug_bt.tsv', index=False, sep="\t")