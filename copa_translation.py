import pandas as pd
import json
from pororo import Pororo
from tqdm import tqdm

jsonObj = pd.read_json(path_or_buf='data/COPA_data/COPA_eng.jsonl', lines=True)

df = pd.DataFrame(jsonObj)

df = df.rename(columns={'premise':'sentence','choice1':'1','choice2':'2','question':'question','label':'Answer','idx':'idx'})

# df = df


mt = Pororo(task="translation", lang="multi")

eng_text_list = []
eng_question_list = []
eng_all_list = []

for idx in tqdm(range(len(df))):
    text, question, fir, sec, labels = df["sentence"][idx], df["question"][idx], df["1"][idx], df["2"][idx], df["Answer"][idx] 

    text_result = mt(text, src="en", tgt="ko")
    fir_result = mt(fir, src="en", tgt="ko")
    sec_result = mt(sec, src="en", tgt="ko")
    question_result = mt(question, src="en", tgt="ko")

    eng_all_list.append([text_result] + [question_result] + [fir_result] + [sec_result] + [labels])

translation_df = pd.DataFrame(eng_all_list, columns=['sentence', 'question', '1', '2', 'Answer'])

# for i in translation_df:
#     if translation_df['question'] == "효과":
#         translation_df['question'] == "결과"

translation_df.to_csv('data/COPA_data/COPA_translation_data.tsv', index=False, sep="\t")

