from pororo import Pororo
import pandas as pd
from tqdm import tqdm


train_data = pd.read_csv("data/COPA_data/COPA_eng_test.tsv", delimiter="\t")

sample_data = train_data

mt = Pororo(task="translation", lang="multi")

# eng_text_list = []
# eng_question_list = []
# eng_all_list = []

# for idx in tqdm(range(len(sample_data))):
#     text, question, fir, sec, labels = sample_data["sentence"][idx], sample_data["question"][idx], sample_data["1"][idx], sample_data["2"][idx], sample_data["Answer"][idx] 

#     text_result = mt(text, src="ko", tgt="en")
#     fir_result = mt(fir, src="ko", tgt="en")
#     sec_result = mt(sec, src="ko", tgt="en")

#     eng_all_list.append([text_result] + [question] + [fir_result] + [sec_result] + [labels])

# # print(eng_all_list)
# eng_df = pd.DataFrame(eng_all_list, columns=['sentence', 'question', '1', '2', 'Answer'])

ko_all_list = []

for idx in tqdm(range(len(sample_data))):
    text, question, fir, sec, labels = sample_data["sentence"][idx], sample_data["question"][idx], sample_data["1"][idx], sample_data["2"][idx], sample_data["Answer"][idx] 

    text_result = mt(text, src="en", tgt="ko")
    fir_result = mt(fir, src="en", tgt="ko")
    sec_result = mt(sec, src="en", tgt="ko")
    question_result= mt(question, src="en", tgt="ko")
   
    ko_all_list.append([text_result] + [question_result] + [fir_result] + [sec_result] + [labels])
    
df = pd.DataFrame(ko_all_list, columns=['sentence', 'question', '1', '2', 'Answer'])

# aug_df = pd.concat([train_data, df])

df.to_csv('data/COPA_data/COPA_test_Eng_trainslation.tsv', index=False, sep="\t")