from pororo import Pororo
import pandas as pd
from tqdm import tqdm


train_data = pd.read_csv("data/BoolQ_data/boolq_all_eng.tsv", delimiter="\t")

sample_data = train_data

mt = Pororo(task="translation", lang="multi")

# eng_text_list = []
# eng_question_list = []
# eng_all_list = []

# for idx in tqdm(range(len(sample_data))):
#     text, question, labels = sample_data["Text"][idx], sample_data["Question"][idx], sample_data["Answer(FALSE = 0, TRUE = 1)"][idx]

#     text_result = mt(text, src="ko", tgt="en")
#     question_result = mt(question, src="ko", tgt="en")

#     eng_all_list.append([text_result] + [question_result] + [labels])

# # print(eng_all_list)

# eng_df = pd.DataFrame(eng_all_list, columns=['Text', 'Question', 'Answer(FALSE = 0, TRUE = 1)'])


ko_all_list = []

for idx in tqdm(range(len(train_data))):
    text, question, labels = train_data["Text"][idx], train_data["Question"][idx], train_data["Answer(FALSE = 0, TRUE = 1)"][idx]

    text_result = mt(text, src="en", tgt="ko")
    question_result = mt(question, src="en", tgt="ko")
   
    ko_all_list.append([text_result] + [question_result] + [labels])
    
df = pd.DataFrame(ko_all_list, columns=['Text', 'Question', 'Answer(FALSE = 0, TRUE = 1)'])

# aug_df = pd.concat([train_data, df])

df.to_csv('data/BoolQ_data/boolq_all_eng_translation.tsv', index=False, sep="\t")