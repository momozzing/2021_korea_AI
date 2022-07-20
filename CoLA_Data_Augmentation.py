import pandas as pd
from tqdm import tqdm
from koeda import AEDA


train_data = pd.read_csv("data/CoLA_data/cola_korsts_data.tsv", delimiter="\t")

train_data['sentence'] = train_data['sentence'].astype(str)
#################################################################################################    AEDA 코드
# aeda = AEDA(
#     morpheme_analyzer="Mecab", punc_ratio=0.1, punctuations=[".", ",", "!", "?", ";", ":"]
# )

# text_list = []
# question_list = []
# all_list = []

# for i in range(4):
#     for idx in tqdm(range(len(train_data))):
#         labels, text = train_data["acceptability_label"][idx], train_data["sentence"][idx]

#         text_result = aeda(text)

#         all_list.append([labels] + [text_result])
    

#     df = pd.DataFrame(all_list, columns=['acceptability_label', 'sentence'])

# # print(df)

# aug_df = pd.concat([train_data, df])

# aug_df.to_csv('data/CoLA_data/CoLA_Aug_train.tsv', index=False, sep="\t")

############################################################################################

########################################################################################### EDA

from koeda import EDA

# sample_data = train_data[:100]

eda = EDA(
    morpheme_analyzer="Mecab", alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.9, prob_rd=0.05
)

text_list = []
question_list = []
all_list = []

# for i in range(4):

for idx in tqdm(range(len(train_data))):
    labels, text = train_data["acceptability_label"][idx], train_data["sentence"][idx]

    text_result = eda(text)

    all_list.append([labels] + [text_result])

df = pd.DataFrame(all_list, columns=['acceptability_label', 'sentence'])

# aug_df = pd.concat([train_data, df])

df.to_csv('data/CoLA_data/cola_korsts_data_eda_0.5.tsv', index=False, sep="\t")


# eda = EDA(
#     morpheme_analyzer="Okt", alpha_sr=0.3, alpha_ri=0.3, alpha_rs=0.3, prob_rd=0.3
# )

# text = "아버지가 방에 들어가신다"

# result = eda(text)
# print(result)
# # 아버지가 정실에 들어가신다

# result = eda(text, p=(0.9, 0.9, 0.9, 0.9), repetition=2)
# print(result)
# # ['아버지가 객실 아빠 안방 방에 정실 들어가신다', '아버지가 탈의실 방 휴게실 에 안방 탈의실 들어가신다']

################################ AEDA
# from koeda import AEDA

# text = "인문과학 또는 인문학(人文學, 영어: humanities)은 인간과 인간의 근원문제, 인간과 인간의 문화에 관심을 갖거나 인간의 가치와 인간만이 지닌 자기표현 능력을 바르게 이해하기 위한 과학적인 연구 방법에 관심을 갖는 학문 분야로서 인간의 사상과 문화에 관해 탐구하는 학문이다. 자연과학과 사회과학이 경험적인 접근을 주로 사용하는 것과는 달리, 분석적이고 비판적이며 사변적인 방법을 폭넓게 사용한다."

# result = aeda(text)
# print(result)
# # 어머니가 ! 집을 , 나가신다

# result = aeda(text, p=0.9, repetition=2)
# print(result)
# # ['! 어머니가 ! 집 ; 을 ? 나가신다', '. 어머니 ? 가 . 집 , 을 , 나가신다']
