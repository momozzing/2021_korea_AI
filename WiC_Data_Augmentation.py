import pandas as pd
from tqdm import tqdm
from koeda import AEDA


train_data = pd.read_csv("data/WiC_data/NIKL_SKT_WiC_Train.tsv", delimiter="\t")

aeda = AEDA(
    morpheme_analyzer="Mecab", punc_ratio=0.1, punctuations=[".", ",", "!", "?", ";", ":"]
)

all_list = []

for idx in tqdm(range(len(train_data))):
    target, sen1, sen2, ans, s1, e1, s2, e2 = train_data["Target"][idx], train_data["SENTENCE1"][idx], train_data["SENTENCE2"][idx], train_data["ANSWER"][idx], train_data["start_s1"][idx], train_data["end_s1"][idx], train_data["start_s2"][idx], train_data["end_s2"][idx] 

    sen1_result = aeda(sen1)
    sen2_result = aeda(sen2)

    all_list.append([target] + [sen1_result] + [sen2_result] + [ans] + [s1] + [e1] + [s2] + [e2])
    
 

df = pd.DataFrame(all_list, columns=['Target', 'SENTENCE1', 'SENTENCE2', 'ANSWER', 'start_s1', 'end_s1', 'start_s2', 'end_s2'])

# print(df)

aug_df = pd.concat([train_data, df])

aug_df.to_csv('data/WiC_data/WiC_Aug_train.tsv', index=False, sep="\t")

print(len(aug_df['SENTENCE2']))
###################################### EDA
# from koeda import EDA

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