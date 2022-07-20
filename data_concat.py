import pandas as pd

# cola_train_data = pd.read_csv("data/CoLA_data/NIKL_CoLA_train.tsv", delimiter="\t")
# cola_test_data = pd.read_csv("data/CoLA_data/NIKL_CoLA_dev.tsv", delimiter="\t")

train_data = pd.read_csv("data/BoolQ_data/boolq_train_eng.tsv", delimiter="\t")
test_data = pd.read_csv("data/BoolQ_data/boolq_dev_eng.tsv", delimiter="\t")
# val_data = pd.read_csv("data/BoolQ_data/bookMRC_dev1000_trim.tsv", delimiter="\t")
# print(copa_test_data['acceptability_label'])
# print(copa_test_data['question'])

# print(train_data['Answer(FALSE = 0, TRUE = 1)'].info())
# print(train_data[:50000].describe())


# copa_train_data['acceptability_label'].replace(0, 1, inplace=True)

# copa_test_data['Answer'] = copa_test_data['Answer'] + 1
# print(copa_train_data['acceptability_label'])

# print(copa_test_data['Answer'])

concat_data = pd.concat([train_data, test_data], axis=0)
# copa_train_test_data = pd.concat([copa_train_data, copa_test_data], axis=0)

# copa_train_test_data = copa_train_test_data.reset_index()
concat_data.to_csv('data/BoolQ_data/boolq_all_eng.tsv', index=False, sep="\t")



# train_data=train_data.sample(frac=1).reset_index(drop=True)
# train_data.to_csv('data/BoolQ_data/boolq_post_train_data.tsv', index=False, sep="\t")

# test_data=test_data.sample(frac=1).reset_index(drop=True)
# test_data.to_csv('data/BoolQ_data/boolq_post_test_data.tsv', index=False, sep="\t")

# val_data=val_data.sample(frac=1).reset_index(drop=True)
# val_data.to_csv('data/BoolQ_data/boolq_post_val_data.tsv', index=False, sep="\t")



# concat_data = pd.read_csv("data/CoLA_data/WiC_train_dev_aug.tsv", delimiter="\t")
# print(concat_data.isnull().sum())

