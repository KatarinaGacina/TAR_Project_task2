import ast
import re
import os
import csv

from transformers import BertTokenizerFast
from sentence_transformers import SentenceTransformer

import pandas as pd

from features import Features

def get_feature_dicts():
    features = Features()
    features.extract_features("C:\\Users\\User\\Documents\\8.semestar\\APT\\Projekt\\TAR_Project\\data\\features.csv")

    dict_features = {}
    dict_feature_num = {}
    duplicates = {}

    k = 1
    for i,f in zip(features.feature_idx, features.features):
        pronaden = 0
        for key, value in dict_features.items():
            if value == f:
                pronaden = 1
                duplicates[i] = key

        if (pronaden == 0):
            dict_features[i] = f
            dict_feature_num[i] = k
            k += 1
    
    return dict_features, dict_feature_num, duplicates

def preproces_text_and_labels(dict_feature_num, duplicates):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    sen_tran = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    train_df = pd.read_csv("./data/train.csv")
    patient_df = pd.read_csv("./data/patient_notes.csv") 

    train_df = train_df.merge(patient_df, on=['pn_num', 'case_num'], how='left')
    train_df = train_df[train_df['annotation'].apply(lambda x: x != "[]")]

    sentences = []
    tokens = []
    labels = []

    for index, p in patient_df.iterrows():
        label = 131 * [0]

        row_df = pd.DataFrame.from_dict([p])
        feature_df = pd.merge(train_df, row_df[['case_num', 'pn_num']], on=['case_num', 'pn_num'], how='inner')
        
        if not feature_df.empty:
            for locs in feature_df["location"]:
                feat_num = feature_df.loc[feature_df["location"] == locs, "feature_num"].item()
                if feat_num in duplicates.keys():
                    feat_num = dict_feature_num[duplicates[feat_num]]
                else:
                    feat_num = dict_feature_num[feat_num]

                label[feat_num-1] = 1

            sen_enc = sen_tran.encode(p["pn_history"])
            word_tokens = tokenizer.tokenize(p["pn_history"])

            sentences.append(sen_enc)
            tokens.append(word_tokens)
            labels.append(label)

            print("processing...")
            print(index)

    inputs_ids = []
    for t in tokens:
        token_ids = tokenizer.convert_tokens_to_ids(list(t))
        inputs_ids.append(token_ids)

    rows = zip(sentences, inputs_ids, labels)
    with open('dataset_3.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sentence encoded', "Words encoded", 'Labels'])
        writer.writerows(rows)
    
    return sentences, labels