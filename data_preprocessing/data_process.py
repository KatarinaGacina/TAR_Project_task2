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
    sen_transf = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    print(os.getcwd())
    train_df = pd.read_csv("./data/train.csv")
    patient_df = pd.read_csv("./data/patient_notes.csv") 

    train_df = train_df.merge(patient_df, on=['pn_num', 'case_num'], how='left')
    train_df = train_df[train_df['annotation'].apply(lambda x: x != "[]")]

    tekst_originals = []
    tekst_preproccesed = []
    labels = []
    labels_binary = []

    for index, p in patient_df.iterrows():
        mapa = tokenizer(p["pn_history"], return_offsets_mapping=True)
        off_map = mapa["offset_mapping"][1:-1]
        
        tekst_p = tokenizer.tokenize(p["pn_history"])
        #tekst_p = re.split(r'[\s,.!?;:{}()/-]+', p["pn_history"].lower().strip())
        
        label_p = len(tekst_p) * [0]
        label_b = len(tekst_p) * [0]

        row_df = pd.DataFrame.from_dict([p])
        feature_df = pd.merge(train_df, row_df[['case_num', 'pn_num']], on=['case_num', 'pn_num'], how='inner')
        
        if not feature_df.empty:
            for locs in feature_df["location"]:
                feat_num = feature_df.loc[feature_df["location"] == locs, "feature_num"].item()
                if feat_num in duplicates.keys():
                    feat_num = dict_feature_num[duplicates[feat_num]]
                else:
                    feat_num = dict_feature_num[feat_num]

                locs = re.sub(r';', "','", locs)
                locs = [tuple(map(int, pair.split())) for pair in ast.literal_eval(locs)]
                for l in locs:
                    x, y = l[0], l[1]

                    for index, (start_i, end_i) in enumerate(off_map):
                        if start_i >= x and y >= end_i:
                            label_p[index] = feat_num
                            label_b[index] = 1

            tekst_org = sen_transf.encode(p["pn_history"])

            tekst_originals.append(tekst_org)
            tekst_preproccesed.append(tekst_p)
            labels.append(label_p)
            labels_binary.append(label_b)

    inputs_ids = []
    for t in tekst_preproccesed:
        token_ids = tokenizer.convert_tokens_to_ids(list(t))
        inputs_ids.append(token_ids)

    rows = zip(tekst_originals, inputs_ids, labels, labels_binary)
    with open('dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sentence encoded', 'Words ecoded', 'Labels', 'Labels binary'])
        writer.writerows(rows)
    
    return tekst_originals, tekst_preproccesed, labels, labels_binary