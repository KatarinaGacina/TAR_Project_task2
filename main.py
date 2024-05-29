import sys
import os

import pandas as pd

import ast
import csv

module_path = os.path.abspath(os.path.join('model'))
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.abspath(os.path.join('data_preprocessing'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn

from transformers import BertTokenizerFast, BertModel

from datasets import CustomDataset
from data_manipulation import pad_collate_fn, pad_collate_fn_main
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from data_process import get_feature_dicts, preproces_text_and_labels

from model import ModelSeqLab, ModelSeqLab2, train, eval

def convert_vector_to_list(string_list):
    numbers = string_list.strip('[]').split()
    return [float(number) for number in numbers]

def preprocess():
    dict_features, dict_feature_num, duplicates = get_feature_dicts()
    preproces_text_and_labels(dict_feature_num, duplicates)

def main():
    #df = pd.read_csv('dataset3.csv')
    #tekst_originals = df['Sentence encoded'].apply(ast.literal_eval)
    df = pd.read_csv('dataset.csv')
    tekst_originals = df['Sentence encoded'].apply(lambda x: convert_vector_to_list(x)).tolist()
    processed_data = df['Words ecoded'].apply(ast.literal_eval)
    labels = df['Labels'].apply(ast.literal_eval)

    num_classes = 132

    hidden_dim = 150
    output_dim = num_classes

    pad_id = BertTokenizerFast.from_pretrained("bert-base-uncased").pad_token_id
    bert = BertModel.from_pretrained('bert-base-uncased')

    custom_dataset = CustomDataset(processed_data, labels, tekst_originals)

    train_size = int(0.8 * len(custom_dataset))
    val_size = len(custom_dataset) - train_size
    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, collate_fn=lambda batch: pad_collate_fn_main(bert, batch, pad_index=pad_id))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=20, shuffle=True, collate_fn=lambda batch: pad_collate_fn_main(bert, batch, pad_index=pad_id))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = ModelSeqLab('bert-base-uncased', hidden_dim, output_dim).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training...")
    for epoch in range(10):
        loss = train(train_dataloader, model, device, criterion, optimizer, num_classes)

        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    torch.save(model.state_dict(), 'model_org_bert.pth')

    targets_all, predictions_all = eval(val_dataloader, model, device, criterion, num_classes)

def main2():
    df = pd.read_csv('dataset.csv')
    tekst_originals = df['Sentence encoded'].apply(lambda x: convert_vector_to_list(x)).tolist()
    processed_data = df['Words ecoded'].apply(ast.literal_eval)
    labels = df['Labels'].apply(ast.literal_eval)

    num_classes = 132

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    pad_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)

    custom_dataset = CustomDataset(processed_data, labels, tekst_originals)

    train_size = int(0.8 * len(custom_dataset))
    val_size = len(custom_dataset) - train_size
    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, collate_fn=lambda batch: pad_collate_fn(batch, pad_index=pad_id))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=20, shuffle=True, collate_fn=lambda batch: pad_collate_fn(batch, pad_index=pad_id))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = ModelSeqLab2(vocab_size).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training...")
    for epoch in range(50):
        loss = train(train_dataloader, model, device, criterion, optimizer, num_classes)

        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    torch.save(model.state_dict(), 'model_embedding.pth')

    targets_all, predictions_all = eval(val_dataloader, model, device, criterion, num_classes)

    targets_list = [item for sublist in targets_all for item in sublist]
    predictions_list = [item for sublist in predictions_all for item in sublist]
    print()
    len_all = 0
    pogodak = 0
    len_nula = 0
    pogodak_nula = 0

    for t, p in zip(targets_list, predictions_list):
        if (t != 133):
            if (t == 0):
                len_nula += 1
            else:
                len_all += 1

            if (t == p):
                if (t == 0):
                    pogodak_nula += 1
                else:
                    pogodak += 1

    print(pogodak)
    print(len_all)
    print(pogodak_nula)
    print(len_nula)

if __name__ == "__main__":
    torch.manual_seed(42)
    #preprocess()
    #main()
    main2()