import sys
import os

import pandas as pd

import itertools

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
from data_manipulation import pad_collate_fn_3
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from data_procces3 import get_feature_dicts, preproces_text_and_labels

from model3 import ModelSeqLab3, train3, eval3

def convert_vector_to_list(string_list):
    numbers = string_list.strip('[]').split()
    return [float(number) for number in numbers]

def preprocess():
    dict_features, dict_feature_num, duplicates = get_feature_dicts()
    preproces_text_and_labels(dict_feature_num, duplicates)

def main():
    df = pd.read_csv('dataset_3.csv')
    #tekst_originals = df['Sentence encoded'].apply(ast.literal_eval)
    tekst_originals = df['Sentence encoded'].apply(lambda x: convert_vector_to_list(x)).tolist()
    processed_data = df['Words encoded'].apply(ast.literal_eval)
    labels = df['Labels'].apply(ast.literal_eval)

    num_classes = 131

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    pad_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)

    custom_dataset = CustomDataset(processed_data, labels, tekst_originals)

    train_size = int(0.8 * len(custom_dataset))
    val_size = len(custom_dataset) - train_size
    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

    layers = [1, 2]
    num_epochs = [50, 100, 120]
    lrs = [1e-2, 1e-3, 1e-4]
    batch_sizes = [10, 32, 64]
    dropouts = [0, 0.2]
    hidden_dims = [150, 300]

    parameter_combinations = itertools.product(layers, num_epochs, lrs, batch_sizes, dropouts, hidden_dims)
    for parameters in parameter_combinations:
        num_layer, num_epoch, lr_rate, batch_size, dropout, hidden_dim = parameters
        print(f"num_layer: {num_layer}, num_epoch: {num_epoch}, lr_rate: {lr_rate}, batch_size: {batch_size}, dropout: {dropout}, hidden_dim: {hidden_dim}")

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: pad_collate_fn_3(batch, pad_index=pad_id))
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=20, shuffle=True, collate_fn=lambda batch: pad_collate_fn_3(batch, pad_index=pad_id))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        model = ModelSeqLab3(vocab_size, hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layer).to(device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

        print("Training...")
        for epoch in range(num_epoch):
            loss = train3(train_dataloader, model, device, criterion, optimizer, num_classes)

            if epoch % 1 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
        #torch.save(model.state_dict(), 'model_3.pth')

        targets_all, predictions_all = eval3(val_dataloader, model, device, criterion, num_classes)
        print()

if __name__ == "__main__":
    torch.manual_seed(42)
    #preprocess()
    main()