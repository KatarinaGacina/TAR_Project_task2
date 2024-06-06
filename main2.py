import sys
import os

import pandas as pd

import ast

module_path = os.path.abspath(os.path.join('model'))
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.abspath(os.path.join('data_preprocessing'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn

from datasets import CustomDataset2
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from data_procces2 import get_feature_dicts, preproces_text_and_labels

from model2 import ModelDetectBase, train, eval, train_with_validation

def convert_vector_to_list(string_list):
    numbers = string_list.strip('[]').split()
    return [float(number) for number in numbers]

def preprocess_sentences():
    dict_features, dict_feature_num, duplicates = get_feature_dicts()
    preproces_text_and_labels(dict_feature_num, duplicates)

def main():
    num_classes = 131

    df = pd.read_csv('dataset_baseline.csv')

    encoded_sentences = df['Sentence encoded'].apply(lambda x: convert_vector_to_list(x)).tolist()
    labels = df['Labels'].apply(ast.literal_eval)

    custom_dataset = CustomDataset2(encoded_sentences, labels)

    train_size = int(0.8 * len(custom_dataset))
    val_size = len(custom_dataset) - train_size
    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = ModelDetectBase().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training...")
    for epoch in range(100):
        loss = train(train_dataloader, model, device, criterion, optimizer, num_classes)

        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    torch.save(model.state_dict(), 'model_baseline.pth')

    targets_all, predictions_all = eval(val_dataloader, model, device, criterion, num_classes)

def main_val():
    df = pd.read_csv('dataset_baseline.csv')

    encoded_sentences = df['Sentence encoded'].apply(lambda x: convert_vector_to_list(x)).tolist()
    labels = df['Labels'].apply(ast.literal_eval)

    dataset = CustomDataset2(encoded_sentences, labels)

    num_epochs = [50, 100, 120]
    lrs = [1e-2, 1e-3, 1e-4]
    batch_sizes = [10, 32, 64]

    i = 0
    best_params = {}
    best_average_metric = 0

    for epochs in num_epochs:
        for lr in lrs:
            for batch_size in batch_sizes:
                params = {"num_epochs": epochs,
                        "lr": lr, "batch_size": batch_size}
                average_metric = train_with_validation(dataset, params)
                if average_metric > best_average_metric:
                    best_average_metric = average_metric
                    best_params = params
                print(i)
                i += 1

    print(best_params, best_average_metric)

if __name__ == "__main__":
    torch.manual_seed(33533)
    main()
    #main_val()
    #preprocess_sentences()