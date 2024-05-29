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

from model2 import ModelDetectBase, train, eval

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
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=20, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = ModelDetectBase().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training...")
    for epoch in range(50):
        loss = train(train_dataloader, model, device, criterion, optimizer, num_classes)

        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    torch.save(model.state_dict(), 'model_baseline.pth')

    targets_all, predictions_all = eval(val_dataloader, model, device, criterion, num_classes)

if __name__ == "__main__":
    torch.manual_seed(42)
    main()
    #preprocess_sentences()