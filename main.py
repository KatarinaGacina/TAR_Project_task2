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

from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader

from model import ModelSeqLab, train, eval

from data_process import encode_data
from data_manipulation import CustomDataset, pad_collate_fn

#from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

def convert_vector_to_list(string_list):
    numbers = string_list.strip('[]').split()
    return [float(number) for number in numbers]


def main():
    #dict_features, dict_feature_num, duplicates = get_feature_dicts()
    #preproces_text_and_labels(dict_feature_num, duplicates)

    df = pd.read_csv('dataset.csv')
    tekst_originals = df['Sentence encoded'].apply(lambda x: convert_vector_to_list(x)).tolist()
    tekst_preproccesed = df['Words ecoded'].apply(ast.literal_eval)
    labels = df['Labels'].apply(ast.literal_eval)
    labels_binary = df['Labels binary'].apply(ast.literal_eval)

    num_classes = 132
    #num_classes = 2

    hidden_dim = 150
    output_dim = num_classes

    pad_id = BertTokenizerFast.from_pretrained("bert-base-uncased").pad_token_id
    processed_data = encode_data(tekst_preproccesed, pad_id)

    custom_dataset = CustomDataset(processed_data, labels, tekst_originals) #labels_binary
    #train_size = int(0.8 * len(custom_dataset))
    #val_size = len(custom_dataset) - train_size
    #train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

    #train_dataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, collate_fn=lambda batch: pad_collate_fn(batch, pad_index=pad_id))
    #val_dataloader = DataLoader(dataset=val_dataset, batch_size=20, shuffle=True, collate_fn=lambda batch: pad_collate_fn(batch, pad_index=pad_id))

    train_dataloader = DataLoader(dataset=custom_dataset, batch_size=10, shuffle=True, collate_fn=lambda batch: pad_collate_fn(batch, pad_index=pad_id))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device="cpu"
    print(device)

    model = ModelSeqLab('bert-base-uncased', hidden_dim, output_dim).to(device)

    weights = [100] * 132
    weights[0] = 1

    weight = torch.tensor(weights)
    criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training...")
    for epoch in range(50):
        loss = train(train_dataloader, model, device, criterion, optimizer, num_classes)

        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    torch.save(model.state_dict(), 'model.pth')

    loss_all, targets_all, predictions_all = eval(train_dataloader, model, device, criterion, num_classes)

    print(loss_all)

    len_all = 0
    pogodak = 0
    for t, p in targets_all, predictions_all:
        if (t != 133):
            loss_all += 1

            if (t == p):
                pogodak += 1
    print(pogodak)
    print(len_all)

    #print(targets_all)
    #print(predictions_all)

if __name__ == "__main__":
    main()