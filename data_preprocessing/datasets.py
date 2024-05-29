import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset): #lstm
    def __init__(self, data, targets, sentences):
        self.data = data
        self.targets = targets
        self.lengths = [len(text) for text in data]
        self.sentences = sentences
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = (torch.tensor(self.data[idx]), self.targets[idx], self.lengths[idx], self.sentences[idx])
        return sample

class CustomDataset2(Dataset): #fc
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
        
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sample = (torch.tensor(self.sentences[idx]), torch.tensor(self.labels[idx]))
        return sample