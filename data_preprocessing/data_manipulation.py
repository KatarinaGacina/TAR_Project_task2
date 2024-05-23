import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, targets, sentences):
        self.data = data #moze i u modelu!!!
        self.targets = targets
        self.lengths = [len(text) for text in data]
        self.sentences = sentences
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = (torch.tensor(self.data[idx]), self.targets[idx], self.lengths[idx], self.sentences[idx])
        return sample

def pad_collate_fn(batch, pad_index):
    texts, labels, lengths, sentences = zip(*batch)

    texts = pad_sequence(texts, batch_first=True, padding_value=pad_index)
    
    max_length = texts.shape[1]
    
    padded_labels = []
    for label, length in zip(labels, lengths):
        padded_label = torch.cat([torch.tensor(label), torch.full((max_length - length,), 133)])
        padded_labels.append(padded_label)
    padded_labels = torch.stack(padded_labels)

    return texts, padded_labels, torch.tensor(lengths), torch.tensor(sentences)