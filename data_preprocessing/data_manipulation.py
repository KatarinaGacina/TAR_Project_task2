import torch

from torch.nn.utils.rnn import pad_sequence

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

def pad_collate_fn_3(batch, pad_index):
    texts, labels, lengths, sentences = zip(*batch)

    texts = pad_sequence(texts, batch_first=True, padding_value=pad_index)

    return texts, torch.tensor(labels), torch.tensor(lengths), torch.tensor(sentences)

def pad_collate_fn_main(bert, batch, pad_index): #bert = BertModel.from_pretrained('bert-base-uncased')
    texts, labels, lengths, sentences = zip(*batch)

    texts = pad_sequence(texts, batch_first=True, padding_value=pad_index)
    attention_masks = texts.ne(pad_index).int() 

    bert_outputs = bert(input_ids=texts, attention_mask=attention_masks)
    embedded_texts = bert_outputs.last_hidden_state.detach().cpu().numpy().tolist()

    max_length = texts.shape[1]
    
    padded_labels = []
    for label, length in zip(labels, lengths):
        padded_label = torch.cat([torch.tensor(label), torch.full((max_length - length,), 133)])
        padded_labels.append(padded_label)
    padded_labels = torch.stack(padded_labels)

    return torch.tensor(embedded_texts), padded_labels, torch.tensor(lengths), torch.tensor(sentences)