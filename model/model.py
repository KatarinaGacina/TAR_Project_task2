import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import BertModel


class ModelSeqLab(nn.Module):
    def __init__(self, bert_model_name, num_layers=1, output_size=132):
        super(ModelSeqLab, self).__init__()
        self.num_layers = num_layers
        self.pad_index = 133

        self.hidden_dim = 150
        self.bert_hidden_size = 768 #self.bert.config.hidden_size
        self.sentence_size = 384

        self.translate_sen = nn.Linear(self.sentence_size, self.hidden_dim)
        
        #self.bert = BertModel.from_pretrained(bert_model_name)
        
        self.translate_vec = nn.Linear(self.bert_hidden_size, self.hidden_dim) #je li bolje koristiti PCA?
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        
        #self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        
        self.fc = nn.Linear(self.hidden_dim * 2, output_size)

    def forward(self, input_ids, lengths, sentences):
        #attention_mask = input_ids.ne(self.pad_index)
        #bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #embeddings_org = bert_outputs.last_hidden_state

        #embeddings = self.translate_vec(embeddings_org)
        embeddings = self.translate_vec(input_ids)

        batch_size  = input_ids.shape[0]
        sentences = self.translate_sen(sentences)
        h_01 = sentences.unsqueeze(0).repeat(self.num_layers * 2, 1, 1).to(input_ids.device)
        c_01 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim).to(input_ids.device)

        packed_input = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input, (h_01, c_01))
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        output_scores = self.fc(output)
        return output_scores

def eval(val_dataloader, model, device, criterion, num_classes):
    print("Evaluating...")
    model.eval()

    targets_all = []
    predictions_all = []
    loss_all = 0

    with torch.no_grad():
        for batch_num, (podaci, targets, lengths, sentences) in enumerate(val_dataloader):
            podaci = podaci.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            sentences = sentences.to(device)

            output_scores = model(podaci, lengths, sentences)
            loss = criterion(output_scores.view(-1, num_classes), targets.long().view(-1))

            loss_all += loss

            predictions = torch.argmax(output_scores.view(-1, num_classes), dim=1)

            targets_all.extend(targets.view(-1).cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
    
    return loss_all, targets_all, predictions_all

def train(train_dataloader, model, device, criterion, optimizer, num_classes):
    model.train()
    
    for batch_num, (podaci, targets, lengths, sentences) in enumerate(train_dataloader):
        #print(batch_num)
        podaci = podaci.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)
        sentences = sentences.to(device)
        
        model.zero_grad()
    
        output_scores = model(podaci, lengths, sentences)

        loss = criterion(output_scores.view(-1, num_classes), targets.long().view(-1))
        
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

    return loss