import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score

class ModelSeqLab(nn.Module):
    def __init__(self, bert_model_name, num_layers=1, output_size=132):
        super(ModelSeqLab, self).__init__()
        self.num_layers = num_layers

        self.hidden_dim = 150

        self.bert_hidden_size = 768
        self.sentence_size = 768

        self.translate_sen = nn.Linear(self.sentence_size, self.hidden_dim)
        
        #self.bert = BertModel.from_pretrained(bert_model_name)
        self.translate_vec = nn.Linear(self.bert_hidden_size, self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        #self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        
        self.fc = nn.Linear(self.hidden_dim * 2, output_size)

    def forward(self, input_ids, lengths, sentences):
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
    
class ModelSeqLab2(nn.Module):
    def __init__(self, vocab_size, num_layers=2, output_size=132):
        super(ModelSeqLab2, self).__init__()
        self.num_layers = num_layers

        self.hidden_dim = 150
        self.sentence_size = 768

        self.translate_sen = nn.Linear(self.sentence_size, self.hidden_dim)
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        
        self.fc = nn.Linear(self.hidden_dim * 2, output_size)

    def forward(self, input_ids, lengths, sentences):
        embeddings = self.embedding_layer(input_ids)

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

    total = 0
    correct = 0

    with torch.no_grad():
        for batch_num, (podaci, targets, lengths, sentences) in enumerate(val_dataloader):
            podaci = podaci.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            sentences = sentences.to(device)

            output_scores = model(podaci, lengths, sentences)
            loss = criterion(output_scores.transpose(1, 2), targets.long())

            loss_all += loss

            predictions = torch.argmax(output_scores.transpose(1, 2), dim=1)

            targets_all.extend(targets.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())

            total += targets.size(0) * targets.size(1)
            correct += (predictions == targets).sum().item()

    indexes_targets = [{x for x in sublist if x not in {0, 133}} for sublist in targets_all]
    indexes_predictions = [{x for x in sublist if x not in {0, 133}} for sublist in predictions_all]

    labels_binary_t = []
    labels_binary_p = []
    for t, p in zip(indexes_targets, indexes_predictions):
        label_t = [0] * 131
        label_p = [0] * 131

        for t_i in t:
            label_t[t_i - 1] = 1

        for p_i in p:
            label_p[p_i - 1] = 1

        labels_binary_p.append(label_p)
        labels_binary_t.append(label_t)

    loss_all /= len(val_dataloader)
    accuracy = 100 * correct / total
    print(f'Validation Loss: {loss_all:.4f}, Accuracy: {accuracy:.2f}%')

    macro_precision = precision_score(labels_binary_t, labels_binary_p, average='macro', zero_division=0)
    macro_recall = recall_score(labels_binary_t, labels_binary_p, average='macro', zero_division=0)
    macro_f1 = f1_score(labels_binary_t, labels_binary_p, average='macro', zero_division=0)

    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    micro_precision = precision_score(labels_binary_t, labels_binary_p, average='micro', zero_division=0)
    micro_recall = recall_score(labels_binary_t, labels_binary_p, average='micro', zero_division=0)
    micro_f1 = f1_score(labels_binary_t, labels_binary_p, average='micro', zero_division=0)

    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")

    conf_matrix = multilabel_confusion_matrix(labels_binary_t, labels_binary_p)
    for i, cm in enumerate(conf_matrix):
        print(f"Confusion Matrix for class {i}:")
        print(cm)
        print()

    tp = 0
    fp = 0
    fn = 0
    for t, p in zip(indexes_targets, indexes_predictions):
        common_elements = t.intersection(p)
        tp += len(common_elements)

        elements_in_set1_not_in_set2 = t - p
        elements_in_set2_not_in_set1 = p - t

        fn += len(elements_in_set1_not_in_set2)
        fp += len(elements_in_set2_not_in_set1)

    print(f'TP: {tp}')
    print(f'FP: {fp}')
    print(f'FN: {fn}')
    print()
    
    return targets_all, predictions_all

def train(train_dataloader, model, device, criterion, optimizer, num_classes):
    model.train()
    
    for batch_num, (podaci, targets, lengths, sentences) in enumerate(train_dataloader):
        podaci = podaci.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)
        sentences = sentences.to(device)
        
        model.zero_grad()
    
        output_scores = model(podaci, lengths, sentences)

        loss = criterion(output_scores.transpose(1, 2), targets.long())
        
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

    return loss