import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score

class ModelSeqLab3(nn.Module):
    def __init__(self, vocab_size, hidden_dim=150, dropout=0.2, num_layers=2, output_size=131):
        super(ModelSeqLab3, self).__init__()
        self.num_layers = num_layers

        self.hidden_dim = hidden_dim
        self.sentence_size = 768

        self.translate_sen = nn.Linear(self.sentence_size, self.hidden_dim)
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        
        self.fc = nn.Linear(self.hidden_dim * 2, output_size)

    def forward(self, input, lengths, sentences):
        embeddings = self.embedding_layer(input)

        batch_size  = input.shape[0]
        sentences = self.translate_sen(sentences)
        h_01 = sentences.unsqueeze(0).repeat(self.num_layers * 2, 1, 1).to(input.device)
        c_01 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim).to(input.device)

        packed_input = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input, (h_01, c_01))
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        output_scores = self.fc(output[:, -1, :])
        return output_scores

def eval3(val_dataloader, model, device, criterion, num_classes):
    print("Evaluating...")
    model.eval()

    targets_all = []
    predictions_all = []

    correct = 0
    total = 0

    loss_all = 0

    with torch.no_grad():
        for batch_num, (podaci, labels, lengths, sentences) in enumerate(val_dataloader):
            sentences = sentences.to(device)
            labels = labels.to(device)
            podaci = podaci.to(device)
            lengths = lengths.to(device)

            output_scores = model(podaci, lengths, sentences)
            loss = criterion(output_scores, labels.float())

            loss_all += loss.item()

            predictions = (torch.sigmoid(output_scores) > 0.5).float()

            total += labels.size(0) * labels.size(1)
            correct += (predictions == labels).sum().item()

            targets_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
    
    print(torch.tensor(targets_all).shape)
    indexes_targets = [{(index+1) for index, element in enumerate(sublist) if element == 1} for sublist in targets_all]
    indexes_predictions = [{(index+1) for index, element in enumerate(sublist) if element == 1} for sublist in predictions_all]

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

    loss_all /= len(val_dataloader)
    accuracy = 100 * correct / total
    print(f'Validation Loss: {loss_all:.4f}, Accuracy: {accuracy:.2f}%')

    macro_precision = precision_score(targets_all, predictions_all, average='macro', zero_division=0)
    macro_recall = recall_score(targets_all, predictions_all, average='macro', zero_division=0)
    macro_f1 = f1_score(targets_all, predictions_all, average='macro', zero_division=0)

    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    micro_precision = precision_score(targets_all, predictions_all, average='micro', zero_division=0)
    micro_recall = recall_score(targets_all, predictions_all, average='micro', zero_division=0)
    micro_f1 = f1_score(targets_all, predictions_all, average='micro', zero_division=0)

    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")

    """conf_matrix = multilabel_confusion_matrix(targets_all, predictions_all)
    for i, cm in enumerate(conf_matrix):
        print(f"Confusion Matrix for class {i}:")
        print(cm)
        print()"""

    return targets_all, predictions_all

def train3(train_dataloader, model, device, criterion, optimizer, num_classes):
    model.train()
    
    for batch_num, (podaci, labels, lengths, sentences) in enumerate(train_dataloader):
        sentences = sentences.to(device)
        labels = labels.to(device)
        podaci = podaci.to(device)
        lengths = lengths.to(device)
        
        model.zero_grad()

        output_scores = model(podaci, lengths, sentences)

        loss = criterion(output_scores, labels.float())
        
        loss.backward()
        optimizer.step()

    return loss