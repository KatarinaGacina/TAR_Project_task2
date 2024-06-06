import torch
import torch.nn as nn

from sklearn.model_selection import KFold

from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score

#{'num_epochs': 120, 'lr': 0.001, 'batch_size': 64} 0.01657996716506667

class ModelDetectBase(nn.Module):
    def __init__(self):
        super(ModelDetectBase, self).__init__()

        self.input_dim = 768
        self.output_dim = 131

        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, sentences):
        output_scores = self.fc(sentences)
        
        return output_scores


def eval(val_dataloader, model, device, criterion, num_classes):
    print("Evaluating...")
    model.eval()

    targets_all = []
    predictions_all = []

    correct = 0
    total = 0

    loss_all = 0

    with torch.no_grad():
        for batch_num, (sentences, labels) in enumerate(val_dataloader):
            sentences = sentences.to(device)
            labels = labels.to(device)

            output_scores = model(sentences)
            loss = criterion(output_scores, labels.float())

            loss_all += loss.item()

            predictions = (torch.sigmoid(output_scores) > 0.5).float()

            #print(predictions.shape)
            #print(labels)

            total += labels.size(0) * labels.size(1)
            correct += (predictions == labels).sum().item()

            targets_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
    
    indexes_targets = [{(index+1) for index, element in enumerate(sublist) if element == 1} for sublist in targets_all]
    indexes_predictions = [{(index+1) for index, element in enumerate(sublist) if element == 1} for sublist in predictions_all]

    #print(len(indexes_targets))
    #print(len(indexes_predictions))

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

    """print(f'TP: {tp}')
    print(f'FP: {fp}')
    print(f'FN: {fn}')
    print()"""


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

    #return targets_all, predictions_all
    return macro_f1, micro_f1

def train(train_dataloader, model, device, criterion, optimizer, num_classes):
    model.train()
    
    for batch_num, (sentences, labels) in enumerate(train_dataloader):
        sentences = sentences.to(device)
        labels = labels.to(device)
        
        model.zero_grad()
    
        output_scores = model(sentences)

        loss = criterion(output_scores, labels.float())
        
        loss.backward()
        optimizer.step()

    return loss

def train_with_validation(dataset, params):

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    average_metric = 0
    for _, (train_idx, val_idx) in enumerate(kfold.split(dataset)):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        model = ModelDetectBase().to(device)

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=params["batch_size"],sampler=train_subsampler,
                                                   drop_last=True)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                 sampler=val_subsampler)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

        for _ in range(params["num_epochs"]):
            model.train()
            epoch_loss = 0
            n = 0
            for batch_num, (sentences, labels) in enumerate(train_loader):

                sentences = sentences.to(device)
                labels = labels.to(device)

                outputs = model(sentences)

                optimizer.zero_grad()

                loss = criterion(outputs, labels.float())

                epoch_loss += loss.item()
                n += 1
                loss.backward()
                optimizer.step()

        macro_f1, micro_f1 = eval(val_loader, model, device, criterion, 131)
        average_metric = macro_f1 * micro_f1

    return average_metric