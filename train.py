import math
import string
import random
import warnings
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix
import os
from imblearn.over_sampling import SMOTE, ADASYN
import wandb
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)  # Set start method to 'spawn' at the very beginning


import torch
from torch import mps
import torch.nn as nn
import torch.nn.functional as F # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from torch.nn.utils.rnn import pad_sequence
from transformers import logging
from torch.optim import AdamW
import torch


from transformers import RobertaTokenizer, RobertaModel
from translate import Translator
import numpy as np
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt


# Suppress all warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = 'mps' if mps.is_available() else 'cpu'
emb_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#emb_model = RobertaModel.from_pretrained('distilroberta-base', output_hidden_states=True)
emb_model = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True)


num_epochs = 20
dropout = 0.2


param_grid = {
    'hidden_dim': [128, 256, 512],
    'learning_rate': [ 0.01, 0.05, 0.001, 0.005,0.1,0.5],
    'batch_size': [32, 64, 128]

}

def early_stopping(val_losses, patience=5):
    if len(val_losses) <= patience:
        return False
    return all(val_losses[-i] >= val_losses[-patience-1] for i in range(1, patience+1))

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, gamma=3, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, max_aspects=6, dropout=0.3):
        super(ContrastiveModel, self).__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.aspects_fc = nn.ModuleList([nn.Linear(hidden_dim // 2, n_classes) for _ in range(max_aspects)])

    def forward(self, embeddings):
        # Convert embeddings to Float tensor
        embeddings = embeddings.float()

        x = self.dropout1(embeddings)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout3(x)
        aspect_probabilities = []
        for fc in self.aspects_fc:
            logits = fc(x)
            probabilities = F.softmax(logits, dim=1)
            aspect_probabilities.append(probabilities)
        return aspect_probabilities
    
class ContrastiveDataset(Dataset):
    def __init__(self, positive_pairs, negative_pairs, positive_sentiments, negative_sentiments):
        self.pairs = positive_pairs + negative_pairs
        self.labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
        self.sentiments = preprocess_sentiments(positive_sentiments + negative_sentiments)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        tensor1, tensor2 = self.pairs[idx]
        label = self.labels[idx]
        sentiment = self.sentiments[idx]
        return (tensor1, tensor2), label, sentiment

def preprocess_sentiments(sentiments):
    if isinstance(sentiments, list):
        sentiments = torch.tensor(sentiments, dtype=torch.long)
    sentiments = sentiments.clone()
    sentiments[sentiments == 2] = 1
    return sentiments

    
def custom_collate_fn(batch, max_seq_length=512):
    tensor_pairs, labels, sentiments = zip(*batch)
    tensor1_list, tensor2_list = zip(*tensor_pairs)

    # Determine the device from the first tensor in the list
    device = tensor1_list[0].device

    # Remove extra dimensions from tensors
    tensor1_list = [tensor.squeeze().long() for tensor in tensor1_list]
    tensor2_list = [tensor.squeeze().long() for tensor in tensor2_list]

    # Replace invalid indices with a valid token index (e.g., 0)
    tensor1_list = [torch.clamp(tensor, min=0) for tensor in tensor1_list]
    tensor2_list = [torch.clamp(tensor, min=0) for tensor in tensor2_list]

    # Pad or truncate the tensors to the maximum sequence length
    tensor1_list = [tensor[:max_seq_length] for tensor in tensor1_list]
    tensor2_list = [tensor[:max_seq_length] for tensor in tensor2_list]

    # Pad the tensors to the maximum sequence length
    tensor1_list = [torch.cat([tensor, torch.zeros(max_seq_length - tensor.size(0), dtype=torch.long).to(device)], dim=0) for tensor in tensor1_list]
    tensor2_list = [torch.cat([tensor, torch.zeros(max_seq_length - tensor.size(0), dtype=torch.long).to(device)], dim=0) for tensor in tensor2_list]

    # Stack the padded tensors
    tensor1 = torch.stack(tensor1_list, dim=0)
    tensor2 = torch.stack(tensor2_list, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32).to(device)
    sentiments = pad_sequence([torch.tensor(s).to(device) for s in sentiments], batch_first=True, padding_value=-1)

    return (tensor1, tensor2), labels, sentiments

def train(model, dataloader, device, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    all_aspect_probabilities = []
    for batch_idx, (inputs, labels, sentiments) in enumerate(dataloader):
        inputs1, inputs2 = inputs
        inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
        labels = labels.to(device)
        sentiments = sentiments.to(device)

        sentiments=preprocess_sentiments(sentiments)
        optimizer.zero_grad()

        outputs1 = model(inputs1)
        outputs2 = model(inputs2)

        loss = 0
        num_aspects = sentiments.size(1)
        aspect_probabilities = []
        
        for i in range(num_aspects):
            aspect_loss = criterion(outputs1[i], sentiments[:, i]) + \
                          criterion(outputs2[i], sentiments[:, i])
            loss += aspect_loss

            # Calculate probabilities for aspect i and add to the list for the batch
            probs = F.softmax(outputs1[i], dim=1).cpu().detach().numpy()
            aspect_probabilities.append(probs)
            batch_idx=+1

            # Print aspect probabilities for the current aspect
            #print(f"Batch {batch_idx}, Aspect {i+1} Probabilities: {probs}")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_aspect_probabilities.append(aspect_probabilities)

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    average_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}], Average Loss: {average_loss:.4f}")
    return average_loss, all_aspect_probabilities

def validate(model, dataloader, device, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No need to track gradients during validation
        for inputs, labels, sentiments in dataloader:
            inputs1, inputs2 = inputs
            inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
            labels = labels.to(device)
            sentiments = sentiments.to(device)
            sentiments=preprocess_sentiments(sentiments)
            outputs1 = model(inputs1.squeeze(1))
            outputs2 = model(inputs2.squeeze(1))

            loss = 0
            num_aspects = sentiments.size(1)
            for i in range(num_aspects):
                aspect_loss = criterion(outputs1[i], sentiments[:, i]) + \
                              criterion(outputs2[i], sentiments[:, i])
                loss += aspect_loss

            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {average_loss:.4f}")
    return average_loss

def expand_labels(tensors, aspects, polarities):
    flat_tensors, flat_aspects, flat_polarities = [], [], []
    for i in range(len(tensors)):
        for tensor in tensors[i]:
            flat_tensors.append(tensor)
            flat_aspects.append(aspects[i])
            flat_polarities.append(polarities[i])
    return flat_tensors, flat_aspects, flat_polarities

def sample_pairs(pairs, sentiments, max_samples):
    valid_pairs = [(pair, sentiment) for pair, sentiment in zip(pairs, sentiments) if pair[0].size(1) > 0 and pair[1].size(1) > 0]
    if len(valid_pairs) == 0:
        return [], []
    if len(valid_pairs) <= max_samples:
        sampled_pairs, sampled_sentiments = zip(*valid_pairs)
        return list(sampled_pairs), list(sampled_sentiments)
    sampled_indices = random.sample(range(len(valid_pairs)), max_samples)
    sampled_pairs = [valid_pairs[i][0] for i in sampled_indices]
    sampled_sentiments = [valid_pairs[i][1] for i in sampled_indices]
    return sampled_pairs, sampled_sentiments

def generate_sentiment_based_pairs(sentence_tensors, aspect_labels, polarity_labels, max_total_pairs=1600):
    all_positive_pairs, all_negative_pairs, positive_sentiments, negative_sentiments = [], [], [], []
    
    print("Generating sentiment-based pairs...")
    smote = SMOTE(random_state=42)

    flat_tensors, flat_aspects, flat_polarities = expand_labels(sentence_tensors, aspect_labels, polarity_labels)

    #print(flat_polarities)
    # Separate positive and negative polarities
    positive_indices = [i for i, polarity in enumerate(flat_polarities) if list(polarity.values())[0] == 2]
    negative_indices = [i for i, polarity in enumerate(flat_polarities) if list(polarity.values())[0] == 0]
    
    #print(negative_indices)
    # Generate positive pairs from positive-positive sentiment aspect windows
    for i in positive_indices:
        for j in positive_indices:
            if i != j:
                aspect_i = list(flat_aspects[i].values())[0]
                aspect_j = list(flat_aspects[j].values())[0]
                if aspect_i == aspect_j:
                    all_positive_pairs.append((flat_tensors[i], flat_tensors[j]))
                    positive_sentiments.append((1, 1))

    # Generate positive pairs from negative-negative sentiment aspect windows
    for i in negative_indices:
        for j in negative_indices:
            if i != j:
                aspect_i = list(flat_aspects[i].values())[0]
                aspect_j = list(flat_aspects[j].values())[0]
                if aspect_i == aspect_j:
                    all_positive_pairs.append((flat_tensors[i], flat_tensors[j]))
                    positive_sentiments.append((0, 0))

    # Generate negative pairs from positive-negative sentiment aspect windows
    for i in positive_indices:
        for j in negative_indices:
            aspect_i = list(flat_aspects[i].values())[0]
            aspect_j = list(flat_aspects[j].values())[0]
            if aspect_i == aspect_j:
                all_negative_pairs.append((flat_tensors[i], flat_tensors[j]))
                negative_sentiments.append((1, 0))

    # print("Going in sampled pairs...")
    # print("Len of all_positive_pairs:", len(all_positive_pairs))
    # print("Len of all_negative_pairs:", len(all_negative_pairs))
    # print("Tensor sizes of positive pairs:")

    # Sample positive pairs, ensuring balance between positive-positive and negative-negative origins
    max_pairs_per_origin = max_total_pairs // 2
    sampled_positives_pos_pos, sampled_positive_sentiments_pos_pos = sample_pairs(
        [pair for pair, sentiment in zip(all_positive_pairs, positive_sentiments) if sentiment == (1, 1)],
        [sentiment for sentiment in positive_sentiments if sentiment == (1, 1)],
        max_pairs_per_origin
    )
    sampled_positives_neg_neg, sampled_positive_sentiments_neg_neg = sample_pairs(
        [pair for pair, sentiment in zip(all_positive_pairs, positive_sentiments) if sentiment == (0, 0)],
        [sentiment for sentiment in positive_sentiments if sentiment == (0, 0)],
        max_pairs_per_origin
    )
    sampled_positives = sampled_positives_pos_pos + sampled_positives_neg_neg
    sampled_positive_sentiments = sampled_positive_sentiments_pos_pos + sampled_positive_sentiments_neg_neg

    sampled_negatives, sampled_negative_sentiments = sample_pairs(all_negative_pairs, negative_sentiments, max_total_pairs)

    # print("Final Len of sampled positives:", len(sampled_positives))
    # print("Final Len of sampled negatives:", len(sampled_negatives))
    # print("Final Len of sampled positive sentiments:", len(sampled_positive_sentiments))
    # print("Final Len of sampled negative sentiments:", len(sampled_negative_sentiments))
    # print("\n")
    return sampled_positives, sampled_negatives, sampled_positive_sentiments, sampled_negative_sentiments

def create_validation_dataloader(val_data, dataset_output_dir, batch_size, max_seq_length=768):
    val_aspect_labels = []
    val_polarity_labels = []
    val_sentence_tensors = []

    for sentence_id, sentence_data in val_data.groupby('id'):
        for _, aspect_data in sentence_data.iterrows():
            aspect_term = aspect_data['Aspect Term'].lower()
            aspect_num = aspect_data['Aspect Number']
            try:
                aspect_tensors = torch.load(f'{dataset_output_dir}/{sentence_id}-{aspect_num}.pk')
            except FileNotFoundError:
                #print(f"File not found: {dataset_output_dir}/{sentence_id}-{aspect_num}.pk")
                continue

            val_aspect_labels.append({aspect_num: aspect_term})
            val_polarity_labels.append({aspect_num: aspect_data['polarity_numeric']})
            val_sentence_tensors.append(aspect_tensors)

    val_positive_pairs, val_negative_pairs, val_positive_sentiments, val_negative_sentiments = generate_sentiment_based_pairs(
        val_sentence_tensors, val_aspect_labels, val_polarity_labels
    )

    val_dataset = ContrastiveDataset(val_positive_pairs, val_negative_pairs, val_positive_sentiments, val_negative_sentiments)
    
    def val_collate_fn(batch, max_seq_length=512):
       
        tensor_pairs, labels, sentiments = zip(*batch)
        tensor1_list, tensor2_list = zip(*tensor_pairs)

        # Determine the device from the first tensor in the list
        device = tensor1_list[0].device

        # Remove extra dimensions from tensors
        tensor1_list = [tensor.squeeze().long() for tensor in tensor1_list]
        tensor2_list = [tensor.squeeze().long() for tensor in tensor2_list]

        # Replace invalid indices with a valid token index (e.g., 0)
        tensor1_list = [torch.clamp(tensor, min=0) for tensor in tensor1_list]
        tensor2_list = [torch.clamp(tensor, min=0) for tensor in tensor2_list]

        # Pad or truncate the tensors to the maximum sequence length
        tensor1_list = [tensor[:max_seq_length] for tensor in tensor1_list]
        tensor2_list = [tensor[:max_seq_length] for tensor in tensor2_list]

        # Pad the tensors to the maximum sequence length
        tensor1_list = [torch.cat([tensor, torch.zeros(max_seq_length - tensor.size(0), dtype=torch.long).to(device)], dim=0) for tensor in tensor1_list]
        tensor2_list = [torch.cat([tensor, torch.zeros(max_seq_length - tensor.size(0), dtype=torch.long).to(device)], dim=0) for tensor in tensor2_list]

        # Stack the padded tensors
        tensor1 = torch.stack(tensor1_list, dim=0)
        tensor2 = torch.stack(tensor2_list, dim=0)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        sentiments = pad_sequence([torch.tensor(s).to(device) for s in sentiments], batch_first=True, padding_value=-1)

        return (tensor1, tensor2), labels, sentiments
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_collate_fn)
    
    if len(val_dataloader) == 0:
        raise ValueError("Validation dataloader is empty. Please check the validation data and the dataloader creation process.")
    
    return val_dataloader

def train_epoch(model, train_data, dataset_output_dir, device, optimizer, criterion, epoch, batch_size, accumulation_steps=4):
    model.train()
    total_loss = 0
    num_batches = 0

    train_aspect_labels = []
    train_polarity_labels = []
    train_sentence_tensors = []

    for sentence_id, sentence_data in train_data.groupby('id', dropna=False):
        for _, aspect_data in sentence_data.iterrows():
            aspect_term = aspect_data.get('Aspect Term', '').lower()
            aspect_num = aspect_data.get('Aspect Number')

            try:
                aspect_tensors = torch.load(f'{dataset_output_dir}/{sentence_id}-{aspect_num}.pk')
            except (FileNotFoundError, RuntimeError):
                print(f"Error loading file: {dataset_output_dir}/{sentence_id}-{aspect_num}.pk. Skipping.")
                continue

            train_aspect_labels.append({aspect_num: aspect_term})
            train_polarity_labels.append({aspect_num: aspect_data['polarity_numeric']})
            train_sentence_tensors.append(aspect_tensors)

    print(f"Generating pairs for epoch {epoch+1}...")
    train_positive_pairs, train_negative_pairs, train_positive_sentiments, train_negative_sentiments = generate_sentiment_based_pairs(
        train_sentence_tensors, train_aspect_labels, train_polarity_labels
    )

    train_dataset = ContrastiveDataset(train_positive_pairs, train_negative_pairs, train_positive_sentiments, train_negative_sentiments)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, collate_fn=custom_collate_fn)

    print(f"Training epoch {epoch+1}...")
    optimizer.zero_grad()
    for batch_idx, (inputs, labels, sentiments) in enumerate(train_dataloader):
        inputs1, inputs2 = inputs
        inputs1, inputs2 = inputs1.to(device), inputs2.to(device)

        labels = labels.to(device)
        sentiments = preprocess_sentiments(sentiments.to(device))
         # Negative: 0, Neutral: 1, Positive: 2, Other: 3
        
       
        
        outputs1 = model(inputs1)
        outputs2 = model(inputs2)

        loss = 0
        num_aspects = sentiments.size(1)
        
        for i in range(num_aspects):
          
            aspect_loss = criterion(outputs1[i], sentiments[:, i]) + \
                          criterion(outputs2[i], sentiments[:, i])
            loss += aspect_loss

        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        num_batches += 1

        if (batch_idx + 1) % (10 * accumulation_steps) == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

           

    average_loss = total_loss / num_batches
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {average_loss:.4f}")

   
    return average_loss


def evaluate(model, dataloader, device, criterion):

    model.eval()
    total_loss = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels, sentiments in dataloader:
            inputs1, inputs2 = inputs
            inputs1, inputs2 = inputs1.to(device), inputs2.to(device)

            labels = labels.to(device)
            sentiments = sentiments.to(device)

            outputs1 = model(inputs1)
            outputs2 = model(inputs2)

            loss = 0
            num_aspects = sentiments.size(1)
            for i in range(num_aspects):
                aspect_loss = criterion(outputs1[i], sentiments[:, i]) + \
                              criterion(outputs2[i], sentiments[:, i])
                loss += aspect_loss

                predictions = torch.argmax(outputs1[i], dim=1)
                true_labels.extend(sentiments[:, i].cpu().tolist())
                predicted_labels.extend(predictions.cpu().tolist())

            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    cm = confusion_matrix(true_labels, predicted_labels)


    print(f"Validation Loss: {average_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    


    return average_loss, accuracy, precision, recall, f1, cm

def train_model(train_df, val_df, train_output_dir, val_output_dir, device, params):
    wandb.init(project="ABSA-final-run", entity="skot0411", config=params)
    model = ContrastiveModel(input_dim=512, hidden_dim=params['hidden_dim'], n_classes=2, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)
    criterion = FocalLoss(alpha=1, gamma=2)

    train_losses = []
    val_losses = []
    val_accuracies = []
    best_model_state = None
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_val_accu = 0.0

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_df, train_output_dir, device, optimizer, criterion, epoch, params['batch_size'])
        train_losses.append(train_loss)

        val_dataloader = create_validation_dataloader(val_df, val_output_dir, params['batch_size'])
        try:
            print("yep in")
            val_loss, val_accuracy, val_precision, val_recall, val_f1, val_cm = evaluate(model, val_dataloader, device, criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
        except (ValueError, RuntimeError):
            print("Error during evaluation. Skipping.")
            continue

        scheduler.step(val_loss)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        if val_f1 > best_val_f1:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_val_accu = val_accuracy
            best_model_state = model.state_dict()

        if early_stopping(val_losses):
            print("Early stopping triggered.")
            break
    #print("here")
    best_model_state = model.state_dict()

    # Log the best validation loss and F1 score
    

    return model, train_losses, val_losses, val_accuracies, best_model_state, best_val_loss, best_val_f1,best_val_accu

def evaluate_model(model, test_df, test_output_dir, device, criterion, batch_size):

    test_dataloader = create_validation_dataloader(test_df, test_output_dir, batch_size)
    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_cm = evaluate(model, test_dataloader, device, criterion)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("Test Confusion Matrix:")
    print(test_cm)

def plot_results(train_losses, val_losses, val_accuracies):
    print("in here")
    # Plot the training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    print("done first plot")
    plt.show()
    

    # Plot the validation accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    print("done second plot")
    plt.show()
def average_state_dicts(state_dicts):
    averaged_state_dict = {}
    for key in state_dicts[0].keys():
        averaged_state_dict[key] = torch.stack([state_dict[key] for state_dict in state_dicts]).mean(dim=0)
    return averaged_state_dict

def pad_sequences(sequences, max_len=None, padding_value=0):
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    padded_sequences = np.full((len(sequences), max_len), padding_value, dtype=np.float64)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    return padded_sequences


def main(output_dir="model/final", subset_size=0.2):
    train_output_dir = "model/embeddings/train"
    val_output_dir = "model/embeddings/valid"
    test_output_dir = "model/embeddings/test-abl"

    train_df = pd.read_pickle('model/dataframe/train.pkl')
    test_df = pd.read_pickle('model/dataframe/test.pkl')
    val_df = pd.read_pickle('model/dataframe/val.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_params = None
    best_avg_val_loss = float('inf')
    best_avg_val_f1 = 0.0
    best_avg_val_acc = 0.0

    criterion = FocalLoss(alpha=1, gamma=2)

    param_combinations = ParameterGrid(param_grid)
    for i, params in enumerate(param_combinations):
        print(f"\nTraining with hyperparameters: {params}")
        wandb.init(project="ABSA-final-run", entity="skot0411", config=params)

        num_subsets = int(1 / subset_size)
        models = []
        all_train_losses = []
        all_val_losses = []
        all_train_accuracies = []
        all_val_accuracies = []
        all_confusion_matrices = []

        for j in range(num_subsets):
            subset_df = train_df.sample(frac=subset_size, random_state=j)

            print(f"\nTraining model {j+1} with subset size {subset_size}...")
            model, train_losses, val_losses, val_accuracies, _, val_loss, val_f1, val_acc = train_model(
                subset_df, val_df, train_output_dir, val_output_dir, device, params
            )
            models.append(model)

            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
            all_val_accuracies.append(val_accuracies)

            # Evaluate the model on the validation set
            val_dataloader = create_validation_dataloader(val_df, val_output_dir, params['batch_size'])
            _, _, _, _, _, confusion_matrix = evaluate(model, val_dataloader, device, criterion)
            all_confusion_matrices.append(confusion_matrix)

        all_train_losses = pad_sequences(all_train_losses)
        all_val_losses = pad_sequences(all_val_losses)
        all_val_accuracies = pad_sequences(all_val_accuracies)
        all_confusion_matrices = np.array(all_confusion_matrices)

        avg_train_losses = np.mean(all_train_losses, axis=0)
        avg_val_losses = np.mean(all_val_losses, axis=0)
        avg_val_accuracies = np.mean(all_val_accuracies, axis=0)
        avg_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
        avg_val_loss = np.mean(all_val_losses, axis=0)[-1]
        avg_val_f1 = val_f1
        avg_val_acc = np.mean(all_val_accuracies, axis=0)[-1]

        print(f"\nAverage validation loss for hyperparameters {params}: {avg_val_loss:.4f}")
        print(f"Average validation F1 score for hyperparameters {params}: {avg_val_f1:.4f}")

        if avg_val_f1 > best_avg_val_f1:
            best_params = params
            best_avg_val_loss = avg_val_loss
            best_avg_val_f1 = avg_val_f1
            best_avg_val_acc = avg_val_acc

        # Save the results to a file
        with open(f"param-{i+1}.txt", "w") as f:
            f.write(f"Hyperparameters: {params}\n\n")
            for epoch in range(len(avg_train_losses)):
                f.write(f"Epoch {epoch+1}:\n")
                f.write(f"Train Loss: {avg_train_losses[epoch]:.4f}\n")
                f.write(f"Val Loss: {avg_val_losses[epoch]:.4f}\n")
                f.write(f"Val Accuracy: {avg_val_accuracies[epoch]:.4f}\n")
                f.write(f"Confusion Matrix:\n{avg_confusion_matrix.tolist()}\n\n")

    #     wandb.log({"avg_val_loss": avg_val_loss, "avg_val_f1": avg_val_f1, "avg_val_acc": avg_val_acc})
    #     wandb.finish()

    # wandb.init(project="final-model-run", entity="skot0411", config=None)

    best_model = ContrastiveModel(input_dim=512, hidden_dim=128, n_classes=2, dropout=dropout).to(device)
    best_model_save_path = os.path.join(output_dir, "best_model")
    torch.save(best_model.state_dict(), best_model_save_path)

    evaluate_model(best_model, test_df, test_output_dir, device, criterion, 64)
    print("Training and evaluation completed.")
    # wandb.finish()

if __name__ == "__main__":
    
    main(subset_size = 1)


