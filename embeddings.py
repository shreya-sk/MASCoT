

import math
import warnings
import pandas as pd
import os

import torch
from torch import cuda
import torch.nn as nn
import torch.nn.functional as F # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from torch.nn.utils.rnn import pad_sequence
from transformers import logging
import torch


from transformers import RobertaTokenizer, RobertaModel


# Suppress all warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

device = 'cuda' if cuda.is_available() else 'cpu'
emb_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
emb_model = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True)
emb_model.to(device)
import torch

def tokenize_and_contextualize(sentence, aspect, tokenizer, max_seq_length):
    # Tokenize sentence and aspect
    sentence_tokens = tokenizer.tokenize(sentence)
    for i in range(len(sentence_tokens)):
        if sentence_tokens[i][0] == "Ġ":
            sentence_tokens[i] = sentence_tokens[i][1:]
        elif sentence_tokens[i][0] == "ł":
            sentence_tokens[i] = "t"
        elif sentence_tokens[i][0] == "Â":
            sentence_tokens[i] = "a"
        
    aspect_tokens = tokenizer.tokenize(aspect)
    for i in range(len(aspect_tokens)):
        if aspect_tokens[i][0] == "Ġ":
            aspect_tokens[i] = aspect_tokens[i][1:]
        elif aspect_tokens[i][0] == "ł":
            aspect_tokens[i] = "t"
        elif aspect_tokens[i][0] == "Â":
            aspect_tokens[i] = "a"
        

    # Truncate sentence tokens if necessary
    max_sentence_length = max_seq_length - len(aspect_tokens) - 3
    sentence_tokens = sentence_tokens[:max_sentence_length]
    # Construct the input sequence
    input_tokens = [tokenizer.cls_token] + sentence_tokens + [tokenizer.sep_token]
    input_tokens += aspect_tokens + [tokenizer.sep_token]

    # Create the aspect mask
    aspect_mask = [0] * (len(sentence_tokens) + 2) + [1] * (len(aspect_tokens) + 1)

    # Convert tokens to IDs and create attention mask
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    attention_mask = [1] * len(input_ids)

    # Pad the sequences
    padding_length = max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    attention_mask += [0] * padding_length
    aspect_mask += [0] * padding_length

    return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(aspect_mask), None, sentence_tokens, aspect_tokens


class AspectBasedSentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_seq_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sentence = row['Sentence']
        aspect = row['Aspect Term']
        label = row['polarity_numeric']

        input_ids, attention_mask, aspect_mask, _, _, _ = tokenize_and_contextualize(
            sentence, aspect, self.tokenizer, self.max_seq_length)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'aspect_mask': aspect_mask,
            'labels': torch.tensor(label, dtype=torch.long),
            'index': idx  # Add the index to the returned batch
        }

class MultiheadAttentionWithAspect(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.2):
        
        super(MultiheadAttentionWithAspect, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = torch.sqrt(torch.FloatTensor([self.depth])).to(device)

        
        # Linear layer for concatenated query and aspect
        self.query_linear = nn.Linear(2 * d_model, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Embedding(768, d_model)
    
    def forward(self, sentence_embedding, aspect_embedding):
        if len(sentence_embedding.size()) == 2:
            sentence_embedding = sentence_embedding.unsqueeze(0)

        if len(aspect_embedding.size()) == 1:
            aspect_embedding = aspect_embedding.unsqueeze(0)
            


        batch_size, seq_length, _ = sentence_embedding.size()
        aspect_length, _ = aspect_embedding.size()

        query = sentence_embedding
        key = sentence_embedding
        value = sentence_embedding

        # Repeat aspect embedding to match the batch size and sequence length
        aspect_embedding = aspect_embedding.unsqueeze(1).repeat(1, seq_length, 1)

        # Concatenate query and aspect embeddings
        concatenated_query = torch.cat((query, aspect_embedding), dim=-1)
        query = self.query_linear(concatenated_query)
        
        # Apply layer normalization to the query
        query = self.layer_norm(query)
        
        # Add positional encodings to the query
        position_ids = torch.arange(seq_length, dtype=torch.long, device=query.device)
        position_embeddings = self.pos_encoding(position_ids)
        query = query + position_embeddings
        
        Q = self.wq(query)
        K = self.wk(query)
        V = self.wv(query)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.depth).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.depth).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.depth).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        weighted_value = torch.matmul(attention, V)
        weighted_value = weighted_value.permute(0, 2, 1, 3).contiguous()
        weighted_value = weighted_value.view(batch_size, -1, self.d_model)
        
        # Apply residual connection and layer normalization
        output = self.layer_norm(weighted_value + query)
        
        output = self.fc_out(output)
        
        return output, attention



def precompute_embeddings(df, tokenizer, model, max_seq_length, pooling_strategy='avg'):
    embeddings = {}
    dataset = AspectBasedSentimentDataset(df, tokenizer, max_seq_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            aspect_mask = batch['aspect_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            hidden_states = outputs.hidden_states

            for i in range(len(input_ids)):
                index = int(batch['index'][i])
                sentence_id = df.iloc[index]['id']
                aspect = df.iloc[index]['Aspect Term']

                token_embeddings = hidden_states[-1][i]
                if pooling_strategy == 'avg':
                    sentence_embedding = torch.mean(token_embeddings, dim=0)
                elif pooling_strategy == 'max':
                    sentence_embedding, _ = torch.max(token_embeddings, dim=0)
                else:
                    raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")

                aspect_embedding = token_embeddings * aspect_mask[i].unsqueeze(-1)
                aspect_embedding = torch.sum(aspect_embedding, dim=0) / torch.sum(aspect_mask[i])
                print("here 2")
                embeddings[(sentence_id, aspect)] = (sentence_embedding.cpu(), aspect_embedding.cpu())

    return embeddings


def split_windows(sentence_tokens, output):
    # Smaller window size for more granularity
    window_size = max(3, math.floor(len(sentence_tokens) / 4))  # minimum size of 3
    stride = 1  # small stride for more overlap

    num_windows = (len(sentence_tokens) - window_size) // stride + 1

    windows_tokens = []
    windows_scores = []

    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        window_tokens = sentence_tokens[start_idx:end_idx]
        window_scores = output[:, start_idx:end_idx, :]

        windows_tokens.append(window_tokens)
        windows_scores.append(window_scores)

    return windows_tokens, windows_scores


def save_tensor_data(output_dir,sentence_id,aspect_number,window_scores):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{output_dir}/{sentence_id}-{aspect_number}.pk"
    
    # Save the tensor using pickle
    with open(filename, 'wb') as f:
        torch.save(window_scores, f)


def generate_embeddings(df, output_dir, precomputed_embeddings):
    for _, row in df.iterrows():
        sentence_id = row['id']
        aspect = row['Aspect Term']
        aspect_number = row['Aspect Number']
        
        sentence_embedding, aspect_embedding = precomputed_embeddings[(sentence_id, aspect)]
        
        sentence_embedding = sentence_embedding.to(device)
        aspect_embedding = aspect_embedding.to(device)
        input_dim = d_model = sentence_embedding.size(-1)
        num_heads = 8
        dropout_rate = 0.2
        #print("here 1")
        attention_model = MultiheadAttentionWithAspect(d_model, num_heads, dropout_rate).to(device)
        attention_model.to(device)
        # Add an extra dimension to sentence_embedding
        sentence_embedding = sentence_embedding.unsqueeze(0)
        sentence_embedding.to(device)
        output, attention = attention_model(sentence_embedding.unsqueeze(0), aspect_embedding.unsqueeze(0))
        
        window_tokens, window_scores = split_windows(row['Sentence'].split(), output)
        
        # Use the aspect number directly from the dataframe
        save_tensor_data(output_dir, sentence_id, aspect_number, window_scores)

def generate_embeddings_for_data(data, output_dir):
    max_seq_length = data['Sentence'].str.len().max()
    precomputed_embeddings = precompute_embeddings(data, emb_tokenizer, emb_model, max_seq_length)
    generate_embeddings(data, output_dir, precomputed_embeddings)


def main(train_data, test_data,val_data, output_dir="embeddings", train_size=1):
    #dataset_name = os.path.basename(train_data).split('.')[0].lower()
    train_output_dir = os.path.join(output_dir, "train")
    val_output_dir = os.path.join(output_dir, "valid")
    test_output_dir = os.path.join(output_dir, "test")

    train_df = pd.read_pickle(train_data)
    val_df = pd.read_pickle(test_data)
    test_df = pd.read_pickle(val_data)

    print(f"Generating Embeddings...")

    generate_embeddings_for_data(train_df, train_output_dir)
    generate_embeddings_for_data(val_df, val_output_dir)
    generate_embeddings_for_data(test_df, test_output_dir)
    


if __name__ == "__main__":
    train_data = "train.pkl"
    test_data = "test.pkl"
    val_data = "val.pkl"

    main(train_data, test_data, val_data)

warnings.filterwarnings("default")

logging.set_verbosity_warning()
