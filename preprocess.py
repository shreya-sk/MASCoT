import string
import random
import warnings
import pandas as pd
from collections import Counter
import os
from sklearn.utils import resample



import torch
import torch.nn.functional as F # type: ignore
from transformers import logging
from transformers import BertForMaskedLM, BertTokenizer

import torch
from translate import Translator
import xml.etree.ElementTree as ET
import pandas as pd

def standardize_sentiment(polarity):
    polarity = polarity.lower().strip(string.whitespace)
    sentiment_set = {"positive", "negative", "neutral"}
    return polarity if polarity in sentiment_set else "other"

def preprocess_xml_data(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = []
    sentence_id = 1
    for sentence in root.findall('sentence'):
        text = sentence.find('text').text
        aspect_categories = sentence.find('aspectCategories')

        if aspect_categories is not None:
            for aspect_category in aspect_categories.findall('aspectCategory'):
                category = aspect_category.get('category')
                polarity = aspect_category.get('polarity')
                data.append({
                    'id': sentence_id,
                    'Sentence': text,
                    'Aspect Term': category,
                    'polarity': polarity
                })
        else:
            data.append({
                'id': sentence_id,
                'Sentence': text,
                'Aspect Term': None,
                'polarity': None
            })
        
        sentence_id += 1

    df = pd.DataFrame(data)
    print(df)
    print(df.columns.values)
    return df

def preprocess_col_data(train):
    train["polarity"] = train["polarity"].apply(standardize_sentiment)
    train['Sentence'] = train['Sentence'].str.lower()
    polarity_mapping = {'negative': 0, 'neutral': 1, 'positive': 2, 'other': 3}
    train['polarity_numeric'] = train['polarity'].map(polarity_mapping)
    return train.drop(['polarity'], axis=1)

def rows_for_augmentation(df):
    unique_ids = df['id'].unique()
    selected_ids = random.sample(list(unique_ids), len(unique_ids) // 2)
    subset_df = df[df['id'].isin(selected_ids)]
    return subset_df


def back_translate(sentence, lang='zh'):
    translator = Translator(to_lang=lang, from_lang='en')
    translated = translator.translate(sentence)
    back_translator = Translator(to_lang='en', from_lang=lang)
    return back_translator.translate(translated)

def get_bert_synonyms(word, context_sentence, top_p=5):
    
    logging.get_logger("transformers.modeling_utils").setLevel(logging.WARNING)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()
    # Mask the target word in the sentence
    masked_sentence = context_sentence.replace(word, tokenizer.mask_token)
    input_ids = tokenizer.encode(masked_sentence, return_tensors='pt')

    # Predict all tokens
    with torch.no_grad():
        outputs = model(input_ids)[0]

    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    if outputs.size(0) > 0:
        if outputs.dim() == 3:
            predicted_token_ids = outputs[0, mask_token_index, :].topk(top_p + 10).indices[0].tolist()  # Increase range to filter out the original word
        else:
            predicted_token_ids = outputs.topk(top_p + 10).indices[0].tolist()  # Increase range to filter out the original word

        # Decode the predicted ids to tokens and filter out the original word
        synonyms = [
                tokenizer.decode([predicted_id]).strip() 
                for predicted_id in predicted_token_ids 
                if tokenizer.decode([predicted_id]).strip() != word.lower() and '#' not in tokenizer.decode([predicted_id]).strip()
            ]

    
        return synonyms[:top_p]
    else:
        return []

def process_aspects(df):
    #print("Current index names:", df.index.names)  # Debug: Check current index names
    if 'Aspect Term' in df.index.names:
        result = df.reset_index(level='Aspect Term', inplace=False)  # Try without inplace to debug
    else:
        result = df  # Just pass the DataFrame as is if Aspect Term is not part of the index
    return result

def extract_multiAspects(df):
    id_counts = df['id'].value_counts()
    non_unique = id_counts[id_counts != 1].index
   # single_aspects = df[df['id'].isin(unique_ids)]
    multiple_aspects = df[df['id'].isin(non_unique)]

    return multiple_aspects

def add_aspect_numbers(group):
    group["Aspect Number"] = range(len(group))
    return group

def apply_augmentation_strategies(df, indices):
    for idx in indices:
        row = df.loc[idx]
        aspect = row['Aspect Term']
        sentence = row['Sentence']
        if aspect.lower() not in sentence.lower():
            continue
        if random.choice(['back_translate', 'synonym']) == 'back_translate':
            bt_sentence = back_translate(sentence)
            if bt_sentence.lower() != sentence.lower() and aspect.lower() in bt_sentence.lower():
                df.at[idx, 'Sentence'] = bt_sentence
                print('bt_sentence--------------')
                print(bt_sentence)
                print(sentence)
            else:
                synonyms = get_bert_synonyms(aspect, sentence)
                if synonyms:
                    new_aspect = synonyms[0]
                    df.at[idx, 'Sentence'] = sentence.replace(aspect, new_aspect)
                    df.at[idx, 'Aspect Term'] = new_aspect
        else:
            synonyms = get_bert_synonyms(aspect, sentence)
            if synonyms:
                new_aspect = synonyms[0]
                
                df.at[idx, 'Sentence'] = sentence.replace(aspect, new_aspect)
                df.at[idx, 'Aspect Term'] = new_aspect
                
                print(aspect, new_aspect)

    return df

def augment(df):
   
    subset_df = rows_for_augmentation(df)
    subset_df.reset_index(drop=True, inplace=True)  # Reset indices for safe operation
    unique_ids = set(subset_df.index)  # Use the reset index for operation
    augmented_df = apply_augmentation_strategies(subset_df, unique_ids)
    return pd.concat([df, augmented_df], ignore_index=True)

def extract_and_process_multiAspects(data):
    data = process_aspects(data)
    data = extract_multiAspects(data)
    data = data.sort_values(by='id')
    data = data.groupby('id').apply(add_aspect_numbers).reset_index(drop=True)
    return data

# def preprocess_data(train_data, test_data, val_data):
#     # Preprocess the training data
#     train_df = preprocess_xml_data(train_data)
#     train_df = preprocess_col_data(train_df)
#     train_df = augment(train_df)
#     train_df = extract_and_process_multiAspects(train_df)

#     val_df = preprocess_xml_data(val_data)
#     val_df = preprocess_col_data(val_df)
#     val_df = extract_and_process_multiAspects(val_df)

#     # Preprocess the test data
#     test_df = preprocess_xml_data(test_data)
#     test_df = preprocess_col_data(test_df)
#     test_df = extract_and_process_multiAspects(test_df)

#     # Balance the classes
#     train_df, val_df, test_df = balance_classes(train_df, val_df, test_df)

#     return train_df, val_df, test_df

from sklearn.utils import resample

def balance_classes(train_df):
    # Get the minority class count
    polarity_counts = train_df['polarity_numeric'].value_counts()
    min_count = polarity_counts.min()

    # Separate the dataframes by polarity
    train_negative = train_df[train_df['polarity_numeric'] == 0]
    train_positive = train_df[train_df['polarity_numeric'] == 2]

    # Downsample the majority class to have equal counts as the minority class
    train_negative_balanced = resample(train_negative, replace=False, n_samples=min_count, random_state=42)
    train_positive_balanced = train_positive

    # Combine the balanced dataframes
    train_df_balanced = pd.concat([train_negative_balanced, train_positive_balanced])

    return train_df_balanced

def preprocess_data(train_data, test_data, val_data):
    # Preprocess the training data
    
    train_df = preprocess_xml_data(train_data)
    train_df = preprocess_col_data(train_df)
    train_df = train_df[train_df['polarity_numeric'].isin([0, 2])]
    train_df = train_df[train_df['Aspect Term'] != 'miscellaneous']
    train_df = extract_and_process_multiAspects(train_df)
    
    train_df = augment(train_df)

    val_df = preprocess_xml_data(val_data)
    val_df = preprocess_col_data(val_df)
    val_df = val_df[val_df['polarity_numeric'].isin([0, 2])]
    val_df = val_df[val_df['Aspect Term'] != 'miscellaneous']
    val_df = extract_and_process_multiAspects(val_df)

    # Preprocess the test data
    test_df = preprocess_xml_data(test_data)
    test_df = preprocess_col_data(test_df)
    test_df = test_df[test_df['polarity_numeric'].isin([0, 2])]
    test_df = test_df[test_df['Aspect Term'] != 'miscellaneous']
    test_df = extract_and_process_multiAspects(test_df)

    # Balance the classes
    train_df = balance_classes(train_df)

    return train_df, val_df, test_df

def main(train_data, test_data,val_data):

    print(f"\nProcessing Data...")
    train_df, val_df, test_df = preprocess_data(train_data, test_data, val_data)
    
    train_df.to_pickle('model/dataframe/train-aug.pkl')
    val_df.to_pickle('model/dataframe/val-aug.pkl')   
    test_df.to_pickle('model/dataframe/test-aug.pkl')      
 

if __name__ == "__main__":
    train_data = "Datasets/MAMS-ACSA/raw/train.xml"
    test_data = "Datasets/MAMS-ACSA/raw/test.xml"
    val_data = "Datasets/MAMS-ACSA/raw/val.xml"
    main(train_data, test_data,val_data)


warnings.filterwarnings("default")

logging.set_verbosity_warning()
