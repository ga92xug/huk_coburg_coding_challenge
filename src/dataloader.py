from typing import Dict, List, Tuple
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import re
from nltk.corpus import stopwords
import nltk

from transformers import DistilBertTokenizer
import torch
from torch.utils.data import DataLoader

def clean_text(
        text: str,
        lower: bool = True,
        letters_numbers_only: str = r"[^a-zA-Z0-9]",
        remove_stop_words: bool = False,
    ) -> str:
    """
    Clean text by removing special characters, numbers, and stopwords.

    - lower: bool, default=True
    - letters_numbers_only: str, default=r"[^a-zA-Z0-9]"
    - remove_stop_words: bool, default=False
    """
    
    if not isinstance(text, str):
        return ""

    # strip leading/trailing whitespaces
    text = text.strip()
    text = text.replace('"', '')
    text = text.replace("'", '') 
    
    if lower:
        text = text.lower()
    if letters_numbers_only is not None:
        text = re.sub(letters_numbers_only, " ", text)
    if remove_stop_words:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text


def get_preprocessed_dfs(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the training and validation dataframes from the given path.

    Args:
    - path: str, the path to the directory containing the training and validation CSV files

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame], containing the training and validation dataframes
    """

    # get dataframes
    column_names = ["id", "product", "label", "comment"]
    train_data = pd.read_csv(os.path.join(path, "training.csv"), header=0, names=column_names)
    test_data = pd.read_csv(os.path.join(path, "validation.csv"), header=0, names=column_names)

    # encode labels
    label_encoder = LabelEncoder()
    train_data['encoded_label'] = label_encoder.fit_transform(train_data['label'])
    test_data['encoded_label'] = label_encoder.transform(test_data['label'])
    label_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
    label_mapping

    # clean text
    train_data['cleaned_comment'] = train_data['comment'].apply(clean_text)
    test_data['cleaned_comment'] = test_data['comment'].apply(clean_text)

    train_data = train_data.drop_duplicates()
    train_data = train_data.dropna(subset=['cleaned_comment'])
    train_data.reset_index(drop=True, inplace=True) 

    return train_data, test_data


def encode_comments(data: pd.DataFrame, tokenizer: DistilBertTokenizer):
    return tokenizer.batch_encode_plus(
        data['cleaned_comment'].tolist(),
        max_length=50,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )

class CommentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # input_ids, attention_mask
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

def get_individual_loader(
        data: pd.DataFrame, 
        tokenizer: DistilBertTokenizer, 
        batch_size: int = 16,
        num_workers: int = 4
    ) -> DataLoader:
    """
    Get the DataLoader for the given data.

    Args:
    - data: pd.DataFrame, the data to be loaded
    - tokenizer: DistilBertTokenizer, the tokenizer to be used
    - batch_size: int, the batch size
    - num_workers: int, the number of workers

    Returns:
    - DataLoader, the DataLoader for the given data
    """
    encodings = encode_comments(data, tokenizer)
    labels = data['encoded_label'].tolist()
    dataset = CommentDataset(encodings, labels)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def get_dataloaders(
        batch_size: int = 16,
        num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Get the training, valid, and test dataloaders.
    """
    train_data, test_data = get_preprocessed_dfs("data")

    train_data, valid_data = train_test_split(
        train_data,
        test_size=0.2,
        stratify=train_data['encoded_label'],
        random_state=42
    )

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    dataloaders = {
        "train": get_individual_loader(train_data, tokenizer, batch_size, num_workers),
        "valid": get_individual_loader(valid_data, tokenizer, batch_size, num_workers),
        "test": get_individual_loader(test_data, tokenizer, batch_size, num_workers)
    }

    return dataloaders


if __name__ == "__main__":
    dataloaders = get_dataloaders()
    print(dataloaders)
    print("Dataloaders loaded successfully!")

    for phase, loader in dataloaders.items():
        print(f"{phase} dataloader length: {len(loader)}")
        for batch in loader:
            print(batch)
            break








