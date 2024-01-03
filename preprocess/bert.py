# Import necessary libraries
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from transformers import logging
logging.set_verbosity_error()

# BERT Model: BERT (Bidirectional Encoder Representations from Transformers) 
# is a deep learning model that's pre-trained on a large corpus of text. 
# It's designed to understand the context and meaning of words in sentences.

# Load the data
df = pd.read_csv('/Users/nurfatinaqilah/Documents/streamlit-test/dataset/severity_processed3.csv')


# BERT Model: BERT (Bidirectional Encoder Representations from Transformers) 
# is a deep learning model that's pre-trained on a large corpus of text. 
# It's designed to understand the context and meaning of words in sentences.

# Map textual labels to integers
label_mapping = {
    "minimum": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3
}
df['label'] = df['label'].map(label_mapping)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization and encoding the dataset
def tokenize_and_encode(texts):
    return tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

# Splitting the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.1
)

# Tokenize and encode
train_encodings = tokenize_and_encode(train_texts.tolist())
val_encodings = tokenize_and_encode(val_texts.tolist())

# Custom dataset class
class SeverityDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = SeverityDataset(train_encodings, train_labels.tolist())
val_dataset = SeverityDataset(val_encodings, val_labels.tolist())

# Load BERT model
# The BertForSequenceClassification model includes a classifier layer for the task of sequence classification (like your text classification task).
# The 'bert-base-uncased' model is a pre-trained BERT model without an additional classifier on top
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_mapping))

# Training settings
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()

from transformers import BertForSequenceClassification

model_path = '/Users/nurfatinaqilah/Documents/streamlit-test/preprocess/my_bert_model'

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.save_pretrained(model_path)
