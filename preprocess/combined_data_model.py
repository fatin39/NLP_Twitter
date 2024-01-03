import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the preprocessed dataset
csv_file_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/dataset_processed.csv'  
df = pd.read_csv(csv_file_path)

# Downsampling
downsample_size = 5000  # Define the size for downsampling
class_0_subset = df[df['label'] == 0].sample(n=downsample_size, random_state=42)
class_1_subset = df[df['label'] == 1].sample(n=downsample_size, random_state=42)
class_2_subset = df[df['label'] == 2].sample(n=downsample_size, random_state=42)

# Combine the subsets and shuffle
downsampled_df = pd.concat([class_0_subset, class_1_subset, class_2_subset])
downsampled_df = downsampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Tokenization and Encoding
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_and_encode(texts):
    texts = [str(text) for text in texts]  # Convert all non-string values to strings
    return tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

# Split the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(downsampled_df['processed_text'], downsampled_df['label'], test_size=0.1)
train_encodings = tokenize_and_encode(train_texts.tolist())
val_encodings = tokenize_and_encode(val_texts.tolist())

# Custom Dataset Class
class MyCustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset objects
train_dataset = MyCustomDataset(train_encodings, train_labels.tolist())
val_dataset = MyCustomDataset(val_encodings, val_labels.tolist())

# Initialize Model
unique_labels = np.unique(downsampled_df['label'].values)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(unique_labels))

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(downsampled_df['label'].values), y=downsampled_df['label'].values)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Define Loss Function and Optimizer
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

# Training Loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs} completed.")

# Save the final model
model_save_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/final_distilbert_model'
model.save_pretrained(model_save_path)