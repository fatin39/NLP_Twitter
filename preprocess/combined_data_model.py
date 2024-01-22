import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed dataset
csv_file_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/dataset_processed.csv'
df = pd.read_csv(csv_file_path)

# Downsampling
downsample_size = 7000  # Define the size for downsampling
class_0_subset = df[df['label'] == 0].sample(n=downsample_size, random_state=42)
class_1_subset = df[df['label'] == 1].sample(n=downsample_size, random_state=42)
class_2_subset = df[df['label'] == 2].sample(n=downsample_size, random_state=42)

# Combine the subsets and shuffle
downsampled_df = pd.concat([class_0_subset, class_1_subset, class_2_subset])
downsampled_df = downsampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(downsampled_df['processed_text'], downsampled_df['label'], test_size=0.2, random_state=42)

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_and_encode(texts):
     # Ensure all inputs are strings
    texts = [str(text) for text in texts]
    return tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

# Before tokenizing, make sure 'processed_text' is of string type
df['processed_text'] = df['processed_text'].astype(str)

# Tokenize and encode texts for the model
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
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(downsampled_df['label'].values), y=downsampled_df['label'].values)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Define Loss Function and Optimizer
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=3e-5)

# Function to calculate accuracy
def calculate_accuracy(preds, labels):
    return torch.sum(preds == labels).item() / len(labels)

# Training Loop with Early Stopping
num_epochs = 4
best_val_accuracy = 0
early_stopping_threshold = 2
no_improvement = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    total_eval_accuracy = 0
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        total_eval_accuracy += calculate_accuracy(predictions, labels)

    avg_train_loss = train_loss / len(train_loader)
    avg_val_accuracy = total_eval_accuracy / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.3f}, Validation Accuracy: {avg_val_accuracy:.3f}")

    # Early stopping logic
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        no_improvement = 0
    else:
        no_improvement += 1

    if no_improvement == early_stopping_threshold:
        print("Early stopping triggered.")
        break

# Save the final model
model_save_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/finals_distilbert_model'
model.save_pretrained(model_save_path)

