import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split

# Define the path to your severity dataset and the pre-trained model
severity_csv_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/severity_processed.csv'  # Update with your severity dataset path
model_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/final_distilbert_model'

# Load the severity dataset
severity_df = pd.read_csv(severity_csv_path)

label_mapping = {
    "minimum": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3
}
severity_df['label'] = severity_df['label'].map(label_mapping)

# Determine the number of unique severity classes
num_severity_classes = severity_df['label'].nunique()

# Load tokenizer and model configuration
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
config = DistilBertConfig.from_pretrained(model_path, num_labels=num_severity_classes)

# Create a new DistilBERT model with the updated configuration
model = DistilBertForSequenceClassification(config)

# Function to tokenize and encode texts
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

# Custom Dataset Class
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

# Preprocessing and splitting the severity dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    severity_df['processed_text'], severity_df['label'], test_size=0.1, random_state=42
)
train_encodings = tokenize_and_encode(train_texts.tolist())
val_encodings = tokenize_and_encode(val_texts.tolist())

# Create dataset objects
train_dataset = SeverityDataset(train_encodings, train_labels.tolist())
val_dataset = SeverityDataset(val_encodings, val_labels.tolist())

# Define device and move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define Loss Function and Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Fine-tuning Loop
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

# Save the fine-tuned model
fine_tuned_model_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/fine_tuned_distilbert_model'
model.save_pretrained(fine_tuned_model_path)