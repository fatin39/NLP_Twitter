import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Load the dataset and select the last 20 entries
csv_file_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/prediction_sentiment_severity.csv'
df = pd.read_csv(csv_file_path)

# Filter the dataset to include only texts classified as 'depressed' or 'suicidal'
df_filtered = df[df['prediction'].isin([1, 2])]

# Map severity labels to numerical values
label_mapping = {
    "minimum": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3
}
df_filtered.loc[:, 'label'] = df_filtered['label'].map(label_mapping)

# Tokenization and Encoding
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to tokenize and encode texts
def tokenize_and_encode(texts):
    # Convert all inputs to strings
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

# Split the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_filtered['text'], df_filtered['label'], test_size=0.1, random_state=42
)
train_encodings = tokenize_and_encode(train_texts.tolist())
val_encodings = tokenize_and_encode(val_texts.tolist())

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

# Create dataset objects
train_dataset = SeverityDataset(train_encodings, train_labels.tolist())
val_dataset = SeverityDataset(val_encodings, val_labels.tolist())

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# Loss Function and Optimizer
criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = AdamW(model.parameters(), lr=5e-5)

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

# Save the fine-tuned model
fine_tuned_model_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/fine_tuned_model'
model.save_pretrained(fine_tuned_model_path)
