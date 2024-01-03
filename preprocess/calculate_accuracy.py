import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the fine-tuned model and tokenizer
model_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/fine_tuned_distilbert_model'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model.eval()  # Set the model to evaluation mode

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

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to tokenize and encode texts
def tokenize_and_encode(texts, tokenizer):
    # Convert all inputs to strings and handle NaN values
    texts = [str(text) if text is not None else "" for text in texts]
    return tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

# Load your dataset (this should be the dataset used for testing/validation)
csv_file_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/dataset_processed.csv'
df = pd.read_csv(csv_file_path)

# Preprocess and split your dataset (if not already split)
# Here we use the entire dataset for demonstration; replace with your test set if available
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['processed_text'], df['label'], test_size=0.1, random_state=42
)

# Ensure val_texts is a list of strings
val_texts = val_texts.astype(str).tolist()

# Tokenize and encode using the correct tokenizer
val_encodings = tokenize_and_encode(val_texts, tokenizer)

# Create the validation dataset
val_dataset = SeverityDataset(val_encodings, val_labels.tolist())

# DataLoader for the validation set
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Calculate accuracy
true_labels = []
pred_labels = []

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    for batch in val_loader:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(inputs, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predictions.cpu().numpy())

accuracy = accuracy_score(true_labels, pred_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")
