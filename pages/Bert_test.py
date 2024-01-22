import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Define the Custom Dataset class
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

# Function to tokenize and encode texts
def tokenize_and_encode(texts, tokenizer):
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

# Load the saved model and tokenizer
model_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/finals_distilbert_model'  # Update this path
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Load your validation data
csv_file_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/dataset_processed.csv'  # Update this path to your CSV file
df = pd.read_csv(csv_file_path)

# Assuming you have already split your data and these are your validation texts and labels
val_texts = df['processed_text'].tolist()  # Replace 'processed_text' with your actual text column name
val_labels = df['label'].tolist()  # Replace 'label' with your actual label column name

# Tokenize and encode the validation texts
val_encodings = tokenize_and_encode(val_texts, tokenizer)

# Create a dataset object for the validation set
val_dataset = MyCustomDataset(val_encodings, val_labels)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize DataLoader for the validation set
val_loader = DataLoader(val_dataset, batch_size=16)

# Evaluate the model
model.eval()  # Set the model to evaluation mode
val_predictions = []
val_true_labels = []

for batch in val_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)

    # Append batch predictions and true labels
    val_predictions.extend(predictions.cpu().numpy())
    val_true_labels.extend(labels.cpu().numpy())

# Calculate the metrics
precision, recall, f1, _ = precision_recall_fscore_support(val_true_labels, val_predictions, average='weighted')
print(f'Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}')
