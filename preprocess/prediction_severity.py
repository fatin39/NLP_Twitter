import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Function to tokenize and encode texts
def tokenize_and_encode(texts, tokenizer):
    return tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/fine_tuned_distilbert_model'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode

# Load the dataset and select the last 20 entries
csv_file_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/dataset_processed.csv'
df = pd.read_csv(csv_file_path)
df_subset = df.tail(50)  # Select the last 50 entries

# Tokenize and encode the subset
encodings = tokenize_and_encode(df_subset['processed_text'].tolist(), tokenizer)

# Predict using the model
with torch.no_grad():  # Disable gradient calculations
    inputs = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    outputs = model(inputs, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=1)

# Add label predictions to the DataFrame
df_subset['prediction'] = predictions.numpy()

# Severity prediction for labels 1 and 2
severity_mapping = {0: "minimum", 1: "mild", 2: "moderate", 3: "severe"}
df_subset['severity_prediction'] = df_subset['prediction'].apply(
    lambda x: severity_mapping.get(x) if x in [1, 2] else None
)

# Save or print the modified DataFrame
df_subset.to_csv('/Users/nurfatinaqilah/Documents/streamlit-test/dataset/predictions_severity.csv', index=False)  # Adjust the path for saving
print(df_subset)
