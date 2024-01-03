import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the trained model
model_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/final_distilbert_model'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to tokenize and encode texts similar to the training phase
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

# Load the dataset and select the last 20 entries
csv_file_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/dataset_processed.csv'
df = pd.read_csv(csv_file_path)
df_subset = df.tail(20)  # Select the last 20 entries

# Tokenize and encode the subset
encodings = tokenize_and_encode(df_subset['processed_text'].tolist())

# Predict using the model
with torch.no_grad():  # Disable gradient calculations
    inputs = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    outputs = model(inputs, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=1)

# Add predictions to the DataFrame
df_subset['prediction'] = predictions.numpy()

# Save or print the modified DataFrame
df_subset.to_csv('/Users/nurfatinaqilah/Documents/streamlit-test/dataset/predictions.csv', index=False)  # Adjust the path for saving
print(df_subset)
