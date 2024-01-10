import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Function to tokenize and encode texts
def tokenize_and_encode(texts, tokenizer):
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

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/fine_tuned_model'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode

# Load the dataset
csv_file_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/dataset_processed.csv'
df = pd.read_csv(csv_file_path)
df_subset = df.tail(50)  # Select the last 20 entries


# # Downsampling
# class_0_samples = 200
# class_1_samples = 200
# class_2_samples = 200

# class_0_subset = df[df['label'] == 0].sample(n=class_0_samples, random_state=42)
# class_1_subset = df[df['label'] == 1].sample(n=class_1_samples, random_state=42)
# class_2_subset = df[df['label'] == 2].sample(n=class_2_samples, random_state=42)

# # Combine and shuffle the subsets
# df_subset = pd.concat([class_0_subset, class_1_subset, class_2_subset])
# df_subset = df_subset.sample(frac=1, random_state=42).reset_index(drop=True)

# Tokenize and encode the subset
encodings = tokenize_and_encode(df_subset['processed_text'].tolist(), tokenizer)

# Predict using the model
with torch.no_grad():  # Disable gradient calculations
    inputs = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    outputs = model(inputs, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=1)

# Add sentiment predictions to the DataFrame
df_subset['sentiment_prediction'] = predictions.numpy()

# Map the sentiment predictions to severity labels
sentiment_to_severity_mapping = {0: "minimum", 1: "mild", 2: "moderate", 3: "severe"}
df_subset['severity_prediction'] = df_subset['sentiment_prediction'].apply(
    lambda x: sentiment_to_severity_mapping.get(x, None)
)

# Save or print the modified DataFrame
df_subset.to_csv('/Users/nurfatinaqilah/Documents/streamlit-test/dataset/predictions_severity_new.csv', index=False)
print(df_subset)
