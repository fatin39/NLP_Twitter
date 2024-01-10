import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the trained model
model_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/final_distilbert_model'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode


# Load the dataset and select the last 20 entries
csv_file_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/severity_depression.csv'
df = pd.read_csv(csv_file_path)

# Downsample the 'minimum' class
df_minimum = df[df['label'] == 'minimum'].sample(n=300, random_state=42)  # Adjust the random_state as needed
df_other = df[df['label'] != 'minimum']

# Combine and shuffle the subsets
df_balanced = pd.concat([df_minimum, df_other]).sample(frac=1, random_state=42).reset_index(drop=True)

# The counts for each severity label and the counts for the labels 'depressed' (1) and 'suicidal' (2) in your dataset are as follows:
#Severity Label Counts:

#Moderate: 394
#Minimum: 300
#Mild: 290
#Severe: 282
#Counts for Predictions:

#Depressed (Label 1): 915
#Suicidal (Label 2): 254
#df_subset = df.tail(50)  # Select the last 20 entries

# # Downsampling
# class_0_samples = 200
# class_1_samples = 200
# class_2_samples = 200

# class_0_subset = df[df['label'] == 0].sample(n=class_0_samples, random_state=42)
# class_1_subset = df[df['label'] == 1].sample(n=class_1_samples, random_state=42)
# class_2_subset = df[df['label'] == 2].sample(n=class_2_samples, random_state=42)


# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Function to tokenize and encode a batch of texts
def tokenize_and_encode_batch(texts):
    return tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

# Process the dataset in smaller batches
batch_size = 100  # Adjust this based on your system's capability
predictions = []

for i in range(0, len(df_balanced), batch_size):
    batch_texts = df_balanced['text'].iloc[i:i+batch_size].tolist()
    encodings = tokenize_and_encode_batch(batch_texts)

    with torch.no_grad():
        inputs = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        outputs = model(inputs, attention_mask=attention_mask)
        batch_predictions = torch.argmax(outputs.logits, dim=1)
        predictions.extend(batch_predictions.numpy())

# Add predictions to the DataFrame
df_balanced['prediction'] = predictions

# Save or print the modified DataFrame
df_balanced.to_csv('/Users/nurfatinaqilah/Documents/streamlit-test/dataset/prediction_sentiment_severity.csv', index=False)  # Adjust the path for saving
print(df_balanced)
