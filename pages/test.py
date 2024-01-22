import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model and tokenizer
model_path = "/Users/nurfatinaqilah/Documents/streamlit-test/dataset/final_distilbert_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model.eval()

# Load and prepare test data
test_data_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/testing_dataset_processed.csv'
test_df = pd.read_csv(test_data_path)

# Shuffle and sample a reduced number of examples from each label
sample_size_per_label = 1500  # Adjust this number as needed
sampled_df = test_df.groupby('label', group_keys=False).apply(lambda x: x.sample(sample_size_per_label))

# Ensure that the text data is in string format
sampled_df['processed_text'] = sampled_df['processed_text'].astype(str)

# Prepare test texts and labels
test_texts = sampled_df['processed_text'].tolist()
test_labels = sampled_df['label'].tolist()

# Initialize lists to store results
predicted_labels = []
batch_size = 100  # Adjust this based on your system's capability

# Process in batches
for i in range(0, len(test_texts), batch_size):
    batch_texts = test_texts[i:i+batch_size]
    encoded_inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        inputs = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']
        outputs = model(inputs, attention_mask=attention_mask)
        batch_predictions = torch.argmax(outputs.logits, dim=1)
        predicted_labels.extend(batch_predictions.numpy())

# Evaluation metrics
accuracy = accuracy_score(test_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predicted_labels, average='weighted')

# Display the metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Confusion Matrix
conf_mat = confusion_matrix(test_labels, predicted_labels)
sns.heatmap(conf_mat, annot=True, fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()