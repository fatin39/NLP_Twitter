import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
from torch.nn.functional import softmax
from transformers import AutoTokenizer, DistilBertForSequenceClassification


# Assuming you have a BERT model and tokenizer already set up:
# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/finals_distilbert_model'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()
# Ensure model is in evaluation mode and on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_tweet_sentiment(tweets):
    encodings = tokenizer.batch_encode_plus(
        tweets,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        inputs = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        outputs = model(inputs, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        return predictions.numpy()


def predict_proba(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()

# Create a LimeTextExplainer
explainer = LimeTextExplainer(class_names=['Normal/Negative/Non-Depressed', 'Depressed', 'Suicidal'])

# Text instance to explain
text_instance = "last year i was depressed. this year i am even more depressed. thats called growth"

# First, predict the label for the text instance
predicted_label = predict_tweet_sentiment([text_instance])[0]
print(f"Predicted Label: {predicted_label}")

# Then, generate a LIME explanation for the prediction
exp = explainer.explain_instance(
    text_instance,
    predict_proba,
    num_features=6,
    labels=[predicted_label]  # explaining the predicted class
)

# Print the explanation
print("LIME Explanation:", exp.as_list(label=predicted_label))

# Optionally, save the explanation to an HTML file
exp.save_to_file('/Users/nurfatinaqilah/Documents/streamlit-test/dataset/explanation.html')
