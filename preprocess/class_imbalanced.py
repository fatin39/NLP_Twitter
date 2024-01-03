import pandas as pd

# Load dataset
csv_file_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/dataset_processed.csv'  # Update this path
df = pd.read_csv(csv_file_path)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch

# Assuming 'label' is the column in your DataFrame that contains class labels
labels = df['label'].values
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Move class_weights to the device (CPU or GPU) where you plan to train your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)

