import pandas as pd

# Replace this with your actual file path
csv_file_path = '/Users/nurfatinaqilah/Documents/streamlit-test/dataset/dataset_processed.csv'

# Load the dataset
df = pd.read_csv(csv_file_path)

# Count the number of samples in each class
class_counts = df['label'].value_counts()
print(class_counts)
