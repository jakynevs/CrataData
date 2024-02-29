from analyser import preprocess_text, predict_sustainability
import pandas as pd
from langdetect import detect

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'  # Fallback in case detection fails
    
# Import Dataset
dataset_path = 'dataset.csv'

df = pd.read_csv(dataset_path, usecols=['about', 'Label'])
df[['lemmatised_text', 'language']] = df['about'].apply(lambda x: pd.Series(preprocess_text(x)))

df['predicted_label'] = [predict_sustainability(text) for text in df['lemmatised text']]

def categorize_prediction(row):
    if row['Label'] == 1 and row['predicted_label'] == 1:
        return 'TP'
    elif row['Label'] == 0 and row['predicted_label'] == 0:
        return 'TN'
    elif row['Label'] == 0 and row['predicted_label'] == 1:
        return 'FP'
    elif row['Label'] == 1 and row['predicted_label'] == 0:
        return 'FN'

# Apply the function to categorize each prediction
df['error_type'] = df.apply(categorize_prediction, axis=1)

error_analysis = df.groupby(['language', 'error_type']).size().unstack(fill_value=0)
print(error_analysis)
