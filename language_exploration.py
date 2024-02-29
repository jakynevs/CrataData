from analyser import predict_sustainability
import pandas as pd

# Import Dataset
dataset_path = 'dataset.csv'
df = pd.read_csv(dataset_path, usecols=['about', 'Label'])

# Detect language and apply predict_sustainability outside of lambda for better performance
def process_row(row):
    predicted_label, language = predict_sustainability(row['about']) 
    return predicted_label, language

# Apply the function to the DataFrame
results = df.apply(process_row, axis=1, result_type='expand')
df[['predicted_label', 'language']] = results

def categorize_prediction(row):
    if row['Label'] == 1 and row['predicted_label'] == 1:
        return 'TP'
    elif row['Label'] == 0 and row['predicted_label'] == 0:
        return 'TN'
    elif row['Label'] == 0 and row['predicted_label'] == 1:
        return 'FP'
    elif row['Label'] == 1 and row['predicted_label'] == 0:
        return 'FN'

df['error_type'] = df.apply(categorize_prediction, axis=1)

error_analysis = df.groupby(['language', 'error_type']).size().unstack(fill_value=0)
print(error_analysis)
