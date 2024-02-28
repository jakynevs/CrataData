import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import dump, load
from analyser import preprocess_text

# Import Dataset
dataset_path = 'dataset.csv'
preprocessed_path = 'preprocessed_df.joblib'

# try:
#     # Try to load the preprocessed DataFrame
#     df = load(preprocessed_path)
#     print("Loaded preprocessed data from cache.")
# except FileNotFoundError:
    # print("Preprocessed data not found, preprocessing now...")
df = pd.read_csv(dataset_path, usecols=['about', 'Label'])
# Function to clean and lemmatise text
df['lemmatised_text'] = df['about'].apply(preprocess_text)
# Save the preprocessed DataFrame for future runs
dump(df, preprocessed_path)

non_sustainable_text = df[df['Label'] == 0]['lemmatised_text']
sustainable_text = df[df['Label'] == 1]['lemmatised_text']

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer_ngrams_path = 'tfidf_vectorizer_ngrams.joblib'
X_tfidf_ngrams_path = 'X_tfidf_ngrams.joblib'
y_path = 'y.joblib'


tfidf_vectorizer_ngrams = TfidfVectorizer(ngram_range=(1, 1))
# Fit and transform the lemmatised text
X_tfidf_ngrams = tfidf_vectorizer_ngrams.fit_transform(df['lemmatised_text'])
# Your target variable
y = df['Label']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf_ngrams, y, test_size=0.2, random_state=42)

# Balance datasets
labels = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=labels, y=y_train)
class_weights_dict = dict(zip(labels, class_weights))

# Initialize the Logistic Regression model with class weights
model = LogisticRegression(class_weight=class_weights_dict)

# Train the model on your training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted') 
cm = confusion_matrix(y_test, y_pred)

# Calculate and print evaluation metrics
print("Accuracy: ", accuracy)
print("Precision:", precision)
print("Recall:   ", recall)
print("F1 Score: ", f1)
print("Confusion Matrix:\n", cm)


dump(tfidf_vectorizer_ngrams, 'tfidf_vectorizer_ngrams.joblib')
dump(model, 'model.joblib')