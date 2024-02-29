# This file was tidied up and moved to model.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import dump, load
from analyser import preprocess_text
from scipy.sparse import hstack, csr_matrix

# Import Dataset
dataset_path = 'dataset.csv'
preprocessed_path = 'preprocessed_df.joblib'

try:
    # Try to load the preprocessed DataFrame
    df = load(preprocessed_path)
    print("Loaded preprocessed data from cache.")
except FileNotFoundError:
    print("Preprocessed data not found, preprocessing now...")

    df = pd.read_csv(dataset_path, usecols=['about', 'Label'])

    # Function to clean and lemmatise text
    df['lemmatised_text'] = df['about'].apply(preprocess_text)

    # Save the preprocessed DataFrame for future runs
    dump(df, preprocessed_path)

non_sustainable_text = df[df['Label'] == 0]['lemmatised_text']
sustainable_text = df[df['Label'] == 1]['lemmatised_text']

# # Initialize the TF-IDF Vectorizer
tfidf_vectorizer_ngrams_path = 'tfidf_vectorizer_ngrams.joblib'
X_tfidf_ngrams_path = 'X_tfidf_ngrams.joblib'
y_path = 'y.joblib'
ngram_range=(1,1)

# Calculate bio length
df['bio_length'] = df['lemmatised_text'].apply(lambda x: len(x.split()))
# Example keywords
sustainable_keywords = ['sustainable', 'renewable energy', 'carbon footprint', 'respeto medio ambiente', 'soluciones energéticas eficientes', 'calidad medio ambiente', 'aportar soluciones energéticas', 'medio ambiente', 'energías renovables', 'gestión residuos']

# Feature creation: Count of sustainability-related keywords
df['sustainability_keyword_count'] = df['about'].apply(lambda x: sum(keyword in x.lower() for keyword in sustainable_keywords))

additional_features = df[['sustainability_keyword_count']].values

# Convert additional features to a sparse matrix
additional_features_sparse = csr_matrix(additional_features)

# Vectorize text features
vectorizer = TfidfVectorizer(ngram_range=ngram_range)
X_text = vectorizer.fit_transform(df['lemmatised_text'])

# Combine text features with bio length
X_combined = hstack([X_text, additional_features_sparse])

# Your target variable
y = df['Label']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Balance datasets
labels = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=labels, y=y_train)
class_weights_dict = dict(zip(labels, class_weights))

# Initialize the Logistic Regression model with class weights
model = LogisticRegression(class_weight=class_weights_dict, max_iter=1000)

# Train the model on your training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted') 
cm = confusion_matrix(y_test, y_pred)

title = f"Ngram of {ngram_range}"

# Calculate and print evaluation metrics
print(title)
print("Accuracy: ", accuracy)
print("Precision:", precision)
print("Recall:   ", recall)
print("F1 Score: ", f1)
print("Confusion Matrix:\n", cm)

dump(vectorizer, 'vectorizer.joblib')
dump(model, 'model.joblib')