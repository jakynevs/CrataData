import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import dump, load
from analyser import preprocess_text
from scipy.sparse import csr_matrix, hstack

# Paths for data and saved models
dataset_path = 'dataset.csv'
vectorizer_path = 'tfidf_vectorizer_ngrams.joblib'
model_path = 'model.joblib'

# Function to load or process and save the dataset
def load_or_process_dataset(path):
    try:
        df = load('preprocessed_df.joblib')
        print("Loaded preprocessed data from cache.")
    except FileNotFoundError:
        print("Preprocessed data not found, preprocessing now...")
        df = pd.read_csv(path, usecols=['about', 'Label'])
        df['lemmatised_text'] = df['about'].apply(preprocess_text)
        dump(df, 'preprocessed_df.joblib')
    return df

# Load or process the dataset
df = load_or_process_dataset(dataset_path)

# Feature creation and vectorization for training phase
def prepare_features(df, vectorizer=None, fit_vectorizer=False):
    # If in training phase, fit the vectorizer; if not, transform using the loaded vectorizer
    if fit_vectorizer:
        vectorizer = TfidfVectorizer(ngram_range=(1,1))
        X_text = vectorizer.fit_transform(df['lemmatised_text'])
        dump(vectorizer, vectorizer_path)  # Save the fitted vectorizer for later use
    else:
        X_text = vectorizer.transform(df['lemmatised_text'])

    X_combined = X_text

    return X_combined

# Training phase
def train_model(df):
    vectorizer = None  # No vectorizer loaded yet, as we'll fit a new one
    X_combined = prepare_features(df, vectorizer, fit_vectorizer=True)
    y = df['Label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    # Balance datasets and train model
    labels = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=labels, y=y_train)
    class_weights_dict = dict(zip(labels, class_weights))

    model = LogisticRegression(class_weight=class_weights_dict, max_iter=1000)
    model.fit(X_train, y_train)

    # Save the trained model
    dump(model, model_path)

    # Predict on the test set and evaluate
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)

# Function to evaluate the model
def evaluate_model(y_test, y_pred):
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:   ", recall_score(y_test, y_pred, average='weighted'))
    print("F1 Score: ", f1_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    df = load_or_process_dataset(dataset_path)
    train_model(df)