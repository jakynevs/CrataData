import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import dump, load
from analyser import preprocess_text
from scipy.sparse import csr_matrix, hstack
from constants import *

# Function to load or process and save the dataset
def load_or_process_dataset(path):
    try:
        df = load('preprocessed_df.joblib')
        print("Loaded preprocessed data from cache.")
    
    except FileNotFoundError:
        print("No preprocessed data in cache. Preprocessing text")
        df = pd.read_csv(DATASET_PATH, usecols=['about', 'Label'])
        df['about'] = df['about'].astype(str)
        df['lemmatised_text'] = df['about'].apply(lambda text: preprocess_text(text))
        dump(df, PREPROCESSED_DATA_PATH)
        print("Preprocessed dataset and saved.")

    return df

# Feature creation and vectorization for training phase
def prepare_features(df):
    df['lemmatised_text'] = df['lemmatised_text'].astype(str)

    # Calculate bio length feature
    df['bio_length'] = df['lemmatised_text'].apply(lambda x: len(x.split()))

    try:
        vectorizer = load(VECTORIZER_PATH)
        print("Loaded existing vectorizer from cache.")
    
    except FileNotFoundError:
        # If not found, initialize and fit a new vectorizer
        print("No vectorizer in cache. Fitting new...")
        
        vectorizer = TfidfVectorizer(ngram_range=(1, 1))

        X_text = vectorizer.fit_transform(df['lemmatised_text'])
        
        # Save vectorizer for future use
        dump(vectorizer, VECTORIZER_PATH)
        print("Vectorizer fitted and saved.")
        
    else:
        # If vectorizer was found, transform data with it
        X_text = vectorizer.transform(df['lemmatised_text'])

    # Convert bio length to a sparse matrix format and stack with text features
    bio_length_sparse = csr_matrix(df['bio_length'].values.reshape(-1, 1))
    X_combined = hstack([X_text, bio_length_sparse])

    return X_combined

# Training phase
def train_model(df):
    X = prepare_features(df)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    try:
        model = load(MODEL_PATH)
        print("Loaded model from cache.")
    
    except FileNotFoundError:
        # If not found, train a new model
        print("No model in cache. Training new model...")

        # Balance datasets and train model
        labels = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=labels, y=y_train)
        class_weights_dict = dict(zip(labels, class_weights))

        model = LogisticRegression(class_weight=class_weights_dict, C=10, max_iter=1000)
        model.fit(X_train, y_train)

        # Save the trained model
        dump(model, MODEL_PATH)

    # Predict on the test set and evaluate
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)

    scores = cross_val_score(model, X_train, y_train, cv=5) 
    print("Cross-validated scores:", scores)

# Function to evaluate the model
def evaluate_model(y_test, y_pred):
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:   ", recall_score(y_test, y_pred, average='weighted'))
    print("F1 Score: ", f1_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    df = load_or_process_dataset(DATASET_PATH)
    print("Data preprocessed, training model...")
    train_model(df)