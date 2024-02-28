import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk.corpus import stopwords
from joblib import dump

# Import Dataset
dataset_path = 'dataset.csv'
df = pd.read_csv(dataset_path)

# Check for duplicate rows
duplicate_rows = df.duplicated().sum()
if duplicate_rows:
    print("Number of duplicate rows:", duplicate_rows)

# Function to clean text data
def clean_text(text):
    # Remove URLs, special characters, and numbers
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase
    text = text.lower().strip()
    return text

# Fill missing textual data with empty strings
df.fillna('', inplace=True)

# Normalisation
df['cleaned_text'] = df['about'].apply(lambda x: x.lower())
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'\W', ' ', x)) # Removes special characters
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))  # Remove single characters
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'\^[a-zA-Z]\s+', ' ', x))  # Remove single characters at the start
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))  # Replace multiple spaces with a single space

# Tokenisation
df['tokens'] = df['cleaned_text'].apply(lambda x: x.split()) 

# Remove stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))
stop_words_list = list(stop_words)

df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

# Lemmatization
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')

df['lemmatised_tokens'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
df['lemmatised_text'] = df['lemmatised_tokens'].apply(lambda x: ' '.join(x))

non_sustainable_text = df[df['Label'] == 0]['lemmatised_text']
sustainable_text = df[df['Label'] == 1]['lemmatised_text']

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer_ngrams = TfidfVectorizer(ngram_range=(1, 2))

# Fit and transform the lemmatised text
# X_tfidf = tfidf_vectorizer.fit_transform(df['lemmatised_text'])
X_tfidf_ngrams = tfidf_vectorizer_ngrams.fit_transform(df['lemmatised_text'])

# Your target variable
y = df['Label']

# Splitting the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
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

dump(tfidf_vectorizer_ngrams, 'tfidf_vectorizer_ngrams.joblib')
dump(model, 'model.joblib')



