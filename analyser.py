import re
import nltk
import spacy
from langdetect import detect, DetectorFactory, LangDetectException
from nltk.corpus import stopwords
from joblib import load
from scipy.sparse import csr_matrix, hstack
from constants import *

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Attempt to find the NLTK data packages before downloading
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load SpaCy models with disabled components for performance
try:
    nlp_spacy_en = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'entity_linker'])
    nlp_spacy_es = spacy.load('es_core_news_sm', disable=['parser', 'ner', 'entity_linker'])
except Exception as e:
    print(f"Error loading SpaCy models: {e}")

# Load the saved TF-IDF Vectorizer and model
try:
    vectorizer = load(VECTORIZER_PATH)
except FileNotFoundError:
    print(f"Vectorizer file not found at {VECTORIZER_PATH}. A new one will be fitted.")
    vectorizer = None 

try:
    model = load(MODEL_PATH)
except FileNotFoundError:
    print(f"Model file not found at {MODEL_PATH}. A new model will be trained.")
    model = None  # You can later check if model is None and then train a new one

def preprocess_text(text):
        
    # Fill missing textual data with an empty string
    text = text if text is not None else ''
    
    # Remove URLs & emails
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    # Normalization
    text = text.lower().strip()  # Convert to lowercase
    text = re.sub(r'\b\d+\b', '', text)  # Remove numbers
    text = re.sub(r'\b[a-zA-Z]\b', ' ', text)  # Remove single characters
    text = re.sub(r'\W+', ' ', text)  # Non-word characters to space
    text = re.sub(r'\bhola\b', '', text)  # Removes whole word 'hola' 
    text = text.replace('ei ei', '') # Removes string 'ei ei' from data

    # Use langdetect to determine the language
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "es" # Default to Spanish

    # Language-specific processing
    doc = nlp_spacy_es(text) if lang == "es" else nlp_spacy_en(text)
    stop_words = set(stopwords.words('spanish' if lang == "es" else 'english'))
    lemmatised_tokens = [token.lemma_ for token in doc if token.text not in stop_words]
    lemmatised_text = ' '.join(lemmatised_tokens)
    
    return lemmatised_text, lang

# Function that uses model to predict text
def predict_sustainability(text):
    preprocessed_text, lang = preprocess_text(text)  # Preprocess and get language
    # Vectorize the preprocessed text
    transformed_text = vectorizer.transform([preprocessed_text])    
    
    # Calculate and transform bio_length into sparse matrix format
    bio_length = len(preprocessed_text.split())
    bio_length_sparse = csr_matrix([[bio_length]]) 

    # Combine vectorized text and bio_length features
    combined_features = hstack([transformed_text, bio_length_sparse])

    prediction = model.predict(combined_features)
    result = (prediction[0], lang)
    return result
