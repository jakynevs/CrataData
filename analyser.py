import re
import nltk
import spacy
from langdetect import detect, DetectorFactory, LangDetectException
from nltk.corpus import stopwords
from joblib import load

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
    tfidf_vectorizer_ngrams = load('tfidf_vectorizer_ngrams.joblib')
    model = load('model.joblib')
except FileNotFoundError:
    print("Model or vectorizer file not found.")
    raise

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

    return ' '.join(lemmatised_tokens), lang
    
def predict_sustainability(text):
    preprocessed_text, lang = preprocess_text(text)  # Preprocess and get language
    transformed_text = tfidf_vectorizer_ngrams.transform([preprocessed_text])
    prediction = model.predict(transformed_text)
    result = (prediction[0], lang)
    return result

def analyse_text(about_section_text):
    result = predict_sustainability(about_section_text)
    return result
