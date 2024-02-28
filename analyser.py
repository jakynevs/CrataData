import re
import nltk
import spacy
from langdetect import detect, DetectorFactory
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from joblib import load

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Load SpaCy Spanish model
nlp_spacy_es = spacy.load('es_core_news_sm')

# Load the saved TF-IDF Vectorizer and model
tfidf_vectorizer_ngrams = load('tfidf_vectorizer_ngrams.joblib')
model = load('model.joblib')

# Attempt to find the NLTK data packages before downloading
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

def preprocess_text(text):
    # Fill missing textual data with an empty string
    text = text if text is not None else ''
    
    # Remove URLs & emails
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    # Normalization
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\b\d+\b', '', text) # Removes numbers
    text = re.sub(r'\W', ' ', text)  # Removes special characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove single characters
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)  # Remove single characters at the start
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Replace multiple spaces with a single space        
    text = re.sub(r'\bhola\b', '', text)  # Removes whole word 'hola' 
    text = text.replace('ei ei', '') # Removes string 'ei ei' from data

    try:
        # Use langdetect to determine the language
        lang = detect(text)
    except:
        lang = "en"

    if lang == "es":
        # Spanish text processing with SpaCy
        doc = nlp_spacy_es(text)
        lemmatised_tokens = [token.lemma_ for token in doc]
    
    else:
        # Tokenization
        tokens = text.split()

        # Remove stopwords
        stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatised_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    lemmatised_text = ' '.join(lemmatised_tokens)
    
    return lemmatised_text

# Example of transforming a single preprocessed text for prediction
def predict_sustainability(text):
    preprocessed_text = preprocess_text(text)
    transformed_text = tfidf_vectorizer_ngrams.transform([preprocessed_text])
    prediction = model.predict(transformed_text)
    print("prediction: ", prediction)
    return 'Yes' if prediction == 1 else 'No'

def analyse_text(about_section_text):
    clean_lemmatised_text = preprocess_text(about_section_text)
    result = predict_sustainability(clean_lemmatised_text)
    return result