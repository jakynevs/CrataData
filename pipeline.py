import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from joblib import dump, load
from model import load_or_process_dataset
from constants import *

# Custom transformer to extract the length of the bio
class BioLengthExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, active=True):
        self.active = active

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        if not self.active:
            return np.zeros((len(data), 1))

        transformed = np.array([len(text.split()) for text in data]).reshape(-1, 1)
        # Ensure output has the same number of rows as input
        assert transformed.shape[0] == len(data), "Output and input row counts do not match."
        
        return transformed

# Custom transformer to count sustainable keywords
class KeywordCounter(BaseEstimator, TransformerMixin):
    def __init__(self, keywords, active=True):
        self.keywords = keywords
        self.active = active

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        if not self.active:
            return np.zeros((len(data), 1))
        
        transformed = np.array([sum(1 for word in text.split() if word.lower() in self.keywords) for text in data]).reshape(-1, 1)
        
        # Ensure output has the same number of rows as input
        assert transformed.shape[0] == len(data), "Output and input row counts do not match."
        
        return transformed

# Define sustainable keywords (adjust this list as needed)
sustainable_keywords = ['sustainable', 'renewable energy', 'carbon footprint', 'respeto medio ambiente', 'soluciones energéticas eficientes', 'calidad medio ambiente', 'aportar soluciones energéticas', 'medio ambiente', 'energías renovables', 'gestión residuos']

# Load or process your dataset
df = load_or_process_dataset(DATASET_PATH)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(df['lemmatised_text'], df['Label'], test_size=0.2, random_state=42)
X_train = X_train.apply(lambda x: ' '.join(x) if isinstance(x, tuple) else x)
X_test = X_test.apply(lambda x: ' '.join(x) if isinstance(x, tuple) else x)

# Pipeline setup
pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('vectorizer', TfidfVectorizer()),
        ])),
        ('bio_length', BioLengthExtractor()), 
        ('keyword_count', KeywordCounter(keywords=sustainable_keywords)),
    ])),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Grid search parameters
param_grid = {
    'features__text__vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (3, 3), (4, 4)],
    'classifier__C': [1, 10, 20, 50],
    'classifier__class_weight': [None, 'balanced'],  # Toggle class_weight
    'features__bio_length__active': [True, False],  # Toggle BioLengthExtractor
    'features__keyword_count__active': [True, False],  # Toggle KeywordCounter
}

# Grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=3, scoring='recall')
grid_search.fit(X_train, y_train)

print("Best parameters set:")
print(grid_search.best_params_)

# Evaluate on test set
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the best model
dump(grid_search.best_estimator_, 'best_model.joblib')
