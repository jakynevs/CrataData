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

# Import Dataset
dataset_path = 'dataset.csv'
df = pd.read_csv(dataset_path)

# Check for duplicate rows
duplicate_rows = df.duplicated().sum()
if duplicate_rows:
    print("Number of duplicate rows:", duplicate_rows)

# Fill missing textual data with empty strings
df.fillna('', inplace=True)

# Normalisation
df['cleaned_text'] = df['about'].apply(lambda x: x.lower())
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'\W', ' ', x)) # Removes special characters
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))  # Remove single characters
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'\^[a-zA-Z]\s+', ' ', x))  # Remove single characters at the start
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))  # Replace multiple spaces with a single space
first_occurrence = df['cleaned_text'][df['cleaned_text'].str.contains('ei ei', na=False)].first_valid_index()

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

df['lemmatized_tokens'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
df['lemmatized_text'] = df['lemmatized_tokens'].apply(lambda x: ' '.join(x))


# Function to plot most common words
def plot_common_words(text, title, n=20):
    vec = CountVectorizer(stop_words=stop_words_list).fit(text)
    bag_of_words = vec.transform(text)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]
    words, freqs = zip(*words_freq)
    
    plt.figure(figsize=(10, 5))
    plt.bar(words, freqs)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.show(block=False)

# Function to plot most common word combos
def plot_top_ngrams(text, title, ngram_range, n=20):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words_list, preprocessor=custom_preprocessor).fit(text)
    bag_of_words = vec.transform(text)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]
    words, freqs = zip(*words_freq)
    plt.figure(figsize=(10, 5))
    plt.bar(words, freqs)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show(block=False)

non_sustainable_text = df[df['Label'] == 0]['lemmatized_text']
sustainable_text = df[df['Label'] == 1]['lemmatized_text']

def custom_preprocessor(text):
    return text.replace('ei ei', '')

plot_common_words(non_sustainable_text, "Non-Sustainable common words", n=20)
plot_common_words(sustainable_text, "Sustainable common words", n=20)
plot_top_ngrams(non_sustainable_text, "Top Bigrams in Non-Sustainable Companies (1,2)", ngram_range=(1,2))
plot_top_ngrams(non_sustainable_text, "Top Bigrams in Non-Sustainable Companies (2,2)", ngram_range=(2,2))
plot_top_ngrams(non_sustainable_text, "Top Bigrams in Non-Sustainable Companies (1,3)", ngram_range=(1,3))
plot_top_ngrams(non_sustainable_text, "Top Bigrams in Non-Sustainable Companies (1,4)", ngram_range=(1,4))
plot_top_ngrams(sustainable_text, "Top Bigrams in Sustainable Companies (1,2)", ngram_range=(1,2))
plot_top_ngrams(sustainable_text, "Top Bigrams in Sustainable Companies (2,2)", ngram_range=(2,2))
plot_top_ngrams(sustainable_text, "Top Bigrams in Sustainable Companies (1,3)", ngram_range=(1,3))
plot_top_ngrams(sustainable_text, "Top Bigrams in Sustainable Companies (1,4)", ngram_range=(1,4))


vec = CountVectorizer(ngram_range=(2, 2), stop_words=stop_words_list, preprocessor=custom_preprocessor)
bag_of_words = vec.fit_transform(non_sustainable_text)  
sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer_ngrams = TfidfVectorizer(ngram_range=(1, 2))

# Fit and transform the lemmatized text
# X_tfidf = tfidf_vectorizer.fit_transform(df['lemmatized_text'])
X_tfidf_ngrams = tfidf_vectorizer_ngrams.fit_transform(df['lemmatized_text'])

# Your target variable
y = df['Label']

# Splitting the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf_ngrams, y, test_size=0.2, random_state=42)

labels = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=labels, y=y_train)
class_weights_dict = dict(zip(labels, class_weights))

# Initialize the Logistic Regression model with class weights
model = LogisticRegression(class_weight=class_weights_dict)

# Train the model on your training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate and print evaluation metrics
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred, average='binary'))
# print("Recall:", recall_score(y_test, y_pred, average='binary'))
# print("F1 Score:", f1_score(y_test, y_pred, average='binary'))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# # Visualize the distribution of labels
# df['Label'].value_counts().plot(kind='bar')
# plt.title('Distribution of Labels')
# plt.xlabel('Label')
# plt.ylabel('Number of Samples')
# plt.xticks(ticks=[0, 1], labels=['Not Sustainable (0)', 'Sustainable (1)'], rotation=0)
# plt.show(block=False)

# Calculate text length for each company
df['Text_Length'] = df['about'].apply(lambda x: len(x.split()))

# # Plot text length distribution for each label
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='Label', y='Text_Length', data=df)
# plt.title('Text Length by Label')
# plt.xlabel('Label')
# plt.ylabel('Number of Words')
# plt.show(block=False)

plt.show()