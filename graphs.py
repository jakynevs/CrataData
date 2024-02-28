import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from analyser import preprocess_text

# Import Dataset
dataset_path = 'dataset.csv'
df = pd.read_csv(dataset_path)

# Check for duplicate rows
duplicate_rows = df.duplicated().sum()
if duplicate_rows:
    print("Number of duplicate rows:", duplicate_rows)

df['lemmatised_text'] = df['about'].apply(preprocess_text)

# Remove stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))
stop_words_list = list(stop_words)

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
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words_list).fit(text)
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

non_sustainable_text = df[df['Label'] == 0]['lemmatised_text']
sustainable_text = df[df['Label'] == 1]['lemmatised_text']

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

# Visualize the distribution of labels
df['Label'].value_counts().plot(kind='bar')
plt.title('Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Number of Samples')
plt.xticks(ticks=[0, 1], labels=['Not Sustainable (0)', 'Sustainable (1)'], rotation=0)
plt.show(block=False)

# Calculate text length for each company
df['Text_Length'] = df['about'].apply(lambda x: len(x.split()))

# Plot text length distribution for each label
plt.figure(figsize=(8, 6))
sns.boxplot(x='Label', y='Text_Length', data=df)
plt.title('Text Length by Label')
plt.xlabel('Label')
plt.ylabel('Number of Words')
plt.show()

