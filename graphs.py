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

# Function to plot most common word combos
def plot_top_ngrams(text, ngram_range, n=20):
    categories = {0: 'Non-Sustainable', 1: 'Sustainable'}
    
    for label, category_name in categories.items():
        text = df[df['Label'] == label]['lemmatised_text']

        vec = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words_list).fit(text)
        bag_of_words = vec.transform(text)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]
        words, freqs = zip(*words_freq)
        
        plt.figure(figsize=(10, 5))
        plt.bar(words, freqs)
        title = f"Top {ngram_range[0]} to {ngram_range[1]}-Grams in {category_name} Companies"
        plt.title(title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show(block=False)

plot_top_ngrams(df, ngram_range=(3,3))

# # Visualize the distribution of labels
# df['Label'].value_counts().plot(kind='bar')
# plt.title('Distribution of Labels')
# plt.xlabel('Label')
# plt.ylabel('Number of Samples')
# plt.xticks(ticks=[0, 1], labels=['Not Sustainable (0)', 'Sustainable (1)'], rotation=0)
# plt.show(block=False)

# # Calculate text length for each company
# df['Text_Length'] = df['about'].apply(lambda x: len(x.split()))

# # Plot text length distribution for each label
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='Label', y='Text_Length', data=df)
# plt.title('Text Length by Label')
# plt.xlabel('Label')
# plt.ylabel('Number of Words')
# plt.show(block=False)

plt.show()

