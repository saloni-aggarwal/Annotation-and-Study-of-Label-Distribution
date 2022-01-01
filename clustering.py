import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
import gensim as gensim
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import Counter

#reading the dataset
data = pd.read_csv('jobQ3_both_dataset.csv', encoding= 'unicode_escape', skipinitialspace=True)
# extracting labels only from the data
labels_only = data.iloc[:,2:]
# filling empty cells with 0
labels_only = labels_only.fillna(0)
columns = labels_only.columns
#setting the no. of clusters to 4
n_clusters = 4

# converting the label columns to numeric
for col in columns:
  labels_only[col] = pd.to_numeric(labels_only[col])


# Compute clustering with Means
k_means = KMeans(init="k-means++", n_clusters=4, n_init=10)
clusters = k_means.fit(labels_only)
print(clusters)
data['Clusters'] = clusters
data.to_csv('clusters.csv')


#text preprocessing
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
# Method to perform stemming and lemmatization on an article body
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
# Method to tokenize, perform stemming and lemmatization using the method above and removing stopwords
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS  and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result



#obtaining the cluster centers from the clusters formed
k_means_cluster_centers = k_means.cluster_centers_
#obtaining the cluster for each item
k_means_labels = pairwise_distances_argmin(labels_only, k_means_cluster_centers)

for k in range(n_clusters):
    my_members = k_means_labels == k
    cluster_tweets = ''
    cluster_center = k_means_cluster_centers[k]
    df = data[my_members].astype(str)
    for msg in df['Message']:
      cluster_tweets = cluster_tweets + msg
    processed_tweets = preprocess(cluster_tweets)
    # Pass the split list to instance of Counter class.
    Counter_obj = Counter(processed_tweets)
    # most_common() produces k frequently encountered
    # input values and their respective counts.
    freq_words = Counter_obj.most_common(10)
    print('Top 10 words in cluster = ',freq_words)
