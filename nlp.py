#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 19:49:45 2018

@author: vivekmishra
"""
#PAth
import os
os.chdir('/Users/vivekmishra/Desktop/USC/599-DSS/project')


#imports

import pandas as pd
import numpy as np
import re
import string
import unicodedata
import seaborn as sns
import matplotlib as plt
from nltk.stem import PorterStemmer
from nltk.corpus import words
from sklearn.cluster import KMeans
import scipy


import nltk
nltk.download('words')
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english')

from sklearn.feature_extraction.text import TfidfVectorizer

import spacy
nlp = spacy.load('en', parse=True, tag=True, entity=True)

#recommendation class
from recommendation import recommendation

#read pickle - Contains LDA And sentiment analysis results
df = pd.read_pickle('df_senti.pkl')


#Preproc func

def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def strip_hashtag(text):
    entity_prefixes = ['#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    word_list = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                word_list.append(word)
    return ' '.join(word_list)

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

    
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, ' ', text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

    
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    whitelist = ["n't","not", "no"]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if (token not in stopword_list or token in whitelist)]
    else:
        filtered_tokens = [token for token in tokens if (token.lower() not in stopword_list or token in whitelist)]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


#TF-IDF for title and desc - features into models
title = list(df['title'])

counter = 0
for sent in title:
    title[counter]=strip_links(sent)
    counter += 1
    
counter = 0
for sent in title:
    title[counter]=strip_hashtag(sent)
    counter += 1

counter = 0
for sent in title:
    title[counter]=remove_special_characters(sent, 
                          remove_digits=True)
    counter += 1
    
counter = 0
for sent in title:
    title[counter]=remove_stopwords(sent)
    counter += 1

counter = 0
for sent in title:
    title[counter]=lemmatize_text(sent)
    counter += 1
    
vectorizer = TfidfVectorizer(strip_accents='unicode')
title_mat = vectorizer.fit_transform(title)
title_mat = title_mat.toarray()

title_mat = pd.DataFrame(title_mat)

#Desc
desc = list(df['desc'])

counter = 0
for sent in desc:
    desc[counter]=strip_links(sent)
    counter += 1
    
counter = 0
for sent in desc:
    desc[counter]=strip_hashtag(sent)
    counter += 1

counter = 0
for sent in desc:
    desc[counter]=remove_special_characters(sent, 
                          remove_digits=True)
    counter += 1
    
counter = 0
for sent in desc:
    desc[counter]=remove_stopwords(sent)
    counter += 1
    
counter = 0
for sent in desc:
    desc[counter]=lemmatize_text(sent)
    counter += 1
    
#Joining desc and title to form a word dictionary

word_dict = []

counter = 0
for text in title:
    tokens_title = tokenizer.tokenize(text)
    tokens_title = [token.strip() for token in tokens_title] 
    
    desc_text = desc[counter]
    tokens_desc = tokenizer.tokenize(desc_text)
    tokens_desc = [token.strip() for token in tokens_desc]
    
    merge = tokens_title+tokens_desc
    word_list = set()
    for item in merge:
        word_list.add(item)
        
    word_dict.append(list(word_list))
    
    counter += 1
    
counter = 0
for item in word_dict:
    word_dict[counter] = ' '.join(item)
    counter += 1    
    
#Subtitle topic 1
subt = list(df['topic1']) 

counter = 0
for item in subt:
    subt[counter] = ' '.join(item)
    counter += 1
       
####TF-IDF    
    
vectorizer = TfidfVectorizer(strip_accents='unicode')
word_mat = vectorizer.fit_transform(word_dict)
word_mat = word_mat.toarray()

word_mat = pd.DataFrame(word_mat)


#Feature Selection
## For time being only use title matrix
vectorizer = TfidfVectorizer(strip_accents='unicode')
title_mat = vectorizer.fit_transform(title)
title_mat = title_mat.toarray()
feature_mat = title_mat
feature_df = pd.DataFrame(title_mat)

#######Clustering

#Adding some more features to tf-idf matrix - scaling required

#title_mat['likes'] = df['likes']
#title_mat['dislike'] = df['dislike']
#title_mat['comment'] = df['comment']
#title_mat['senti_title'] = df['senti_title']
#title_mat['senti_desc'] = df['senti_desc']
#title_mat['senti_subt'] = df['senti_subt']

#Conversion of dataframe to spare matrix

no_of_cluster = 5

dense_matrix = np.array(feature_df.as_matrix(columns = None), dtype=bool).astype(np.int)
sparse_matrix = scipy.sparse.csr_matrix(dense_matrix)
kmeans = KMeans(n_clusters=no_of_cluster, random_state=0)
kmeans.fit(sparse_matrix)

print("Top terms per cluster:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(5):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print
    
clusters = kmeans.labels_.tolist()


#Counter for each cluster - To check cluster distribution
from collections import Counter
el = Counter(clusters)

#Cosine similarity

from sklearn.metrics.pairwise import cosine_similarity
dist = cosine_similarity(feature_mat)

#Silhouette  score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(dist, kmeans.labels_)

#Recommendation
#Testing with video id
df = df.reset_index()
df = df.drop(['index'],axis=1)
vid = 'iUdgD8kYU-E'
feature_df['id'] = df['id']
feature_df['clusters'] = clusters
df['clusters'] = clusters
rec_obj = recommendation()
least_rel,most_rel = rec_obj.getRecommendation(vid,df,feature_df)

print("The titles of most relevent recommendation")

for item in most_rel:
    title_str = df[df['id']== item[0]]['title']
    print('Title: ' + str(title_str))
    
print("The title of least relevent recommendation")

for item in least_rel:
    title_str = df[df['id']== item[0]]['title']
    print('Title: ' + str(title_str))    





