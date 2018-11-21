#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 16:29:29 2018

@author: vivekmishra
"""


from sklearn.feature_extraction.text import  CountVectorizer
# from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
import nltk
nltk.download('punkt')
from nltk import sent_tokenize
import re
import unicodedata

class LDA:
    
    
    def remove_special_characters(self,text):
        pattern = r'[^a-zA-z0-9\s.,]'
        text = re.sub(pattern, ' ', text)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    
    def display_topics(self,model, feature_names, no_top_words):
       topics = [] 
       for topic_idx, topic in enumerate(model.components_):
           #print ("Topic %d:" % (topic_idx))
           #print (" ".join([feature_names[i]
           #                for i in topic.argsort()[:-no_top_words - 1:-1]]))
           topics.append([feature_names[i]
                           for i in topic.argsort()[:-no_top_words - 1:-1]]) 
           
       return topics
                
    
    def main(self,text,counter):
     
        #text = df['subtitle'].iloc[12]    
        text = text.replace('"',' ')
        text = text.replace('“',' ')
        text = text.replace('”',' ')
        text = text.replace("'",' ')
        text = text.replace("…",' . ')
        text = text.replace(":",' ')
        text = text.replace("&amp;#39;","'")
        no_features = 100
        text = self.remove_special_characters(text)
        documents=sent_tokenize(text)
        
        print(text)
    
        tf_vectorizer = CountVectorizer(max_df=0.60, min_df=1, max_features=no_features,stop_words="english")
        tf = tf_vectorizer.fit_transform(documents)
        tf_feature_names = tf_vectorizer.get_feature_names()
        
        no_topics = 3
    
    
        # Run LDA
        lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
        
        no_top_words = 9
        # display_topics(nmf, tfidf_feature_names, no_top_words)
        topics = self.display_topics(lda, tf_feature_names, no_top_words)
        
        return topics[counter]