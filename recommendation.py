#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:49:47 2018

@author: vivekmishra
"""
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import string
import unicodedata
import nltk
nltk.download('words')
from nltk.tokenize.toktok import ToktokTokenizer

nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english')

from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
nlp = spacy.load('en', parse=True, tag=True, entity=True)
import operator
#vid = 'iUdgD8kYU-E'

class recommendation:
    
    def getRecommendation(self,vid,df,feature_df):
        cluster_no = feature_df[feature_df['id'] == vid]['clusters']
        cluster_no = cluster_no.item()
        #select data of that cluster
        feature_df = feature_df.set_index('id')
        slice_df = feature_df[feature_df['clusters'] == cluster_no]
        #Here we do all feature engineering again for this cluster
        title_temp = list(df[df['clusters'] == cluster_no]['title'])

        counter = 0
        for sent in title_temp:
            title_temp[counter]=self.strip_links(sent)
            counter += 1
            
        counter = 0
        for sent in title_temp:
            title_temp[counter]=self.strip_hashtag(sent)
            counter += 1
        
        counter = 0
        for sent in title_temp:
            title_temp[counter]=self.remove_special_characters(sent, 
                                  remove_digits=True)
            counter += 1
            
        counter = 0
        for sent in title_temp:
            title_temp[counter]=self.remove_stopwords(sent)
            counter += 1
        
        counter = 0
        for sent in title_temp:
            title_temp[counter]=self.lemmatize_text(sent)
            counter += 1
            
        vectorizer = TfidfVectorizer(strip_accents='unicode')
        title_temp_mat = vectorizer.fit_transform(title_temp)
        title_temp_mat = title_temp_mat.toarray()
        
        title_temp_mat = pd.DataFrame(title_temp_mat)
        #Cosine similarity
        cos_sim = cosine_similarity(title_temp_mat.values)
        df_temp = pd.DataFrame(cos_sim, columns=slice_df.index.values, index=slice_df.index)
        
        interest_list = dict(df_temp[vid])
        index_min = sorted(interest_list.items(), key=operator.itemgetter(1))[:3]
        index_max = sorted(interest_list.items(), key=operator.itemgetter(1),reverse=True)[:4]
        
        return index_min,index_max
    
    def strip_links(self,text):
        link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
        links         = re.findall(link_regex, text)
        for link in links:
            text = text.replace(link[0], ', ')    
        return text

    def strip_hashtag(self,text):
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
    
    def lemmatize_text(self,text):
        text = nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
        return text
    
        
    def remove_special_characters(self,text, remove_digits=False):
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, ' ', text)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text
    
        
    def remove_stopwords(self,text, is_lower_case=False):
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        whitelist = ["n't","not", "no"]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if (token not in stopword_list or token in whitelist)]
        else:
            filtered_tokens = [token for token in tokens if (token.lower() not in stopword_list or token in whitelist)]
        filtered_text = ' '.join(filtered_tokens)    
        return filtered_text

    
    
    
    