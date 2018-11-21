#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 18:52:41 2018

@author: vivekmishra
"""
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import unicodedata
import re
from gensim import corpora, models


import nltk
nltk.download('wordnet')

class LDA_new:

    def remove_special_characters(self,text):
        pattern = r'[^a-zA-z0-9\s.,]'
        text = re.sub(pattern, ' ', text)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text
    
    def lemmatize_stemming(self,text):
        #stemmer = SnowballStemmer('english')
        return WordNetLemmatizer().lemmatize(text, pos='v')
    
    def preprocess(self,text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(self.lemmatize_stemming(token))
        return result
     
    
    
    def main(self,text,counter):
        
        text = text.replace("&amp;#39;","'")
        text = self.remove_special_characters(text)
        text = text.strip()
        if len(text) > 0:
            text = text.split(".")
        
            temp = list(map(self.preprocess,text))
            dictionary = gensim.corpora.Dictionary(temp)
            bow_corpus = [dictionary.doc2bow(doc) for doc in temp]
            
            lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
            topic_list =  lda_model.show_topic(counter,10)
            
            return_list = []
            
            for items in topic_list:
                return_list.append(items[0])
                
            return return_list
        else:
            return []
        
    