#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 19:37:06 2018

@author: vivekmishra
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import unicodedata
import re

class senti:

    def sentimentFinder(self,sent):
        analyser = SentimentIntensityAnalyzer()
        snt = analyser.polarity_scores(sent)
        return(snt['compound'])
        
    def remove_special_characters(self,text):
        pattern = r'[^a-zA-z0-9\s.,]'
        text = re.sub(pattern, ' ', text)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text
    
    def main(self,text):
        text = text.replace("&amp;#39;","'")
        #text = remove_special_characters(text)    
    
        score =  self.sentimentFinder(text)
        return score               