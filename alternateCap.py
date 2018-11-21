#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 02:39:00 2018

@author: vivekmishra
"""

import requests
import re

class alternateCap:
    
    def downloadCap(self,id):
        
        print(id)
        url = "http://video.google.com/timedtext?lang=en&v="+id+"&fmt=srv1"
        r = requests.get(url)
        data = r.text
        text = re.sub('<[^<]+>', "",data)
        
        return text