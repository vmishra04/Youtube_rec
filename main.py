#CapDownload#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 20:10:44 2018

@author: vivekmishra
"""

import os
os.chdir('/Users/vivekmishra/Desktop/USC/599-DSS/project')
import requests


import tensorflow as tf
import numpy as np
import pandas as pd
from IPython.display import YouTubeVideo

from yt_api import Description
#from cap import CapDownload
from alternateCap import alternateCap
from LDA import LDA
from LDA_new import LDA_new
from senti import senti


directory = '/Users/vivekmishra/Desktop/USC/599-DSS/project/video/'
vid_ids = []
labels = []
mean_rgb = []
mean_audio = []

for filename in os.listdir(directory):
    if filename.endswith(".tfrecord"):
        if filename.startswith("train"):
            video_lvl_record = directory+'/'+filename
            for example in tf.python_io.tf_record_iterator(video_lvl_record):
                tf_example = tf.train.Example.FromString(example)
        
                vid_ids.append(tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
                labels.append(tf_example.features.feature['labels'].int64_list.value)
                mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)
                mean_audio.append(tf_example.features.feature['mean_audio'].float_list.value)
                
 
temp_df = pd.DataFrame()               
temp_df['id'] = vid_ids
temp_df['mean_rgb'] = mean_rgb
temp_df['mean_audio'] = mean_audio
temp_df['labels'] = labels

temp_df.to_csv("tfrecords.csv")

head = temp_df.head(100)
    
    

def getVideoID(id):
    url = 'http://data.yt8m.org/2/j/i/'+id[0]+id[1]+'/'+id+'.js'
    r = requests.get(url)
    data = r.text
    arr = data.split(",")
    if len(arr) > 1:
        new_id = arr[1]
        new_id = new_id.strip('"')
        new_id = new_id.rstrip('");')
        return new_id
    else:
        return np.nan
    
####Analysis - Not required
print('Number of videos in this tfrecord: ',len(mean_rgb))
print('First video feature length',len(mean_rgb[0]))
print('First 20 features of the first youtube video (',vid_ids[0],')')
print(mean_rgb[0][:20])

def play_one_vid(record_name, video_index):
    id = vid_ids[video_index]
    new_id = getVideoID(id)
    return new_id
    
# Show Video
YouTubeVideo(play_one_vid(video_lvl_record, 0))

####

### From this part we create the main dataframe for sentiment analysis part
### DF contains video_id,description,title,likes,views for now

df = pd.DataFrame()



#Get Video_id for each video
new_id = []
count = 1
for vid_id in vid_ids:
    print(count) 
    new_id.append(getVideoID(vid_id))
    count = count+1
       
df['id'] = new_id
df.to_pickle("df.pkl")

#
<<<<<<< HEAD
df = pd.read_pickle("news.pkl")
=======
df = pd.read_pickle("df_new.pkl")
>>>>>>> 5723a44c9754a697cb591c5cb11d3debef07dce7
new_id = df['id']
    
#Get Desc,Title, Tags, Views
#Description of label
text = pd.read_csv('https://research.google.com/youtube8m/csv/2/vocabulary.csv')
description = Description()

desc_data = []
title = []
caption = []
views = []
likes = []
dislike = []
favorite = []
comment = []
count = 1
desc_id = []
tags = []
label_tag = []
lang = []
index = 0
new_id_temp = new_id[20829:len(new_id)-1]
for vid_id in new_id_temp:
    #client = description.get_authenticated_service()
    print(count)
    data = description.videos_list_by_id(client,part='snippet,contentDetails,statistics',id=vid_id)
    if 'items' in data.keys():
        if len(data['items']) > 0:
            desc_id.append(vid_id)
            #label_tag.append([text['Name'][id] for id in labels[index]])
            label_tag.append(data['items'][0]['snippet']['categoryId'])
            desc_data.append(data['items'][0]['snippet']['description'])  
            title.append(data['items'][0]['snippet']['title'])
            caption.append(data['items'][0]['contentDetails']['caption'])
            
            if 'defaultAudioLanguage' in data['items'][0]['snippet'].keys():
                lang.append(data['items'][0]['snippet']['defaultAudioLanguage'])
            else:
                lang.append('default')
            
            if 'tags' in data['items'][0]['snippet'].keys():
                tags.append(data['items'][0]['snippet']['tags'])
            else:
                tags.append(np.nan)
            
            if 'viewCount' in data['items'][0]['statistics'].keys():
                views.append(int(data['items'][0]['statistics']['viewCount']))
            else:
                views.append(np.nan)
                
            if 'likeCount' in data['items'][0]['statistics'].keys():
                likes.append(int(data['items'][0]['statistics']['likeCount']))
            else:
                likes.append(np.nan)
                
            if 'dislikeCount' in data['items'][0]['statistics'].keys():   
                dislike.append(int(data['items'][0]['statistics']['dislikeCount']))
            else:
                dislike.append(np.nan)
                
            if 'favoriteCount'  in  data['items'][0]['statistics'].keys():  
                favorite.append(int(data['items'][0]['statistics']['favoriteCount']))
            else:
                favorite.append(np.nan)
                
            if 'commentCount' in  data['items'][0]['statistics'].keys():  
                comment.append(int(data['items'][0]['statistics']['commentCount']))
            else:
                comment.append(np.nan)
                
            count = count+1
    index+=1
    


df['id'] = desc_id
df['title'] = title
df['desc'] = desc_data
df['views'] = views
df['likes'] = likes
df['dislike'] = dislike
df['favorite'] = favorite
df['comment'] = comment
df['tags'] = tags
df['caption'] = caption
df['labels'] = label_tag


df.to_pickle("df_new.pkl")
df.to_csv("df_new.csv")



------------

#Get List of Captions + download

cap = alternateCap() 
head['subtitle'] = head.apply(lambda row: cap.downloadCap(row['id']) if row['caption'] == 'true' 
                                else np.nan,axis=1)
<<<<<<< HEAD



news = pd.DataFrame()
=======
>>>>>>> 5723a44c9754a697cb591c5cb11d3debef07dce7

news['id'] = desc_id
?/?.//news['title'] = title
news['desc'] = desc_data
news['views'] = views
news['likes'] = likes
news['dislike'] = dislike
news['favorite'] = favorite
news['comment'] = comment
news['tags'] = tags
news['caption'] = caption
news['labels'] = label_tag


news.to_pickle("news.pkl")
news.to_csv("news_politics.csv")



df = pd.read_pickle('news.pkl')
head = df.head(1000)

cap = alternateCap() 
df['subtitle'] = df.apply(lambda row: cap.downloadCap(row['id']) if row['caption'] == 'true' 
                                else np.nan,axis=1)

head = df.head(1000)
df.to_csv('news_politics_caption.csv')
df.to_pickle("df_latest.pkl")

df = pd.read_pickle('df_latest.pkl')

#----------------------------------------

#Subset by availability of caption
df = df[df['subtitle'].isnull() == False]
df = df[df['subtitle'].str.len() > 0]

#LDA
lda = LDA_new()
topic1_list = []
count = 0
for i,j in df.iterrows():
    print(count)
    topic1_list.append(lda.main((str(j["subtitle"])),0))
    count += 1

df['topic1'] = topic1_list

lda = LDA_new()
topic2_list = []
count = 0
for i,j in df.iterrows():
    print(count)
    topic2_list.append(lda.main((str(j["subtitle"])),1))
    count += 1
    
df['topic2'] = topic2_list

df.to_pickle("df_lda.pkl")

#Sentiment analysis

senti = senti()
df['senti_title'] = df['title'].apply(lambda x : senti.main(x))
df['senti_desc'] = df['desc'].apply(lambda x : senti.main(x))
df['senti_subt'] = df['subtitle'].apply(lambda x: senti.main(x))

df.to_pickle("df_senti.pkl")