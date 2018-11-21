#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 18:12:53 2018

@author: vivekmishra
"""


# -*- coding: utf-8 -*-

import os

import google.oauth2.credentials

import google_auth_oauthlib.flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow

# The CLIENT_SECRETS_FILE variable specifies the name of a file that contains
# the OAuth 2.0 information for this application, including its client_id and
# client_secret.
CLIENT_SECRETS_FILE = "client_secret.json"

# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account and requires requests to use an SSL connection.
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

def get_authenticated_service():
  flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
  credentials = flow.run_console()
  return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

def print_response(response):
  print(response)

# Build a resource based on a list of properties given as key-value pairs.
# Leave properties with empty values out of the inserted resource.
def build_resource(properties):
  resource = {}
  for p in properties:
    # Given a key like "snippet.title", split into "snippet" and "title", where
    # "snippet" will be an object and "title" will be a property in that object.
    prop_array = p.split('.')
    ref = resource
    for pa in range(0, len(prop_array)):
      is_array = False
      key = prop_array[pa]

      # For properties that have array values, convert a name like
      # "snippet.tags[]" to snippet.tags, and set a flag to handle
      # the value as an array.
      if key[-2:] == '[]':
        key = key[0:len(key)-2:]
        is_array = True

      if pa == (len(prop_array) - 1):
        # Leave properties without values out of inserted resource.
        if properties[p]:
          if is_array:
            ref[key] = properties[p].split(',')
          else:
            ref[key] = properties[p]
      elif key not in ref:
        # For example, the property is "snippet.title", but the resource does
        # not yet have a "snippet" object. Create the snippet object here.
        # Setting "ref = ref[key]" means that in the next time through the
        # "for pa in range ..." loop, we will be setting a property in the
        # resource's "snippet" object.
        ref[key] = {}
        ref = ref[key]
      else:
        # For example, the property is "snippet.description", and the resource
        # already has a "snippet" object.
        ref = ref[key]
  return resource

# Remove keyword arguments that are not set
def remove_empty_kwargs(**kwargs):
  good_kwargs = {}
  if kwargs is not None:
    for key, value in kwargs.items():
      if value:
        good_kwargs[key] = value
  return good_kwargs

def search_list_by_keyword(client, **kwargs):
  # See full sample for function
  kwargs = remove_empty_kwargs(**kwargs)

  response = client.search().list(
    **kwargs
  ).execute()

  return response



  # When running locally, disable OAuthlib's HTTPs verification. When
  # running in production *do not* leave this option enabled.
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
  #client = get_authenticated_service()
  
# 1. Search by category  
data = search_list_by_keyword(client,
part='snippet',
maxResults=50,
videoCategoryId=25,
regionCode='US',
type='video')
  
temp = data['items']
#Store the results and get next page token
vid_ids = set()
nextPageToken = data['nextPageToken']
print(nextPageToken)
for row in temp:
    vid_ids.add(row['id']['videoId'])
    
    chanelID = row['snippet']['channelId']
    # 2. Search by channelid
    channel_data = search_list_by_keyword(client,
                    part='snippet',
                    maxResults=50,
                    channelId=chanelID,
                    regionCode='US',
                    type='video')
    
    temp_channel = channel_data['items']
    
    nextPageTokenChannel = channel_data['nextPageToken']
    for row_channel in temp_channel:
        vid_ids.add(row_channel['id']['videoId'])
    
    #keep making request to subsequent channels
    channel_counter = 0
    while nextPageTokenChannel:
        print("channel counter:" +str(channel_counter))
        
        channel_data = search_list_by_keyword(client,
                    part='snippet',
                    maxResults=50,
                    channelId=chanelID,
                    regionCode='US',
                    pageToken=nextPageTokenChannel,
                    type='video')
        
        temp_channel = channel_data['items']
    
        if 'nextPageToken' in channel_data.keys():
            nextPageTokenChannel = channel_data['nextPageToken']
        else:   
            break
        for row_channel in temp_channel:
            vid_ids.add(row_channel['id']['videoId'])
        
        channel_counter += 1

import numpy as np

#Next request
count = 1    
while  nextPageToken:   
    print("token count"+str(count))
    #print(nextPageToken)
    data = search_list_by_keyword(client,
    part='snippet',
    maxResults=50,
    videoCategoryId=25,
    regionCode='US',
    type='video',
    pageToken=nextPageToken)    
    
    temp = data['items']
  
    if 'nextPageToken' in data.keys():
        nextPageToken = data['nextPageToken']
    else:
        break
    for row in temp:
        vid_ids.add(row['id']['videoId'])
        
        chanelID = row['snippet']['channelId']
        # 2. Search by channelid
        channel_data = search_list_by_keyword(client,
                        part='snippet',
                        maxResults=50,
                        channelId=chanelID,
                        regionCode='US',
                        type='video')
        
        temp_channel = channel_data['items']
        
        if 'nextPageToken' in channel_data.keys():
            nextPageTokenChannel = channel_data['nextPageToken']
        else:
            nextPageTokenChannel = ''
        for row_channel in temp_channel:
            vid_ids.add(row_channel['id']['videoId'])
        
        #keep making request to subsequent channels
        channel_counter = 0
        while nextPageTokenChannel:
            print("channel counter:" +str(channel_counter))
            
            channel_data = search_list_by_keyword(client,
                        part='snippet',
                        maxResults=50,
                        channelId=chanelID,
                        regionCode='US',
                        pageToken=nextPageTokenChannel,
                        type='video')
            
            temp_channel = channel_data['items']
        
            if 'nextPageToken' in channel_data.keys():
                nextPageTokenChannel = channel_data['nextPageToken']
            else:   
                break
            for row_channel in temp_channel:
                vid_ids.add(row_channel['id']['videoId'])
            
            channel_counter += 1
        
    
    count += 1
    
import pandas as pd    
news = pd.DataFrame()

news['id'] = list(vid_ids)
news.to_pickle('news.pkl')
