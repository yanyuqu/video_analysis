# -*- coding: utf-8 -*-

# Sample Python code for youtubeAnalytics.reports.query
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/guides/code_samples#python

import os

import google_auth_oauthlib.flow
import googleapiclient.discovery
import pandas as pd
import numpy as np
import datetime
import googleapiclient.errors
global response
global results_final
global df_empty
from pandas.io.json import json_normalize

scopes = ["https://www.googleapis.com/auth/youtube.readonly"]

def main():
    global response
    global results_final
    global df_final
    global df_empty
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtubeAnalytics"
    api_version = "v2"
    client_secrets_file = "XXXXXXXX"

    # Get credentials and create an API client
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        client_secrets_file, scopes)
    credentials = flow.run_console()
    youtube_analytics = googleapiclient.discovery.build(
        api_service_name, api_version, credentials=credentials)

    columns = ['Date']
    df_empty = pd.DataFrame(columns=columns)
    df_empty['Date'] = pd.date_range(start='1/1/2018', end='31/12/2019')
    df_empty['Date'] = df_empty['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
	
	#Load all the video id for which you need to pull the stats.
    data = pd.read_csv("C:/Users/Tushar/Documents/Serato_Video_Intelligence/list_off_five_hundred_videos.csv")
    video_ids = data['video_id'].values.tolist()
    
    for video in video_ids:
        print(video)
        request = youtube_analytics.reports().query(dimensions="day",endDate="2019-12-31",filters="video=="+video,ids="channel==MINE",metrics="likes",startDate="2018-01-01")
        response = request.execute()
        df = json_normalize(response, 'rows')
        results = df.rename(columns={0: "Date", 1: video})
        df_empty = df_empty.merge(results, on='Date', how='left')
        
        #results_final = df_empty.merge(df_final, on='Date', how='left')

if __name__ == "__main__":
    main()