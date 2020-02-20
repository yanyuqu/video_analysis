

import numpy as np
import pandas as pd
import base64
import os
import os.path
import pickle
import pandas as pd
import argparse
import io
import google.cloud
import google.oauth2.credentials
import google_auth_oauthlib.flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.cloud import bigquery
import json
import httplib2
from oauth2client.client import OAuth2WebServerFlow
from oauth2client.file import Storage
from oauth2client import client
from pandas.io.json import json_normalize
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from google.auth.transport.requests import Request
from google.cloud.videointelligence import enums
import re


global frame
global statistics



CLIENT_SECRETS_FILE = 'C:/Users/Tushar/Documents/DataOPs/client_secret_youtube.json'


def get_analytics_service():
    SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]
    TOKEN_FILE = 'C:/Users/Tushar/Documents/DataOPs/token_data.pickle'
    API_SERVICE_NAME = 'youtubeAnalytics'
    API_VERSION = 'v2'
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server()
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
    return build(API_SERVICE_NAME, API_VERSION, credentials=creds)


def get_data_service():
    SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]
    TOKEN_FILE = 'C:/Users/Tushar/Documents/DataOPs/token_data.pickle'
    API_SERVICE_NAME = 'youtube'
    API_VERSION = 'v3'
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server()
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
    return build(API_SERVICE_NAME, API_VERSION, credentials=creds)


def execute_api_request(client_library_function, **kwargs):
    global response_analytics
    global results
    response_analytics = client_library_function(
        **kwargs
    ).execute()
    df = json_normalize(response_analytics, 'rows')
    results = df.rename(columns={0: "Time_Ratio", 1: "audiencewatchratio", 2: "relativeRetentionPerformance"})
    return results


def analyze_labels(path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Tushar/Documents/DataOPs/video_api.json"
    from google.cloud import videointelligence
    # [START video_analyze_labels]
    """Detect labels given a file path."""
    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.enums.Feature.LABEL_DETECTION]

    mode = videointelligence.enums.LabelDetectionMode.SHOT_AND_FRAME_MODE
    config = videointelligence.types.LabelDetectionConfig(
        label_detection_mode=mode)
    context = videointelligence.types.VideoContext(
        label_detection_config=config)

    operation = video_client.annotate_video(
        path, features=features, video_context=context)
    print('\nProcessing video for label annotations:')

    result = operation.result(timeout=180)
    print('\nFinished processing.')

    df1 = []
    # Process shot level label annotations
    shot_labels = result.annotation_results[0].shot_label_annotations
    label_row1 = {}
    for i, shot_label in enumerate(shot_labels):
        print('Shot label description: {}'.format(
            shot_label.entity.description))
        label_row1['Description'] = shot_label.entity.description

        for category_entity in shot_label.category_entities:
            print('\tLabel category description: {}'.format(
                category_entity.description))
        for i, shot in enumerate(shot_label.segments):
            start_time = (shot.segment.start_time_offset.seconds +
                          shot.segment.start_time_offset.nanos / 1e9)
            end_time = (shot.segment.end_time_offset.seconds +
                        shot.segment.end_time_offset.nanos / 1e9)
            positions = '{}s to {}s'.format(start_time, end_time)
            confidence = shot.confidence
            row_segment_info1 = ({'Confidence': shot.confidence, 'Start': start_time, 'End': end_time})
            print(row_segment_info1)
            label_row1.update(row_segment_info1)
            print(label_row1)
            df1.append(label_row1.copy())
            print('\tSegment {}: {}'.format(i, positions))
            print('\tConfidence: {}'.format(confidence))
        print('\n')
    frame_shot = pd.DataFrame(df1)
    frame_shot = frame_shot.sort_values('Start')
    frame_shot = frame_shot[['Start', 'End', 'Description', 'Confidence']]
    return frame_shot


def analyze_videos(url):
    global statistics
    global frame

    split = url.split("=")

    youtube = get_data_service()
    request = youtube.videos().list(part="snippet,contentDetails,statistics", id=split[1])
    response = request.execute()
    title = response["items"][0]["snippet"]["title"]
    statistics = response["items"][0]["statistics"]

    youtubeAnalytics = get_analytics_service()
    results = execute_api_request(
        youtubeAnalytics.reports().query,
        ids='channel==UCuA4QDy-VPkbhkiLUclP1SA',
        startDate='2019-04-12',
        endDate='2019-08-20',
        filters="video==" + split[1] + ";audienceType==ORGANIC",
        metrics="audienceWatchRatio,relativeRetentionPerformance",
        dimensions='elapsedVideoTimeRatio')

    path = 'gs://email_match_shiny/' + split[1] + '.mp4'

    # Calling the analyze labels function for the video analysis. Currently shot level analysis is being done.
    frame_shot = analyze_labels(path)

    # Data Procssing to map and create retention metrics associated with each label
    video_length = round(frame_shot['End'].iloc[-1])
    results['time_elapsed'] = results['Time_Ratio'] * video_length
    results['delta_retention'] = results['relativeRetentionPerformance'].diff().fillna(
        results['relativeRetentionPerformance']).astype(float)
    results['delta_retention'].iloc[0] = 0.00
    print(frame_shot)
    print(results)

    df1_map = results.set_index('time_elapsed')['delta_retention'].to_dict()
    frame_shot['cum_delta_retention'] = frame_shot.apply(
        lambda x: (['{v:.6f}'.format(v=v) for k, v in df1_map.items() if k >= x.Start and k <= x.End]), axis=1)
    frame = frame_shot[frame_shot.astype(str)['cum_delta_retention'] != '[]']
    frame = frame.loc[frame['Confidence'] >= 0.6]
    frame['cum_delta_retention'] = frame['cum_delta_retention'].apply(lambda x: pd.to_numeric(x))
    frame['sum'] = frame['cum_delta_retention'].apply(sum)
    frame = frame.drop(columns=['cum_delta_retention'])
    return frame
	
def bq_upload(frame):
    PROJECT = 'shiny-demo'
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Tushar/Documents/DataOPs/bq_api.json"
    full_table_id = 'dataops.new_table'
    frame.to_gbq(full_table_id, project_id=PROJECT, if_exists='replace')
    
    
