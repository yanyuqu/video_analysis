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
from google.cloud import videointelligence
from google.cloud.videointelligence import enums
global results
global frame_shot
global frame_segment
global time_span_label
global label_retention
global title

CLIENT_SECRETS_FILE = 'C:/Users/Tushar/Documents/Serato_Video_Intelligence/client_secret_youtube.json'

#Function to get the service for youtube analytics api
def get_analytics_service():
    SCOPES = ['https://www.googleapis.com/auth/yt-analytics.readonly']
    TOKEN_FILE = 'C:/Users/Tushar/Documents/Serato_Video_Intelligence/token.pickle'
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

#Function to get the service for youtube data api
def get_data_service():
    SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]
    TOKEN_FILE = 'C:/Users/Tushar/Documents/token_data.pickle'
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

#Execution function to get the analytics data  
def execute_api_request(client_library_function, **kwargs):
  global response
  global results
  response = client_library_function(
    **kwargs
  ).execute()
  df = json_normalize(response, 'rows')
  results = df.rename(columns={0: "Time_Ratio", 1: "audiencewatchratio", 2: "relativeRetentionPerformance"})


#Video analysis function taken from the google documentation
def analyze_labels(path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/Tushar/Documents/Serato_Video_Intelligence/video_api.json"
    global df
    global df1
    global df2
    global frame_shot
    global frame_segment
    global shot_labels
    # [START video_analyze_labels_gcs]
    """ Detects labels given a GCS path. """
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
    df = []
    # Process video/segment level label annotations
    segment_labels = result.annotation_results[0].segment_label_annotations
    for i, segment_label in enumerate(segment_labels):
        print('Video label description: {}'.format(
            segment_label.entity.description))
        label_row = {}
        for category_entity in segment_label.category_entities:
            print('\tLabel category description: {}'.format(
                category_entity.description))
            label_row['Description'] = category_entity.description

        for i, segment in enumerate(segment_label.segments):
            start_time = (segment.segment.start_time_offset.seconds +
                          segment.segment.start_time_offset.nanos / 1e9)
            end_time = (segment.segment.end_time_offset.seconds +
                        segment.segment.end_time_offset.nanos / 1e9)
            positions = '{}s to {}s'.format(start_time, end_time)
            confidence = segment.confidence
            row_segment_info = ({'Confidence': segment.confidence, 'Start': start_time, 'End': end_time})
            label_row.update(row_segment_info)
            df.append(label_row)
            print('\tSegment {}: {}'.format(i, positions))
            print('\tConfidence: {}'.format(confidence))
        print('\n')
    
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

    # Process frame level label annotations
    frame_labels = result.annotation_results[0].frame_label_annotations
    for i, frame_label in enumerate(frame_labels):
        print('Frame label description: {}'.format(
            frame_label.entity.description))
        for category_entity in frame_label.category_entities:
            print('\tLabel category description: {}'.format(
                category_entity.description))

        # Each frame_label_annotation has many frames,
        # here we print information only about the first frame.
        frame = frame_label.frames[0]
        time_offset = (frame.time_offset.seconds +
                       frame.time_offset.nanos / 1e9)
        print('\tFirst frame time offset: {}s'.format(time_offset))
        print('\tFirst frame confidence: {}'.format(frame.confidence))
        print('\n')
    frame_shot = pd.DataFrame(df1)
    frame_segment = pd.DataFrame(df)
    frame_shot = frame_shot.sort_values('Start')
    frame_segment = frame_segment.sort_values('Start')
    frame_segment = frame_segment[['Start', 'End', 'Description', 'Confidence']]
    frame_shot = frame_shot[['Start', 'End', 'Description', 'Confidence']]
    # [END video_analyze_labels_gcs]
    

if __name__ == '__main__':
  # Disable OAuthlib's HTTPs verification when running locally.
  # *DO NOT* leave this option enabled when running in production.
  #os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
  global time_span_label
  global label_retention
  global title

  #Authenticating again to GCP to dump the data to big query table
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/Tushar/Documents/Serato_Video_Intelligence/video_api.json"
  
  #Target Project ID
  PROJECT = 'serato-big-query'
  
  #Target big query table to dump the data
  full_table_id = 'serato_video_analysis.serato_studio'
  
  #Reading all the video urls stored as a pickle file. This can be stored on gcs when deployed in production
  load_url = pickle.load(open('C:/Users/Tushar/Documents/Serato_Video_Intelligence/studio_url', 'rb'))
  
  #Extracting the Video ID from the loaded url
  split = load_url[0].split("=")
  
  #Connecting to the youtube analytics api and fetching the analytics data
  youtubeAnalytics = get_analytics_service()
  execute_api_request(
      youtubeAnalytics.reports().query,
      ids='channel==UCuA4QDy-VPkbhkiLUclP1SA',
      startDate='2019-04-12',
      endDate='2019-08-13',
      filters= 'video=='+split[1]+';audienceType==ORGANIC',
      metrics='audienceWatchRatio,relativeRetentionPerformance',
      dimensions='elapsedVideoTimeRatio')

  #Connecting to the youtube data api and the fetching the title of the video
  youtube = get_data_service()
  request = youtube.videos().list(part="snippet,contentDetails,statistics",id=split[1])
  response = request.execute()
  title = response["items"][0]["snippet"]["title"]
  
  #Path to the video stored on gcs, the video is stored as renamed as id. This needs to be tested if we can use the title of the video in place of the video id. 
  path = 'gs://serato_youtube_videos/Serato_Studio_Tutorials/'+title+'.mp4'
  
  #Calling the analyze labels function for the video analysis. Currently shot level analysis is being done.
  analyze_labels(path)

  #Data Procssing to map and create retention metrics associated with each label
  video_length = round(frame_shot['End'].iloc[-1])	  
  results['time_elapsed'] = results['Time_Ratio'] * video_length
  results['delta_retention'] = results['relativeRetentionPerformance'].diff().fillna(results['relativeRetentionPerformance']).astype(float)
  results['delta_retention'].iloc[0] = 0.00
  
  df1_map = results.set_index('time_elapsed')['delta_retention'].to_dict()
  frame_shot['cum_delta_retention'] = frame_shot.apply(lambda x: ([f'{v:.6f}' for k, v in df1_map.items() if k >= x.Start and k <= x.End]), axis=1)
  frame = frame_shot[frame_shot.astype(str)['cum_delta_retention'] != '[]']
  frame = frame.loc[frame['Confidence'] >= 0.6]
  frame['cum_delta_retention'] = frame['cum_delta_retention'].apply(lambda x : pd.to_numeric(x))
  frame['sum'] = frame['cum_delta_retention'].apply(sum)
  time_span_label = frame.groupby(['Start', 'End', 'Description'])['sum'].mean()
  label_retention = frame.groupby(['Description'])['sum'].mean()
  label_retention = label_retention.to_frame().reset_index()
  label_retention['Title'] = title
  #label_retention.to_gbq(full_table_id, project_id=PROJECT, if_exists='append')

  
  
  
  
