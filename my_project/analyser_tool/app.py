from textwrap import dedent

global frame
global statistics
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_player as player
import numpy as np
import pandas as pd
import dash_table as table
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import base64
import os
import os.path
import pickle
import pandas as pd
import youtube_dl
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
import re

from .server import app
# statistics = {'viewCount': '8351', 'likeCount': '151', 'dislikeCount': '13', 'favoriteCount': '0', 'commentCount': '31'}
CLIENT_SECRETS_FILE = 'credentials.json'


def get_analytics_service():
    SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]
    TOKEN_FILE = 'token_data.pickle'
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
    TOKEN_FILE = 'token_data.pickle'
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
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "video_api.json"
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

    result = operation.result(timeout=90)
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
        ids='channel==UCBKHQZ6GqOmFtnMZU4Fb-6g',
        startDate='2017-06-24',
        endDate='2019-08-20',
        filters="video==" + split[1] + ";audienceType==ORGANIC",
        metrics="audienceWatchRatio,relativeRetentionPerformance",
        dimensions='elapsedVideoTimeRatio')

    path = 'gs://artifacts.modular-granite-265122.appspot.com/' + split[1] + '.mp4'

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


def indicator(color, text, id_value):
    return html.Div(
        [

            html.P(
                text,
                className="twelve columns indicator_text",
            ),
            html.P(
                id=id_value,
                className="indicator_value",
                style={'font-weight': 'bold'}
            ),
        ],
        className="three columns indicator",
        style={
            'backgroundColor': 'white',
            'text-align': 'center',
            'border-radius': '10px',
            'margin-right': '5em'
        }
    )


#image_filename = 'Ascibe_logo.png'
#encoded_image = base64.b64encode(open(image_filename, 'rb').read())

DEBUG = True
FRAMERATE = 24.0


def markdown_popup():
    return html.Div(
        id='markdown',
        className="model",
        style={'display': 'none'},
        children=(
            html.Div(
                className="markdown-container",
                children=[
                    html.Div(
                        className='close-container',
                        children=html.Button(
                            "Close",
                            id="markdown_close",
                            n_clicks=0,
                            className="closeButton",
                            style={'border': 'none', 'height': '100%'}
                        )
                    ),
                    html.Div(
                        className='markdown-text',
                        children=[dcc.Markdown(
                            children=dedent(
                                '''
                                ##### More About This App

                                This app analyses and detect labels in your video using state-of-the-art Google Video Intelligence API.
                                Video Intelligence API has pre-trained machine learning models that automatically recognize a vast number 
                                of objects, places and actions in stored and streaming video. The app then maps those labels to the audience 
                                retention metrics for your video posted on youtube to give you a sense of what are the labels that 
                                has more audience retention power over others.


                                '''
                            ))
                        ]
                    )
                ]
            )
        )
    )


# Main App

app.layout = html.Div(
    children=[
        html.Div(
            id='top-bar',
            className='row',
            style={'backgroundColor': '#fa4f56',
                   'height': '5px',
                   }
        ),
        html.Div(
            className='container',
            children=[
                html.Div(
                    id='left-side-column',
                    className='eight columns',
                    style={'display': 'flex',
                           'flexDirection': 'column',
                           'flex': 1,
                           'height': 'calc(100vh - 5px)',
                           'backgroundColor': '#F2F2F2',
                           'overflow-y': 'scroll',
                           'marginLeft': '0px',
                           'justifyContent': 'flex-start',
                           'alignItems': 'center'},
                    children=[
                        html.Div(
                            id='header-section',
                            children=[
                                html.H4(
                                    'Label Detection Explorer'
                                ),
                                html.P(
                                    'To get started, provide a valid youtube url for any video from your youtube channel'
                                    ' and then click on Run Analysis. The visualization will be displayed once the processing'
                                    ' is complete. For more relativity, you can play the video to understand the data'
                                ),
                                html.Button("Learn More", id="learn-more-button", n_clicks=0)
                            ]
                        ),
                        html.Div(
                            className='video-outer-container',
                            children=html.Div(
                                style={'width': '100%', 'paddingBottom': '56.25%', 'position': 'relative'},
                                children=player.DashPlayer(
                                    id='video-display',
                                    style={'position': 'absolute', 'width': '100%',
                                           'height': '100%', 'top': '0', 'left': '0', 'bottom': '0', 'right': '0'},
                                    url='',
                                    controls=True,
                                    playing=False,
                                    volume=1,
                                    width='100%',
                                    height='100%'
                                )
                            )
                        ),
                        html.Div(
                            className='control-section',
                            children=[
                                html.Div(
                                    className='control-element',
                                    children=[
                                        html.Div(children=["Provide A Valid YouTube URL:"], style={'width': '40%'}),
                                        html.Div(dcc.Input(
                                            id='video_url',
                                            type='text',
                                            className='twelve columns'
                                        ), style={'width': '40%'}),
                                        html.Button('Run Analysis', id='submit_button',
                                                    style={'background-color': '#2D91C3', 'color': 'white',
                                                           'font-size': '1em',
                                                           'margin-left': '1em'})
                                    ]
                                ),

                                html.Div(
                                    className='control-element',
                                    children=[
                                        html.Div(children=["Graph View Mode:"], style={'width': '40%'}),
                                        dcc.Dropdown(
                                            id="dropdown-graph-view-mode",
                                            options=[
                                                {'label': 'Table Mode', 'value': 'tabular'},
                                                {'label': 'Graph Mode', 'value': 'graphical'}
                                            ],
                                            value='',
                                            searchable=False,
                                            clearable=False,
                                            style={'width': '60%'}
                                        )
                                    ]
                                ),

                                html.Div(
                                    className='control-element',
                                    children=[
                                        html.Div(children=["Video Statistics:"], style={'width': '40%'}),
                                        indicator(
                                            "#b5deff",
                                            "View Count",
                                            "view_indicator",
                                        ),
                                        indicator(
                                            "#119DFF",
                                            "Comment Count",
                                            "comment_indicator",
                                        ),
                                    ]
                                ),

                                html.Div(
                                    className='control-element',
                                    children=[
                                        html.Div(children=[""], style={'width': '40%'}),
                                        indicator(
                                            "#EF553B",
                                            "Like Count",
                                            "like_indicator",
                                        ),
                                        indicator(
                                            "#EF553B",
                                            "Dislike Count",
                                            "dislike_indicator",
                                        ),
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                html.Div(
                    id='right-side-column',
                    className='four columns',
                    style={
                        'height': 'calc(100vh - 5px)',
                        'overflow-y': 'scroll',
                        'marginLeft': '1%',
                        'display': 'flex',
                        'backgroundColor': '#F9F9F9',
                        'flexDirection': 'column'
                    },
                    children=[
                        html.Div(
                            className='img-container',
                            children=html.Img(
                                style={'height': '100%', 'margin': '2px'},
                                #src='data:image/png;base64,{}'.format(encoded_image.decode())
                            )
                        ),
                        html.Div(id="div-table-mode"),
                        html.Div(id="div-graph-mode")
                    ]
                )
            ],
            style={'backgroundColor': '#F2F2F2'}),
        markdown_popup(),
        html.Div(id='intermediate-value', style={'display': 'none'})
    ]
)


# Footage Selection
@app.callback(Output("video-display", "url"),
              [Input("submit_button", "n_clicks")],
              [State('video_url', 'value')])
def select_footage(n_clicks, video_url):
    # Find desired footage and update player video
    if n_clicks is not None and n_clicks > 0:
        url = video_url
        return url


# Learn more popup
@app.callback(Output("markdown", "style"),
              [Input("learn-more-button", "n_clicks"), Input("markdown_close", "n_clicks")])
def update_click_output(button_click, close_click):
    if button_click > close_click:
        return {"display": "block"}
    else:
        return {"display": "none"}


# Processing and Storing the results in intermediate div
@app.callback(Output("intermediate-value", "children"),
              [Input("submit_button", "n_clicks")],
              [State('video_url', 'value')]
              )
def video_processing(n_clicks, video_url):
    global frame
    if n_clicks is not None and n_clicks > 0:
        frame = analyze_videos(video_url)
        # frame = pd.read_csv("newframes.csv")
        print("video_results")

        return frame.to_json(orient='split')


@app.callback(Output("div-table-mode", "children"),
              [Input("dropdown-graph-view-mode", "value")])
def update_table_mode(dropdown_value):
    global frame

    if dropdown_value == "tabular":
        labels = frame[['Description', 'sum']]
        labels['Frequency'] = labels.groupby('Description')['Description'].transform('count')
        label_retention = labels.groupby(['Description', 'Frequency'])['sum'].mean().reset_index()
        label_retention['Description'] = label_retention['Description'].str.upper()
        label_retention = label_retention.rename(
            columns={'Description': 'Labels', 'sum': 'Relative Audience Retention'})
        label_retention = label_retention.sort_values(by='Relative Audience Retention', ascending=False)
        label_retention['Relative Audience Retention'] = label_retention['Relative Audience Retention'].round(6)
        timeline_retention = frame[['Start', 'End', 'Description', 'sum']]
        timeline_retention = timeline_retention.groupby(['Start', 'End', 'sum'])['Description'].apply(
            lambda x: ','.join(x)).reset_index()
        timeline_retention = timeline_retention.round({'Start': 2, 'End': 2, 'sum': 6})
        timeline_retention = timeline_retention.rename(
            columns={'Description': 'Labels', 'sum': 'Relative Audience Retention'})
        return [
            html.Div(
                children=[
                    html.P(children="Retention By Label",
                           className='plot-title', style={'margin': '0 0 1em 0'}),
                    html.Div([
                        table.DataTable(
                            id="label_retention",
                            columns=[{"name": i, "id": i} for i in label_retention.columns],

                            data=label_retention.to_dict("rows"),

                            style_table={'maxHeight': '40vh', 'width': '100%', 'overflowY': 'scroll'},
                            style_cell_conditional=[
                                {
                                    'if': {'column_id': c},
                                    'textAlign': 'left'
                                } for c in ['Labels']
                            ],
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(248, 248, 248)'
                                }
                            ],

                            style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold'
                            }
                        )],
                        style={'height': '40vh'}),

                    html.P(children="Retention by Time Stamp",
                           className='plot-title', style={'margin': '1em 0 1em 0'}),
                    html.Div([
                        table.DataTable(
                            id="timestamp_retention",
                            columns=[{"name": i, "id": i} for i in timeline_retention.columns],
                            data=timeline_retention.to_dict("rows"),
                            style_table={'maxHeight': '40vh', 'width': '100%', 'overflowY': 'scroll'},
                            style_cell={'textAlign': 'left', 'minWidth': '20px', 'width': '20px', 'maxWidth': '50px',
                                        'whiteSpace': 'normal'},
                            css=[{
                                'selector': '.dash-cell div.dash-cell-value',
                                'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                            }],
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(248, 248, 248)'
                                }
                            ],

                            style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold'
                            }
                        )],
                        style={'height': '40vh'}
                    )
                ],
                style={'backgroundColor': '#F2F2F2'}
            )
        ]
    else:
        return []


@app.callback(Output("div-graph-mode", "children"),
              [Input("dropdown-graph-view-mode", "value")])
def update_graph_mode(value):
    global frame
    if value == "graphical":
        label_retention = frame
        label_average = frame[['Description', 'sum']]
        #tips = px.data.tips()
       # minval = label_retention['sum'].min()
        #maxval = label_retention['sum'].max()
        #colors = list('rgb(250,79,86)' for i in range(len(label_retention)))
        return [
            html.Div(
                children=[
                    html.P(children="Retention Score of Detected Labels",
                           className='plot-title', style={'margin': '0 0 1em 0', 'width': '100%'}),
                    dcc.Graph(
                        id="bar-score-graph",
                        figure=go.Figure({
                            'data': [go.Box(y=label_average['sum'],
                                            x=label_average['Description'],
                                            name='Audience Retention',
                                            boxpoints=False,
                                            marker=dict(color='rgb(8,81,156)'))
                                     ]})
                            # 'layout': {'showlegend': False,
                            #            'autosize': False,
                            #            'paper_bgcolor': 'rgb(249,249,249)',
                            #            'plot_bgcolor': 'rgb(249,249,249)',
                            #            'xaxis': {'automargin': True, 'tickangle': -45},
                            #            'yaxis': {'automargin': True, 'range': [minval, maxval],
                            #                      'title': {'text': 'Score'}}}

                        ,
                        style={'height': '55vh', 'width': '100%'}
                    ),
                    html.P(children="Audience Retention Behavior",
                           className='plot-title', style={'margin': '0 0 1em 0', 'width': '100%'}),
                    dcc.Graph(
                        id="line_chart_retention",
                        figure=go.Figure({
                            'data': [go.Scatter(x=label_retention['Start'], y=label_retention['sum'], mode='lines',
                                                name='Audience Retention',
                                                line=dict(color='firebrick', width=4))],
                            'layout': {
                                'yaxis': {'title': {'text': 'Audience Retention'}, 'automargin': True},
                                'xaxis': {'title': {'text': 'Time Segment'}, 'automargin': True},
                                'paper_bgcolor': 'rgb(249,249,249)',
                                'plot_bgcolor': 'rgb(249,249,249)',

                            }
                        }),
                        style={'height': '45vh', 'width': '100%'}
                    )
                ],
                style={'backgroundColor': '#F2F2F2', 'width': '100%'}
            )
        ]
    else:
        return []


@app.callback(
    Output("view_indicator", "children"),
    [Input("intermediate-value", "children")],
)
def view_indicator_callback(frame):
    if frame is not None:
        view_count = statistics["viewCount"]
        return view_count


@app.callback(
    Output("comment_indicator", "children"),
    [Input("intermediate-value", "children")],
)
def comment_indicator_callback(frame):
    if frame is not None:
        comment_count = statistics["commentCount"]
        return comment_count


@app.callback(
    Output("like_indicator", "children"),
    [Input("intermediate-value", "children")],
)
def like_indicator_callback(frame):
    if frame is not None:
        like_count = statistics["likeCount"]
        return like_count


@app.callback(
    Output("dislike_indicator", "children"),
    [Input("intermediate-value", "children")],
)
def like_indicator_callback(frame):
    if frame is not None:
        dislike_count = statistics["dislikeCount"]
        return dislike_count



