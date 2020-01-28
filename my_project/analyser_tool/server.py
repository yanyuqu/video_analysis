from flask import Flask
from dash import Dash

import os




# should start and end with a '/'
URL_BASE_PATHNAME = '/'
#STATIC_URL = '/static/'


external_stylesheets = [
    "https://storage.cloud.google.com/serato_external/static/base.css",
    "https://storage.cloud.google.com/serato_external/static/fonts.css",
    "https://storage.cloud.google.com/serato_external/static/internal.css"
]

server = Flask(__name__)

#assets_path = 'C:/Users/Tushar/Documents/serato_video_analyser/video_analyser/my_project/analyser_tool/assets'

app = Dash(
    __name__,
    server=server,
    url_base_pathname=URL_BASE_PATHNAME,
    external_stylesheets=external_stylesheets

)

# css_directory = os.getcwd()
# stylesheets = ['base.css', 'fonts.css', 'internal.css']
# static_css_route = '/static/'
#
# @app.server.route('{}<stylesheet>'.format(static_css_route))
# def serve_stylesheet(stylesheet):
#     if stylesheet not in stylesheets:
#         raise Exception(
#             '"{}" is excluded from the allowed static files'.format(
#                 stylesheet
#             )
#         )
#     return Flask.send_from_directory(css_directory, stylesheet)
#
#
# for stylesheet in stylesheets:
#     app.css.append_css({"external_url": "/static/{}".format(stylesheet)})
#
# app.scripts.config.serve_locally = True
# app.config['suppress_callback_exceptions'] = True
#

