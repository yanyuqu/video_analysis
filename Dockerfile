From python:3.7.6

WORKDIR /usr/src/app/my_project

COPY my_project .

RUN pip install -r requirements.txt

CMD gunicorn -t 300 -b 0.0.0.0:8000 my_project.wsgi