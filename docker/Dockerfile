FROM tiangolo/uwsgi-nginx-flask:python3.8

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN python3 -m pip install matplotlib appdirs geojson_rewind geomet requests_cache wheel
RUN python3 -m pip install git+https://github.com/rvanasa/pygbif.git
RUN python3 -m pip install git+https://github.com/morand-g/geoenrich@develop

RUN rm /app/*
RUN curl https://raw.githubusercontent.com/morand-g/geoenrich/develop/docker/initialize.py -o /home/initialize.py

RUN sed -i "s+\'./\'+\'/app/data/\'+" /usr/local/lib/python3.8/site-packages/geoenrich/credentials_example.py
