FROM tiangolo/uwsgi-nginx-flask:python3.8

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN python3 -m pip install matplotlib appdirs geojson_rewind geomet requests_cache wheel
RUN python3 -m pip install git+https://github.com/rvanasa/pygbif.git
RUN python3 -m pip install git+https://github.com/morand-g/geoenrich@develop

RUN mkdir /app
RUN mkdir /app/static
RUN mkdir /app/static/styles
RUN mkdir /app/templates

