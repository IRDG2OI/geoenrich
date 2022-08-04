FROM tiangolo/uwsgi-nginx-flask:python3.8

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN python3 -m pip install matplotlib appdirs geojson_rewind geomet requests_cache wheel
RUN python3 -m pip install git+https://github.com/rvanasa/pygbif.git
RUN python3 -m pip install git+https://github.com/morand-g/geoenrich@develop

RUN mkdir /app/static
RUN mkdir /app/static/styles
RUN mkdir /app/templates

RUN curl https://raw.githubusercontent.com/morand-g/geoenrich/develop/docker/app/templates/download.html -o /app/templates/download.html
RUN curl https://raw.githubusercontent.com/morand-g/geoenrich/develop/docker/app/templates/home.html -o /app/templates/home.html
RUN curl https://raw.githubusercontent.com/morand-g/geoenrich/develop/docker/app/static/styles/style.css -o /app/static/styles/style.css

RUN sed -i "s+Path('./')+Path('/app/Data/')" /usr/local/lib/python3.8/site-packages/geoenrich/credentials.py