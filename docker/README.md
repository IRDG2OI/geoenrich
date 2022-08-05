## Docker image

You may use GeoEnrich and its associated webapp locally by loading a Docker container.

To do this you can download *docker-compose.yml* and *Dockerfile*, and load the container the following way:

```
docker-compose up -d --build
```

You can then use geoenrich from the command line:

```
docker exec -it python-flask-2 python
```

Or launch the web app in a browser:
```
localhost:8080
```