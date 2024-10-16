from pathlib import Path 
import requests
import shutil

print('Initialization...')

# Download files from github if they don't already exist

if not(Path('/app/data/').exists()):
    Path('/app/data/').mkdir()

if not(Path('/app/templates/').exists()):
    Path('/app/templates/').mkdir()

if not(Path('/app/conf/').exists()):
    Path('/app/conf/').mkdir()

if not(Path('/app/static/styles/').exists()):
    Path('/app/static/styles/').mkdir(parents=True)

if not(Path('/app/static/uploads/').exists()):
    Path('/app/static/uploads/').mkdir(parents=True)

if not(Path('/app/static/stats/').exists()):   
    Path('/app/static/stats/').mkdir(parents=True)

if not(Path('/app/static/assets/').exists()):   
    Path('/app/static/assets/').mkdir(parents=True)

if not(Path('/app/static/styles/style.css').exists()): 
    r = requests.get('https://raw.githubusercontent.com/morand-g/geoenrich/main/docker/app/static/styles/style.css') 
    Path('/app/static/styles/style.css').open('wb').write(r.content)

if not(Path('/app/templates/home.html').exists()): 
    r = requests.get('https://raw.githubusercontent.com/morand-g/geoenrich/main/docker/app/templates/home.html') 
    Path('/app/templates/home.html').open('wb').write(r.content)

if not(Path('/app/templates/download.html').exists()): 
    r = requests.get('https://raw.githubusercontent.com/morand-g/geoenrich/main/docker/app/templates/download.html') 
    Path('/app/templates/download.html').open('wb').write(r.content)

if not(Path('/app/main.py').exists()): 
    r = requests.get('https://raw.githubusercontent.com/morand-g/geoenrich/main/docker/app/main.py') 
    Path('/app/main.py').open('wb').write(r.content)


assets = ['favicon.ico', 'logo_france.png', 'logo_g2oi.png', 'logo_github.png', 'logo_ird.png',
          'logo_reunion.png', 'logo_rtd.png', 'logo_ue.png']

for asset in assets:
    if not(Path('/app/static/assets/' + asset).exists()): 
        r = requests.get('https://raw.githubusercontent.com/morand-g/geoenrich/main/docker/app/static/assets/' + asset) 
        Path('/app/static/assets/' + asset).open('wb').write(r.content)

if Path('/app/conf/credentials.py').exists():
    shutil.copy(Path('/app/conf/credentials.py'), Path('/usr/local/lib/python3.10/site-packages/geoenrich/credentials.py'))

if Path('/app/conf/personal_catalog.csv').exists():
    shutil.copy(Path('/app/conf/personal_catalog.csv'), Path('/usr/local/lib/python3.10/site-packages/geoenrich/data/personal_catalog.csv'))

print('Initialization complete.')
