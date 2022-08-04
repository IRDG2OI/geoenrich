from pathlib import Path 
import requests

app_path = Path('/app')

# Download files from github if they don't already exist


# if not(Path('app/static/styles/style.css').exists()):
# 	r = requests.get('https://raw.githubusercontent.com/morand-g/geoenrich/develop/docker/app/static/styles/style.css')
# 	Path('app/static/styles/').mkdir(parents=True)
# 	Path('app/static/styles/style.css').open('wb').write(r.content)
# 	Path('app/static/styles/style.css').open('wb').close()