from pathlib import Path 
import requests

# if not(Path('app/static/styles/style.css').exists()):
# 	r = requests.get('https://raw.githubusercontent.com/morand-g/geoenrich/develop/docker/app/static/styles/style.css')
# 	Path('app/static/styles/').mkdir(parents=True)
# 	Path('app/static/styles/style.css').open('wb').write(r.content)
# 	Path('app/static/styles/style.css').open('wb').close()