from pathlib import Path

# Paths

########################################################
root_path = Path('./') # <---- Enter your chosen root path here
########################################################

biodiv_path = root_path / 'biodiv'
sat_path = root_path / 'sat'

if not root_path.exists() :
	root_path.mkdir()

if not sat_path.exists() :
	sat_path.mkdir()

if not biodiv_path.exists() :
	biodiv_path.mkdir()



################### GBIF Credentials ###################

email = ''
gbif_username = ''
gbif_pw = ''


################## OpenDAP credentials #################

# Dictionary key must be domain name

dap_creds = {
	'cmems-du.eu': { # Copernicus
		'user': '',
		'pw'  : ''}
			}