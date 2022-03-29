import os

# Paths

root_path = '/media/Data/Data/'

biodiv_path = root_path + 'biodiv/'
sat_path = root_path + 'sat/'

if not(os.path.exists(root_path)):
	os.mkdir(root_path)

if not(os.path.exists(biodiv_path)):
	os.mkdir(biodiv_path)

if not(os.path.exists(sat_path)):
	os.mkdir(sat_path)


# Gbif Credentials

email = ''
gbif_username = ''
gbif_pw = ''


# Opendap credentials
# Dictionary key must be domain name

dap_creds = {
	'cmems-du.eu': { # Copernicus
		'user': '',
		'pw'  : ''}
			}