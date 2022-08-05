import os
import time
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, Response

from geoenrich.dataloader import import_occurrences_csv, load_areas_file
from geoenrich.enrichment import create_enrichment_file, enrich
from geoenrich.exports import produce_stats

app = Flask(__name__)

# enable debugging mode
app.config["DEBUG"] = True

# App variables
app.config['UPLOAD_FOLDER'] =  'static/uploads/'
app.config['UPLOAD_EXTENSIONS'] = ['.csv']
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024 # Maximum filesize of 20MB
app.config['DS_REF'] = ''


# Root URL
@app.route('/')
def index():
    return render_template('home.html')



# Process uploaded file
@app.route("/", methods=['POST'])
def uploadFiles():
     # get the uploaded file
     uploaded_file = request.files['file']
     if uploaded_file.filename != '':
          csv_filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
          uploaded_file.save(csv_filepath)

          var_id = request.form['var_id']
          geo_buff = int(request.form['geo_buff'])
          time_buff = [int(request.form['tbuff1']), int(request.form['tbuff2'])]

          if request.form.get('checkbox'):
               depth_request = 'all'
          else:
               depth_request = 'surface'

          ########## TO DO #########
          # Check csv file integrity

          try:
               df = import_occurrences_csv(path = csv_filepath,
                         id_col = 'id', date_col = 'date', lat_col = 'latitude', lon_col = 'longitude')
          except:
               df = load_areas_file(csv_filepath)


          ds_ref = datetime.now().__str__().replace(' ','_')
          app.config['DS_REF'] = ds_ref
          create_enrichment_file(df, ds_ref)
          enrich(ds_ref, var_id, geo_buff, time_buff, depth_request, maxpoints = 2000000)
          produce_stats(ds_ref, var_id, out_path = 'static/stats/')

          os.remove(csv_filepath)
          return render_template('download.html')


# Provide stats file
@app.route("/getStats")
def getStats():
     with open('static/stats/' + app.config['DS_REF'] + '_0_stats.csv') as fp:
         csv = fp.read()
         return Response(csv,
                         mimetype="text/csv",
                         headers={"Content-disposition":
                                   f"attachment; filename={app.config['DS_REF']}_0_stats.csv"})


# Run app
if (__name__ == "__main__"):
     app.run(host='0.0.0.0', port=8080, debug=True)
     #app.run(host='172.22.0.2', port=80, debug=True)