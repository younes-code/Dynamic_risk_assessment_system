from flask import Flask, request
import pandas as pd
import numpy as np
import json
import os
from diagnostics import *
from scoring import *


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 


# Define a route for the root URL ("/")
@app.route("/", methods=['GET'])
def root():
    return jsonify({'message': 'Welcome to the API'})

# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    data_path = request.get_json()['datapath']

    df = pd.read_csv(data_path)
    y_pred = model_predictions(df)
    return str(y_pred)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
    model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    score = score_model(test_data_path, model_path)
    return str(score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
   summary_stats = dataframe_summary()
   return jsonify({'summary_stats': summary_stats})  # Return summary statistics as JSON

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    # Check timing and percent NA values
    timing = execution_time()
    missing = missing_data()
    outdated = outdated_packages_list()
    
    diagnostics_data = {
        'timing': timing,
        'missing_data': missing,
        'outdated_packages': outdated
    }
    return jsonify(diagnostics_data) 

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
