from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import json
import shutil

def deploy():
    # Load config.json and correct path variable
    with open('config.json', 'r') as f:
        config = json.load(f)

    output_model_path = os.path.join(config['output_model_path'])
    prod_deployment_path = os.path.join(config['prod_deployment_path'])
    ingesteddata_path = os.path.join(config['output_folder_path'])

    # Function for deployment
    def store_model_into_pickle():
        # Copy the trained model file (trainedmodel.pkl) into the deployment directory
        model_file_src = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
        model_file_dest = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
        shutil.copyfile(model_file_src, model_file_dest)

    def store_latest_score():
        # Copy the latestscore.txt file into the deployment directory
        score_file_src = os.path.join(output_model_path, 'latestscore.txt')
        score_file_dest = os.path.join(prod_deployment_path, 'latestscore.txt')
        shutil.copyfile(score_file_src, score_file_dest)

    def store_ingested_files():
        # Copy the ingestedfiles.txt file into the deployment directory
        ingested_file_src = os.path.join(ingesteddata_path, 'ingestedfiles.txt')
        ingested_file_dest = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
        shutil.copyfile(ingested_file_src, ingested_file_dest)

    print("Starting deployment ...")
    store_model_into_pickle()
    store_latest_score()
    store_ingested_files()
    print("Deployment completed.")

if __name__ == '__main__':
    deploy()
