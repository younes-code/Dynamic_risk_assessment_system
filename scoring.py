from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 


# Function for model scoring
def score_model(test_data_path, model_path):
    # This function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # It should write the result to the latestscore.txt file

    # Load the trained model from trainedmodel.pkl
    model = pickle.load(open(model_path, 'rb'))

    # Read in test data from the directory specified in test_data_path
    test_df = pd.read_csv(test_data_path)

    # Remove the categorical column 'corporation' from both training and test data
    categorical_column = 'corporation'
    if categorical_column in test_df.columns:
        test_df = test_df.drop(columns=[categorical_column])

    # Split the test data into features (X) and the target variable (y)
    X_test = test_df.drop(columns=['exited'])
    y_test = test_df['exited']

    # Predict using the trained model
    y_pred = model.predict(X_test)

    # Calculate the F1 score
    f1_score = metrics.f1_score(y_test, y_pred)

    # Write the F1 score to latestscore.txt
    score_file_path = os.path.join(config['output_model_path'], 'latestscore.txt')
    with open(score_file_path, 'w') as score_file:
        score_file.write(str(f1_score))
    return (f1_score)
if __name__ == '__main__':
    model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
    test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
    score_model(test_data_path,model_path)
