
import pandas as pd
import numpy as np
import timeit
import os,subprocess
import json
import pickle
import pip
import diagnostics
################## Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(data):
    # Read the deployed model from prod_deployment_path
    model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')

    # Load the trained model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Remove the categorical column 'corporation' from the input data (if present)
    categorical_column = 'corporation'
    if categorical_column in data.columns:
        data = data.drop(columns=[categorical_column])

    # Extract features from the test data
    X_test = data.drop(columns=['exited'])

    # Make predictions using the model
    predictions = model.predict(X_test)

    return predictions.tolist()  # Convert predictions to a list

##################Function to get summary statistics
def dataframe_summary():

    dataset_path = os.path.join(config['output_folder_path'], 'finaldata.csv')

    # Read the dataset
    df = pd.read_csv(dataset_path)

    # Calculate means, medians, and standard deviations for numeric columns
    summary_stats = df.describe().loc[['mean', '50%', 'std']]

    return summary_stats.values.tolist()

################## Function to check for missing data
def missing_data():

    dataset_path = os.path.join(config['output_folder_path'], 'finaldata.csv')

    # Read the dataset
    df = pd.read_csv(dataset_path)

    # Calculate the percentage of missing data for each column
    missing_percentages = (df.isnull().sum() / len(df)) * 100

    return missing_percentages.values.tolist()

##################Function to get timings
def execution_time():
    # Define paths to the ingestion and training scripts
    ingestion_script_path = 'ingestion.py'
    training_script_path = 'training.py'

    # Measure execution time of data ingestion
    ingestion_time = timeit.timeit(lambda: os.system(f'python {ingestion_script_path}'), number=1)

    # Measure execution time of model training
    training_time = timeit.timeit(lambda: os.system(f'python {training_script_path}'), number=1)

    return [ingestion_time, training_time]

##################Function to check dependencies
def outdated_packages_list():
    # Run 'pip list --outdated' command to check outdated packages
    result = subprocess.run(['pip', 'list', '--outdated', '--format', 'json'], stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError("Failed to check outdated packages.")

    # Parse the JSON output of 'pip list --outdated'
    outdated_packages = json.loads(result.stdout)

    return outdated_packages


if __name__ == '__main__':


    # Diagnostics and reporting (you can run this regardless of model drift)
    data = pd.read_csv(os.path.join(config['test_data_path'], 'testdata.csv'))
    y_pred = diagnostics.model_predictions(data)
    stats = diagnostics.dataframe_summary()
    missing = diagnostics.missing_data()
    time_check = diagnostics.execution_time()
    outdated = diagnostics.outdated_packages_list()

    # Print or save the results as needed
    print("Summary Statistics:")
    print(stats)

    print("\nMissing Data Percentages:")
    print(missing)

    print("\nTiming (seconds):")
    print("Data Ingestion Time:", time_check[0])
    print("Model Training Time:", time_check[1])

    print("\nOutdated Dependencies:")
    print(outdated)

    print("\nPredictions:")
    print(y_pred)