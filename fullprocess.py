import os
import json
import scoring
import training
import deployment
import reporting  # Import the reporting module
import pandas as pd

# Load config.json to get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

prod_deployment_path = config['prod_deployment_path']
input_folder_path = config['input_folder_path']
output_folder_path = os.path.join(config['output_folder_path'])

# Function to read the list of ingested files
def read_ingested_files():
    ingested_files = set()
    ingested_files_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')

    if os.path.exists(ingested_files_path):
        with open(ingested_files_path, 'r') as f:
            for line in f:
                ingested_files.add(line.strip())

    return ingested_files

# Function to check for model drift (you can customize this logic)
def check_model_drift():
    # Read the score from the latest model
    latest_score_path = os.path.join(prod_deployment_path, 'latestscore.txt')
    if os.path.exists(latest_score_path):
        with open(latest_score_path, 'r') as f:
            latest_score = float(f.read().strip())
    else:
        latest_score = None

    # Run scoring on the most recent data
    # Run scoring on the most recent data
    new_data_path = os.path.join(output_folder_path, 'finaldata.csv')  # Replace with your new data file path
    model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    new_score = scoring.score_model(new_data_path, model_path)  # Replace with your model path
    print(f'latest score: {latest_score}, new score: {new_score}')
    # Check for model drift (you can define your own threshold)
    if  new_score >= latest_score:
        return True
    else:
        return False

if __name__ == '__main__':
    # Check and read new data
    ingested_files = read_ingested_files()
    files_to_ingest = []

    for root, dirs, files in os.walk(input_folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Check if the file is not already ingested
            if file not in ingested_files:
                files_to_ingest.append(file_path)

    # Deciding whether to proceed, part 1
    if not files_to_ingest:
        print("No new data found. Ending the process.")
    else:
        print(f"New data found. Ingesting {len(files_to_ingest)} new file(s).")

        # Ingest new data (replace with your ingestion logic)
        for file_path in files_to_ingest:
            print(f"Ingesting new data from: {file_path}")
            # Replace this with your ingestion logic

            # After ingestion, update the ingested files list
            ingested_files.add(os.path.basename(file_path))

        # Update the ingestedfiles.txt file
        ingested_files_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
        with open(ingested_files_path, 'w') as f:
            f.write("\n".join(ingested_files))

        # Checking for model drift
        if check_model_drift():
            print("Model drift detected. Proceeding with re-deployment.")
            # Re-train and re-deploy the model 
            training.train_model()
            new_data_path = os.path.join(output_folder_path, 'finaldata.csv')  
            model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
            scoring.score_model(new_data_path, model_path)

            deployment.deploy()

            reporting.generate_confusion_matrix_plot()
        else:
            print("No model drift detected. Ending the process.")

