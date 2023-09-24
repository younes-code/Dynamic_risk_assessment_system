import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

# List to store the filenames of ingested CSV files
ingested_files = []


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    #List all CSV files in the input folder
    csv_files = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]

    # Initialize an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()

    # Iterate through each CSV file and append its data to the combined DataFrame
    for csv_file in csv_files:
        file_path = os.path.join(input_folder_path, csv_file)
        df = pd.read_csv(file_path)
        combined_df = combined_df.append(df, ignore_index=True)

        # Add the filename to the list of ingested files
        ingested_files.append(csv_file)
        
    # Deduplicate the combined DataFrame to ensure only unique rows are kept
    combined_df.drop_duplicates(inplace=True)

    # Write the combined DataFrame to an output file (e.g., combined_data.csv)
    combined_output_path = os.path.join(output_folder_path, 'finaldata.csv')
    combined_df.to_csv(combined_output_path, index=False)

    # Save the list of ingested files to ingestedfiles.txt
    ingested_files_path = os.path.join(output_folder_path, 'ingestedfiles.txt')
    with open(ingested_files_path, 'w') as file:
        file.write('\n'.join(ingested_files))

if __name__ == '__main__':
    merge_multiple_dataframe()
