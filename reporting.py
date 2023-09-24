import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions  # Import the model_predictions function


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

output_model_path = config['output_model_path']  # Updated variable name

# Function for reporting
def generate_confusion_matrix_plot():
    test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
    # true labels for test data
    y_true = pd.read_csv(test_data_path)['exited'].values

    # labels predicted by the model
    data = pd.read_csv(test_data_path)
    y_pred = model_predictions(data)

    # plot the confusion matrix
    confusion = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(confusion, annot=True, cmap='Blues')

    ax.set_title('Confusion matrix')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.savefig(os.path.join(config['output_model_path'], 'confusionmatrix.png'))

if __name__ == '__main__':
    generate_confusion_matrix_plot()
