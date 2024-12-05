import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse


os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/anuroop.3.pratap/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="anuroop.3.pratap"
os.environ["MLFLOW_TRACKING_PASSWORD"]="e5f48ce0815c6f99f6890556b9d036c483f30684"


# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]


# The purpose of this function is to evaluate a machine learning model on a given dataset, calculate its accuracy, and log the results to MLflow.
def evaluate(data_path,model_path):
    data=pd.read_csv(data_path) # Reads the data from the provided CSV file (data_path) using pandas. The data variable will contain the entire dataset
    X = data.drop(columns=["Outcome"]) # The feature data (X) is extracted by removing the target column Outcome. This will be used as input for the model.
    y = data["Outcome"] # The target variable (y) is the Outcome column from the dataset, which will be used to compare against the model's predictions.


    # This sets up the URI for MLflow’s tracking server, where metrics and model information will be logged. In this case, it points to a remote MLflow server hosted on Dagshub.
    mlflow.set_tracking_uri("https://dagshub.com/anuroop.3.pratap/machinelearningpipeline.mlflow")

    # load the model from the disk
    #  Loads the saved machine learning model from the file specified by model_path. The model is saved in the pickle format (a binary format used to serialize Python objects), and it is deserialized here so it can be used for predictions.
    model=pickle.load(open(model_path,'rb'))

    # The model is used to predict the outcomes based on the feature data (X). The model uses the learned patterns to make predictions on the input data.
    predictions=model.predict(X)

    # This calculates the accuracy score, which is a metric that compares the predicted values (predictions) with the true values (y). It gives a percentage of how accurate the model is at predicting the outcome.
    accuracy=accuracy_score(y,predictions)
    
 
    ## log metrics to MLFLOW
    # This logs the accuracy value to MLflow, so you can track and compare the performance of models over time. The key is "accuracy", and the value is the computed accuracy score.
    mlflow.log_metric("accuracy",accuracy)
    print("Model accuracy:{accuracy}")

if __name__=="__main__":
    evaluate(params["data"],params["model"])
