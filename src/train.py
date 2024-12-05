# This script defines a complete pipeline for training a Random Forest Classifier, tuning its hyperparameters, evaluating its performance, and 
# logging the entire process to MLflow, a platform for tracking machine learning experiments

import pandas as pd # Used for reading and handling data
from sklearn.ensemble import RandomForestClassifier # The machine learning model used in this pipeline
import pickle # For saving the trained model to a file
import yaml # For reading the configuration parameters from params.yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report # Metrics to evaluate model performance
from mlflow.models import infer_signature # Used to capture input-output schema for logging models to MLflow
import os # For file and directory operations

from sklearn.model_selection import train_test_split,GridSearchCV # Tools for splitting the dataset and performing hyperparameter tuning
from urllib.parse import urlparse 

# The urlparse function from the urllib.parse module in Python is used to parse (or break down) a URL into its individual components, making it easy to work with the different parts of a URL
    # How urlparse Works
    # When you pass a URL to urlparse, it breaks the URL into the following components:

    # scheme: The protocol (e.g., http, https, ftp).
    # netloc: The network location (e.g., domain name and port).
    # path: The path to the resource (e.g., /page.html).
    # params: Parameters for the last path segment (rarely used).
    # query: The query string (e.g., ?key=value&key2=value2).
    # fragment: The fragment identifier (e.g., #section).
    # The result is a ParseResult object that behaves like a tuple.

    # from urllib.parse import urlparse

    # url = "https://www.example.com:8080/path/to/page?key=value&key2=value2#section"

    # # Parse the URL
    # parsed_url = urlparse(url)

    # # Access components
    # print("Scheme:", parsed_url.scheme)      # Output: https
    # print("Netloc:", parsed_url.netloc)      # Output: www.example.com:8080
    # print("Path:", parsed_url.path)          # Output: /path/to/page
    # print("Query:", parsed_url.query)        # Output: key=value&key2=value2
    # print("Fragment:", parsed_url.fragment)  # Output: section


import mlflow # # A library for logging experiment details

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/anuroop.3.pratap/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="anuroop.3.pratap"
os.environ["MLFLOW_TRACKING_PASSWORD"]="e5f48ce0815c6f99f6890556b9d036c483f30684"

# These environment variables configure MLflow to connect to a DagsHub server (an MLflow-compatible remote server).
# MLFLOW_TRACKING_URI: The URI of the MLflow tracking server.
# Authentication: Username and password are set to allow access to the server.

def hyperparameter_tuning(X_train,y_train,param_grid):
    rf=RandomForestClassifier() 
    # The model being tuned
    # param_grid: A dictionary defining the hyperparameters to tune
    
    grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2) 
    # Performs an exhaustive search over hyperparameters using cross-validation (cv=3)
    # n_jobs=-1: Utilizes all available CPU cores
    
    grid_search.fit(X_train,y_train)
    
    return grid_search  
    # Returns the grid_search object, which contains the best model and its parameters 

    # What is GridSearchCV?
    # GridSearchCV is a method from Scikit-learn used to perform an exhaustive search over a grid of hyperparameter combinations for a given model.
    # It tries out every possible combination of specified hyperparameter values and evaluates the model using cross-validation (in this case, cv=3).

    # What Happens in fit?
    # When you call fit on the GridSearchCV object:
    # Model Training:
    # A RandomForestClassifier (or the specified estimator) is trained multiple times, each time with a different combination of hyperparameters from the param_grid.
    # There are 2 x 3 x 2 x 2 = 24 combinations of hyperparameters to test

    # Cross-Validation:
    # For each combination of hyperparameters, the training data (X_train, y_train) is split into cv=3 folds.
    # The model is trained on 2 folds and validated on the remaining 1 fold. This process is repeated 3 times (once for each fold), and the average performance score (e.g., accuracy) is recorded.

    # Performance Evaluation:
    # After testing all combinations, GridSearchCV identifies the set of hyperparameters that gives the best performance based on the chosen metric (e.g., accuracy by default)

    # Final Model:
    # fit refits the model using the best hyperparameters on the entire training dataset (X_train and y_train)

    # What is Stored After fit?
    # Once fit is complete, the grid_search object contains:
    # grid_search.best_params_: The best combination of hyperparameters.
    # grid_search.best_score_: The highest average performance score during cross-validation.
    # grid_search.best_estimator_: The model trained with the best hyperparameters

    # Output
    # Best Parameters: {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2}

## Load the parameters from params.yaml

params=yaml.safe_load(open("params.yaml"))["train"]
# Reads params.yaml: A configuration file with a structure like this:
    #   train:
    #   data: data/raw/data.csv
    #   model: models/model.pkl
    #   random_state: 42
    #   n_estimators: 100
    #   max_depth: 5
# Extracts the train section into the params dictionary:
    # {
    #     'data': 'path/to/dataset.csv',
    #     'model': 'path/to/model.pkl',
    #     'random_state': 42,
    #     'n_estimators': 100,
    #     'max_depth': None
    # }


def train(data_path,model_path,random_state,n_estimators,max_depth):
    # The main function in the code is train(), which takes several parameters:

    # data_path: Path to the dataset.
    # model_path: Path to save the trained model.
    # random_state: A random seed for reproducibility.
    # n_estimators: Number of trees in the Random Forest.
    # max_depth: Maximum depth of the trees in the Random Forest.
    data=pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"])
    y=data['Outcome']
    # data: Reads the dataset from the CSV file at the specified data_path.
    # X: The feature matrix (all columns except the target column Outcome).
    # y: The target vector (the Outcome column).

    mlflow.set_tracking_uri("https://dagshub.com/anuroop.3.pratap/machinelearningpipeline.mlflow")
    # mlflow.set_tracking_uri(): Specifies the MLflow tracking URI, which is the URL where experiment data will be logged (in this case, a remote URL on Dagshub)


    ## start the MLFLOW run
    with mlflow.start_run():
        # mlflow.start_run(): Begins an MLflow experiment run. All logging of metrics, parameters, and models happens within this context.
        # split the dataset into training and test sets
        
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
        # train_test_split(): Splits the data into training (80%) and testing (20%) sets
        
        signature=infer_signature(X_train,y_train)
        # train_test_split(): Splits the data into training (80%) and testing (20%) sets


        ## Define hyperparameter grid

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        # param_grid: A dictionary defining the hyperparameters to tune for the Random Forest Classifier.
        
        # Perform hyperparameter tuning
        grid_search=hyperparameter_tuning(X_train,y_train,param_grid)
        # hyperparameter_tuning(): This function performs GridSearchCV over the specified parameter grid, using cross-validation to find the best combination of hyperparameters

        ## get the best model
        best_model=grid_search.best_estimator_
        # The model trained with the best hyperparameters from the grid search

        ## predict and evaluate the model

        y_pred=best_model.predict(X_test)
        # Predictions made on the test set

        accuracy=accuracy_score(y_test,y_pred)
        # The accuracy of the model, calculated by comparing the predicted values to the true values (y_test).
        
        print(f"Accuracy:{accuracy}")

        ## Log additional metrics \
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_param("best_n_estimatios",grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_param("best_sample_split", grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_samples_leaf", grid_search.best_params_['min_samples_leaf'])
        # the purpose of these lines is to log important model performance metrics (like accuracy) and hyperparameters (like n_estimators, max_depth, etc.) during the MLflow run. This allows you to track and compare model performance across different runs, making it easier to choose the best configuration


        ## log the confusion matrix and classification report

        cm=confusion_matrix(y_test,y_pred)
        cr=classification_report(y_test,y_pred)
        mlflow.log_text(str(cm),"confusion_matrix.txt")
        mlflow.log_text(cr,"classification_report.txt")
        # Confusion Matrix (cm): A matrix that shows the number of correct and incorrect predictions, classified by each category.
        # Classification Report (cr): Provides additional evaluation metrics like precision, recall, and F1-score.
        # Both are logged as text files in the MLflow run.

        
        # This portion of code is dealing with logging the trained model to MLflow, but it also checks whether the model is being logged to a remote store (e.g., DagsHub or a cloud-based storage) or to a local file system. The logging behavior is adjusted based on this check.
        
        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
        
        # Explanation:
        # mlflow.get_tracking_uri(): This function retrieves the MLflow tracking URI (Uniform Resource Identifier), which tells MLflow where to log and track experiments, metrics, and models. It could be a local file path (like file:///path/to/directory) or a remote URL (like https://dagshub.com or another cloud platform URL).

        # urlparse(mlflow.get_tracking_uri()): The urlparse function from Python’s urllib.parse module is used to break down the URI into its components (scheme, netloc, path, params, query, fragment). Here, the scheme represents the part of the URI that indicates the protocol or storage type. For example:

            # file:// for local file systems.
            # https:// for cloud-based storage (e.g., DagsHub, AWS, etc.).
        # .scheme: Extracts just the scheme (protocol or storage type) from the parsed URI. If the URI is https://dagshub.com, the scheme would be https, meaning it's using a cloud-based remote store.

        # Why it's used:
        # This line determines if the model logging is happening locally or remotely. It looks at the tracking URI and extracts the storage type (remote vs. file system). The next steps will use this information to choose how to log the model.       

        if tracking_url_type_store!='file':
            # This if condition checks if the scheme is not 'file'. In other words, it checks if the tracking URI is not pointing to a local file system but a remote storage solution like a cloud-based service (e.g., DagsHub, AWS, GCP, Azure, etc.)
            mlflow.sklearn.log_model(best_model,"model",registered_model_name="Best Model")
            # If the scheme is anything other than 'file' (meaning the URI is pointing to remote storage), the condition evaluates as True, and the following block of code will execute:

            # mlflow.sklearn.log_model(best_model, "model", registered_model_name="Best Model") will log the trained model to a remote MLflow server (e.g., DagsHub or another cloud service) under the model name "Best Model".
            # Why it's used:
            # The model is logged remotely, which is common in production or collaborative environments. By logging to a remote store, the model can be accessed by multiple users or systems. It also ensures that all experiments and models are saved in a centralized location.
        
        else:
            # Explanation:
            # If the scheme is 'file' (i.e., the tracking URI is pointing to a local file system), then the condition tracking_url_type_store != 'file' will evaluate as False, and the else block will execute.

            # The else block contains the following code:

            # mlflow.sklearn.log_model(best_model, "model", signature=signature) will log the trained model to the local file system.
            # The "model" string indicates that the model is being stored under the name "model."
            # The signature variable is passed as an argument. The signature represents the input and output schema of the model (i.e., the structure of the data it expects as input and the structure of its output predictions). The signature helps keep track of the model's input/output data format, which is useful for model validation and deployment.
            
            # Why it's used:
            # When the tracking URI points to a local file system (e.g., file:///path/to/folder), the model is logged locally. This is useful for experimentation or testing on a local machine without needing to use a remote server.
            
            mlflow.sklearn.log_model(best_model, "model",signature=signature)

        ## create the directory to save the model
        os.makedirs(os.path.dirname(model_path),exist_ok=True)
        # The line os.makedirs(os.path.dirname(model_path), exist_ok=True) is used to create a directory (or directories) in the file system where a model will be saved

        # os.path.dirname(model_path)
        # Explanation:
        # model_path: This is a variable that contains the path where the model will be saved. For example, it might be something like "./models/model.pkl" or "/user/data/model/first_model.pkl".
        # os.path.dirname(): This function returns the directory part of a given file path. It effectively extracts the path to the directory, removing the file name. For example:
            # If model_path = "./models/model.pkl", then os.path.dirname(model_path) will return "./models".
            # If model_path = "/user/data/model/first_model.pkl", then os.path.dirname(model_path) will return "/user/data/model".        
        # Why it's used:
        # We want to create the directory where the model will be saved, but not the actual file (i.e., we only need the directory path, not the file itself). This helps us ensure that the directory exists before we attempt to save the model file there.

        # os.makedirs(...)
        # Explanation:
        # os.makedirs(): This function is used to create a directory and any necessary intermediate directories. For example, if you're saving a file at "./models/model.pkl" and the "./models" directory does not exist, os.makedirs() will create it. If the directory already exists, it won't raise an error (unless the exist_ok parameter is set to False).
        # Why it's used:
        # We are using os.makedirs() to ensure that the directory where the model is going to be saved exists. If it doesn't exist yet, it will be created. This is helpful in cases where the parent directories (like "./models") may not be pre-created, avoiding errors when saving the model.

        # exist_ok=True
        # Explanation:
        # exist_ok=True: This is a parameter passed to the os.makedirs() function. It tells Python to not raise an error if the directory already exists. If this parameter were set to False, an error would be raised if the directory already exists.
        # Why it's used:
        # By setting exist_ok=True, we make sure that the directory creation process is safe. If the directory already exists, no error is raised, and the code continues to execute without interruption. This makes the code more robust and user-friendly, as you don't have to worry about whether the directory is already present.


        filename=model_path
        pickle.dump(best_model,open(filename,'wb'))
        # Explanation:
        # pickle.dump(): The pickle.dump() function is used to serialize and save a Python object (in this case, the trained model) to a file. "Serialization" means converting the object (like a machine learning model) into a byte stream, so it can be stored or transmitted.
        # best_model: This is the model object that was trained and is now being saved. It could be a RandomForestClassifier, for example, or any other type of model object from scikit-learn.
        # open(filename, 'wb'): The open() function is used to open a file in a specific mode:
            # 'wb' stands for "write binary". It means the file will be opened in binary write mode, which is required for saving non-text files (like the model). It ensures that the model is written as a binary file.
        # filename: This is the path to the file where the model should be saved (e.g., ./models/model.pkl).
        # The open() function returns a file object that is passed to pickle.dump(), allowing the model to be written to the file.

        print(f"Model saved to {model_path}")

if __name__=="__main__":
    train(params['data'],params['model'],params['random_state'],params['n_estimators'],params['max_depth'])
