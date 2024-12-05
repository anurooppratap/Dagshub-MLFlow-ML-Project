import pandas as pd # Used for data manipulation and analysis. Here, it will read and save CSV files
import sys # Provides access to Python runtime system functions (not used explicitly here)
import yaml # Allows reading and writing YAML files, which are commonly used for configuration
import os # Provides utilities to interact with the operating system, like creating directories

## Loading parameters from param.yaml

params = yaml.safe_load(open("params.yaml"))['preprocess']

# params.yaml: A YAML file likely containing configurations for the script.
# yaml.safe_load: Reads and parses the file into a Python dictionary.
# ['preprocess']: Extracts the preprocess section from the YAML configuration. It should look something like this:
    # preprocess:
    #   input: "data/raw/data.csv"
    #   output: "data/processed/data.csv"
# After this, params will contain
    # params = {
    #     "input": "data/raw/data.csv",
    #     "output": "data/processed/data.csv"
    # }

## Data preprocessing

def preprocess(input_path, output_path):
    data = pd.read_csv(input_path) # Reads the CSV file from input_path into a DataFrame called data
    data # This line does nothing because it’s just referencing the DataFrame without assigning or printing

    os.makedirs(os.path.dirname(output_path), exist_ok=True) 
    # Creates the directories needed for the output_path if they don’t exist
    # exist_ok=True: Prevents errors if the directories already exist
    
    data.to_csv(output_path, header=None, index=False) 
    # data.to_csv(output_path): Saves the DataFrame data to a CSV file at output_path.
    # header=None: Removes the column names from the output file.
    # index=False: Prevents saving the index column.
    
    print(f"Preprocessed data saved to {output_path}")

if __name__=="__main__":
    preprocess(params["input"],params["output"])
    # if __name__ == "__main__":: Ensures the script only runs when executed directly, not when imported as a module.
    # preprocess(params["input"], params["output"]): Calls the preprocess function, passing the input and output paths from the params dictionary.