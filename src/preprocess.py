"""
PREPROCESS

'preprocess.py' contains methods to preprocess current dataset (PROMISE dataset)
into a proper data format that fits the machine learning model(s) and print out the reuslts.

This file contains 2 methods with the following details:
    1) 'preprocess' method, which will return features and label from the extracted dataset
        after being preprocessed by handling the empty value and imbalanced
    
    2) 'log' method, which will print out the result of 'preprocess' method along side with 
        the timestamp and running machine info
        and write it on preprocess.txt inside reports/preprocess.txt folder

author: anonymouse
        Sep 2021
"""

from datetime import datetime
import imblearn.over_sampling as SMOTE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.model_selection as train_test_split
import platform


def preprocess(file):
    """Load the dataset"""
    dataset = pd.read_csv(file)

    """
    Features & label extraction
    X = features
        To get all of the features we drop the 'defects' column
        axis=1 tells pandas to delete column not row 

    y = label
        The label is the 'defects' column in dataset table
    """
    X = pd.DataFrame(dataset.drop(["defects"], axis=1))
    y = pd.DataFrame(dataset["defects"])

    # Count how many defects and not defects data
    defect = y[y["defects"] == True].count()
    not_defect = y[y["defects"] == False].count()


    """
    Handle missing value
    """


    """
    Handle/encode categorical data
    """


    """
    Handle noisy data
    """


    """
    Handle imbalance dataset
    """


    """
    Split data train & data test
    """


    """
    Feature scaling
    """




    # return {"X": X, "y": y, "dataset": dataset}
    # return {"X": X, "y": y}
    return {"defect": defect, "not": not_defect}


def log():
    with open("reports/preprocess.txt", "w") as preprocess_file:
        print("+----- PREPROCESSING LOGGER -----+", file=preprocess_file)
        print("Timestamp: ", datetime.now().strftime("%H:%M:%S %d-%m-%Y"), file=preprocess_file)
        print(
            "Platform: ",
            platform.uname().system
            + " - "
            + platform.uname().node
            + " - "
            + platform.uname().machine,
            file=preprocess_file,
        )
        print(file=preprocess_file)
        print(preprocess("datasets/processed/pc4.csv"), file=preprocess_file)
        print("+--------------------------------+", file=preprocess_file)


log()