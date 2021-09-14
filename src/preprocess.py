"""
PREPROCESS

'preprocess.py' contains method to preprocess current dataset (PROMISE dataset)
into a proper data format that fits the machine learning model(s).

This file contain a method called 'preprocess' which will return features and label from the extracted dataset

author: anonymouse
        Sep 2021
"""

import imblearn.over_sampling as SMOTE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.model_selection as train_test_split


def preprocess(file):
    """Load the dataset"""
    dataset = pd.read_csv(file)

    """
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




    # return {"X": X, "y": y, "dataset": dataset}
    # return {"X": X, "y": y}
    return {"defect": defect, "not": not_defect}


print(preprocess("datasets/processed/pc4.csv"))
