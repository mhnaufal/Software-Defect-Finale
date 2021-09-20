"""
--- PREPROCESS ---

'preprocess.py' contains methods to preprocess current dataset (PROMISE dataset)
into a proper data format that fits the machine learning model(s) and print out the results.

This file contains 3 methods with the following details:
    1) 'preprocess' method, which will return features and label from the extracted dataset
        after being preprocessed by handling the empty value and imbalanced
    
    2) 'preprocess_log' method, which will print out the result of 'preprocess' method along side with 
        the timestamp and running machine info
        and write it on preprocess.txt inside reports/preprocess.txt folder
    
    3) 'print_bar_diagram', which will print out the pyplot bar diagram representing the balance of the dataset

Author: anonymouse
        Sep 2021
-------------------
"""

from datetime import datetime
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    ADASYN,
    SMOTEN,
    SMOTENC,
    KMeansSMOTE,
)
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import getpass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import platform


def preprocess(file):
    """
    preprocess(file) -> features and label
        This 'preprocess' method take an argument which is the dataset and 
        will return the features and label of the given dataset
    """

    """
    Load the dataset
    """
    dataset = pd.read_csv(file)
    # preprocess_log(file)

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

    ### Count how many defects and not defects data
    defect = y[y["defects"] == True].count()
    not_defect = y[y["defects"] == False].count()
    print_bar_diagram(y, "imbalanced")

    """
    Handle missing value
    NOTE: There is no missing value here
    """
    # print(X.isnull().values.any())    # To check the missing value in the features column

    """
    Handle/encode categorical data
    NOTE: There is no categorical data anymore here
    """

    """
    Handle noisy data
    NOTE: There is no noisy data here (hopefully)
    """

    """
    Handle imbalance dataset
    NOTE: We use some of different techniques to handle the imbalance data
    TODO: Try different techniques listed below!
    """
    #1) RandomOverSampler #
    ros = RandomOverSampler(random_state=3301)
    X_ros, y_ros = ros.fit_resample(X.values, y.values)
    print_bar_diagram(y_ros, "balanced")

    #2) RandomUnderSampler #
    # rus = RandomUnderSampler()

    #3) TomekLinks ???#

    #4) SMOTE #
    # smote = SMOTE()

    #5) ADASYN #
    # adasyn = ADASYN()

    #6) NearMiss ???#

    #7) SMOTENC #
    # smotenc = SMOTENC()

    #8) SMOTEN #
    # smoten = SMOTEN()

    #9) KMeanSmote #
    # k_mean_smote = KMeansSMOTE()

    """
    Split the data train & data test & also the data validation
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_ros, y_ros, test_size=0.3, random_state=3301
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=0.2, random_state=3301
    )

    """
    Feature scaling
    NOTE: No need for feature scaling (for now)
    """

    return X, y, X_ros, y_ros, X_train, y_train, X_test, y_test, X_validation, y_validation


def preprocess_log(file):
    """
    preprocess_log() -> pyplot diagram
        This method will return a 'running log' of the above 'preprocess' method
    """

    with open("reports/preprocess.txt", "w") as preprocess_file:
        print("+----- PREPROCESSING LOGGER -----+", file=preprocess_file)
        print(
            "Timestamp: ",
            datetime.now().strftime("%H:%M:%S %d-%m-%Y"),
            file=preprocess_file,
        )
        print(
            "Platform: ",
            platform.uname().system
            + " - "
            + platform.uname().node
            + " - "
            + platform.uname().machine,
            file=preprocess_file,
        )
        print("Author: ", getpass.getuser(), file=preprocess_file)
        print(file=preprocess_file)
        print(preprocess(file), file=preprocess_file)
        print("+--------------------------------+", file=preprocess_file)


def print_bar_diagram(label, type):
    """
    print_bar_diagram(label, type) -> pyplot diagram
        This method take two parameters,
        - the first one is label which is the data that will be described in the diagram
        - the second one is the type of data:
            'imbalanced' = the data is in format of a single column with rows represent the value
            'balanced' = the data is in format a list with all of the rows value
        and will return a bar diagram saved in reports/figures/ folder or Exception error if the plotting type is unknown
    """

    if type == "imbalanced":
        plt.bar(
            ["True", "False"],
            [
                len(label[label["defects"] == True].index),
                len(label[label["defects"] == False].index),
            ],
            color="royalblue",
            edgecolor="green",
            width=0.4,
            alpha=0.8,
        )
        plt.grid(color="#95a5a6", linestyle="--", linewidth=2, axis="y", alpha=0.7)
        plt.xlabel("Defect Status")
        plt.ylabel("Number of files")
        plt.suptitle("Software Defect Prediction", fontsize=12, fontweight="bold")
        plt.title(["Imbalanced data", datetime.now().strftime("%H:%M:%S %d-%m-%Y")])

        prompt = input(f'''Choose how you will display the imbalanced data diagram: 
                    [1] Save diagram to 'reports/figures' folder
                    [2] Show up the diagram to desktop
                    [3] Save diagram to 'reports/figures' folder and show up to desktop
        ''')

        if prompt == str(1):
            plt.savefig("reports/figures/preprocess_imbalanced.png", bbox_inches="tight")
        elif prompt == str(2):
            plt.show()
        elif prompt == str(3):
            plt.show()
            plt.savefig("reports/figures/preprocess_imbalanced.png", bbox_inches="tight")
        else:
            raise Exception("ERROR: No such option")

    elif type == "balanced":
        plt.bar(
            ["True", "False"],
            [label.tolist().count(True), label.tolist().count(False)],
            color="lime",
            edgecolor="green",
            width=0.4,
            alpha=0.8,
        )
        plt.grid(color="#95a5a6", linestyle="--", linewidth=2, axis="y", alpha=0.7)
        plt.xlabel("Defect Status")
        plt.ylabel("Number of files")
        plt.suptitle("Software Defect Prediction", fontsize=12, fontweight="bold")
        plt.title(["Balanced data", datetime.now().strftime("%H:%M:%S %d-%m-%Y")])

        prompt = input(f'''Choose how you will display the balanced diagram: 
                    [1] Save diagram to 'reports/figures' folder
                    [2] Show up the diagram to desktop
                    [3] Save diagram to 'reports/figures' folder and show up to desktop
        ''')

        if prompt == str(1):
            plt.savefig("reports/figures/preprocess_balanced.png", bbox_inches="tight")
        elif prompt == str(2):
            plt.show()
        elif prompt == str(3):
            plt.show()
            plt.savefig("reports/figures/preprocess_balanced.png", bbox_inches="tight")
        else:
            raise Exception("ERROR: No such option")

    else:
        raise Exception("Unknown plotting type")

