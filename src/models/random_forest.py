"""
--- RANDOM FOREST ---

'random_forest.py' contains method to create and run the random forest (classifier) model. 

This file contains 2 methods with the following details:
    1) 'random_forest' method, which will create the random forest model with the following 'adjustable' settings
        number of tree (n_estimators) = 100
        maximum depth of the tree (max_depth) = 10
        random_state = 3301
    
    2) 'random_forest_score_log' method, will print out the random forest model score
        by using some well-known performance matrix scoring techniques such as accuracy, recall, AUC, and precision
        and save it in reports/results/random_forest.txt


Author: Group of Anonymouse
        Sep 2021
-------------------
"""

import os
import sys
import platform
import getpass

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # Not a good practice but it's ok for now

from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    recall_score,
    roc_auc_score,
    precision_score,
)
from preprocess import preprocess
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier


def random_forest(X_train, y_train):
    """
    random_forest(X_train, y_train) -> random forest model
        This method will form a random forest model based on
        the training data provided
    """

    clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    clf.fit(X_train, y_train)

    return clf


def random_forest_score_log(y_test, y_test_predict, y_validation, y_validation_predict):
    """
    random_forest_score_log(y_test, y_test_predict, y_validation, y_validation_predict) -> model scores
        This method will print out, inside the reports/results/random_forest.txt, the random forest model score 
        by using some of the imported performance matrix measurement
    """
    
    prompt = input(f'''Choose result output format: 
                    [1] Print output to console
                    [2] Print output to reports folder
                    [3] Print output both to console and reports folder
    ''')

    if prompt == str(1):
        print("+----- RANDOM FOREST LOGGER ------+")
        print(
            "Timestamp: ",
            datetime.now().strftime("%H:%M:%S %d-%m-%Y"),
        )
        print(
            "Platform: ",
            platform.uname().system
            + " - "
            + platform.uname().node
            + " - "
            + platform.uname().machine,
        )
        print("Author: ", getpass.getuser())
        print()
        print("|---------- Test Score -----------|")
        print("|---------------------------------|")
        print("| Accuracy     : ", accuracy_score(y_test, y_test_predict))
        print("| Precision    : ", precision_score(y_test, y_test_predict))
        print("| AUC          : ", roc_auc_score(y_test, y_test_predict))
        print("| Recal        : ", recall_score(y_test, y_test_predict))
        print("|------- Validation Score --------|")
        print("|---------------------------------|")
        print("| Accuracy     : ", accuracy_score(y_validation, y_validation_predict))
        print("| Precision    : ", precision_score(y_validation, y_validation_predict))
        print("| AUC          : ", roc_auc_score(y_validation, y_validation_predict))
        print("| Recal        : ", recall_score(y_validation, y_validation_predict))
        print("|---------------------------------|")
        print("+---------------------------------+")
    elif prompt == str(2):
        with open("reports/results/random_forest.txt", "a") as random_forest_file:
            print("+----- RANDOM FOREST LOGGER ------+", file=random_forest_file)
            print(
                "Timestamp: ",
                datetime.now().strftime("%H:%M:%S %d-%m-%Y"),
                file=random_forest_file,
            )
            print(
                "Platform: ",
                platform.uname().system
                + " - "
                + platform.uname().node
                + " - "
                + platform.uname().machine,
                file=random_forest_file,
            )
            print("Author: ", getpass.getuser(), file=random_forest_file)
            print(file=random_forest_file)
            print("|---------- Test Score -----------|", file=random_forest_file)
            print("|---------------------------------|", file=random_forest_file)
            print("| Accuracy     : ", accuracy_score(y_test, y_test_predict), file=random_forest_file)
            print("| Precision    : ", precision_score(y_test, y_test_predict), file=random_forest_file)
            print("| AUC          : ", roc_auc_score(y_test, y_test_predict), file=random_forest_file)
            print("| Recal        : ", recall_score(y_test, y_test_predict), file=random_forest_file)
            print("|------- Validation Score --------|", file=random_forest_file)
            print("|---------------------------------|", file=random_forest_file)
            print("| Accuracy     : ", accuracy_score(y_validation, y_validation_predict), file=random_forest_file)
            print("| Precision    : ", precision_score(y_validation, y_validation_predict), file=random_forest_file)
            print("| AUC          : ", roc_auc_score(y_validation, y_validation_predict), file=random_forest_file)
            print("| Recal        : ", recall_score(y_validation, y_validation_predict), file=random_forest_file)
            print("|---------------------------------|", file=random_forest_file)
            print("+---------------------------------+", file=random_forest_file)
    elif prompt == str(3):
        print("+----- RANDOM FOREST LOGGER ------+")
        print(
            "Timestamp: ",
            datetime.now().strftime("%H:%M:%S %d-%m-%Y"),
        )
        print(
            "Platform: ",
            platform.uname().system
            + " - "
            + platform.uname().node
            + " - "
            + platform.uname().machine,
        )
        print("Author: ", getpass.getuser())
        print()
        print("|---------- Test Score -----------|")
        print("|---------------------------------|")
        print("| Accuracy     : ", accuracy_score(y_test, y_test_predict))
        print("| Precision    : ", precision_score(y_test, y_test_predict))
        print("| AUC          : ", roc_auc_score(y_test, y_test_predict))
        print("| Recal        : ", recall_score(y_test, y_test_predict))
        print("|------- Validation Score --------|")
        print("|---------------------------------|")
        print("| Accuracy     : ", accuracy_score(y_validation, y_validation_predict))
        print("| Precision    : ", precision_score(y_validation, y_validation_predict))
        print("| AUC          : ", roc_auc_score(y_validation, y_validation_predict))
        print("| Recal        : ", recall_score(y_validation, y_validation_predict))
        print("|---------------------------------|")
        print("+---------------------------------+")

        with open("reports/results/random_forest.txt", "a") as random_forest_file:
            print("+----- RANDOM FOREST LOGGER ------+", file=random_forest_file)
            print(
                "Timestamp: ",
                datetime.now().strftime("%H:%M:%S %d-%m-%Y"),
                file=random_forest_file,
            )
            print(
                "Platform: ",
                platform.uname().system
                + " - "
                + platform.uname().node
                + " - "
                + platform.uname().machine,
                file=random_forest_file,
            )
            print("Author: ", getpass.getuser(), file=random_forest_file)
            print(file=random_forest_file)
            print("|---------- Test Score -----------|", file=random_forest_file)
            print("|---------------------------------|", file=random_forest_file)
            print("| Accuracy     : ", accuracy_score(y_test, y_test_predict), file=random_forest_file)
            print("| Precision    : ", precision_score(y_test, y_test_predict), file=random_forest_file)
            print("| AUC          : ", roc_auc_score(y_test, y_test_predict), file=random_forest_file)
            print("| Recal        : ", recall_score(y_test, y_test_predict), file=random_forest_file)
            print("|------- Validation Score --------|", file=random_forest_file)
            print("|---------------------------------|", file=random_forest_file)
            print("| Accuracy     : ", accuracy_score(y_validation, y_validation_predict), file=random_forest_file)
            print("| Precision    : ", precision_score(y_validation, y_validation_predict), file=random_forest_file)
            print("| AUC          : ", roc_auc_score(y_validation, y_validation_predict), file=random_forest_file)
            print("| Recal        : ", recall_score(y_validation, y_validation_predict), file=random_forest_file)
            print("|---------------------------------|", file=random_forest_file)
            print("+---------------------------------+", file=random_forest_file)
    else:
        raise Exception("ERROR: No such option")


""" Main section """
(
    X,
    y,
    X_ros,
    y_ros,
    X_train,
    y_train,
    X_test,
    y_test,
    X_validation,
    y_validation,
) = preprocess("datasets/processed/pc4.csv")    # NOTE: To use different dataset, change the dataset file HERE!

rf_model = random_forest(X_ros, y_ros)
y_test_prediction = rf_model.predict(X_test)
y_validation_prediction = rf_model.predict(X_validation)

random_forest_score_log(y_test, y_test_prediction, y_validation, y_validation_prediction)

