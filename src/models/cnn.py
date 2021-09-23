"""
--- CNN ---

'cnn.py', file contains cnn model for classifying whether a software is categorized as defect or not

This file contains 2 methods with the following details:
    1) 'cnn' method, which is the main method and take parameters of the features and label from the given
        dataset after being preprocessed and will return the model itself
    
    2) 'cnn_score_log' method, will print out the cnn model score
        by using some well-known performance matrix scoring techniques such as accuracy, recall, AUC, and precision
        and save it in reports/results/cnn.txt folder


Author: Group of Anonymouse
        Sep 2021
-------------------
"""

import os
import sys
import platform
import getpass
import tensorflow as tf

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy


def cnn(X, X_train, X_validation, y_train, y_validation, epochs, batch_size):
    """
    cnn(X_train, X_validation, y_train, y_validation, epochs, batch_size) -> cnn model
        This method will form a deep learning CNN model and also train the model
        NOTE: The CNN model below is for classification, not (yet) for feature extraction!
    """


    # NOTE: input_shape format is (number_of_rows (y axis), number_of_columns (x axis), number_of_depth (z axis))
    #     : Check whether we need the 'id' column or not for CNN input_shape
    rows = 1
    columns = len(X.columns)
    channels = 1
    input_shape = (rows, columns, channels)

    # Reformat the data train and validation to fit CNN input model
    X_train_model = X_train.reshape(X_train.shape[0], rows, columns, channels)
    X_validation_model = X_validation.reshape(
        X_validation.shape[0], rows, columns, channels
    )

    # Create the CNN model
    model = Sequential()
    model.add(Conv2D(64, activation="relu", kernel_size=1, input_shape=input_shape))
    model.add(Conv2D(32, activation="relu", kernel_size=1))
    model.add(Conv2D(16, activation="relu", kernel_size=1))
    model.add(Flatten())
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.summary()

    # Compile the model
    model.compile(
        optimizer="adam",
        loss=BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    # Train the model
    model.fit(
        x=X_train_model,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_validation_model, y_validation),
        verbose=1,
    )

    return model


def cnn_score_log(y_test, y_test_prediction, y_validation, y_validation_prediction):
    """
    cnn_score_log(y_test, y_test_prediction, y_validation, y_validation_prediction) -> model scores
        This method will print out, inside the reports/results/cnn.txt, the random forest model score 
        by using some of the imported performance matrix measurement
    """


    prompt = input(f'''Choose result output format: 
                    [1] Print output to console
                    [2] Print output to reports folder
                    [3] Print output both to console and reports folder
    ''')

    if prompt == str(1):
        print("+----- CNN LOGGER ------+")
        print(
            "Timestamp: ",
            datetime.now().strftime("%H:%M:%S %d-%m-%Y")
        )
        print(
            "Platform: ",
            platform.uname().system
            + " - "
            + platform.uname().node
            + " - "
            + platform.uname().machine
        )
        print("Author: ", getpass.getuser())
        print()
        print("|---------- Test Score -----------|")
        print("|---------------------------------|")
        print("| Accuracy     : ", accuracy_score(y_test, y_test_prediction))
        print("| Precision    : ", precision_score(y_test, y_test_prediction))
        print("| AUC          : ", roc_auc_score(y_test, y_test_prediction))
        print("| Recal        : ", recall_score(y_test, y_test_prediction))
        print("|------- Validation Score --------|")
        print("|---------------------------------|")
        print("| Accuracy     : ", accuracy_score(y_validation, y_validation_prediction))
        print("| Precision    : ", precision_score(y_validation, y_validation_prediction))
        print("| AUC          : ", roc_auc_score(y_validation, y_validation_prediction))
        print("| Recal        : ", recall_score(y_validation, y_validation_prediction))
        print("|---------------------------------|")
        print("+---------------------------------+")
    elif prompt == str(2):
        with open("reports/results/cnn.txt", "a") as cnn_file:
            print("+----- CNN LOGGER ------+", file=cnn_file)
            print(
                "Timestamp: ",
                datetime.now().strftime("%H:%M:%S %d-%m-%Y"),
                file=cnn_file,
            )
            print(
                "Platform: ",
                platform.uname().system
                + " - "
                + platform.uname().node
                + " - "
                + platform.uname().machine,
                file=cnn_file,
            )
            print("Author: ", getpass.getuser(), file=cnn_file)
            print(file=cnn_file)
            print("|---------- Test Score -----------|", file=cnn_file)
            print("|---------------------------------|", file=cnn_file)
            print("| Accuracy     : ", accuracy_score(y_test, y_test_prediction), file=cnn_file)
            print("| Precision    : ", precision_score(y_test, y_test_prediction), file=cnn_file)
            print("| AUC          : ", roc_auc_score(y_test, y_test_prediction), file=cnn_file)
            print("| Recal        : ", recall_score(y_test, y_test_prediction), file=cnn_file)
            print("|------- Validation Score --------|", file=cnn_file)
            print("|---------------------------------|", file=cnn_file)
            print("| Accuracy     : ", accuracy_score(y_validation, y_validation_prediction), file=cnn_file)
            print("| Precision    : ", precision_score(y_validation, y_validation_prediction), file=cnn_file)
            print("| AUC          : ", roc_auc_score(y_validation, y_validation_prediction), file=cnn_file)
            print("| Recal        : ", recall_score(y_validation, y_validation_prediction), file=cnn_file)
            print("|---------------------------------|", file=cnn_file)
            print("+---------------------------------+", file=cnn_file)
    elif prompt == str(3):
        print("+----- CNN LOGGER ------+")
        print(
            "Timestamp: ",
            datetime.now().strftime("%H:%M:%S %d-%m-%Y")
        )
        print(
            "Platform: ",
            platform.uname().system
            + " - "
            + platform.uname().node
            + " - "
            + platform.uname().machine
        )
        print("Author: ", getpass.getuser())
        print()
        print("|---------- Test Score -----------|")
        print("|---------------------------------|")
        print("| Accuracy     : ", accuracy_score(y_test, y_test_prediction))
        print("| Precision    : ", precision_score(y_test, y_test_prediction))
        print("| AUC          : ", roc_auc_score(y_test, y_test_prediction))
        print("| Recal        : ", recall_score(y_test, y_test_prediction))
        print("|------- Validation Score --------|")
        print("|---------------------------------|")
        print("| Accuracy     : ", accuracy_score(y_validation, y_validation_prediction))
        print("| Precision    : ", precision_score(y_validation, y_validation_prediction))
        print("| AUC          : ", roc_auc_score(y_validation, y_validation_prediction))
        print("| Recal        : ", recall_score(y_validation, y_validation_prediction))
        print("|---------------------------------|")
        print("+---------------------------------+")

        with open("reports/results/cnn.txt", "a") as cnn_file:
            print("+----- CNN LOGGER ------+", file=cnn_file)
            print(
                "Timestamp: ",
                datetime.now().strftime("%H:%M:%S %d-%m-%Y"),
                file=cnn_file,
            )
            print(
                "Platform: ",
                platform.uname().system
                + " - "
                + platform.uname().node
                + " - "
                + platform.uname().machine,
                file=cnn_file,
            )
            print("Author: ", getpass.getuser(), file=cnn_file)
            print(file=cnn_file)
            print("|---------- Test Score -----------|", file=cnn_file)
            print("|---------------------------------|", file=cnn_file)
            print("| Accuracy     : ", accuracy_score(y_test, y_test_prediction), file=cnn_file)
            print("| Precision    : ", precision_score(y_test, y_test_prediction), file=cnn_file)
            print("| AUC          : ", roc_auc_score(y_test, y_test_prediction), file=cnn_file)
            print("| Recal        : ", recall_score(y_test, y_test_prediction), file=cnn_file)
            print("|------- Validation Score --------|", file=cnn_file)
            print("|---------------------------------|", file=cnn_file)
            print("| Accuracy     : ", accuracy_score(y_validation, y_validation_prediction), file=cnn_file)
            print("| Precision    : ", precision_score(y_validation, y_validation_prediction), file=cnn_file)
            print("| AUC          : ", roc_auc_score(y_validation, y_validation_prediction), file=cnn_file)
            print("| Recal        : ", recall_score(y_validation, y_validation_prediction), file=cnn_file)
            print("|---------------------------------|", file=cnn_file)
            print("+---------------------------------+", file=cnn_file)
    else:
        raise Exception("ERROR: No such option")


""" Main section """
(
    X,
    y,
    X_scaled,
    y_scaled,
    X_train,
    y_train,
    X_test,
    y_test,
    X_validation,
    y_validation,
) = preprocess("datasets/processed/pc4.csv")  # NOTE: To use different dataset, change the dataset file HERE!

# Instantiate the model
cnn_model = cnn(X, X_train, X_validation, y_train, y_validation, 300, 32)

# Make a prediction
y_test_prediction = cnn_model.predict(X_test.reshape(X_test.shape[0], 1, len(X.columns), 1)) > 0.5
y_validation_prediction = cnn_model.predict(X_validation.reshape(X_validation.shape[0], 1, len(X.columns), 1)) > 0.5

cnn_score_log(y_test, y_test_prediction, y_validation, y_validation_prediction)

