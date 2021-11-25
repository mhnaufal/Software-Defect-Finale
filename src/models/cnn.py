"""
--- CNN ---

'cnn.py', file contains cnn model for classifying whether a software is categorized as defect or not

This file contains 2 methods with the following details:
    1) 'cnn' method, which is the main method and take parameters of the features and label from the given
        dataset after being preprocessed and will return the model itself
    
    2) 'cnn_score_log' method, will print out the cnn model score
        by using some well-known performance matrix scoring techniques such as accuracy, recall, AUC, and precision
        and save it in reports/results/cnn.txt folder

    3) 'console_log' method, the actual cnn logger which is used in 'cnn_score_log' method


Author: Group of Anonymouse
        Sep 2021
-------------------
"""

import os
import sys
import platform
import getpass
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

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
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy


EPOCHS = 300
BATCH_SIZE = 32

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

    # NOTE: Optional model remover before creating a new model
    tf.keras.backend.clear_session()

    # Create the CNN model
    model = Sequential()
    model.add(Conv2D(64, activation="relu", kernel_size=1, input_shape=input_shape))
    model.add(Conv2D(32, activation="relu", kernel_size=1))
    model.add(Flatten())
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.summary()

    # Compile the model
    model.compile(
        optimizer="adam",
        loss=BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    # Train the model
    history = model.fit(
        x=X_train_model,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_validation_model, y_validation),
        verbose=1,
    )

    plt.plot(history.history["val_loss"], color='g', label="val_loss")  
    plt.plot(history.history["loss"], color='r', label="loss")
    plt.ylim(bottom=0)
    plt.ylim(top=1)
    plt.title("loss")
    plt.legend()
    plt.show()

    plt.plot(history.history["accuracy"], color='r', label="accuracy")  
    plt.plot(history.history["val_accuracy"], color='g', label="val_accuracy")
    plt.ylim(bottom=0)
    plt.ylim(top=1)
    plt.title("accuracy")
    plt.legend()
    plt.show()
    
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
        console_log(y_test, y_test_prediction, y_validation, y_validation_prediction)

        ### Create the confussion matrix
        cm = confusion_matrix(y_test, y_test_prediction)
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_prediction).ravel()
        cm = [[tp, fp], [fn, tn]]
        
        ax = sns.heatmap(cm, annot=True, fmt = "d", cmap="YlGnBu", linewidths=.5, linecolor="gray")
        ax.set_xlabel('ACTUAL LABELS', fontweight="bold")
        ax.set_ylabel('PREDICTED LABELS', fontweight="bold") 
        ax.set_title(datetime.now().strftime("%H:%M:%S %d-%m-%Y"))
        ax.xaxis.set_ticklabels(['Defect', 'Not Defect'])
        ax.yaxis.set_ticklabels(['Defect', 'Not Defect'])
        plt.suptitle("CNN Confussion Matrix", fontsize=12, fontweight="bold")
        plt.show()

    elif prompt == str(2):
        ### Create the confussion matrix
        cm = confusion_matrix(y_test, y_test_prediction)
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_prediction).ravel()
        cm = [[tp, fp], [fn, tn]]
        
        ax = sns.heatmap(cm, annot=True, fmt = "d", cmap="YlGnBu", linewidths=.5, linecolor="gray")
        ax.set_xlabel('ACTUAL LABELS', fontweight="bold")
        ax.set_ylabel('PREDICTED LABELS', fontweight="bold") 
        ax.set_title(datetime.now().strftime("%H:%M:%S %d-%m-%Y"))
        ax.xaxis.set_ticklabels(['Defect', 'Not Defect'])
        ax.yaxis.set_ticklabels(['Defect', 'Not Defect'])
        plt.suptitle("CNN Confussion Matrix", fontsize=12, fontweight="bold")
        plt.savefig("reports/figures/confussion matrixs/cnn.png", bbox_inches="tight")

        with open("reports/results/cnn.txt", "a") as cnn_file:
            console_log(y_test, y_test_prediction, y_validation, y_validation_prediction, cnn_file)

    elif prompt == str(3):
        console_log(y_test, y_test_prediction, y_validation, y_validation_prediction)

        ### Create the confussion matrix
        cm = confusion_matrix(y_test, y_test_prediction)
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_prediction).ravel()
        cm = [[tp, fp], [fn, tn]]
        
        ax = sns.heatmap(cm, annot=True, fmt = "d", cmap="YlGnBu", linewidths=.5, linecolor="gray")
        ax.set_xlabel('ACTUAL LABELS', fontweight="bold")
        ax.set_ylabel('PREDICTED LABELS', fontweight="bold") 
        ax.set_title(datetime.now().strftime("%H:%M:%S %d-%m-%Y"))
        ax.xaxis.set_ticklabels(['Defect', 'Not Defect'])
        ax.yaxis.set_ticklabels(['Defect', 'Not Defect'])
        plt.suptitle("CNN Confussion Matrix", fontsize=12, fontweight="bold")
        plt.savefig("reports/figures/confussion matrixs/cnn.png", bbox_inches="tight")
        plt.show()

        with open("reports/results/cnn.txt", "a") as cnn_file:
            console_log(y_test, y_test_prediction, y_validation, y_validation_prediction, cnn_file)
            
    else:
        raise Exception("ERROR: No such option")


def console_log(y_test, y_test_prediction, y_validation, y_validation_prediction, file=None):
    print("\n+----- CNN LOGGER ------+", file=file)
    print(
        "Timestamp: ",
        datetime.now().strftime("%H:%M:%S %d-%m-%Y"), 
        file=file
    )
    print(
        "Platform: ",
        platform.uname().system
        + " - "
        + platform.uname().node
        + " - "
        + platform.uname().machine, 
        file=file
    )
    print("Author: ", getpass.getuser(), file=file)
    print(file=file)
    print("|---------- Test Score -----------|", file=file)
    print("|---------------------------------|", file=file)
    print("| Accuracy     : ", accuracy_score(y_test, y_test_prediction), file=file)
    print("| Precision    : ", precision_score(y_test, y_test_prediction), file=file)
    print("| AUC          : ", roc_auc_score(y_test, y_test_prediction), file=file)
    print("| Recal        : ", recall_score(y_test, y_test_prediction), file=file)
    print("|------- Validation Score --------|", file=file)
    print("|---------------------------------|", file=file)
    print("| Accuracy     : ", accuracy_score(y_validation, y_validation_prediction), file=file)
    print("| Precision    : ", precision_score(y_validation, y_validation_prediction), file=file)
    print("| AUC          : ", roc_auc_score(y_validation, y_validation_prediction), file=file)
    print("| Recal        : ", recall_score(y_validation, y_validation_prediction), file=file)
    print("|---------------------------------|", file=file)
    print("+---------------------------------+", file=file)
    print()

    return

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
) = preprocess("datasets/processed/big_data1.csv")  # NOTE: To use different dataset, change the dataset file HERE!

# Initiate the model
cnn_model = cnn(X, X_train, X_validation, y_train, y_validation, EPOCHS, BATCH_SIZE)

# Make a prediction
y_test_prediction = cnn_model.predict(X_test.reshape(X_test.shape[0], 1, len(X.columns), 1)) > 0.5
y_validation_prediction = cnn_model.predict(X_validation.reshape(X_validation.shape[0], 1, len(X.columns), 1)) > 0.5

cnn_score_log(y_test, y_test_prediction, y_validation, y_validation_prediction)
