import os

import joblib
import pandas as pd 
from sklearn import metrics
from sklearn import tree
import src.config as config
import src.model_dispatcher as model_dispatcher

def run(fold, model, label_name="targets"):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)
    
    # training data is where kfold is not equal to provided fold also, note that we reset the index.
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data is where kfold is equal to provided fold.
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the label column form dataframe and convert it to a numpy array by using values.
    # target is label column in the dataframe.
    x_train = df_train.drop(label_name, axis=1).values
    y_train = df_train[label_name].values

    # similarly, for validation, we have
    x_valid = df_valid.drop(label_name, axis=1).values
    y_valid = df_valid[label_name].values

    # initialize simple decision tree classifier from sklearn
    clf = model_dispatcher.models[model]

    # fit the model on training data
    clf.fit(x_train, y_train)

    # create predictions for validation samples
    preds = clf.predict(x_valid)

    # calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    # save the model
    joblib.dump(clf, 
                os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )