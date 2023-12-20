"""
Docstring
---------

Document to create dataset using scikit-learn dataset and saving to input.
""" 

# create folds and save data in input
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import manifold
from sklearn.model_selection import train_test_split, KFold

def create_folds(data: pd.DataFrame):
    """
    Take in a dataframe (training data) and create folds
    data: the input pd.DataFrame on which to create folds for training
    """
    df = data

    # we create a new column called kfold and fil it with -1.
    df["kfold"] = -1

    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # initiate the kfold class from model_selection module
    kf = KFold(n_splits=5)

    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold

    # save the new csv with kfold column
    return df

def save_mnist_df():

    data = datasets.fetch_openml(
        'mnist_784',
        version=1,
        return_X_y=True
    )

    print("mnist dataset loaded from openml.")

    pixel_values, targets = data
    targets = targets.astype(int)
    
    print("creating tsne manifolds")
    # T-distributed Stochastic Neighbor Embedding.
    tsne = manifold.TSNE(n_components=2, random_state=42)
    transformed_data = tsne.fit_transform(pixel_values[:3000, :])
    tsne_df = pd.DataFrame(
        np.column_stack((transformed_data, targets[:3000])),
        columns=["x", "y", "targets"]
    )
    
    tsne_df.loc[:, "targets"] = tsne_df.targets.astype(int)
    # Split X and y variables for train test split.
    y = tsne_df['targets']
    X = tsne_df.drop(columns=['targets'])
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print("train-test split of data created")
    
    # create folds of the training data
    train_df = create_folds(train_df)

    print("train folds created")
    
    # save the train and test set
    train_df.to_csv("input/mnist_train_folds.csv")
    test_df.to_csv("input/mnist_test_folds.csv")
    print("train and test data saved to .csv for mnist")
