import joblib
import pandas as pd 
from sklearn import metrics
from sklearn import tree

def run(fold):
    # read the training data with folds
    df = pd.read_csv("../input/mnist_train_folds.csv")
    