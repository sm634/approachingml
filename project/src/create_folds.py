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

def save_mnist_df():

    data = datasets.fetch_openml(
        'mnist_784',
        version=1,
        return_X_y=True
    )
    pixel_values, targets = data
    print(pixel_values[:5], targets[:5])
    targets = targets.astype(int)
    # T-distributed Stochastic Neighbor Embedding.
    tsne = manifold.TSNE(n_components=2, random_state=42)
    transformed_data = tsne.fit_transform(pixel_values[:3000, :])
    tsne_df = pd.DataFrame(
        np.column_stack((transformed_data, targets[:3000])),
        columns=["x", "y", "targets"]
    )
    tsne_df.loc[:, "targets"] = tsne_df.targets.astype(int)
    tsne_df.to_csv("../input/mnist_train_folds.csv")

# run the dataset creating and saving function.
save_mnist_df()
