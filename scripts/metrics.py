from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
import pandas as pd


# Macro-averaged MSE used for ordinal regression when there is imbalanced classes
def macro_mse(y_true, y_pred):
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()

    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy()

    unique, counts = np.unique(y_true, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    mse = 0

    for i in range(len(y_true)):
        mse += ((y_true[i]-y_pred[i])**2)/counts_dict[y_true[i]]

    return mse / len(unique)