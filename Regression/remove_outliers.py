import numpy as np
import pandas as pd


def remove_outliers_Z(X, y, threshold=3):
    z_scores = np.abs((X - X.mean()) / X.std())
    outliers = (z_scores >= threshold).any(axis=1)

    X_no_outliers = X[~outliers]
    y_no_outliers = y[~outliers]

    return X_no_outliers, y_no_outliers


def remove_outliers_iqr(X, y):
    q1 = X.quantile(0.25)
    q3 = X.quantile(0.75)
    iqr = q3 - q1

    outliers = ((X < (q1 - 1.5 * iqr)) | (X > (q3 + 1.5 * iqr))).any(axis=1)

    X_no_outliers = X[~outliers]
    y_no_outliers = y[~outliers]

    return X_no_outliers, y_no_outliers
