import pandas as pd

def test_features(path=""):
    return pd.read_csv(
        "./Classification/" + path + "classification-data/test_features.csv"
    )


def train_features(path=""):
    return pd.read_csv(
        "./Classification/" + path + "classification-data/train_features.csv"
    )


def train_label(path=""):
    return pd.read_csv(
        "./Classification/" + path + "classification-data/train_label.csv"
    )
