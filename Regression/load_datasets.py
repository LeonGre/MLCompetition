import pandas as pd


def test_features(str=""):
    return pd.read_csv("./" + str + "regression-data/test_features.csv")


def train_features(str=""):
    return pd.read_csv("./" + str + "regression-data/train_features.csv")


def train_label(str=""):
    return pd.read_csv("./" + str + "regression-data/train_label.csv")


def test_label(str=""):
    return pd.read_csv("./" + str + "regression-data/test_label.csv")


def train_df(str=""):
    train_features_df = train_features(str)
    train_label_df = train_label(str)
    train_df = pd.merge(train_features_df, train_label_df, on="Id")
    # remove duplicates
    train_df = train_df.drop_duplicates(subset=["user", "item"], keep="last")

    train_df['rating'] = train_df['rating'].astype(float)
    # Specify the column order
    column_order = ['Id', 'user', 'item', 'rating', 'timestamp']

    # Reorder the columns
    train_df = train_df[column_order]
    return train_df
