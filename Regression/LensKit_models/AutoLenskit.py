import numpy as np
from lkauto.lkauto import get_best_recommender_model
from lkauto.lkauto import get_best_prediction_model
from sklearn.model_selection import train_test_split
from lenskit.metrics.predict import rmse


from load_datasets import train_features, train_label, test_features
import pandas as pd

# Load the dataset
train_features = train_features("../")
train_label = train_label("../")
test_features = test_features("../")
X_test = test_features.drop("Id", axis=1)

train_df = pd.merge(train_features, train_label, on="Id")
# remove duplicates
train_df = train_df.drop_duplicates(subset=["user", "item"], keep="last")
train_df.drop("Id", axis=1, inplace=True)

train_df['rating'] = train_df['rating'].astype(float)
# Specify the column order
column_order = ['user', 'item', 'rating', 'timestamp']

# Reorder the columns
train_df = train_df[column_order]

best_model = get_best_prediction_model(
    optimization_metric=rmse,
    time_limit_in_sec=60*60*3,
    train=train_df,
    split_folds=1,
    split_frac=0.2,
    random_state=42,
    user_column='user',
    item_column='item',
    rating_column='rating',
    timestamp_col='timestamp'
)