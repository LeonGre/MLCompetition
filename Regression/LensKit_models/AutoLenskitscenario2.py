import json

from ConfigSpace import Constant
from lenskit import batch, Recommender
from lkauto import get_best_recommender_model, get_best_prediction_model
from lkauto.algorithms.als import BiasedMF
from lkauto.algorithms.funksvd import FunkSVD

from lenskit.metrics.predict import rmse
from lkauto.algorithms.item_knn import ItemItem
from LensKit_models.test import algo
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

# initialize ItemItem ConfigurationSpace
cs = FunkSVD.get_default_configspace()

cs.add_hyperparameters([Constant("algo", "FunkSVD")])
# set a random seed for reproducible results
cs.seed(42)

# Provide the ItemItem ConfigurationSpace to the get_best_recommender_model function.
model_SVD, config_SVD = get_best_prediction_model(train=train_df, cs=cs, time_limit_in_sec=60*60*10, split_folds=3, split_frac=0.2, random_state=42, user_column='user', item_column='item', rating_column='rating', timestamp_col='timestamp')

# Convert the configuration to a string
config_str = json.dumps(config_SVD)

# Write the configuration to a file
with open('config_FunkSVD.txt', 'w') as f:
    f.write(config_str)

fittable = Recommender.adapt(model_SVD)
model_SVD.fit(train_df)
preds = model_SVD.predict(test_features)

# print the RMSE value
print("RMSE: {}".format(rmse(predictions=preds, truth=test_features['rating'])))

"""
#preds = batch.predict(algo, n_jobs=1)

cs_bmf = BiasedMF.get_default_configspace()
cs_bmf.add_hyperparameters([Constant("algo", "BiasedMF")])
cs_bmf.seed(42)
# Provide the ItemItem ConfigurationSpace to the get_best_recommender_model function.
model_BMF, config_BMF = get_best_prediction_model(train=train_df, cs=cs_bmf)

# Convert the configuration to a string
config_str = json.dumps(config_BMF)

# Write the configuration to a file
with open('config_BMF.txt', 'w') as f:
    f.write(config_str)

fittable = Recommender.adapt(model_BMF)
model_BMF.fit(train_df)
preds = model_BMF.predict(test_features)

# print the RMSE value
print("RMSE: {}".format(rmse(predictions=preds, truth=test_features['rating'])))
"""