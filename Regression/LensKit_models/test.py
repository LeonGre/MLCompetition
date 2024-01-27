from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, funksvd
from lenskit.algorithms.bias import Bias
from lenskit.metrics.predict import rmse
from lenskit import datasets
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


# Split the data into training and test sets
train, test = next(xf.partition_users(train_df, 1, xf.SampleFrac(0.2)))

# Initialize the model
#algo = als.BiasedMF(50, iterations=50, reg=0.01)
algo = funksvd.FunkSVD(50, iterations=50, reg=0.01)
# Fit the model
model = algo.fit(train)

# Generate predictions for the test set
preds = batch.predict(algo, test, n_jobs=1)

# Calculate the RMSE
error = rmse(preds['prediction'], preds['rating'])
print(f'Test RMSE: {error:.2f}')