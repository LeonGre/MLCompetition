from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als
from lenskit import batch
from lenskit.algorithms import Recommender, funksvd

from lenskit import crossfold as xf

from sklearn.metrics import mean_squared_error
from math import sqrt

def calculate_rmse(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = sqrt(mse)
    return rmse

def cross_validation(train_features, train_labels, test_features, n_folds=5):
    # Create a copy of the training data and add the labels
    data = train_features.copy()
    data['rating'] = train_labels

    # Initialize a list to store the results for each fold
    results = []

    # Split the data into training and test sets
    for train, test in xf.partition_users(data, n_folds, xf.SampleFrac(0.2)):
        # Train and test the model
        preds = biased_mf(train, test)
        # Save the results
        results.append(preds)

    # Calculate the average result
    avg_result = sum(results) / len(results)

    return avg_result


def biased_mf(train_data, test_features):
    # Create a Recommender
    algo = als.BiasedMF(50, iterations=50, reg=0.01)
    # Create a recommender
    rec = Recommender.adapt(algo)

    # Train the recommender
    rec.fit(train_data)
    # Make predictions
    preds = batch.predict(rec, test_features, n_jobs=1)
    return preds


def unbiased_mf(train_features, train_labels, test_features):
    # Create a Recommender
    algo = als.ImplicitMF(50, iterations=50, reg=0.01)
    # Create a recommender
    rec = Recommender.adapt(algo)

    train_data = train_features.copy()
    train_data['rating'] = train_labels

    # Train the recommender
    rec.fit(train_data)
    # Make predictions
    preds = batch.predict(rec, test_features)
    # Save the predictions
    preds.to_csv("./regression-data/test_label.csv", index=False)
    return preds


def funk_svd(train_features, train_labels, test_features):
    # Create a Recommender
    algo = funksvd.FunkSVD(50, iterations=50)
    # Create a recommender
    rec = Recommender.adapt(algo)

    train_data = train_features.copy()
    train_data['rating'] = train_labels

    # Train the recommender
    rec.fit(train_data)
    # Make predictions
    preds = batch.predict(rec, test_features)
    return preds
