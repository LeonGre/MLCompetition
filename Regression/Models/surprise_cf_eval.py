from surprise import Dataset, Reader, SVD, SVDpp, accuracy
from surprise.model_selection import cross_validate
from surprise import Dataset, Reader, SVDpp
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms.matrix_factorization import SVDpp

from surprise import Dataset, Reader, SVDpp
from surprise.model_selection import cross_validate, KFold, ShuffleSplit

from surprise import Dataset, Reader, SVDpp
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, GroupKFold
import numpy as np


def surprise_collaborative_filtering(train_df, X_test, random_state1, cv_method):
    # Surprise library expects a different format, so we need to create a Reader and a Dataset
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df[["user", "item", "rating"]], reader)

    # Use the SVD algorithm for collaborative filtering
    svd = SVD(random_state=random_state1, n_epochs=20, lr_all=0.005, reg_all=0.2)

    # Define the cross validation method
    if cv_method == 'StratifiedKFold':
        cv = StratifiedKFold(n_splits=5)
    elif cv_method == 'TimeSeriesSplit':
        cv = TimeSeriesSplit(n_splits=5)
    elif cv_method == 'KFold':
        cv = KFold(n_splits=5)
    elif cv_method == 'GroupKFold':
        cv = GroupKFold(n_splits=5)
    elif cv_method == 'ShuffleSplit':
        cv = ShuffleSplit(n_splits=5)

    # Perform cross validation
    rmse_scores = []
    for train_index, test_index in cv.split(train_df, train_df["rating"]):
        train_data = Dataset.load_from_df(train_df.iloc[train_index][["user", "item", "rating"]],
                                          reader).build_full_trainset()
        test_data = [tuple(x) for x in train_df.iloc[test_index][["user", "item", "rating"]].values]

        svd.fit(train_data)
        predictions = svd.test(test_data)

        # Compute RMSE score
        rmse = accuracy.rmse(predictions, verbose=False)
        rmse_scores.append(rmse)
        print(f'RMSE for this fold: {rmse}')

    # Build full trainset and fit the model
    train_data = data.build_full_trainset()
    svd.fit(train_data)

    # Predict ratings for test set
    test_predictions = []
    for index, row in X_test.iterrows():
        prediction = svd.predict(row["user"], row["item"]).est
        test_predictions.append(prediction)

    print(f'Average RMSE: {np.mean(rmse_scores)}')

    return test_predictions
