import numpy as np
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate


def surprise_collaborative_filtering_cv(train_df, X_test, random_state1, cv_method):
    # Surprise library expects a different format, so we need to create a Reader and a Dataset
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df[["user", "item", "rating"]], reader)

    # Use the SVD algorithm for collaborative filtering
    svd = SVD(random_state=random_state1, n_epochs=20, lr_all=0.005, reg_all=0.2)

    # Perform cross validation
    cv_results = cross_validate(svd, data, measures=['RMSE'], cv=5, verbose=True, n_jobs=-1)

    # Build full trainset and fit the model
    train_data = data.build_full_trainset()
    svd.fit(train_data)

    # Predict ratings for test set
    test_predictions = []
    for index, row in X_test.iterrows():
        prediction = svd.predict(row["user"], row["item"]).est
        test_predictions.append(prediction)

    print(f'Average RMSE: {np.mean(cv_results["test_rmse"])}')

    return test_predictions
