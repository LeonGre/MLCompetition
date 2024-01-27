# train_regression_model.py
import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import randint

from Regression import load_datasets
from feature_engineering import prepare_features, prepare_test_features  # Import the function


def train_regression_model():
    # Load and preprocess the data
    train_data, test_data = prepare_features()

    # Define feature columns
    feature_columns = [
        'user', 'item', 'timestamp', 'user_avg_rating', 'user_rating_count',
        'user_rating_var', 'item_avg_rating', 'item_rating_count', 'item_rating_var'
    ]

    X_train = train_data[feature_columns]
    y_train = train_data['rating'].astype(float)
    X_test = test_data[feature_columns]
    # Fill NaN values with the mean value of the column
    X_test = X_test.fillna(X_test.mean())
    y_test = test_data['rating'].astype(float)

    print("X_train isnan " + str(X_train.isna().any().any()))
    print("y_train isnan " + str(y_train.isna().any().any()))
    print("X_test isnan " + str(X_test.isna().any().any()))
    print("y_test isnan " + str(y_test.isna().any().any()))

    # Hyperparameter distribution for RandomizedSearchCV
    param_dist = {
        'max_iter': randint(100, 200),
        'max_depth': [10, 20, None],
        'min_samples_leaf': randint(1, 11),
        'l2_regularization': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'learning_rate': [0.1, 0.01, 0.001, 0.0001],
        'max_leaf_nodes': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }

    # Initialize the base model
    model = HistGradientBoostingRegressor(random_state=161, max_depth=5, verbose=1)

    cv = StratifiedKFold(n_splits=5)

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=cv,
                                       scoring='neg_mean_squared_error', random_state=66, n_jobs=-1, verbose=1)
    random_search.fit(X_train, y_train)

    # Initialoze lowest_rmse to infinity
    lowest_rmse = float('inf')

    # Get the RMSE for each fold
    cv_results = random_search.cv_results_
    for i in range(random_search.cv.get_n_splits()):
        fold_score_key = f'split{i}_test_score'
        fold_rmse = np.sqrt(-cv_results[fold_score_key][0])  # Convert from negative MSE
        print(f"RMSE for fold {i + 1}: {fold_rmse}")

        # Update lowest_rmse if current fold's RMSE is lower
        if fold_rmse < lowest_rmse:
            lowest_rmse = fold_rmse

    # Print average RMSE
    mean_test_score = cv_results['mean_test_score']
    avg_rmse = np.sqrt(-mean_test_score[0])  # Convert from negative MSE
    print(f"Average RMSE: {avg_rmse}")

    # Print the best hyperparameters
    print(f"Best Score: {random_search.best_score_}")
    print(random_search.best_params_)
    display(pd.DataFrame(random_search.cv_results_).to_string())

    # Best model
    best_model = random_search.best_estimator_

    # Retrain the best model on the entire dataset (train_data + test_data)
    all_data = pd.concat([train_data, test_data])
    # Fill NaN values with the mean value of the column
    all_data = all_data.fillna(all_data.mean())
    # fit the model
    best_model.fit(all_data[feature_columns], all_data['rating'].astype(float))

    # Load test features
    test_features = load_datasets.test_features("../")

    test_features_prepared = prepare_test_features(test_features, train_data)
    test_features_prepared = test_features_prepared.fillna(test_features_prepared.mean())
    X_test_features = test_features_prepared[feature_columns]
    predictions = best_model.predict(X_test_features)


    # Save predictions to a CSV file

    predictions_df = pd.DataFrame(predictions, columns=['Predicted'])
    predictions_df.insert(0, 'Id', range(0, len(predictions_df)))
    predictions_df.to_csv('regression_test_predictions.csv', index=False)

    print("Regression model predictions saved to regression_test_predictions.csv")
    print("Lowest RMSE train_regression_model:", lowest_rmse)
    return lowest_rmse

if __name__ == "__main__":
    rmse = train_regression_model()
