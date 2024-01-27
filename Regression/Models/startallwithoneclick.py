# ensemble_predictions.py

import pandas as pd

from load_datasets import test_features
from train_regression_model import train_regression_model
from feature_engineering import prepare_features

def ensemble_predictions(svd_predictions, regression_predictions, alpha=0.65):
    return alpha * svd_predictions + (1 - alpha) * regression_predictions

if __name__ == "__main__":
    train_regression_model()

    # Load model predictions
    regression_predictions = pd.read_csv('regression_test_predictions.csv')['Predicted']
    other_model_predictions = pd.read_csv('./../regression-data/')['Predicted']  # Adjust the path

    # Ensemble predictions
    final_predictions = ensemble_predictions(regression_predictions, other_model_predictions)

    final_predictions_df = pd.DataFrame(final_predictions, columns=['Predicted'])
    final_predictions_df.insert(0, 'Id', range(0, len(final_predictions_df)))
    final_predictions_df.to_csv('final_predictions.csv', index=False)
    print("Predictions saved to final_predictions.csv")