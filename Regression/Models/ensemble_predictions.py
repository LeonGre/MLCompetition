import pandas as pd


def ensemble_predictions(svd_predictions, regression_predictions, alpha=0.5):
    return alpha * svd_predictions + (1 - alpha) * regression_predictions

if __name__ == "__main__":
    # Load model predictions
    regression_predictions = pd.read_csv('regression_test_predictions.csv')['Predicted']
    other_model_predictions = pd.read_csv('./../regression-data/test_label.csv')['Predicted']

    # Ensemble predictions
    final_predictions = ensemble_predictions(regression_predictions, other_model_predictions)

    final_predictions_df = pd.DataFrame(final_predictions, columns=['Predicted'])
    final_predictions_df.insert(0, 'Id', range(0, len(final_predictions_df)))
    final_predictions_df.to_csv('final_predictions.csv', index=False)
    print("Predictions saved to final_predictions.csv")