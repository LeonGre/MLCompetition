import os

import pandas as pd
from ConfigSpace import UniformFloatHyperparameter, ConfigurationSpace
from sklearn.metrics import mean_squared_error
from smac import MultiFidelityFacade as MFFacade, Scenario
from smac import HyperparameterOptimizationFacade as HPOFacade

import load_datasets
from Models.train_regression_model import train_regression_model
from train_svdpp_model import train_svdpp_model


def time_based_split(data: pd.DataFrame, test_size=0.2):
    # Assuming 'timestamp' is your time column and it's properly formatted
    data = data.sort_values('timestamp')
    # Calculate the index to split the data
    split_index = int((1 - test_size) * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data


# sadly this is not working because there was a logic mistake i couldnt fix in time
# so i just used the rmse to calculate the weights
def ensemble_error(config, seed=0):
    print("ensemble_error function called")
    weights = [float(config[k]) for k in config if k.startswith('weight')]
    weights = [w / sum(weights) for w in weights]

    # Load model predictions
    regression_predictions = pd.read_csv('regression_test_predictions.csv')
    regression_predictions = pd.merge(regression_predictions, load_datasets.train_features("../"), on='Id')
    regression_predictions = regression_predictions[['Id', 'Predicted', 'timestamp']]
    regression_train, regression_test = time_based_split(regression_predictions)

    other_model_predictions = pd.read_csv('./../regression-data/test_label.csv')
    other_model_predictions = pd.merge(other_model_predictions, load_datasets.train_features("../"), on='Id')
    other_model_predictions = other_model_predictions[['Id', 'Predicted', 'timestamp']]
    other_model_train, other_model_test = time_based_split(other_model_predictions)

    # Load validation targets
    validation_targets = load_datasets.train_df("../")
    _, validation_targets = time_based_split(validation_targets)

    # Ensemble predictions
    final_predictions = weights[0] * regression_train + weights[1] * other_model_train
    print("Length of regression_train:", len(regression_train))
    print("Length of other_model_train:", len(other_model_train))
    print("Length of validation_targets:", len(validation_targets))
    print("Length of final_predictions:", len(final_predictions))

    # Compute error on validation set
    error = mean_squared_error(validation_targets, final_predictions, squared=False)
    print("RMSE error calculated:", error)
    return error


if __name__ == "__main__":
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '0'

    rmse_regressor = train_regression_model()
    rmse_svdpp = train_svdpp_model("../")


    inverse_error_regressor = 1 / rmse_regressor
    inverse_error_svdpp = 1 / rmse_svdpp

    sum_inverse_errors = inverse_error_regressor + inverse_error_svdpp
    weight_regressor = inverse_error_regressor / sum_inverse_errors
    weight_svdpp = inverse_error_svdpp / sum_inverse_errors

    print("Weight for regressor:", weight_regressor)
    print("Weight for SVD++:", weight_svdpp)

    regression_predictions = pd.read_csv('regression_test_predictions.csv')
    other_model_predictions = pd.read_csv('./../regression-data/test_label.csv')
    final_predictions = weight_regressor * regression_predictions + weight_svdpp * other_model_predictions

    final_predictions_df = pd.DataFrame(final_predictions, columns=['Predicted'])
    final_predictions_df.insert(0, 'Id', range(0, len(final_predictions_df)))
    final_predictions_df.to_csv('final_predictions.csv', index=False)
    print("Predictions saved to final_predictions.csv")

"""
    cs = ConfigurationSpace()
    weight1 = UniformFloatHyperparameter("weight1", 0.0, 1.0)
    weight2 = UniformFloatHyperparameter("weight2", 0.0, 1.0)
    cs.add_hyperparameters([weight1, weight2])

    # Define your Scenario
    scenario = Scenario(configspace=cs,
                        n_trials=300,
                        n_workers=16,
                        seed=-1,
                        )

    # Instantiate SMAC4HPO object
    smac = HPOFacade(scenario=scenario, target_function=ensemble_error)

    optimized_weights = smac.optimize()

    # Load model predictions
    reg = pd.read_csv('regression_test_predictions.csv')['Predicted']
    other = pd.read_csv('./../regression-data/test_label.csv')['Predicted']

    optimized_weight1 = optimized_weights.get('weight1')
    optimized_weight2 = optimized_weights.get('weight2')

    # Ensemble predictions
    final_predictions = optimized_weight1 * reg + optimized_weight2 * other
"""

