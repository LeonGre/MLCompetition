from ConfigSpace import ConfigurationSpace
from smac import Scenario, RunHistory
from surprise import SVDpp, Dataset, Reader, accuracy
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
import pandas as pd
import os
from smac import HyperparameterOptimizationFacade as HPOFacade

from Regression import load_datasets


def train_svdpp_model(str=""):
    # Load and sort the full dataset by timestamp
    train_data = load_datasets.train_df(str)
    train_data = train_data.sort_values('timestamp')

    # Convert to Surprise dataset format
    reader = Reader(rating_scale=(1, 5))

    # Define a time-based train-test split
    test_size = 0.2  # 20% of the data for testing
    split_index = int((1 - test_size) * len(train_data))
    train_df = train_data[:split_index]
    train_df = train_df.sort_values('Id')
    test_df = train_data[split_index:]
    test_df = test_df.sort_values('Id')

    # Configuration Space for hyperparameters
    cs = ConfigurationSpace()
    n_factors = UniformIntegerHyperparameter("n_factors", 50, 150, default_value=100)
    n_epochs = UniformIntegerHyperparameter("n_epochs", 5, 30, default_value=20)
    lr_all = UniformFloatHyperparameter("lr_all", 0.002, 0.010, default_value=0.005)
    reg_all = UniformFloatHyperparameter("reg_all", 0.02, 0.1, default_value=0.04)
    cs.add_hyperparameters([n_factors, n_epochs, lr_all, reg_all])

    # Objective function for hyperparameter optimization
    def objective_function(cfg, seed=0):
        cfg = {k: cfg[k] for k in cfg if cfg[k]}
        algo = SVDpp(**cfg)
        trainset = Dataset.load_from_df(train_df[['user', 'item', 'rating']], reader).build_full_trainset()
        algo.fit(trainset)
        testset = list(test_df[['user', 'item', 'rating']].itertuples(index=False, name=None))
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        return rmse  # SMAC minimizes the objective

    # Scenario object
    scenario = Scenario(configspace=cs,
                        n_trials=100,
                        n_workers=16
                        )

    # Optimize
    smac = HPOFacade(scenario=scenario, target_function=objective_function)
    best_config = smac.optimize()

    # Access the RunHistory object from the SMAC object
    run_history: RunHistory = smac.runhistory
    cost = run_history.get_min_cost(best_config)

    print("Best configuration:", best_config)
    print("Best RMSE:", cost)

    # Extract the parameters of the best configuration
    best_params = {k: best_config[k] for k in best_config}

    # Retrain the best model on the entire dataset
    algo = SVDpp(**best_params)
    full_trainset = Dataset.load_from_df(train_data[['user', 'item', 'rating']], reader).build_full_trainset()
    algo.fit(full_trainset)

    # Load test features and prepare for prediction
    test_data = load_datasets.test_features("../")

    # Create testset for prediction and keep track of Ids
    testset_to_predict = [(row.user, row.item, 0) for row in test_data.itertuples(index=False)]
    ids = test_data['Id'].tolist()  # Keep track of Ids

    # Predict
    predictions = algo.test(testset_to_predict)

    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(predictions, columns=['uid', 'iid', 'r_ui', 'est', 'details'])

    # Attach Ids to predictions and save to CSV
    predictions_df['Id'] = ids  # Reattach the Ids
    predictions_df.rename(columns={'est': 'Predicted'}, inplace=True)
    predictions_df = predictions_df[['Id', 'Predicted']]
    predictions_df.sort_values('Id', inplace=True)  # Ensure the order is based on 'Id'
    predictions_df.to_csv("./" + str + "regression-data/test_label.csv", index=False)

    print("Predictions saved to test_label.csv")
    return cost  # return the RMSE of the best model


if __name__ == "__main__":
    best_rmse = train_svdpp_model()
    print("Best RMSE:", best_rmse)
