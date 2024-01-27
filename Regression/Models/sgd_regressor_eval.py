from sklearn.linear_model import SGDRegressor

def sgd_regressor(train_features, train_labels, test_features):
    
    # Create and fit the model
    model = SGDRegressor()
    model.fit(train_features, train_labels)

    # Make predictions
    predictions = model.predict(test_features)

    return predictions