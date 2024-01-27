from sklearn.tree import DecisionTreeRegressor


def decisionTree(X_train, y_train, X_test):
    # create a DecisionTreeRegressor object
    tree = DecisionTreeRegressor()

    # fit the model to the training data
    tree.fit(X_train, y_train)

    # make predictions on the test data
    predictions = tree.predict(X_test)

    return predictions
