from sklearn.neighbors import KNeighborsClassifier

def knn_eval(train_features, train_labels, test_features, knn):
    knn = KNeighborsClassifier(n_neighbors=knn)
    knn.fit(train_features, train_labels)
    predicted_labels = knn.predict(test_features)
    return predicted_labels
