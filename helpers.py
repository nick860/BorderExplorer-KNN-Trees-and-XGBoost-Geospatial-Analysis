import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn.tree
from knn import KNNClassifier
from xgboost import XGBClassifier


def decision_tree_demo(x_train, y_train, x_test, y_test):
    # Create random data
    # Initialize Decision Tree classifier
    tree_classifier = DecisionTreeClassifier(random_state=42)

    # Train the Decision Tree on the training data
    tree_classifier.fit(x_train, y_train)

    # Make predictions on the test data
    y_pred = tree_classifier.predict(x_test)

    # Compute the accuracy of the predictions
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy}")


def loading_random_forest():
    model = RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=4)


def loading_xgboost(X_train, Y_train, X_test, Y_test, X_valid, Y_valid):
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=4)
    model.fit(X_train, Y_train)
    print("the xgboost accurcy on test set is " + str(np.mean(model.predict(X_test) == Y_test)))
    print("the xgboost accurcy on validation set is " + str(np.mean(model.predict(X_valid) == Y_valid)))
    print("the xgboost accurcy on train set is " + str(np.mean(model.predict(X_train) == Y_train)))
    plot_decision_boundaries(model, X_test, Y_test, title=' xgboost on test set')
    plot_decision_boundaries(model, X_valid, Y_valid, title=' xgboost on validation set')
    plot_decision_boundaries(model, X_train, Y_train, title=' xgboost on train set')
    
def plot_decision_boundaries(model, X, y, title='Decision Boundaries'):
    """
    Plots decision boundaries of a classifier and colors the space by the prediction of each point.

    Parameters:
    - model: The trained classifier (sklearn model).
    - X: Numpy Feature matrix.
    - y: Numpy array of Labels.
    - title: Title for the plot.
    """
    # h = .02  # Step size in the mesh

    # enumerate y
    y_map = {v: i for i, v in enumerate(np.unique(y))}
    enum_y = np.array([y_map[v] for v in y]).astype(int)

    h_x = (np.max(X[:, 0]) - np.min(X[:, 0])) / 200
    h_y = (np.max(X[:, 1]) - np.min(X[:, 1])) / 200

    # Plot the decision boundary.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

    # Make predictions on the meshgrid points.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([y_map[v] for v in Z])
    Z = Z.reshape(xx.shape)
    vmin = np.min([np.min(enum_y), np.min(Z)])
    vmax = np.min([np.max(enum_y), np.max(Z)])

    # Plot the decision boundary.
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8, vmin=vmin, vmax=vmax)

    # Scatter plot of the data points with matching colors.
    plt.scatter(X[:, 0], X[:, 1], c=enum_y, cmap=plt.cm.Paired, edgecolors='k', s=40, alpha=0.7, vmin=vmin, vmax=vmax)

    plt.title("Decision Boundaries")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.show()


def knn_examples(X_train, Y_train, X_test, Y_test, k , distance_metric):
    """
    Notice the similarity to the decision tree demo above.
    This is the sklearn standard format for models.
    """

    # Initialize the KNNClassifier with k=5 and L2 distance metric
    knn_classifier = KNNClassifier(k, distance_metric)

    # Train the classifier
    knn_classifier.fit(X_train, Y_train)

    # Predict the labels for the test set
    y_pred = knn_classifier.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = np.mean(y_pred == Y_test)
    return knn_classifier , accuracy

def read_data_demo(filename):
    """
    Read the data from the csv file and return the features and labels as numpy arrays.
    """
    return pd.read_csv(filename).to_numpy()
def anomaly_detection_demo(X_train, Y_train, X_test):
    # x_test should be the one from the AD_test.csv
    knn_classifier = KNNClassifier(5, 'l2')
    knn_classifier.fit(X_train, Y_train)
    distance, indx = knn_classifier.knn_distance(X_test)
    distance_anomaly_scores = np.sum(distance, axis=1)
    anomalies = ad_set[np.argsort(distance_anomaly_scores)[-50:]]
    normal_points = ad_set[np.argsort(distance_anomaly_scores)[:distance_anomaly_scores.shape[0] - 50]]
    plt.scatter(X_train[:, 0], X_train[:, 1], color='black', alpha=0.01)
    plt.scatter([], [], color='black', label='train') # dummy points for legend
    plt.scatter(normal_points[:, 0], normal_points[:, 1], color='blue', label='normal')
    plt.scatter(anomalies[:, 0], anomalies[:, 1], color='red', label='anomaly')
    plt.title("Anomaly Detection")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.show()
    
def train_decision_trees(X_train, Y_train, X_test, Y_test, X_valid, Y_valid):
    np.random.seed(42)
    decision_trees = []

    save_model = []

    for max_depth in  [1, 2, 4, 6, 10, 20, 50, 100]:
        for max_leaf_nodes in [50, 100, 1000]:
            tree = DecisionTreeClassifier(random_state=42, max_leaf_nodes=max_leaf_nodes, max_depth=max_depth)
            tree.fit(X_train, Y_train)

            y_pred = tree.predict(X_test)
            accuracy_test = np.mean(y_pred == Y_test)

            y_pred_train = tree.predict(X_train)
            accuracy_train = np.mean(y_pred_train == Y_train)

            
            y_pred_valid = tree.predict(X_valid)
            accuracy_valid = np.mean(y_pred_valid == Y_valid)

            save_model.append((tree, max_depth, max_leaf_nodes,accuracy_train, accuracy_test,accuracy_valid ))

    valid_accurcy_idx = np.argmax([i[5] for i in save_model])
    valid_accurcy_max = save_model[valid_accurcy_idx][5]
    test_accurcy = save_model[valid_accurcy_idx][4]
    train_accurcy = save_model[valid_accurcy_idx][3]
    print("the max valid accurcy is " + str(valid_accurcy_max) + " and the test accurcy is " + str(test_accurcy) + " and the train accurcy is " + str(train_accurcy))
    print(save_model[valid_accurcy_idx])
    plot_decision_boundaries(save_model[valid_accurcy_idx][0], X_test, Y_test, title=' max valid accurcy tree on test set')
    plot_decision_boundaries(save_model[valid_accurcy_idx][0], X_valid, Y_valid, title=' max valid accurcy tree on validation set')
    plot_decision_boundaries(save_model[valid_accurcy_idx][0], X_train, Y_train, title=' max valid accurcy tree on train set')
    tree_of_50_leafs = [i for i in save_model if i[2] == 50]
    best_of_50_leafs_idx = np.argmax([i[5] for i in tree_of_50_leafs])
    print("the best of 50 leafs is " + str(tree_of_50_leafs[5]))
    plot_decision_boundaries(tree_of_50_leafs[best_of_50_leafs_idx][0], X_test, Y_test, title=' best of 50 leafs tree on test set')
    plot_decision_boundaries(tree_of_50_leafs[best_of_50_leafs_idx][0], X_valid, Y_valid, title=' best of 50 leafs tree on validation set')
    plot_decision_boundaries(tree_of_50_leafs[best_of_50_leafs_idx][0], X_train, Y_train, title=' best of 50 leafs tree on train set')

    tree_of_max_depth = [i for i in save_model if i[1] <= 6]   
    best_of_max_depth_idx = np.argmax([i[5] for i in tree_of_max_depth])
    print("the best of max depth is " + str(tree_of_max_depth[best_of_max_depth_idx]))
    plot_decision_boundaries(tree_of_max_depth[best_of_max_depth_idx][0], X_test, Y_test, title=' best of max depth tree on test set')
    plot_decision_boundaries(tree_of_max_depth[best_of_max_depth_idx][0], X_valid, Y_valid, title=' best of max depth tree on validation set')
    plot_decision_boundaries(tree_of_max_depth[best_of_max_depth_idx][0], X_train, Y_train, title=' best of max depth tree on train set')

    random_forest = RandomForestClassifier(n_estimators=300, max_depth=6)
    random_forest.fit(X_train, Y_train)
    plot_decision_boundaries(random_forest, X_test, Y_test, title=' random forest on test set')
    plot_decision_boundaries(random_forest, X_valid, Y_valid, title=' random forest on validation set')
    plot_decision_boundaries(random_forest, X_train, Y_train, title=' random forest on train set')
    print("the random forest accurcy on test set is " + str(np.mean(random_forest.predict(X_test) == Y_test)))
    print("the random forest accurcy on validation set is " + str(np.mean(random_forest.predict(X_valid) == Y_valid)))
    print("the random forest accurcy on train set is " + str(np.mean(random_forest.predict(X_train) == Y_train)))
    return decision_trees   


def KNN (X_train, Y_train, X_test, Y_test):
    knn_modle = {'l1': [], 'l2': []}
    k_values = [1, 10 , 100, 1000, 3000]
    index = [f"k={k}" for k in k_values]
    data = {'l1': [] , 'l2': []}
    for k in k_values:
        for metric in ['l1', 'l2']:
            knn_classifier, accuracy = knn_examples(X_train , Y_train, X_test, Y_test, k , metric)
            knn_modle[metric].append((knn_classifier, accuracy))
            data[metric].append(accuracy)

    print(pd.DataFrame(data, index=index))
    k_max_l2 = k_values[np.argmax(data['l2'])]
    k_min_l2 = k_values[np.argmin(data['l2'])]
    print("the max k " + str(k_max_l2))
    print("the min k " + str(k_min_l2))
    plot_decision_boundaries(knn_modle['l2'][np.argmax(data['l2'])][0], X_test, Y_test, title='Decision Boundaries l2 k max')
    plot_decision_boundaries(knn_modle['l2'][np.argmin(data['l2'])][0], X_test, Y_test, title='Decision Boundaries l2 k min')
    plot_decision_boundaries(knn_modle['l1'][np.argmax(data['l2'])][0], X_test, Y_test, title='Decision Boundaries l1 k max')

if __name__ == '__main__':
    train_set = read_data_demo('train.csv')
    validation_set = read_data_demo('validation.csv')
    test_set = read_data_demo('test.csv')
    ad_set = read_data_demo('AD_test.csv')
    X_train = train_set[:, :2]
    Y_train = train_set[:, 2]

    X_test  = test_set[:, :2]
    Y_test = test_set[:, 2]  

    X_valid = validation_set[:, :2]
    Y_valid = validation_set[:, 2]

    X_ad_set = ad_set[:, :2]
    
    KNN(X_train, Y_train, X_test, Y_test)
    anomaly_detection_demo(X_train, Y_train, X_ad_set)
    train_decision_trees(X_train, Y_train, X_test, Y_test, X_valid, Y_valid)
    loading_xgboost(X_train, Y_train, X_test, Y_test, X_valid, Y_valid)