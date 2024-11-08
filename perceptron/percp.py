import numpy as np
import numpy.typing as npt
import pandas as pd
import warnings
import plotly.express as px



def map_labels_to_binary(labels: npt.NDArray[np.str_], to_ones: str, to_zeroes: str) -> npt.NDArray[np.int_]:
    
    with pd.option_context('future.no_silent_downcasting', True):
        return pd.Series(labels).replace({to_ones : 1, to_zeroes : 0}).to_numpy()

def map_binary_to_labels(labels: npt.NDArray[np.int_], from_ones: str, from_zeroes: str) -> npt.NDArray[np.str_]:

    with pd.option_context('future.no_silent_downcasting', True):
        return pd.Series(labels).replace({1 : from_ones, 0 : from_zeroes}).to_numpy()

iris_mapping = {
    0 : 'sepal_length',
    1 : 'sepal_width',
    2 : 'petal_length',
    3 : 'petal_width',
    4 : 'species'
}

df = pd.read_csv(
 'https://archive.ics.uci.edu/ml/'
 'machine-learning-databases/iris/iris.data',
 header=None, encoding='utf-8').rename(columns=iris_mapping)

# 'Iris-versicolor', 'Iris-setosa'
# sepal_length_feature = df.loc[0:100, 'sepal_length']
# petal_length_feature = df.loc[0:100, 'petal_width']
features = df.loc[:99, ['sepal_length', 'petal_length']].values
labels = df.loc[:99, 'species'].values

#features = df.iloc[0:100, [0, 2]].values # Raschka's feature selection 
#labels = df.iloc[0:100, 4].values # Raschka's target selection
labels_mapped_to_binary: npt.NDArray[np.int_] = map_labels_to_binary(labels, 'Iris-versicolor', 'Iris-setosa')
binary_mapped_to_labels: npt.NDArray[np.str_] = map_binary_to_labels(labels_mapped_to_binary, 'Iris-versicolor', 'Iris-setosa')

def fit():
    pass

def sigma(z: float):
    
    if z >= 0:
        return 1 
    else:
        return 0
    
def net_input(features: npt.NDArray[np.float64], weights: npt.NDArray[np.float64], bias: float):

    net_input = np.dot(weights.T, features) + bias # overflow here when matmul
    return net_input

def output_value(true_class_label, predicted_class_label):

    return true_class_label - predicted_class_label


eta = 0.1
number_of_iterations = 10
seed = 1
bias = 0.0

rgen = np.random.RandomState(seed)
weights = rgen.normal(loc=0.0, scale=0.01, size=features.shape[1])

errors = []
for _ in range(number_of_iterations):
    # zip features per category, e.g one example (training row) with two features (columns)
    # per one class label
    error_count_per_training_iteration = 0
    for xi, y in zip(features, labels_mapped_to_binary):
        
        sigma_prediction = sigma(net_input(xi, weights, bias))
        class_label_delta = output_value(y, sigma_prediction)
        # print(class_label_delta, sigma_prediction)
        # Update the weights and bias unit
        # xi does not have to represent just a single value
        # We are using the power of linear algebra to do everything using matrices
        # E.g xi can be [1.35, 5,35], remember, it must have same dimensions as weights
        # Do not confuse xi with X (capital X)

        # with warnings.catch_warnings(record=True) as w:
        weights += eta * class_label_delta * xi 
        bias += eta * class_label_delta

        error_count_per_training_iteration += int((eta * class_label_delta) != 0.0)

        # if class_label_delta != 0:
        #     error_count_per_training_iteration += 1


    errors.append(error_count_per_training_iteration)


px.scatter(
y=errors, 
x=range(1, len(errors) + 1),
labels={'x' : 'Epochs', 'y' : 'Number of updates'},
title='Perceptron training stats')