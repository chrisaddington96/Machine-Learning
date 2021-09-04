# -*- coding: utf-8 -*-
"""
Author: Chris Addington
Assignment: k-NN and k-Means
Course: CS 460 Machine Learning
Code provided by Dr. Siddique at the University of Kentucky

File Description: File to implement k Nearest Neighbor algorithm and work
with datasets.
"""
# Import statements
from math import *
from decimal import Decimal
from random import seed
from random import randrange
from csv import reader
import os

# Dataset for testing purporses
'''
dataset = [
            [2.7810836,2.550537003,0],[1.465489372,2.362125076,0],[3.396561688,4.400293529,0],
            [1.38807019,1.850220317,0],[3.06407232,3.005305973,0],[7.627531214,2.759262235,1],
            [5.332441248,2.088626775,1],[6.922596716,1.77106367,1],[8.675418651,-0.242068655,1],
            [7.673756466,3.508563011,1]]
    
row0 = dataset[0]
'''

'''
Calculate distances between points
'''

def p_root(value, root):
    # Calculate the root value
    root_value = 1 / float(root)
    # Return the value ^ root_value to 6 decimal values
    return round (Decimal(value) ** Decimal(root_value), 6)

def calculate_distance(row1, row2, distance_func=2):
    # Remove labels and copy all other entries
    row1, row2 = row1[:-1], row2[:-1]
    # Return the p_root for each entry
    return float(p_root(sum(pow(abs(a-b), distance_func) for a,b in zip(row1, row2)), distance_func))

def test_dist_funcs():
    # compute distance from data point at index 0
    distances = [calculate_distance(row0,row,4) for row in dataset]
    print(distances)
    
'''
Get nearest neighbors
'''

def get_neighbors(train, test_row, num_neighbors=3, distance_func=2):
    # Compute distances
    distances = [(train_row, calculate_distance(test_row, train_row, distance_func))
    for train_row in train]
    
    # Sort distances
    distances.sort(key=lambda tup: tup[1])
    # Get top k-neighbors
    neighbors = [distances[i][0] for i in range(num_neighbors)]
    
    return neighbors

def test_neighbors():
    # Test top 3 neighbors
    neighbors = get_neighbors(dataset, row0, distance_func=3)
    print(neighbors)  
    
'''
Make predictions
'''

def predict_classification(train, test_row, num_neighbors=3, distance_func=2):
    # Find neighbors
    neighbors = get_neighbors(train, test_row, num_neighbors, distance_func)
    # Get output values
    output_values = [row[-1] for row in neighbors]
    # Make the prediction
    prediction = max(set(output_values), key=output_values.count)
    return prediction

def test_prediction():
    prediction = predict_classification(dataset, row0)
    print('Expected %d, Got %d.' % (dataset[0][-1], prediction))
    
'''
Run from csv files
'''

def load_csv(filename):
    __location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # Load dataset from csv file
    dataset = list()
    with open(os.path.join(__location__, filename)) as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def str_column_to_float(dataset, column):
    # Convert string to float in dataset
    for row in dataset:
        row[column] = float(row[column].strip())
        
def str_column_to_int(dataset, column):
    # Convert string to int in dataset
    unique = set([row[column] for row in dataset])
    lookup = {value : i for i, value in enumerate(unique)}
    
    for row in dataset:
        row[column] = lookup[row[column]]
        
    return lookup

def dataset_minmax(dataset):
    # Calculate min/max values for each column
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

def normalize_dataset(dataset, minmax):
    # Rescale dataset to range 0-1
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0] / (minmax[i][i] - minmax[i][0]))
            
def cross_validation_split(dataset, n_folds):
    # Split dataset into k folds
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def accuracy_metric(actual, predicted):
    # Calculate accuracy percentage
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    # Evaluate the algorithm using cross validation split
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

def k_nearest_neighbors(train, test, num_neighbors=3, distance_func=2):
    #kNN algorithm
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors, distance_func)
        predictions.append(output)
    return predictions

# Global vars for testing
seed(1)
n_folds = 5
num_neighbors = 5
distance_func = 2

def kNN_driver(filename='iris.csv'):
    # Load data set
    dataset = load_csv(filename)
    
    # Change string to float for each entry
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
        
    # Convert class column to ints
    str_column_to_int(dataset, len(dataset[0])-1)
    
    # Evaluate algorithm

    scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors, distance_func)
    print('Scores %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
'''
Execute main function
'''
if __name__ == "__main__":
    #test_dist_funcs()
    #test_neighbors()
    #test_prediction()
    kNN_driver()