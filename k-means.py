# -*- coding: utf-8 -*-
"""
Author: Chris Addington
Assignment: k-NN and k-Means
Course: CS 460 Machine Learning
Some code provided by Dr. Siddique at the University of Kentucky

File Description: File to implement k-means algorithm and work
with datasets.
"""
# Import plotting library
import matplotlib.pyplot as plot
from matplotlib import style

# Set plotting style
style.use("ggplot")

# Other imports
from math import *
from decimal import Decimal
from random import choice, uniform
from csv import reader, writer
import os
from copy import deepcopy
import numpy as np

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

# Random dataset generator
def make_random_dataset(num_dimension, num_rows):
    # Make an empty list
    dataset = []
    
    # Populate the list with random numbers
    for i in range(num_rows):
        indv_data = []
        data_sum = 0
        for j in range(num_dimension):
            value = uniform(0.0, 10.0)
            indv_data.append(value)
            data_sum += value

        # Create labels
        if data_sum >= num_dimension * 5.0:
            label = 'pos'
        else:
            label = 'neg'
        indv_data.append(label)
        
        dataset.append(indv_data)
    return dataset

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

# Part 0: select k
k = 3
MAX_RUNS = 1000

# Step 1: randomly select k initial cluster seeds
def rand_select_seeds(dataset):
    # Make dict to hold clusters
    clusters = dict()
    data_copy = deepcopy(dataset)
    
    # Assign clusters
    for i in range(k):
        rand_cluster = choice(data_copy)
        data_copy.remove(rand_cluster)
        rand_cluster = rand_cluster[:-1]
        clusters[i] = rand_cluster
        
    return clusters

# Step 2: calculate distance from each object to each cluster seed
def calc_dist(clusters, dataset):
    # remove labels from dataset
    data_copy = deepcopy(dataset)
    for i in range(len(data_copy)):
        data_copy[i] = data_copy[i][:-1]   
    # calculate euclidean distance between each point and each cluster
    distances = dict()
    i = 0
    for row in data_copy:
        row_dist = list()
        for cluster in clusters.values():
            row_dist.append(calculate_distance(row, cluster))
        distances[i] = row_dist
        i += 1
        
    return distances

# Step 3: Assign each object to the closest cluster
def assign_class(distances, clusters, dataset):
    # Make dictionary for classifications
    classifications = dict()
    
    # Check each data item for it's closest cluster
    for i in range(len(dataset)):
        classification = distances[i].index(min(distances[i]))
        classifications[i] = classification
        
    return classifications
        
# Step 4: Compute the new centroid for each cluster
def compute_new_centroid(clusters, classifications, dataset):
    # Store previous centroids
    prev_cent = dict(clusters)
    new_cent = dict()

    # For every cluster
    for i in range(len(clusters)):
        # For every data vector sum the two dimensions
        d1 = 0
        d2 = 0
        j = 0
        for row in dataset:
            if classifications[j] == i:
                d1 += row[0]
                d2 += row[1]
            j += 1
        d1 = d1 / len(dataset)
        d2 = d2 / len(dataset)
        new_cent[i] = [d1, d2]
    print("new centroids: ", new_cent)
        
    return prev_cent, new_cent

# Calculate difference between old and new cluster
def check_diff(old_clusters, new_clusters):
    # Store total difference and number of dimensions
    total_difference = 0
    num_dimensions = len(old_clusters[0])
    
    # Find the difference in each dimension
    for i in range(k):
        for j in range(len(old_clusters[i])):
            total_difference = abs(old_clusters[i][j] - new_clusters[i][j])
    
    difference = total_difference #/ num_dimensions
    return difference

# Main function
def k_means(num_dimension, num_row):
    # Make dataset with dimensions and rows
    dataset = make_random_dataset(num_dimension, num_row)
        
    # Convert class column to ints
    str_column_to_int(dataset, len(dataset[0])-1)
    
    # Step 1: get k
    clusters = rand_select_seeds(dataset)
    
    # Set tolerance to halt
    tol = 0.001
    diff = 1
    num_runs = 0
    
    # Steps 2-4
    while diff > tol and num_runs < MAX_RUNS:
        # Step 2: Calculate distances
        distances = calc_dist(clusters, dataset)
        
        # Step 3: Assign to clusters
        classifications =  assign_class(distances, clusters, dataset)
        
        # Step 4: Compute new centroid
        prev_clusters, clusters = compute_new_centroid(clusters, classifications, dataset)
        
        # Check difference between old and new clusters
        diff = check_diff(prev_clusters, clusters)
        
        num_runs += 1
    
    print("Number of runs: ", num_runs)
    print("Difference between last clusters: ", diff)    
    return classifications, clusters, dataset
    
        
# Plot the results
def plot_results():
    # Run k-means
    classifications, clusters, dataset = k_means(2, 20)
    
    # Plot cluster seeds
    for i in range(len(clusters)):
        plot.scatter(clusters[i][0], clusters[i][1], marker="o", color="k", s=150, linewidths=5)
        
    # Plot data points
    for i in range(len(dataset)):
        plot.scatter(dataset[i][0], dataset[i][1], marker="x", color="b", s=150, linewidth=5)
    
    
        
if __name__ == "__main__":

    plot_results()
    
        
                    
        



        
    
            
            
    