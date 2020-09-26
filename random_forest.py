# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 12:45:21 2020

@author: elisa
"""
import pandas as pd
import seaborn as sns
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from decisiontree import DecisionTree


def bootstrap_sample(X,y):
    n_sample = X.shape[0]
    idxs = np.random.choice(n_sample, size = 6000, replace = True)
    return X[idxs], y[idxs]
    
def most_frequent_label(y):
    most_frequent = Counter(y).most_common(1)[0][0] #since most_common is a list of tuples we look for the first element of the first tuple
    return most_frequent

class RandomForest:
    
    def __init__(self, n_trees= 10, min_samples_split= 2, max_depth = 10, n_feats = 25):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []
        

    
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split = self.min_samples_split, max_depth= self.max_depth, n_feats= self.n_feats)
            X_sample, y_sample = bootstrap_sample(X, y)    
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_frequent_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

def accuracy(real, predicted):
	correct = 0
	for i in range(len(real)):
		if real[i] == predicted[i]:
			correct += 1
	return correct / float(len(real)) * 100.0


def error(real, predicted):
    return 100 - accuracy(real, predicted)
