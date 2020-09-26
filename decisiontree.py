# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 12:44:28 2020

@author: elisa
"""

##DecisionTree
import pandas as pd
import seaborn as sns
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def scaled_entropy(y):
    counts = np.bincount(y)
    probs = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probs if p > 0]) #since log is not defined for negative numbers

class Node: 
    def __init__(self, feature = None, threshold = None, left = None, right = None,value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value    
    
    
    def leaf_node(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth= 10, n_feats=25):
        self.min_samples_split= min_samples_split
        self.max_depth= max_depth
        self.n_feats= n_feats
        self.root= None
        
   
    def split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs
    
        
    def build_tree(self, X, y , depth = 0):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_labels = len(np.unique(y))
        
        #stopping criteria
        if (n_labels ==1
            or depth >= self.max_depth
            or n_samples < self.min_samples_split):
            leaf_value = self.most_frequent_label(y)
            return Node(value= leaf_value)  #so if it meets stoppingcrit. ok, otherwise it continues
        
        feat_idxs = np.random.choice(n_features, self.n_feats, replace= False) #it selectes indexes from the n of features randomly without replacement since we don't want same index many times and the array has lenght= self.n_feats
        
        #greedy search
        best_feat, best_threshold = self.Best_criteria(X, y, feat_idxs)
        left_idxs, right_idxs = self.split(X[:, best_feat], best_threshold)
        left = self.build_tree(X[left_idxs, :], y[left_idxs], depth = depth+1) #iterate and grow the tree at the left of the node
        right = self.build_tree(X[right_idxs, :], y[right_idxs], depth = depth+1) #iterate and grow the tree at the right of the node
        return Node(best_feat, best_threshold, left, right)
    
                                            
    
    def Information_Gain(self, y, X_column, threshold):
        e_parent = scaled_entropy(y)
        #generate split
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0 #as information gain
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = scaled_entropy(y[left_idxs]), scaled_entropy(y[right_idxs])
        #weighted avg of children' entropy
        e_children = (n_left/len(y))*e_left + (n_right/len(y))*e_right                                          
        #return information gain
        Information_Gain = e_parent- e_children
        return Information_Gain 
    
    
    def Best_criteria(self, X, y, feat_idxs):
        best_gain = 0
        split_idx, split_threshold = None, None
        for f in feat_idxs:
            X_column= X[:, f]
            thresholds = np.unique(X_column)
            for t in thresholds:
                infogain = self.Information_Gain(y, X_column, t)
                
                if infogain > best_gain:
                    best_gain= infogain
                    split_idx = f
                    split_threshold = t
        return split_idx, split_threshold
    
    
    def most_frequent_label(self,y):
        most_frequent = Counter(y).most_common(1)[0][0] #since most_common is a list of tuples we look for the first element of the first tuple
        return most_frequent



    def fit(self,X,y):
        #grow tree
        self.n_feats= X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1]) #if not specified n of features for the split take the n of rows in X, otherwise the mmin among the nfeats specified above and the n of rows of X
        self.root = self.build_tree(X,y)
        
    
    
    #predict tree
    def along_tree(self, x, node):
        if node.leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.along_tree(x,node.left)
        return self.along_tree(x, node.right)
         
    
    def predict(self, X):
        return np.array([self.along_tree(x, self.root) for x in X])
    
def accuracy(real, predicted):
	correct = 0
	for i in range(len(real)):
		if real[i] == predicted[i]:
			correct += 1
	return correct / float(len(real)) * 100.0


def error(real, predicted):
    return 100 - accuracy(real, predicted)