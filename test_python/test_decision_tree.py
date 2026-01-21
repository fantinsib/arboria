from arboria import DecisionTree, accuracy
import pytest 
import numpy as np

def test_decision_tree_instanciation():
    tree = DecisionTree(max_depth=5)
    assert isinstance(tree, DecisionTree)

def test_decision_tree_fit():

    X = np.array([[1,2,1],[4,5,5], [7,8,9]])
    y = np.array([0,1,1])

    tree = DecisionTree(max_depth=5)
    tree.fit(X,y)
    y_pred = tree.predict(np.array([1,1,1]))
    assert( y_pred[0] == 0)

def test_decision_tree_default_args():
    X = np.array([[1,2,1],[4,5,5], [7,8,9]])
    y = np.array([0,1,1])

    tree = DecisionTree() # fits the tree with no arguments
    tree.fit(X,y)
    y_pred = tree.predict(np.array([1,1,1]))
    assert( y_pred[0] == 0)