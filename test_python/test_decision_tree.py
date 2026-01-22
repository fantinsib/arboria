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

def test_decision_tree_min_sample_split():
    X = np.array([[1,2,1],[4,5,5], [7,8,9]])
    y = np.array([0,1,1])

    tree = DecisionTree(min_sample_split=5) # fits the tree with no arguments
    tree.fit(X,y)

def test_decision_tree_min_sample_effective():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    bc = load_breast_cancer()
    X = bc.data.astype(np.float32)  
    y = bc.target.astype(np.int32) 
    x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=10)

    tree1 = DecisionTree(min_sample_split=400)
    tree2 = DecisionTree(min_sample_split=2)

    tree1.fit(x_train, y_train)
    tree2.fit(x_train, y_train)

    pred1 = tree1.predict(x_test)
    pred2 = tree2.predict(x_test)


    assert(accuracy(pred1, pred2) != 1)

