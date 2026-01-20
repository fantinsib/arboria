from ._arboria import DecisionTree as _DecisionTree
from ._arboria import RandomForest as _RandomForest
from ._arboria import _accuracy 

import math

class DecisionTree(_DecisionTree):
    def __init__(self, max_depth: int):
        """
        Decision tree classifier.

        Parameters
        ----------
        max_depth : int
            Maximum depth of the tree.
        """
        super().__init__(max_depth)

    def fit(self, X, y, criterion="gini"):
        """
        Fit the decision tree.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
        criterion : {"gini", "entropy"}, default="gini"
        """
        return self._fit(X, y, criterion)
    
    def predict(self, X):
        """
        Returns predicted class for samples X.

        Parameters
        ----------
        X : ndarray with same shape as training data

        Returns
        -------
        np.ndarray : array of predicted class as integers.
        """
        return self._predict(X)


class RandomForest(_RandomForest):
    def __init__(self, max_depth: int, max_features: int | str ="sqrt", n_estimators: int =70, seed : int | None = None):
        """
        Random Forest classifier.

        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest. Default is 70
        max_features: int | str
            Number of features to sample at each split. Can be int or
            "sqrt" : value set as the square root of the number of features.
        max_depth : int
            Maximum depth of the tree.
        seed : int
            Seed of the tree. Default None will result in a random seed.
        """
        if max_features == "sqrt":
            self.mtry = -99
        elif max_features == "log":
            self.mtry = -98
        else:
            self.mtry = max_features
        return super().__init__(n_estimators, self.mtry, max_depth, seed)

    def fit(self, X, y, criterion= 'gini'):
        """
        Fit the Random Forest.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
        criterion : {"gini", "entropy"}, default="gini"
        """
        if self.mtry == -99:
            self.mtry = max(1, int(math.sqrt(X.shape[1])))
        if self.mtry == -98:
            self.mtry = max(1, int(math.log2(X.shape[1])))
        return self._fit(X, y, criterion, self.mtry)
    
    def predict(self, X):
        """
        Returns predicted class for samples X.

        Parameters
        ----------
        X : ndarray with same shape as training data

        Returns
        -------
        np.ndarray : array of predicted class as integers.
        """
        return self._predict(X)
    
    def predict_proba(self, X):
        """
        Returns predicted class for samples X as float as the average
        of each tree votes. 

        Parameters
        ----------
        X : ndarray with same shape as training data

        Returns
        -------
        np.ndarray : array of predicted class as float.
        """
        return self._predict_proba(X)
    
    def out_of_bag(self, X, y):
        """
        Returns the out-of-bag accuracy of the Random Forest.

        Parameters
        ----------
        X : ndarray of samples passed as training data
        y : ndarray of target values passed as training data

        Returns
        -------
        float : the score of the Random Forest classifier on 
        the samples not bootstrapped during training
        """

        return self._out_of_bag(X,y)


def accuracy(y_true, y_pred):
    """
    Computes the accuracy score.

    Parameters
    ----------
    y_true : ndarray of target classes
    y_pred : ndarray of predicted classes

    Returns
    -------
    float : the percentage of correct predictions.
    """

    return _accuracy(y_true, y_pred)