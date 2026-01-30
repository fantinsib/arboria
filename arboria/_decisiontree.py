
from ._arboria import DecisionTree as _DecisionTreeBase


class _DecisionTree(_DecisionTreeBase):
    def __init__(
        self,
        max_depth: int | None = None,
        min_sample_split: int | None = None,
        type: str = "classification",
    ):
        """
        Decision tree classifier.

        Parameters
        ----------
        max_depth : int
            Maximum depth of the tree. Default is None
        min_sample_split : int
            Minimum of samples allowed in a leaf. Default None will set no limit
        """

        super().__init__(
            max_depth=max_depth,
            min_sample_split=min_sample_split,
            type=type,
        )

    def fit(self, X, y, criterion="gini"):
        """
        Fit the decision tree.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
        criterion : {"gini", "entropy"}, default="gini"
        """
        if not hasattr(X, "__array_interface__"):
            raise TypeError("X must be a NumPy-compatible array")

        if not hasattr(y, "__array_interface__"):
            raise TypeError("y must be a NumPy-compatible array")
        
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
        if not hasattr(X, "__array_interface__"):
            raise TypeError("X must be a NumPy-compatible array")

        return self._predict(X)
