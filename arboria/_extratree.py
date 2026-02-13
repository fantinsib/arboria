

from ._arboria import ExtraTree as _ExtraTreeBase
import math

class _ExtraTree(_ExtraTreeBase):
    def __init__(self, n_estimators: int = 70,
                 max_features: int | str ="sqrt", 
                 max_depth: int = None, 
                 max_samples: float = None,
                 min_sample_split: int = None,
                 n_random_split: int = 1,
                 n_jobs: int = 1,
                 seed : int | None = None,
                 type : str = "classification"):
    

        if max_features == "sqrt":
            self.mtry = -99
        elif max_features == "log":
            self.mtry = -98
        else:
            self.mtry = max_features
        super().__init__(
            n_estimators=n_estimators,
            m_try=self.mtry,
            max_depth=max_depth,
            min_sample_split=min_sample_split,
            max_samples=max_samples,
            n_random_split = n_random_split,
            n_jobs=n_jobs,
            seed=seed,
            type=type,
        )

    def fit(self, X, y, criterion= 'gini'):
        """
        Fit the Random Forest.

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
        if not hasattr(X, "__array_interface__"):
            raise TypeError("X must be a NumPy-compatible array")

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
        if not hasattr(X, "__array_interface__"):
            raise TypeError("X must be a NumPy-compatible array")
        return self._predict_proba(X)