from arboria import ExtraTreeRegressor, ExtraTreeClassifier
import numpy as np
import pytest

def test_extratree_constructor():

    tree = ExtraTreeRegressor(n_estimators=100,
                              n_random_split=2)
    assert(isinstance(tree, ExtraTreeRegressor))
    
def test_extratree_constructor_error():
    
    with pytest.raises(ValueError):
        tree = ExtraTreeRegressor(n_estimators=100,
                              n_random_split=-1)
    
def test_extratree_simple_predict():
        X = np.array([[0,0,1],[1,1,2], [1,2,1],[4,5,5], [7,8,9], [10,11, 12]])
        y = np.array([0,0,0,1,1,1])

        et = ExtraTreeClassifier(n_estimators=100, 
                                 n_random_split=1,
                                 seed = 1)
        
        et.fit(X,y)
        