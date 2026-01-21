from arboria import RandomForest, accuracy
import pytest
import numpy as np

def test_random_forest_init():
    rf = RandomForest(max_depth=5, max_features=3, n_estimators=10, seed=1)
    assert isinstance(rf, RandomForest)

@pytest.mark.parametrize(
        "kwargs",
        [
            dict(max_features = 2, max_depth= 5),
            dict(n_estimators = 10, max_depth= 5),
        ]
)
def test_random_forest_default_params(kwargs):
    rf = RandomForest(**kwargs)
    assert isinstance(rf, RandomForest)

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_estimators=0,  max_features=3, max_depth=5),
        dict(n_estimators=10, max_features=0, max_depth=5),
        dict(n_estimators=10, max_features=-97, max_depth=5),
        dict(n_estimators=10, max_features=3, max_depth=0),
    ]
)
def test_random_forest_bad_params(kwargs):

    with pytest.raises(ValueError):
        RandomForest(**kwargs)

def test_random_forest_reproductible():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    bc = load_breast_cancer()
    X = bc.data.astype(np.float32)  
    y = bc.target.astype(np.int32) 
    x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=10)

    rf1 = RandomForest(max_depth = 10, seed = 10)
    rf2 = RandomForest(max_depth = 10, seed = 10)

    rf1.fit(x_train, y_train, criterion="entropy")
    rf2.fit(x_train, y_train, criterion="entropy")

    prob1 = rf1.predict_proba(x_test)
    prob2= rf2.predict_proba(x_test)

    assert (accuracy(prob1, prob2)==1)







