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
            dict(n_estimators = 10),
            dict(),
            dict(seed = 1),
            dict(max_features = 2),
            dict(n_estimators = 10, seed = 1),
        ]
)
def test_random_forest_default_params(kwargs):
    rf = RandomForest(**kwargs)
    assert isinstance(rf, RandomForest)
    X = np.array([[1,2,1],[4,5,5], [7,8,9]])
    y = np.array([0,1,1])
    
    rf.fit(X,y)
    y_pred = rf.predict(np.array([1,1,1]))
    assert( y_pred[0] == 0)

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

    X = np.array([[2,3,5],
        [2,3,5],
        [4, 6, 10],
        [4, 6, 10],
        [8, 12, 20],
        [8, 12, 20]])
    y = np.array([0,1,0,1,0,1])

    rf1 = RandomForest(n_estimators=2, min_sample_split=2, seed = 10)
    rf2 = RandomForest(n_estimators=2, min_sample_split=2, seed = 10)

    rf1.fit(X, y, criterion="entropy")
    rf2.fit(X, y, criterion="entropy")

    s = np.array([4, 12, 5])
    prob1 = rf1.predict_proba(s)
    prob2= rf2.predict_proba(s)

    assert (prob1 == prob2)

    rf3 = RandomForest(n_estimators=2,max_features=1,min_sample_split=1, seed = 123)
    rf4 = RandomForest(n_estimators=2, max_features= 1, min_sample_split=1, seed = 321)

    rf3.fit(X, y, criterion="entropy")
    rf4.fit(X, y, criterion="entropy")

    prob3 = rf3.predict_proba(s)
    prob4= rf4.predict_proba(s)

    assert (prob3 != prob4)

def test_random_forest_max_samples():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    bc = load_breast_cancer()
    X = bc.data.astype(np.float32)  
    y = bc.target.astype(np.int32) 
    x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=10)

    rf1 = RandomForest(max_samples=1.2, max_depth=6, seed = 10)
    rf2 = RandomForest(max_samples = 0.1, max_depth =6, seed = 10)

    rf1.fit(x_train, y_train, criterion="entropy")
    rf2.fit(x_train, y_train, criterion="entropy")

    assert(round(rf1.get_max_samples(),2) == 1.2)
    assert(round(rf2.get_max_samples(),2) == 0.1)

    prob1 = rf1.predict_proba(x_test)
    prob2 = rf2.predict_proba(x_test)

    assert (prob1 != prob2)

def test_random_forest_min_sample_split():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    bc = load_breast_cancer()
    X = bc.data.astype(np.float32)  
    y = bc.target.astype(np.int32) 
    x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=10)

    rf1 = RandomForest(min_sample_split=10, n_estimators=1, max_depth=6, seed = 10)
    rf2 = RandomForest(min_sample_split=500, n_estimators=1, max_depth =6, seed = 10)

    rf1.fit(x_train, y_train, criterion="entropy")
    rf2.fit(x_train, y_train, criterion="entropy")

    prob1 = rf1.predict_proba(x_test)
    prob2 = rf2.predict_proba(x_test)

    assert np.any(prob1 != prob2)

def test_random_forest_parallelism():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    bc = load_breast_cancer()
    X = bc.data.astype(np.float32)  
    y = bc.target.astype(np.int32) 
    x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=10)

    rf1 = RandomForest(min_sample_split=10, n_estimators=1, max_depth=6, seed = 10)
    rf2 = RandomForest(min_sample_split=500, n_estimators=1, max_depth =6, seed = 10)

    rf1.fit(x_train, y_train, criterion="entropy")
    rf2.fit(x_train, y_train, criterion="entropy")

    prob1 = rf1.predict_proba(x_test)
    prob2 = rf2.predict_proba(x_test)

    assert np.any(prob1 != prob2)