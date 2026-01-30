from arboria import RandomForestRegressor
import numpy as np


def test_random_forest_regressor_instanciation():
    rf = RandomForestRegressor(max_depth=2, max_features=1, n_estimators=2, seed=1)
    assert isinstance(rf, RandomForestRegressor)


def test_random_forest_regressor_fit_predict():
    X = np.array([[0.0], [0.0], [10.0], [10.0]], dtype=np.float32)
    y = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)

    rf = RandomForestRegressor(n_estimators=1, max_features=1, max_depth=1, max_samples=1.0, seed=123)
    rf.fit(X, y, criterion="sse")

    preds = np.array(rf.predict(np.array([[0.0], [10.0]], dtype=np.float32)))
    assert preds.shape == (2,)
    assert np.isfinite(preds).all()


def test_random_forest_regressor_reproductible():
    X = np.array([[0.0], [0.0], [10.0], [10.0]], dtype=np.float32)
    y = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)

    rf1 = RandomForestRegressor(n_estimators=2, max_features=1, max_depth=1, max_samples=1.0, seed=10)
    rf2 = RandomForestRegressor(n_estimators=2, max_features=1, max_depth=1, max_samples=1.0, seed=10)

    rf1.fit(X, y, criterion="sse")
    rf2.fit(X, y, criterion="sse")

    sample = np.array([0.0], dtype=np.float32)
    pred1 = rf1.predict(sample)
    pred2 = rf2.predict(sample)

    assert np.all(pred1 == pred2)
