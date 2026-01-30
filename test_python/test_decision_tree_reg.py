from arboria import DecisionTreeRegressor
import numpy as np


def test_decision_tree_regressor_instanciation():
    tree = DecisionTreeRegressor(max_depth=2)
    assert isinstance(tree, DecisionTreeRegressor)


def test_decision_tree_regressor_fit_predict():
    X = np.array([[0.0], [0.0], [10.0], [10.0]], dtype=np.float32)
    y = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)

    tree = DecisionTreeRegressor(max_depth=1)
    tree.fit(X, y, criterion="sse")

    pred_left = tree.predict(np.array([0.0], dtype=np.float32))[0]
    pred_right = tree.predict(np.array([10.0], dtype=np.float32))[0]

    assert abs(pred_left - 2.0) < 1e-6
    assert abs(pred_right - 6.0) < 1e-6


def test_decision_tree_regressor_default_args():
    X = np.array([[1.0], [1.0], [1.0], [1.0]], dtype=np.float32)
    y = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

    tree = DecisionTreeRegressor()
    tree.fit(X, y, criterion="sse")

    pred = tree.predict(np.array([1.0], dtype=np.float32))[0]
    assert abs(pred - 5.0) < 1e-6
