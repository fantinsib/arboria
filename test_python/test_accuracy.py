from arboria import accuracy
import pytest

@pytest.mark.parametrize(
    "args",
    [
        ([0, 0, 1, 1], [0, 0, 1, 1]),
        ([3, 0, 0, 1, 10], [3, 0, 0, 1, 10]),
    ]
)
def test_perf_accuracy(args):
    assert accuracy(*args) == 1

@pytest.mark.parametrize(
    "args",
    [
        ([0, 0, 1, 1], [0, 0, 1, 0]),
        ([3, 0, 0, 1, 10,11,12,13], [3, 0, 0, 1, 10, 11, 1, 1]),
    ]
)
def test_other_accuracy(args):
    assert accuracy(*args) == 0.75


def test_accuracy_dim_mismatch():
    x1 = [1,1,0,0,0]
    x2 = [1,1,0,0]
    with pytest.raises(ValueError):
        accuracy(x1,x2)
