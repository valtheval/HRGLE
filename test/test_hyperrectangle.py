import sys
import os
import pytest

cwd = os.getcwd().split("/")[0:-1]
code_src_path = "/".join(cwd) + "/"
sys.path.append(code_src_path)

from src.hyperrectangle import Hyperrectangle
import numpy as np
import pandas as pd

@pytest.fixture
def exple_X():
    return np.array([[1, 2, 3], [4, 5, 6]]).reshape((2, 3))


@pytest.fixture
def exple_df():
    X = np.array([[1, 2, 3], [4, 5, 6]]).reshape((2, 3))
    return pd.DataFrame(X, columns=list("abc"))


@pytest.fixture
def exple_y():
    return np.array([0,1]).reshape((2,1))


def test_init_around_random_point(exple_X, exple_df):
    expected1 = [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]]
    bounds = Hyperrectangle.init_around_random_point(exple_X, 0.5, 2)
    bounds_df = Hyperrectangle.init_around_random_point(exple_df, 0.5, 2)
    assert bounds == expected1
    assert bounds_df == expected1


def test_filter_dataset(exple_X, exple_y):
    hr = Hyperrectangle(bounds=[[0,2], [0,2], [2,3]])
    x1, y1 = hr.filter_dataset(exple_X, exple_y)
    expected_x, expected_y = np.array([[1, 2, 3]]).reshape((1, 3)), np.array([0]).reshape((1,1))
    np.testing.assert_equal(x1, expected_x)
    np.testing.assert_equal(y1, expected_y)


def test_compute_metric(exple_X, exple_y):
    #TODO mettre à jour
    hr = Hyperrectangle(bounds=[[0,2], [0,2], [2,3]])

    def user_defined_metric(X, y):
        return y.sum() + 12


    s = hr.compute_metric(exple_X, exple_y, "size")
    c = hr.compute_metric(exple_X, exple_y, "concentration")
    z = hr.compute_metric(exple_X, exple_y, "z_score")
    f = hr.compute_metric(exple_X, exple_y, "f_beta")
    udm = hr.compute_metric(exple_X, exple_y, user_defined_metric)

    assert s == 1
    assert c == 0
    assert z == -0.5
    assert f == 0
    assert udm == 12

    with pytest.raises(Exception):
        assert hr.compute_metric(exple_X, exple_y, "blablabgrrrzzz")
    with pytest.raises(ValueError):
        assert hr.compute_metric(exple_X, exple_y, 2)


def test_change_bounds(exple_X):
    hr = Hyperrectangle(bounds=[[1, 10], [-2, 2], [3, 9], [-7, -1]])
    hr.change_bound(1, "left", 1)
    assert hr.bounds == [[1, 10], [-1, 2], [3, 9], [-7, -1]]

    hr.change_bound(1, "left", -3)
    assert hr.bounds == [[1, 10], [-4, 2], [3, 9], [-7, -1]]

    hr.change_bound(3, "left", 8)
    assert hr.bounds == [[1, 10], [-4, 2], [3, 9], [-1, -1]]

    hr.change_bound(2, "right", 10)
    assert hr.bounds == [[1, 10], [-4, 2], [3, 19], [-1, -1]]

    hr.change_bound(2, "right", -3)
    assert hr.bounds == [[1, 10], [-4, 2], [3, 16], [-1, -1]]

    hr.change_bound(1, "right", -10)
    assert hr.bounds == [[1, 10], [-4, -4], [3, 16], [-1, -1]]

    hr = Hyperrectangle(bounds=[[1, 10], [-2, 2], [3, 9]], X=exple_X, compute_min_max_dataset=True)
    hr.change_bound(0, "left", -3)
    assert hr.bounds == [[1, 10], [-2, 2], [3, 9]]

    hr.change_bound(1, "left", -1)
    assert hr.bounds == [[1, 10], [2, 2], [3, 9]]

    hr.change_bound(1, "right", 3)
    assert hr.bounds == [[1, 10], [2, 5], [3, 9]]

    hr.change_bound(1, "right", 3)
    assert hr.bounds == [[1, 10], [2, 5], [3, 9]]


def test_fit(exple_X, exple_y):
    pass


