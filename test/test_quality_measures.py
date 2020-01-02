import sys
import os
import pytest

cwd = os.getcwd().split("/")[0:-1]
code_src_path = "/".join(cwd) + "/"
sys.path.append(code_src_path)

from src.hyperrectangle import Hyperrectangle
import numpy as np
import pandas as pd
from src.quality_measures import *


@pytest.fixture
def y_true():
    return np.array([0,1,1,0,1]).reshape((5,1))

@pytest.fixture
def y_pred():
    return np.array([1,1,0,0,1]).reshape((5,1))


def test_metrics(y_true, y_pred):

    size = Size()
    assert size.compute(y_true, y_pred) == 3

    coverage = Coverage()
    assert coverage.compute(y_true, y_pred) == 3/5

    support = Support()
    assert support.compute(y_true, y_pred) == 2/5

    confidence = Confidence()
    assert confidence.compute(y_true, y_pred) == 2/3

    precision = Precision()
    assert precision.compute(y_true, y_pred) == 2/3

    precision = Precision()
    assert precision.compute(y_true, y_pred) == confidence.compute(y_true, y_pred)

    precision_qc2 = PrecisionQc(2)
    assert precision_qc2.compute(y_true, y_pred) == 0

    precision_qc1 = PrecisionQc(1)
    assert precision_qc1.compute(y_true, y_pred) == 1

    precision_qc3 = PrecisionQc(3)
    assert precision_qc3.compute(y_true, y_pred) == -1

    precision_qc0 = PrecisionQc(0)
    assert precision_qc0.compute(y_true, y_pred) == 2

    precision_qg2 = PrecisionQg(2)
    assert precision_qg2.compute(y_true, y_pred) == 2/3

    precision_qg3 = PrecisionQg(3)
    assert precision_qg3.compute(y_true, y_pred) == 0.5

    precision_qg05 = PrecisionQg(0.5)
    assert precision_qg05.compute(y_true, y_pred) == 4/3

    sensitivity = Sensitivity()
    assert sensitivity.compute(y_true, y_pred) == 2/3

    fpr = FPr()
    assert fpr.compute(y_true, y_pred) == 1/2

    specificity = Specificity()
    assert specificity.compute(y_true, y_pred) == 1/2


