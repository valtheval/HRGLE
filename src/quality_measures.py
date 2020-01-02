import numpy as np
from abc import ABC, abstractmethod

# The metrics are computed based on the definition from the paper "An overview on subgroup discovery: foundations and \\
# applications" Francisco Herrera et. Al.


class Metric(ABC):

    @abstractmethod
    def compute(self, y_true, y_pred):
        pass

    #abstractmethod
    def __str__(self):
        pass

class Size(Metric):

    def __init__(self):
        pass

    def compute(self, y_true, y_pred):
        return y_pred.sum()

    def __str__(self):
        return "size"


class Coverage(Metric):

    def __init__(self):
        pass

    def compute(self, y_true, y_pred):
        return y_pred.sum()/len(y_true)

    def __str__(self):
        return "coverage"


class Support(Metric):

    def __init__(self):
        pass

    def compute(self, y_true, y_pred):
        return (y_true*y_pred).sum()/len(y_true)

    def __str__(self):
        return "support"


class Confidence(Metric): #Also precision

    def __init__(self):
        pass

    def compute(self, y_true, y_pred):
        return (y_true*y_pred).sum()/y_pred.sum()

    def __str__(self):
        return "confidence"


class Precision(Metric):#Also confidence

    def __init__(self):
        pass

    def compute(self, y_true, y_pred):
        return (y_true*y_pred).sum()/y_pred.sum()

    def __str__(self):
        return "precision"


class PrecisionQc(Metric):

    def __init__(self, c):
        self.c = c

    def compute(self, y_true, y_pred):
        y_true_bar = np.logical_not(y_true).astype(int)
        return (y_true*y_pred).sum() - self.c*(y_true_bar*y_pred).sum()

    def __str__(self):
        return "precision_qc"


class PrecisionQg(Metric):

    def __init__(self, g):
        self.g = g

    def compute(self, y_true, y_pred):
        y_true_bar = np.logical_not(y_true).astype(int)
        return (y_true*y_pred).sum()/((y_true_bar*y_pred).sum() + self.g)

    def __str__(self):
        return "precision_qg"


class Sensitivity(Metric):

    def __init__(self):
        pass

    def compute(self, y_true, y_pred):
        return (y_true*y_pred).sum()/y_true.sum()

    def __str__(self):
        return "sensitivity"


class FPr(Metric):

    def __init__(self):
        pass

    def compute(self, y_true, y_pred):
        y_true_bar = np.logical_not(y_true).astype(int)
        return (y_true_bar*y_pred).sum()/y_true_bar.sum()

    def __str__(self):
        return "false_positive_rate"


class Specificity(Metric):

    def __init__(self):
        pass

    def compute(self, y_true, y_pred):
        y_true_bar = np.logical_not(y_true).astype(int)
        y_pred_bar = np.logical_not(y_pred).astype(int)
        return (y_true_bar*y_pred_bar).sum()/y_true_bar.sum()

    def __str__(self):
        return "specificity"
