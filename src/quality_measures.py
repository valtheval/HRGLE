import numpy as np

from abc import ABC, abstractmethod

# The metrics are computed based on the definition from the paper "An overview on subgroup discovery: foundations and \\
# applications" Francisco Herrera et. Al.


class Metric(ABC):

    @abstractmethod
    def compute(self, y_true, y_pred):
        pass

class Coverage(Metric):

    def __init__(self):
        pass

    def compute(self, y_true, y_pred):
        return y_pred.sum()/len(y_true)


class Support(Metric):

    def __init__(self):
        pass

    def compute(self, y_true, y_pred):
        return (y_true*y_pred).sum()/len(y_true)

class Confidence(Metric):

    def __init__(self):
        pass

    def compute(self, y_true, y_pred):
        return (y_true*y_pred).sum()/y_pred.sum()


class Accuracy(Metric):

    def __init__(self):
        pass

    def compute(self, y_true, y_pred):
        return (y_true*y_pred).sum()/y_pred.sum()


class PrecisionQc(Metric):

    def __init__(self, c):
        self.c = c

    def compute(self, y_true, y_pred):
        y_true_bar = ~y_true
        return (y_true*y_pred).sum() - self.c*(y_true_bar*y_pred).sum()


class PrecisionQg(Metric):

    def __init__(self, g):
        self.g = g

    def compute(self, y_true, y_pred):
        y_true_bar = ~y_true
        return (y_true*y_pred).sum()/((y_true_bar*y_pred).sum() + self.g)

class Sensitivity(Metric):

    def __init__(self):
        pass

    def compute(self, y_true, y_pred):
        return (y_true*y_pred).sum()/y_true.sum()

class FPr(Metric):

    def __init__(self):
        pass

    def compute(self, y_true, y_pred):
        y_true_bar = ~y_true
        return (y_true_bar*y_pred).sum()/y_true_bar.sum()


class Specificity(Metric):

    def __init__(self):
        pass

    def compute(self, y_true, y_pred):
        y_true_bar = ~y_true
        y_pred_bar = ~y_pred
        return (y_true_bar*y_pred_bar).sum()/y_true_bar.sum()
