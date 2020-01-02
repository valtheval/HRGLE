import numpy as np
from copy import deepcopy
from src.utils import transform_df_to_np_array
from src.utils import vprint
import pandas as pd
from sklearn.metrics import recall_score, precision_score, accuracy_score


class Hyperrectangle():


    def __init__(self, bounds=None, init_around_random_point=None, X=None, min_size=10, nb_steps=100, alpha=0.1,
                 metric="concentration", epsilon=0.1, beta=1, margin=0.5, compute_min_max_dataset=False, verbose=False):
        if bounds:
            self.bounds = bounds
        elif init_around_random_point:
            self.bounds = Hyperrectangle.init_around_random_point(X, margin=margin)
        else:
            ValueError("Either bounds of init_around_random_point must be provided")
        self.min_size = min_size
        self.nb_steps = nb_steps
        self.metric = metric
        self.epsilon = epsilon
        self.beta = beta
        self.alpha = alpha
        self.verbose = verbose

        self.bounds_history = [deepcopy(self.bounds)]
        self.metric_history = []
        self.exploration_steps = []

        # For prediction
        self.concentration = None
        self.proba = None

        if compute_min_max_dataset:
            self.info_min_max = [[min(X[:, i]), max(X[:, i])] for i in range(X.shape[1])]
            if isinstance(X, pd.DataFrame):
                self.var_names = X.columns
            else:
                self.var_names = None
        else:
            self.info_min_max = None


    def fit(self, X, y):
        current_size = self.compute_size(X, y)
        nb_steps = 0

        best_metric = self.compute_metric(X, y, self.metric)
        self.metric_history.append(best_metric)

        v = self.verbose
        while (current_size > self.min_size) and (nb_steps < self.nb_steps):
            original_metric = best_metric
            current_bounds = deepcopy(self.bounds)
            var_opt, bound_opt, trend_opt = None, None, None
            # Explore around the hyperrectangle
            vprint(v, "step - current best metric -- bound - trend - metric")
            for var in range(X.shape[1]):
                for bound in ["left", "right"]:
                    for k in ["pos", "neg"]:
                        if k == "pos":
                            trend = self.epsilon
                        else:
                            trend = - self.epsilon
                        self.change_bound(var, bound, trend)
                        new_metric = self.compute_metric(X, y, self.metric)
                        vprint(v, "%d - %.3f -- %s - %.2f - %.3f"%(nb_steps, original_metric, bound, trend, new_metric))
                        # Record history
                        self.bounds_history.append(deepcopy(self.bounds))
                        self.metric_history.append(new_metric)

                        if new_metric > best_metric:
                            var_opt, bound_opt, trend_opt = var, bound, trend
                            best_metric = new_metric
                        self.bounds = deepcopy(current_bounds)
            # Update the bounds if necessary
            if best_metric > original_metric:
                grad = (best_metric - original_metric)/trend_opt
                self.change_bound(var_opt, bound_opt, self.alpha*grad)
                best_metric = self.compute_metric(X, y, self.metric)
                if v:
                    bound_to_print = [list(map(lambda x: round(x, 2), b)) for b in self.bounds]
                    curr_bounds_to_print = [list(map(lambda x: round(x, 2), b)) for b in current_bounds]
                    vprint(v, "---update bounds--- alpha*grad = %.3f (%.3f, %.3f) - %s -> %s"%(self.alpha*grad, self.alpha,
                                                                                              grad, curr_bounds_to_print,
                                                                                              bound_to_print))
                # Record history
                self.bounds_history.append(deepcopy(self.bounds))
                self.metric_history.append(best_metric)
            else:
                vprint(v, "---no bounds update---")
                # Record history
                self.bounds_history.append(deepcopy(self.bounds))
                self.metric_history.append(best_metric)

            # Calcul stop criteria
            current_size = self.compute_size(X, y)
            nb_steps += 1
        # Record concentration for predictions
        self.concentration = self.compute_concentration(X, y)


    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.get_mask(X).astype(int)


    def get_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        mask = self.get_mask(X).astype(int)
        return self.concentration*mask



    def change_bound(self, dim, bound, epsilon):
        """
        Change the boundaries of the dim-th variable by extending the left or right bound by epsilon (might be negative)\\
        For instance :
        >>> hr = Hyperrectangle(bounds=[[1, 10], [-2, 2], [3, 9], [-7, -1]])
        >>> hr.change_bound(1, "left", 1)
        >>> hr.bounds
        [[1, 10], [-1, 2], [3, 9], [-7, -1]]
        The changes ensure that left bound is always lower than or equal to the right bound. If the hyperrectangle\\
         has min/max dataset information, the function ensure that lower bound is always greater or equal to the\\
         minimum observed in the variable the dataset and the right bound lower or equal than the maximum observed in\\
         the variable on the dataset
        :param dim: int, the boundary number to change
        :param bound: string, must be 'left' or 'right'. Indicates which side to change
        :param epsilon: float, the amount of change to apply.
        :return: None
        """
        new_bounds = self.bounds
        if bound == "left":
            if (epsilon > 0):
                new_bounds[dim][0] = min(new_bounds[dim][0] + epsilon, new_bounds[dim][1])
            elif (epsilon < 0) and (self.info_min_max is not None):
                new_bounds[dim][0] = max(new_bounds[dim][0] + epsilon, self.info_min_max[dim][0])
            else:
                new_bounds[dim][0] = new_bounds[dim][0] + epsilon
        elif bound == "right":
            if (epsilon < 0):
                new_bounds[dim][1] = max(new_bounds[dim][1] + epsilon, new_bounds[dim][0])
            elif (epsilon > 0) and (self.info_min_max is not None):
                new_bounds[dim][1] = min(new_bounds[dim][1] + epsilon, self.info_min_max[dim][1])
            else:
                new_bounds[dim][1] = new_bounds[dim][1] + epsilon
        else:
            raise ValueError("bound must take either 'left' or 'right'")


    def compute_metric(self, X, y):
        """
        Compute metric on the dataset X,y provided. Metric available are 'size', 'concentration', 'z_score', 'f_beta' \\
        based on size and concentration, 'lift'. A user defined function can also be provided. It must take a numpy array \\
        representing X and one representing y after they have been filtered by the hyperrectangle.
        :param X: numpy array or datafram pandas
        :param y: numpy array or pandas series
        :param metric: string or function
        :return: float
        """
        if isinstance(X, pd.DataFrame):
            X = transform_df_to_np_array(X)

        if isinstance(y, pd.Series):
            y = y.values

        y_pred = self.predict(X)

        if isinstance(metric, str):
            if metric == "size":
                m = self.compute_size(X, y)
            elif metric == "concentration":
                m = self.compute_concentration(X, y)
            elif metric == "z_score":
                m = self.compute_z_score(X, y)
            elif metric == "f_beta":
                m = self.compute_f_beta_score(X, y)
            elif metric == "lift": #TODO ajouter ce test aux tests unitaires
                m = self.compute_lift(X, y)
            #TODO ajouter recall, precision, accuracy
            else:
                raise NotImplementedError("Please provide a string metric among those available")
        elif callable(metric):
            _, y_filter = self.filter_dataset(X, y)
            m = metric(X, y_filter)
        else:
            raise ValueError("metric takes either string or function")
        return m


    def compute_size(self, X, y):
        _, y_filter = self.filter_dataset(X, y)
        return len(y_filter)


    def compute_concentration(self, X, y):
        _, y_filter = self.filter_dataset(X, y)
        if len(y_filter) > 0:
            return y_filter.sum()/len(y_filter)
        else:
            return 0


    def compute_z_score(self, X, y):
        original_concentration = y.sum()/len(y)
        _, y_filter = self.filter_dataset(X, y)
        if len(y_filter) > 0:
            current_concentration = y_filter.sum()/len(y_filter)
            return (current_concentration - original_concentration)/len(y_filter)
        else:
            return None


    def compute_f_beta_score(self, X, y):
        c = self.compute_concentration(X, y)
        s = self.compute_size(X, y)
        d = (self.beta**2)*s + c
        if d != 0:
            return (1+self.beta**2)*c*s/d
        else:
            return 0


    def compute_lift(self, X, y):
        original_concentration = y.sum()/len(y)
        c = self.compute_concentration(X, y)
        if original_concentration != 0:
            return c/original_concentration
        else:
            raise Exception("There is no 1 in the y vector")


    def compute_recall(self, X, y):#todo tu à faire
        X_filter, y_filter = self.filter_dataset(X, y)
        y_pred = self.predict(X_filter)
        return recall_score(y_filter, y_pred)


    def compute_precision(self, X, y):#todo tu à faire
        X_filter, y_filter = self.filter_dataset(X, y)
        y_pred = self.predict(X_filter)
        return precision_score(y_filter, y_pred)


    def compute_accuracy(self, X, y):#todo tu à faire
        X_filter, y_filter = self.filter_dataset(X, y)
        y_pred = self.predict(X_filter)
        return accuracy_score(y_filter, y_pred)


    def filter_dataset(self, X, y):
        mask = self.get_mask(X).reshape(-1)
        return X[mask, :], y[mask].reshape(mask.sum(), 1)


    def get_mask(self, X):
        mask = True
        for i in range(len(self.bounds)):
            mask = mask * ((X[:,i]>=self.bounds[i][0])*(X[:,i]<=self.bounds[i][1]))
        return mask.reshape(X.shape[0], 1)


    @staticmethod
    def init_around_random_point(X, margin=0.5, seed=None):
        """
        Choose one point (one line) within the dataset and create a hypercube that as this point as center and \
        vertices of length 2*margin
        :param X: array of 1 line per point and variables in columns. Can be a pandas dataframe to
        :param margin: distance between the center and one face of the hypercube
        :param seed: random seed
        :return: list of list describing bounds of the hypercube
        """
        if seed:
            np.random.seed(seed)
        if isinstance(X, pd.DataFrame):
            X = transform_df_to_np_array(X)
        point = X[np.random.randint(0, X.shape[0]),:]
        bounds = []
        for i in range(len(point)):
            low = point[i] - margin
            high = point[i] + margin
            bounds.append([low, high])
        return bounds

