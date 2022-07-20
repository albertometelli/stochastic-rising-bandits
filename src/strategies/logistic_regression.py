import numpy as np
from IMDB_experiment.strategy import Strategy
from scipy.special import expit


class LogisticRegression(Strategy):
    """Online Logistic Regression"""

    def __init__(self, n_features :int, alpha = 1):
        self.n_features = n_features
        self.w_0 = np.ones(self.n_features) / self.n_features
        self.alpha = alpha
        self.w_t = self.w_0
        self.reset()


    def predict(self, x) -> float:
        def sigmoid(x):
            return min( max( expit(x), 1e-4 ), 1-1e-4 )

        y = sigmoid( np.dot(self.w_t, x) )
        return y


    def prediction_result(self, x, prediction, target):
        self.w_t = self.w_t - self.alpha * (prediction - target) * x


    def reset(self) -> None:
        self.w_t = self.w_0


    def copy(self):
        return LogisticRegression(self.n_features, self.alpha)


    def __str__(self):
        return f"Logistic Regression alpha = {self.alpha}"
