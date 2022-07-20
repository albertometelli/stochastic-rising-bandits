import numpy as np
from IMDB_experiment.strategy import Strategy
from sklearn.naive_bayes import MultinomialNB
from sklearn.exceptions import NotFittedError

class NaiveBayes(Strategy):
    """not used in the final experiments"""

    def __init__(self) -> None:
        self.reset()


    def predict(self, x) -> float:
        try:
            return max( min( self.classifier.predict_proba(x.reshape(1,-1)) [0,1], 1-1e-4), 1e-4)
        except NotFittedError:
            return 0.5


    def prediction_result(self, x, prediction, target):
        self.classifier.partial_fit(x.reshape(1,-1), [target], classes=[0,1])


    def reset(self) -> None:
        self.classifier = MultinomialNB()


    def copy(self):
        return NaiveBayes()


    def __str__(self):
        return f"Naive Bayes"
