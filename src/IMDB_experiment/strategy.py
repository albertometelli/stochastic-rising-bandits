class Strategy:
    """class which represents a BaseAlgorithm for the IMDB experiment"""

    def __init__(self) -> None:
        raise NotImplementedError


    def predict(self, x) -> int:
        """predict the class of point x (negative or positive)"""
        raise NotImplementedError


    def prediction_result(self, x, prediction, target):
        """update the base algorithm using point x, its prediction and the real classification of x"""
        raise NotImplementedError


    def reset(self):
        raise NotImplementedError