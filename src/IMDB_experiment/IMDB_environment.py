import numpy as np
from numpy.random import default_rng
from scipy.sparse import vstack
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle


class IMDB_Environment():
    """class to handle the IMDB dataset"""

    X = None
    t = None
    initialized = False


    def __init__(self, n_data, n_feat, T):
        self.n_data = n_data
        self.n_feat = n_feat
        self.T = T

        if not IMDB_Environment.initialized:
            self.load_data()
            IMDB_Environment.initialized = True
        
        self.shuffle_indexes()


    def load_data(self) -> None:
        """load IMDB dataset from /data"""
        
        X_train, t_train = load_svmlight_file(f"data/train_labeledBoW.feat")
        X_test, t_test = load_svmlight_file(f"data/test_labeledBoW.feat", n_features=X_train.shape[1])

        # merge train and test set
        IMDB_Environment.X = vstack([X_train, X_test])
        IMDB_Environment.t = np.append(t_train, t_test)

        # shuffle points
        IMDB_Environment.X, IMDB_Environment.t = shuffle(IMDB_Environment.X, IMDB_Environment.t)

        # select only a part of the dataset
        IMDB_Environment.X = IMDB_Environment.X    [:self.n_data, :self.n_feat]
        IMDB_Environment.t = IMDB_Environment.t    [:self.n_data]

        # normalize X, t
        IMDB_Environment.X = IMDB_Environment.X.toarray()
        IMDB_Environment.t = (IMDB_Environment.t > 5).astype(int)


    def shuffle_indexes(self) -> None:
        """shuffle indexes, i.e. the order in which we are popping the data from the dataset"""
        
        self.indexes = np.arange(self.n_data)
        rng = default_rng()
        rng.shuffle(self.indexes)
        self.time = 0


    def get_next_point(self) -> "tuple(np.ndarray, float)":
        """returns a point of the dataset (x_t) and the related target value (t)"""

        if self.time >= self.indexes.size:   # I have visited all the points, revisit them in a different order (notice that this does not happen in our experiments)
            rng = default_rng()
            rng.shuffle(self.indexes)

        idx = self.indexes[self.time % self.indexes.size]
        x = IMDB_Environment.X[idx, :]
        t = IMDB_Environment.t[idx]

        self.time += 1

        assert not np.any(np.isnan(x))

        return x,t


    def copy(self):
        """create & return a copy of the IMDB environment"""
        return IMDB_Environment(self.n_data, self.n_feat, self.T)
