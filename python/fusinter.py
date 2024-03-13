from functools import partial
from multiprocessing import Pool
from typing import List

import numpy as np
from FUSINTER_cpp_pybind import FUSINTERDiscretizer as FUSINTERSingleFeatureDiscretizer


def get_splits(x, y, alpha, lam):
    fsfd = FUSINTERSingleFeatureDiscretizer(alpha, lam)
    return fsfd.fit(x, y)


class FUSINTERDiscretizer:
    def __init__(self, alpha, lam, **kwargs):

        self.not_concurrent = True if kwargs.get("not_concurrent") else False

        self.splits: List[np.ndarray] = []
        self.alpha = alpha
        self.lam = lam

    def fit(self, X, y):

        if self.not_concurrent:
            for x in X.T:
                fsfd = FUSINTERSingleFeatureDiscretizer(self.alpha, self.lam)
                splits = fsfd.fit(x, y)
                self.splits.append(splits)
        else:
            with Pool(5) as p:
                self.splits = p.map(partial(get_splits, alpha=self.alpha, lam=self.lam, y=y), X.T[:])

        return self.splits

    def transform(self, X):
        results = np.zeros_like(X)
        for i, x in enumerate(X.T):
            results[:, i] = np.digitize(x, self.splits[i])

        return results
