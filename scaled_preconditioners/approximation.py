import functools
import operator
import warnings

import scipy.linalg
from scipy.linalg import cholesky
from scipy.sparse.linalg import aslinearoperator
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_range_finder, randomized_svd

__all__ = ["approximate", "linear_operator_composition"]


def approximate(
    X,
    algorithm: str,
    rank_approx: int,
    n_oversamples: int = 10,
    n_power_iter: int = 5,
    random_state=None,
):
    if algorithm in "truncated_svd":
        svd = TruncatedSVD(
            n_components=rank_approx,
            algorithm="arpack",
            n_iter=n_power_iter,
            n_oversamples=n_oversamples,
            random_state=random_state,
        )
        if scipy.sparse.issparse(X):
            warnings.warn(
                "Converted sparse matrix to dense since a truncated SVD was requested."
            )
            X = X.toarray()
        Us = svd.fit_transform(X)
        return Us, svd.components_
    elif algorithm == "randomized":
        U, s, VT = randomized_svd(
            X,
            n_components=rank_approx,
            n_oversamples=n_oversamples,
            n_iter=n_power_iter,
            random_state=random_state,
            power_iteration_normalizer="QR",
        )
        return U * s, VT
    elif algorithm == "nystrom":
        Q = randomized_range_finder(
            X,
            size=rank_approx + n_oversamples,
            n_iter=n_power_iter,
            random_state=random_state,
        )
        B_1 = X @ Q
        B_2 = Q.T @ B_1
        C = cholesky(B_2, lower=True)
        FT = scipy.linalg.solve(C, B_1.T)
        return FT.T, FT
    else:
        raise NotImplementedError


def linear_operator_composition(args):
    if len(args) == 0:
        raise ValueError("Empty list passed to function.")
    return functools.reduce(operator.mul, map(aslinearoperator, args))
