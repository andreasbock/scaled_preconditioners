import functools
import operator

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
    n_oversamples: int = 6,
    n_power_iter: int = 2,
    random_state=None,
):
    """
    Computes a low rank approximation of a matrix.

    Args:
        X: Original matrix.
        algorithm: Can be either 'truncated_svd', 'randomized' or 'nystrom'.
        rank_approx: rank of the approximation (must be less than rank(X)).
        n_oversamples: Oversampling parameter.
        n_power_iter: Number of power iterations used in range finding.
        random_state: Seed.

    Returns:
        A low rank approximation of `X`.

    Raises:
        ValueError: If 'truncated_svd' is requested for a sparse matrix.
        NotImplementedError: If `algorithm` is not recognised.
    """
    if algorithm in "truncated_svd":
        if scipy.sparse.issparse(X):
            raise ValueError(
                "Input matrix must be dense if you specify algorithm='truncated_svd'."
            )
        svd = TruncatedSVD(
            n_components=rank_approx,
            algorithm="arpack",
            n_iter=n_power_iter,
            n_oversamples=n_oversamples,
            random_state=random_state,
        )
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
