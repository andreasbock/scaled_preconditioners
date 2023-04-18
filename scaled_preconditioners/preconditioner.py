import functools

import numpy as np
import scipy.sparse
from scipy.sparse import identity as sparse_identity
from scipy.sparse.linalg import LinearOperator

from scaled_preconditioners.approximation import approximate

__all__ = ["compute_preconditioner"]


def compute_preconditioner(
    factor,
    B,
    algorithm: str,
    rank_approx: int,
    n_oversamples: int = 1,
    n_power_iter: int = 0,
    random_state: int = 0,
) -> LinearOperator:
    """
    For a Hermitian matrix S = A + B, this method computes the preconditioner:

        P = Q(I + X)Q^*,

    where X is a low rank approximation G = Q^{-1} B Q^{-*}. The preconditioner
    is provided as a `LinearOperator`.

    Args:
        factor: {array-like, sparse matrix} of shape (n, n)
            Factor of A.
        B: {array-like, sparse matrix} of shape (n, n)
            Positive semidefinite matrix.
        algorithm: Can be either 'truncated_svd', 'randomized' or 'nystrom'.
        rank_approx: rank of the approximation (must be less than rank(X)).
        n_oversamples: Oversampling parameter.
        n_power_iter: Number of power iterations used in range finding.
        random_state: Seed.

    Returns:
        A low rank approximation of `X`.

    Raises:
        ValueError: If `factor` is sparse and `B` is not, or vice versa.
    """
    is_sparse = scipy.sparse.issparse(factor)
    if is_sparse ^ scipy.sparse.issparse(B):
        raise ValueError(
            f"Type mismatch between inputs. Factor is {type(factor)}"
            f" but PSD matrix is {type(B)}. Both need to be either"
            f" dense or CSR matrices."
        )
    if is_sparse:
        _solve_fn = scipy.sparse.linalg.spsolve
    else:
        _solve_fn = scipy.linalg.solve

    right_inv = _solve_fn(factor, B.T).T
    G = _solve_fn(factor, right_inv)

    factors_approx = approximate(
        G,
        algorithm,
        rank_approx=rank_approx,
        n_oversamples=n_oversamples,
        n_power_iter=n_power_iter,
        random_state=random_state,
    )
    prod = functools.reduce(lambda a, b: a @ b, factors_approx)
    if is_sparse:
        inner = sparse_identity(factor.shape[0]) + prod
    else:
        inner = np.eye(factor.shape[0]) + prod

    m = scipy.linalg.inv(factor @ inner @ factor.T)

    def action(vector):
        return m.dot(vector)

    return LinearOperator(factor.shape, matvec=action)
