import functools

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse import identity as sparse_identity
from scipy.sparse.linalg import LinearOperator

from scaled_preconditioners.approximation import approximate

__all__ = ["compute_preconditioner"]


def compute_preconditioner(
    Q,
    B,
    algorithm: str,
    rank_approx: int,
    n_oversamples: int = 1,
    n_power_iter: int = 0,
    random_state: int = 0,
) -> LinearOperator:
    """For

        S = A + B,

    computes the preconditioner:

        P = Q(I + X)Q^*,

    where X is an approximation G = Q^{-1}BQ^{-*}. The preconditioner is provided as a
     `LinearOperator`.

    Parameters
    ----------
    Q : {array-like, sparse matrix} of shape (n, n)
        Factor of A.
    B : {array-like, sparse matrix} of shape (n, n)
        Positive semidefinite matrix.
    algorithm
    rank_approx
    n_oversamples
    n_power_iter
    random_state
    """
    is_sparse = scipy.sparse.issparse(Q)
    if is_sparse ^ scipy.sparse.issparse(B):
        raise ValueError(
            f"Type mismatch between inputs. Factor is {type(Q)}"
            f" but PSD matrix is {type(B)}. Both need to be either"
            f" dense or CSR matrices."
        )
    if is_sparse:
        _solve_fn = scipy.sparse.linalg.spsolve
    else:
        _solve_fn = scipy.linalg.solve

    BQinvT = _solve_fn(Q, B.T).T
    G = _solve_fn(Q, BQinvT)

    factors = approximate(
        G,
        algorithm,
        rank_approx=rank_approx,
        n_oversamples=n_oversamples,
        n_power_iter=n_power_iter,
        random_state=random_state,
    )
    prod = functools.reduce(lambda a, b: a @ b, factors)
    if is_sparse:
        inner = sparse_identity(Q.shape[0], dtype=np.float64) + csr_matrix(prod)
    else:
        inner = np.eye(Q.shape[0], dtype=np.float64) + prod

    def action(vector):
        return _solve_fn(Q @ inner @ Q.T, vector)

    return LinearOperator(Q.shape, matvec=action)
