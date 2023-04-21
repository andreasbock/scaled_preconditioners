import numpy as np
import scipy.sparse
from scipy.sparse import identity as sparse_identity
from scipy.sparse.linalg import LinearOperator

from scaled_preconditioners.approximation import Factor, approximate

__all__ = ["compute_preconditioner", "Factor"]


def compute_preconditioner(
    factor: Factor,
    psd_term: LinearOperator,
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
    is provided as a `LinearOperator`. The type of approximation is given by the
    `algorithm` parameter.

    Args:
        factor: a Factor object.
        psd_term: a Symmetric positive semidefinite matrix as a LinearOperator.
        algorithm: Can be either 'truncated_svd', 'randomized' or 'nystrom'.
        rank_approx: rank of the approximation (must be less than rank(X)).
        n_oversamples: Oversampling parameter. Not currently supported for
            algorithm="nystrom".
        n_power_iter: Number of power iterations used in range finding.
        random_state: Seed.

    Returns:
        A low rank approximation of `X` as a LinearOperator.
    """
    scaled_psd_term = factor.inv() @  psd_term @ factor.inv().T
    f, ft = approximate(
        scaled_psd_term,
        algorithm,
        rank_approx=rank_approx,
        n_oversamples=n_oversamples,
        n_power_iter=n_power_iter,
        random_state=random_state,
    )
    inner = sparse_identity(rank_approx) + ft @ f

    def apply_inner(vector):
        # v -> f.T v
        w = ft.dot(vector)

        # w -> (I_r + f.T f)^{1} w
        u = scipy.linalg.solve(inner, w)

        # action of  I_r - f(I_r + f.T @ f)^{1} f.T
        return vector - np.dot(f, u)

    def action(vector):
        vector = factor.solve(vector)
        vector = apply_inner(vector)
        vector = factor.rsolve(vector)
        return vector

    return LinearOperator(factor.shape, matvec=action)
