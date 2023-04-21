from typing import Union

import numpy as np
import scipy.linalg
from scipy.linalg import cholesky
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_range_finder

__all__ = ["approximate", "Factor"]

SparseMatrix = Union[csc_matrix, csr_matrix]


class Factor(LinearOperator):
    """ A LinearOperator object including how to apply its inverse action.

    This subclasses LinearOperator to provide an interface which along with the
    usual actions:

        x -> L x,
        x-> L.T x,

    allows provides:

        x -> L^{-1} x,
        x -> L^{-T} x.

    The inverse action is computed using the `solve` argument (or the default).

    Attributes:
        matrix: A boolean indicating if we like SPAM or not.
        solve: A method for solving Lx = b.
    """
    def __init__(
        self,
        matrix: Union[np.ndarray, SparseMatrix, LinearOperator],
        solve=None,
    ):
        super().__init__(matrix.dtype, matrix.shape)
        self.linear_operator = aslinearoperator(matrix)
        self.matrix = matrix
        self.shape = self.matrix.shape
        self.is_sparse = scipy.sparse.issparse(matrix)

        if solve is not None:
            self._solve = solve
        else:
            self._solve = scipy.sparse.linalg.spsolve

    def inv(self) -> LinearOperator:
        """
        Returns a LinearOperator object given by the inverse action of `self`.
        Allows composition of factors, e.g., the following is valid:

            l = LinearOperator(...)
            f = Factor(...)
            a = f @ l @ f.inv()
        """
        return LinearOperator(
            shape=self.shape,
            dtype=self.dtype,
            matvec=self.solve,
            rmatvec=self.rsolve,
        )

    def solve(self, x):
        return self._solve(self.matrix, x)

    def rsolve(self, x):
        return self._solve(self.matrix.T, x)

    def _matvec(self, x):
        return self.linear_operator.matvec(x)

    def _rmatvec(self, x):
        return self.linear_operator.rmatvec(x)


def approximate(
        linear_operator: LinearOperator,
        algorithm: str,
        rank_approx: int,
        n_oversamples: int = 1,
        n_power_iter: int = 0,
        random_state=None,
):
    """
    Computes a low rank approximation of a LinearOperator.

    For a LinearOperator L, The result is a tuple, (F, F.T), such that

        L(I) ~ F @ F.T, where I is the identity matrix of appropriate dimension.

    Args:
        linear_operator: LinearOperator.
        algorithm: Can be either 'truncated_svd', 'randomized' or 'nystrom'.
        rank_approx: rank of the approximation (must be less than rank(X)).
        n_oversamples: Oversampling parameter.
        n_power_iter: Number of power iterations used in range finding.
        random_state: Seed.

    Returns:
        The factors F, F.T of a low rank approximation of `X` as np.arrays of
        size n x r and r x n, respectively.

    Raises:
        NotImplementedError: If `algorithm` is not recognised.
    """
    if algorithm == "truncated_svd":
        d, _ = linear_operator.shape
        _matrix = linear_operator.dot(np.identity(d))
        svd = TruncatedSVD(
            n_components=rank_approx,
            algorithm="arpack",
            n_iter=n_power_iter,
            n_oversamples=n_oversamples,
            random_state=random_state,
        )
        Us = svd.fit_transform(_matrix)
        return Us, svd.components_
    elif algorithm == "randomized":
        Q = randomized_range_finder(
            linear_operator,
            size=rank_approx + n_oversamples,
            n_iter=n_power_iter,
            random_state=random_state,
            power_iteration_normalizer="QR",
        )
        qmq = Q.T @ (linear_operator @ Q)
        s_rev, Uhat_rev = scipy.linalg.eigh(qmq)
        Uhat = np.flip(Uhat_rev, axis=1)
        s = np.flip(s_rev)
        U = Q @ Uhat
        return U[:, :rank_approx] * s[:rank_approx], U[:, :rank_approx].T
    elif algorithm == "nystrom":
        Q = randomized_range_finder(
            linear_operator,
            size=rank_approx + n_oversamples,
            n_iter=n_power_iter,
            random_state=random_state,
            power_iteration_normalizer="QR",
        )
        B_1 = linear_operator @ Q
        B_2 = Q.T @ B_1
        C = cholesky(B_2, lower=True)
        FT = scipy.linalg.solve(C, B_1.T)
        return FT.T, FT
    else:
        raise NotImplementedError
