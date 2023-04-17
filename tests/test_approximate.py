import numpy as np
import pytest
from numpy.linalg import svd as np_svd
from sklearn.utils.extmath import randomized_range_finder

from scaled_preconditioners.approximation import approximate
from tests.conftest import matrices


def test_approximate_truncated_svd(dense_matrix):
    X, rank = dense_matrix
    Us, VT = approximate(X, algorithm="truncated_svd", rank_approx=rank)
    X_r = Us @ VT
    _check_error(X - X_r)


@pytest.mark.parametrize("matrix", matrices)
@pytest.mark.parametrize("n_power_iterations", [0, 1, 2])
def test_approximate_randomised_svd(
    matrix, oversampling, random_state, n_power_iterations, request,
):
    X, rank_approx = request.getfixturevalue(matrix)
    Us, VT = approximate(
        X,
        algorithm="randomized",
        rank_approx=rank_approx,
        n_oversamples=oversampling,
        n_power_iter=n_power_iterations,
        random_state=random_state,
    )
    X_rsvd = Us @ VT

    Q = randomized_range_finder(
        X,
        size=rank_approx + oversampling,
        n_iter=n_power_iterations,
        power_iteration_normalizer="QR",
        random_state=random_state,
    )
    hat_U_np, s_np, VT_np = np_svd(Q.T @ X)
    U_np = Q @ hat_U_np
    X_rsvd_naive = (U_np[:, :rank_approx] * s_np[:rank_approx]) @ VT_np[:rank_approx, :]

    _check_error(X - X_rsvd_naive)
    _check_error(X - X_rsvd)


@pytest.mark.parametrize("matrix", matrices)
@pytest.mark.parametrize("n_power_iterations", [0, 1, 2])
def test_approximate_nystrom(
    matrix, oversampling, random_state, n_power_iterations, request
):
    X, rank_approx = request.getfixturevalue(matrix)
    F, FT = approximate(
        X,
        algorithm="nystrom",
        rank_approx=rank_approx,
        n_oversamples=0,
        n_power_iter=n_power_iterations,
        random_state=random_state,
    )
    Q = randomized_range_finder(
        X,
        size=rank_approx,
        n_iter=n_power_iterations,
        power_iteration_normalizer="none",
        random_state=random_state,
    )
    X_nys = F @ FT
    XQ = X @ Q
    X_nys_naive = XQ @ np.linalg.inv(Q.T @ XQ) @ XQ.T
    _check_error(X - X_nys_naive)
    _check_error(X - X_nys)


def _check_error(error):
    error = np.linalg.norm(error)
    success = np.allclose(error, 0)
    if not success:
        print(f"error = {np.linalg.norm(error)}")
    assert success
