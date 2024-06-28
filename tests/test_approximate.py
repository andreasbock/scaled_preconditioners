import numpy as np
import pytest
from numpy.linalg import svd as np_svd
from scipy.sparse.linalg import LinearOperator
from sklearn.utils.extmath import randomized_range_finder

from scaled_preconditioners.approximation import Factor, approximate
from tests.conftest import matrices, spsd_linops


def test_factor(dimension):
    X = np.random.rand(dimension, dimension).astype(np.float64)

    f = Factor(m=X)
    vec = np.random.rand(dimension)

    # test left actions
    assert np.allclose(f.matvec(vec), np.dot(X, vec))

    # test right actions
    assert np.allclose(f.rmatvec(vec), np.dot(X.T, vec))

    # same for inverses
    finv = f.inv()

    # test inverse left action
    result = scipy.linalg.solve(X, vec)
    assert np.allclose(finv.matvec(vec), result)
    assert np.allclose(f.solve(vec), result)

    # test right actions
    tresult = scipy.linalg.solve(X, vec, transposed=True)
    assert np.allclose(f.rmatvec(vec), np.dot(X.T, vec))
    assert np.allclose(f.rsolve(vec), tresult)


@pytest.mark.parametrize("matrix", matrices)
def test_compose_factors(matrix, request):
    matrix = request.getfixturevalue(matrix)
    f = Factor(matrix)
    f = f @ f
    vec = np.random.rand(matrix.shape[0])

    # test left actions
    assert np.allclose(f.matvec(vec), matrix.dot(matrix.dot(vec)))

    # test right actions
    assert np.allclose(f.rmatvec(vec), matrix.T.dot(matrix.T.dot(vec)))


@pytest.mark.parametrize("matrix", spsd_linops)
def test_approximate_truncated_svd(matrix, psd_rank, request):
    matrix = request.getfixturevalue(matrix)
    Us, VT = approximate(matrix, algorithm="truncated_svd", rank_approx=psd_rank)
    matrix_rsvd = Us @ VT
    _check_error(matrix, matrix_rsvd)


@pytest.mark.parametrize("matrix", spsd_linops)
@pytest.mark.parametrize("n_power_iterations", [0, 1, 2])
def test_approximate_randomised_svd(
    matrix, oversampling, psd_rank, random_state, n_power_iterations, request,
):
    matrix = request.getfixturevalue(matrix)
    Us, VT = approximate(
        matrix,
        algorithm="randomized",
        rank_approx=psd_rank,
        n_oversamples=oversampling,
        n_power_iter=n_power_iterations,
        random_state=random_state,
    )
    matrix_rsvd = Us @ VT
    Q = randomized_range_finder(
        matrix,
        size=psd_rank + oversampling,
        n_iter=n_power_iterations,
        power_iteration_normalizer="QR",
        random_state=random_state,
    )
    hat_U_np, s_np, _ = np_svd((matrix.A.T @ Q).T)
    U_np = Q @ hat_U_np
    matrix_rsvd_naive = (U_np[:, :psd_rank] * s_np[:psd_rank]) @ U_np[:, :psd_rank].T
    _check_error(matrix, matrix_rsvd_naive)
    _check_error(matrix, matrix_rsvd)


@pytest.mark.parametrize("matrix", spsd_linops)
@pytest.mark.parametrize("n_power_iterations", [0, 1, 2])
def test_approximate_nystrom(
    matrix, oversampling, psd_rank, random_state, n_power_iterations, request
):
    matrix = request.getfixturevalue(matrix)
    F, FT = approximate(
        matrix,
        algorithm="nystrom",
        rank_approx=psd_rank,
        n_oversamples=0,
        n_power_iter=n_power_iterations,
        random_state=random_state,
    )
    Q = randomized_range_finder(
        matrix,
        size=psd_rank,
        n_iter=n_power_iterations,
        power_iteration_normalizer="none",
        random_state=random_state,
    )
    matrix_nys = F @ FT
    mq = matrix @ Q
    matrix_nys_naive = mq @ np.linalg.inv(Q.T @ mq) @ mq.T
    _check_error(matrix, matrix_nys_naive)
    _check_error(matrix, matrix_nys)


def _check_error(m1, m2, atol=1e-08):
    m1 = m1.A if isinstance(m1, LinearOperator) else m1
    m2 = m2.A if isinstance(m2, LinearOperator) else m2
    error = m1 - m2
    success = np.allclose(error, 0, atol=atol)
    if not success:
        print(f"error = {np.linalg.norm(error)}")
    assert success
