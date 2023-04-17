import numpy as np
import pytest
import scipy.sparse.linalg as linalg

from scaled_preconditioners.preconditioner import compute_preconditioner
from scaled_preconditioners.utils import ConjugateGradientCounter
from tests.conftest import problems


@pytest.mark.parametrize("problem", problems)
def test_pcg_exact_pc_truncated_svd(problem, psd_rank, random_state, request):
    Q, B = request.getfixturevalue(problem)
    counter = ConjugateGradientCounter()
    rhs = np.random.rand(Q.shape[0])

    tsvd_pc = compute_preconditioner(
        Q, B, algorithm="truncated_svd", rank_approx=psd_rank
    )
    _, exit_code = linalg.cg(Q @ Q.T + B, rhs, M=tsvd_pc, callback=counter)
    assert exit_code == 0
    assert counter.n_iter == 1


@pytest.mark.parametrize("problem", problems)
@pytest.mark.parametrize("n_power_iterations", [0, 1, 2])
def test_pcg_exact_pc_randomised_svd(
    problem, psd_rank, random_state, oversampling, n_power_iterations, request
):
    Q, B = request.getfixturevalue(problem)
    counter = ConjugateGradientCounter()
    rhs = np.random.rand(Q.shape[0])
    rsvd_pc = compute_preconditioner(
        Q,
        B,
        algorithm="randomized",
        rank_approx=psd_rank,
        n_oversamples=oversampling,
        n_power_iter=n_power_iterations,
        random_state=random_state,
    )
    _, exit_code = linalg.cg(Q @ Q.T + B, rhs, M=rsvd_pc, callback=counter)
    assert exit_code == 0
    assert counter.n_iter == 1


@pytest.mark.parametrize("problem", problems)
@pytest.mark.parametrize("n_power_iterations", [0, 1, 2])
def test_pcg_exact_pc_nystrom(
    problem, psd_rank, random_state, oversampling, n_power_iterations, request
) -> None:
    Q, B = request.getfixturevalue(problem)
    B = B @ B.T  # symmetrise
    counter = ConjugateGradientCounter()
    rsvd_pc = compute_preconditioner(
        Q,
        B,
        algorithm="nystrom",
        rank_approx=psd_rank,
        n_oversamples=0,
        n_power_iter=n_power_iterations,
        random_state=random_state,
    )
    rhs = np.random.rand(Q.shape[0])
    _, exit_code = linalg.cg(Q @ Q.T + B, rhs, M=rsvd_pc, callback=counter)
    assert exit_code == 0
    assert counter.n_iter == 1
