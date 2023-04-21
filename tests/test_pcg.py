import numpy as np
import pytest
import scipy.sparse.linalg as linalg

from scaled_preconditioners.preconditioner import compute_preconditioner
from scaled_preconditioners.utils import ConjugateGradientCounter
from tests.conftest import factors, spsd_linops


@pytest.mark.parametrize("factor", factors)
@pytest.mark.parametrize("spsd", spsd_linops)
def test_pcg_exact_pc_truncated_svd(factor, spsd, psd_rank, request):
    factor = request.getfixturevalue(factor)
    spsd = request.getfixturevalue(spsd)

    counter = ConjugateGradientCounter()
    rhs = np.random.rand(factor.shape[0])

    pc = compute_preconditioner(
        factor, spsd, algorithm="truncated_svd", rank_approx=psd_rank,
    )
    _, exit_code = linalg.cg(factor @ factor.T + spsd, rhs, M=pc, callback=counter)
    assert exit_code == 0
    assert counter.n_iter <= 2


@pytest.mark.parametrize("factor", factors)
@pytest.mark.parametrize("spsd", spsd_linops)
@pytest.mark.parametrize("n_power_iterations", [0, 1, 2])
def test_pcg_exact_pc_randomised_svd(
    factor, spsd, random_state, psd_rank, oversampling, n_power_iterations, request
):
    factor = request.getfixturevalue(factor)
    spsd = request.getfixturevalue(spsd)
    counter = ConjugateGradientCounter()
    rhs = np.random.rand(factor.shape[0])
    pc = compute_preconditioner(
        factor,
        spsd,
        algorithm="randomized",
        rank_approx=psd_rank,
        n_oversamples=oversampling,
        n_power_iter=n_power_iterations,
        random_state=random_state,
    )
    _, exit_code = linalg.cg(factor @ factor.T + spsd, rhs, M=pc, callback=counter)

    assert exit_code == 0
    assert counter.n_iter <= 2


@pytest.mark.parametrize("factor", factors)
@pytest.mark.parametrize("spsd", spsd_linops)
@pytest.mark.parametrize("n_power_iterations", [0, 1, 2])
def test_pcg_exact_pc_nystrom(
    factor, spsd, random_state, psd_rank, n_power_iterations, request
):
    factor = request.getfixturevalue(factor)
    spsd = request.getfixturevalue(spsd)
    counter = ConjugateGradientCounter()
    rhs = np.random.rand(factor.shape[0])
    pc = compute_preconditioner(
        factor,
        spsd,
        algorithm="nystrom",
        rank_approx=psd_rank,
        n_oversamples=0,
        n_power_iter=n_power_iterations,
        random_state=random_state,
    )
    _, exit_code = linalg.cg(factor @ factor.T + spsd, rhs, M=pc, callback=counter)

    assert exit_code == 0
    assert counter.n_iter <= 2
