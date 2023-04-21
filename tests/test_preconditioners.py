import numpy as np
import pytest

from scaled_preconditioners.preconditioner import compute_preconditioner
from tests.conftest import factors, spsd_linops


@pytest.mark.parametrize("factor", factors)
@pytest.mark.parametrize("spsd", spsd_linops)
def test_preconditioner_truncated_svd(factor, spsd, psd_rank, request):
    factor = request.getfixturevalue(factor)
    spsd = request.getfixturevalue(spsd)
    S = factor @ factor.T + spsd
    pc = compute_preconditioner(
        factor,
        spsd,
        algorithm="truncated_svd",
        rank_approx=psd_rank,
    )
    assert _action_inverts(S, pc)


@pytest.mark.parametrize("factor", factors)
@pytest.mark.parametrize("spsd", spsd_linops)
@pytest.mark.parametrize("n_power_iterations", [0, 1, 2])
def test_preconditioner_randomised_svd(
    factor, spsd, psd_rank, random_state, n_power_iterations, request
):
    factor = request.getfixturevalue(factor)
    spsd = request.getfixturevalue(spsd)
    S = factor @ factor.T + spsd
    pc = compute_preconditioner(
        factor,
        spsd,
        algorithm="randomized",
        rank_approx=psd_rank,
        n_oversamples=2,  # a bit of slack for randomisation
        n_power_iter=n_power_iterations,
        random_state=random_state,
    )
    assert _action_inverts(S, pc)


@pytest.mark.parametrize("factor", factors)
@pytest.mark.parametrize("spsd", spsd_linops)
@pytest.mark.parametrize("n_power_iterations", [0, 1, 2])
def test_preconditioner_nystrom(
    factor, spsd, psd_rank, random_state, n_power_iterations, request
):
    factor = request.getfixturevalue(factor)
    spsd = request.getfixturevalue(spsd)
    S = factor @ factor.T + spsd
    pc = compute_preconditioner(
        factor,
        spsd,
        algorithm="nystrom",
        rank_approx=psd_rank,
        n_oversamples=0,
        n_power_iter=n_power_iterations,
        random_state=random_state,
    )
    assert _action_inverts(S, pc, atol=1e-07)


def _action_inverts(matrix, preconditioner, atol=1e-5) -> bool:
    v = np.ones(matrix.shape[0])
    pv = matrix.dot(v)
    u = preconditioner.matvec(pv)
    rel = np.linalg.norm(u - v) / np.linalg.norm(v)
    close = np.allclose(rel, 0, atol=atol)
    if not close:
        print(f"|| v - inv(PC)*M*v|| / ||v|| = {rel}")
    return close
