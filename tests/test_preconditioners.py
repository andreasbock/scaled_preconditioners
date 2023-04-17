import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator

from scaled_preconditioners.approximation import linear_operator_composition
from scaled_preconditioners.preconditioner import compute_preconditioner
from tests.conftest import problems


def test_linear_operator_composition_empty():
    with pytest.raises(ValueError):
        linear_operator_composition([])


def test_linear_operator_composition(dimension):
    A1 = np.random.rand(dimension, dimension)
    A2 = np.random.rand(dimension, dimension)

    def mv0(v):
        return 2 * v

    def mv1(v):
        return A1.dot(v)

    def mv2(v):
        return A2.dot(v)

    shape = (dimension, dimension)
    l0 = LinearOperator(shape, matvec=mv0)
    l1 = LinearOperator(shape, matvec=mv1)
    l2 = LinearOperator(shape, matvec=mv2)

    ls = [l2, l1, l0]
    chain = linear_operator_composition(ls)

    for _ in range(10):
        v = np.random.rand(dimension)
        r2 = A2.dot(A1.dot(2 * v))
        rl = chain.matvec(v)
        assert np.allclose(r2, rl)


@pytest.mark.parametrize("problem", problems)
def test_preconditioner_truncated_svd(problem, psd_rank, request):
    Q, B = request.getfixturevalue(problem)
    S = Q @ Q.T + B
    pc = compute_preconditioner(Q, B, algorithm="truncated_svd", rank_approx=psd_rank)
    assert _action_inverts(S, pc)


@pytest.mark.parametrize("problem", problems)
@pytest.mark.parametrize("n_power_iterations", [0, 1, 2])
def test_preconditioner_randomised_svd(
    problem, psd_rank, random_state, oversampling, n_power_iterations, request
):
    Q, B = request.getfixturevalue(problem)
    S = Q @ Q.T + B
    pc = compute_preconditioner(
        Q,
        B,
        algorithm="randomized",
        rank_approx=psd_rank,
        n_oversamples=1,
        n_power_iter=n_power_iterations,
        random_state=random_state,
    )
    assert _action_inverts(S, pc, atol=1e-05)


@pytest.mark.parametrize("problem", problems)
@pytest.mark.parametrize("n_power_iterations", [0, 1, 2])
def test_preconditioner_nystrom(
    problem, psd_rank, random_state, n_power_iterations, request
):
    Q, B = request.getfixturevalue(problem)
    B = B @ B.T  # symmetrise
    S = Q @ Q.T + B
    pc = compute_preconditioner(
        Q,
        B,
        algorithm="nystrom",
        rank_approx=psd_rank,
        n_oversamples=0,
        n_power_iter=n_power_iterations,
        random_state=random_state,
    )
    assert _action_inverts(S, pc)


def test_approximate_mismatch_problem(dimension, psd_rank):
    B = np.random.rand(dimension, psd_rank).astype(np.float64)
    Q = csr_matrix(np.eye(dimension).astype(np.float64), dtype=np.float64)
    with pytest.raises(ValueError):
        compute_preconditioner(Q, B, algorithm="truncated_svd", rank_approx=psd_rank)


def _action_inverts(matrix, preconditioner, atol=1e-5) -> bool:
    v = np.ones(matrix.shape[0])
    pv = matrix.dot(v)
    u = preconditioner.matvec(pv)
    rel = np.linalg.norm(u - v) / np.linalg.norm(v)
    close = np.allclose(rel, 0, atol=atol)
    if not close:
        print(f"|| v - inv(PC)*M*v|| / ||v|| = {rel}")
    return close
