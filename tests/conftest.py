import numpy as np
import pytest
from scipy.sparse import csr_matrix


@pytest.fixture(scope="module")
def dimension() -> int:
    return 50


@pytest.fixture(scope="module")
def psd_rank(dimension):
    return int(dimension / 2)


@pytest.fixture(scope="module")
def rank_approx(psd_rank) -> int:
    return int(psd_rank / 2)


@pytest.fixture(scope="module")
def oversampling() -> int:
    return 5


@pytest.fixture(scope="module")
def random_state() -> int:
    return 1


@pytest.fixture(scope="module")
def dense_matrix(dimension, psd_rank):
    m = np.random.rand(dimension, psd_rank).astype(np.float64)
    return m @ m.T, psd_rank


@pytest.fixture(scope="module")
def sparse_matrix(dense_matrix):
    matrix, rank = dense_matrix
    return csr_matrix(matrix, dtype=np.float64), rank


@pytest.fixture(scope="module")
def dense_problem(dimension, psd_rank, dense_matrix):
    Q = np.random.rand(dimension, dimension).astype(np.float64)
    B, _ = dense_matrix
    return Q, B


@pytest.fixture(scope="module")
def sparse_problem(dimension, psd_rank):
    Q = np.random.rand(dimension, dimension)
    F = np.random.rand(dimension, psd_rank)
    B = F @ F.T
    Q = csr_matrix(Q, dtype=np.float64)
    B = csr_matrix(B, dtype=np.float64)
    return Q, B


matrices = ["dense_matrix", "sparse_matrix"]
problems = ["dense_problem", "sparse_problem"]
