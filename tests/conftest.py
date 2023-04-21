import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import aslinearoperator

from scaled_preconditioners.approximation import Factor


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


# ------- Positive definite matrices/factors

@pytest.fixture(scope="module")
def dense_pd_matrix(dimension):
    return np.random.rand(dimension, dimension).astype(np.float64)


@pytest.fixture(scope="module")
def sparse_pd_matrix(dense_pd_matrix):
    return csr_matrix(dense_pd_matrix)


@pytest.fixture(scope="module")
def dense_factor(dense_pd_matrix):
    return Factor(m=dense_pd_matrix)


@pytest.fixture(scope="module")
def sparse_factor(sparse_pd_matrix):
    return Factor(m=sparse_pd_matrix)


# ------- Symmetric positive semi-definite matrices/factors

@pytest.fixture(scope="module")
def dense_spsd_matrix(dimension, psd_rank):
    m = np.random.rand(dimension, psd_rank).astype(np.float64)
    return m @ m.T


@pytest.fixture(scope="module")
def sparse_spsd_matrix(dense_spsd_matrix):
    return csr_matrix(dense_spsd_matrix)


@pytest.fixture(scope="module")
def dense_spsd_linop(dense_spsd_matrix):
    return aslinearoperator(dense_spsd_matrix)


@pytest.fixture(scope="module")
def sparse_spsd_linop(sparse_spsd_matrix):
    return aslinearoperator(sparse_spsd_matrix)


pd_mats = ["dense_pd_matrix", "sparse_pd_matrix"]
spsd_mats = ["dense_spsd_matrix", "sparse_spsd_matrix"]
matrices = pd_mats + spsd_mats
spsd_linops = ["dense_spsd_linop", "sparse_spsd_linop"]
factors = ["dense_factor", "sparse_factor"]
