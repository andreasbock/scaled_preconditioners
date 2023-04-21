import numpy as np
import scipy.sparse.linalg as linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import aslinearoperator

from scaled_preconditioners.preconditioner import Factor, compute_preconditioner
from scaled_preconditioners.utils import ConjugateGradientCounter

# Define some parameters
dimension = 100
spsd_rank = 50
rank_approx = 8
oversampling = 2
power_iters = 1

# Construct S = A + B
m = np.random.rand(dimension, spsd_rank)
psd = aslinearoperator(csr_matrix(m @ m.T))
factor = Factor(csr_matrix(np.random.rand(dimension, dimension)))
S = factor @ factor.T + psd

# Construct S = A + B
tsvd_pc = compute_preconditioner(
    factor, psd, algorithm="truncated_svd", rank_approx=rank_approx
)
rsvd_pc = compute_preconditioner(
    factor,
    psd,
    algorithm="randomized",
    rank_approx=rank_approx,
    n_oversamples=oversampling,
    n_power_iter=0,
)
rsvd_pc_pwr = compute_preconditioner(
    factor,
    psd,
    algorithm="randomized",
    rank_approx=rank_approx,
    n_oversamples=oversampling,
    n_power_iter=power_iters,
)
nys_pc = compute_preconditioner(
    factor,
    psd,
    algorithm="nystrom",
    rank_approx=rank_approx,
    n_oversamples=0,
    n_power_iter=0,
)

rhs = np.random.rand(dimension)

counter = ConjugateGradientCounter()
_, info = linalg.cg(S, rhs, callback=counter)
print("No preconditioner:")
print(f"\t Converged: {info == 0}")
print(f"\t Iterations: {counter.n_iter}\n")

counter.reset()
_, info = linalg.cg(S, rhs, M=tsvd_pc, callback=counter)
print("Truncated SVD preconditioner:")
print(f"\t Converged: {info == 0}")
print(f"\t Iterations: {counter.n_iter}\n")

counter.reset()
_, info = linalg.cg(S, rhs, M=rsvd_pc, callback=counter)
print("Randomised SVD preconditioner:")
print(f"\t Converged: {info == 0}")
print(f"\t Iterations: {counter.n_iter}\n")

counter.reset()
_, info = linalg.cg(S, rhs, M=rsvd_pc_pwr, callback=counter)
print("Randomised SVD preconditioner (power iteration):")
print(f"\t Converged: {info == 0}")
print(f"\t Iterations: {counter.n_iter}\n")

counter.reset()
_, info = linalg.cg(S, rhs, M=nys_pc, callback=counter)
print("Nyström preconditioner (power iteration):")
print(f"\t Converged: {info == 0}")
print(f"\t Iterations: {counter.n_iter}")
