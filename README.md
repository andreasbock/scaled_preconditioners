# Preconditioner Design via Bregman Divergences

This package implements the preconditioners in [1]. A simple use case
is demonstrated below. See `examples/example_pcg.py` for a demo of all the 
preconditioners defined in this package.

[1] TODO

## Build Status

[![Tests](https://github.com/andreasbock/scaled_preconditioners/actions/workflows/tests.yml/badge.svg)](https://github.com/andreasbock/scaled_preconditioners/actions/workflows/tests.yml)

## Installation

``pip install scaled_preconditioners``

## A simple example

#### Define some parameters
```
dimension = 100
psd_rank = 50
```

#### Construct S = A + B
```
F = csc_matrix(np.random.rand(dimension, psd_rank))
B = F @ F.T
Q = csc_matrix(np.random.rand(dimension, dimension))
S = Q @ Q.T + B
```

#### Construct the preconditioner
Here we use a randomised SVD, other options include truncated SVD, the
Nyström approximation. There is support for oversampling and power iteration
schemes.
```
rank_approx = 15
pc = compute_preconditioner(
    Q,
    B,
    algorithm="randomized",
    rank_approx=rank_approx,
    n_oversamples=4,
    n_power_iter=0,
)
```

#### Set up a right-hand side

```
rhs = np.random.rand(dimension)
counter = ConjugateGradientCounter()
```

#### Solve `Sx=b` with and without a preconditioner 
```
_, info = linalg.cg(S, rhs, callback=counter)
print("No preconditioner:")
print(f"\t Converged: {info == 0}")
print(f"\t Iterations: {counter.n_iter}\n")

counter.reset()
_, info = linalg.cg(S, rhs, M=rsvd_pc, callback=counter)
print("Randomised SVD preconditioner:")
print(f"\t Converged: {info == 0}")
print(f"\t Iterations: {counter.n_iter}\n")
```