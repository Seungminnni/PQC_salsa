import os

import numpy as np
from scipy.linalg import circulant


def sample_uniform_lwe_matrix(rng, num_rows, num_cols, modulus):
    """Sample a standard LWE matrix with iid entries in Z_q."""
    matrix = rng.randint(0, modulus, size=(num_rows, num_cols), dtype=np.int64)
    assert matrix.shape == (num_rows, num_cols)
    assert np.min(matrix) >= 0 and np.max(matrix) < modulus
    return matrix


def sample_negacyclic_rlwe_matrix(rng, dimension, modulus):
    """Sample the structured negacyclic matrix used by the RLWE path."""
    a = rng.randint(0, modulus, size=dimension, dtype=np.int64)
    matrix = circulant(a)
    tri = np.triu_indices(dimension, 1)
    matrix[tri] *= -1
    matrix %= modulus
    assert matrix.shape == (dimension, dimension)
    assert np.min(matrix) >= 0 and np.max(matrix) < modulus
    return matrix


class OriginalLWESamples:
    """Create the orig_A.npy file expected by the tiny-sample preprocessing."""

    def __init__(self, params):
        self.N = params.N
        self.Q = params.Q
        self.lwe = params.lwe
        self.dump_path = params.dump_path
        self.global_rank = getattr(params, "global_rank", 0)
        self.env_base_seed = params.env_base_seed
        self.num_samples = 4 * self.N if params.num_orig_samples == -1 else params.num_orig_samples

        if self.N <= 0:
            raise ValueError("step origA requires --N > 0")
        if self.Q <= 0:
            raise ValueError("step origA requires --Q > 0")
        if self.num_samples <= 0:
            raise ValueError("step origA requires --num_orig_samples > 0")
        if not self.lwe:
            raise ValueError("step origA only supports standard LWE. Pass --lwe true.")

    def generate(self):
        rng = np.random.RandomState(
            [self.global_rank, self.env_base_seed, self.N, self.Q, self.num_samples]
        )
        return sample_uniform_lwe_matrix(rng, self.num_samples, self.N, self.Q)

    def save(self):
        matrix = self.generate()
        output_path = os.path.join(self.dump_path, "orig_A.npy")
        np.save(output_path, matrix)
        return output_path, matrix.shape
