import numpy as np
import matplotlib.pyplot as plt

def check_gpe(H_U, H_Y, plot):
    """Check the Generalized Persistence of Excitation (GPE) condition."""
    # Compute the rank of the Hankel matrices
    rank_H_U = np.linalg.matrix_rank(H_U)
    rank_H_Y = np.linalg.matrix_rank(H_Y)

    # Check if the rank is full
    mL = H_U.shape[0]  # number of rows in H_U

    gpe_satisfied = (rank_H_U == mL)

    if(plot):
        plt.figure(figsize=(12, 5))
        plt.title("Singular Values of H_Y")
        plt.loglog(np.linalg.svd(H_Y, compute_uv=False), marker='o')
        plt.tight_layout()
        plt.show()

    
    return gpe_satisfied, rank_H_U, rank_H_Y

def hankel(U, Y, L):
    """Construct Hankel matrices from input and output data."""
    """L is the depth of the Hankel matrix, i.e., the number of block rows."""
    # Number of samples
    N = U.shape[1]
    m = U.shape[0]  # number of inputs
    p = Y.shape[0]  # number of outputs

    # Number of columns in the Hankel matrix
    K = N - L + 1

    # Construct Hankel matrices for inputs and outputs
    H_U = np.zeros((m * L, K))
    H_Y = np.zeros((p * L, K))

    for i in range(K):
        H_U[:, i] = U[:, i:i+L].flatten()
        H_Y[:, i] = Y[:, i:i+L].flatten()

    return H_U, H_Y

