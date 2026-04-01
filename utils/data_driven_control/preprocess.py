import numpy as np
import matplotlib.pyplot as plt

def subspace_ID(H_U, H_Y, k):
    """Obtain k-dim subspace basis matrix """
    # Perform SVD on the output Hankel matrix
    H_W = np.stack((H_U, H_Y), axis=0)  # Stack H_U and H_Y vertically
    U, _, _ = np.linalg.svd(H_W, full_matrices=False)

    # Take the first k columns of U as the subspace basis
    subspace_basis = U[:, :k]

    return subspace_basis

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


def describe_hankel_gpe(U, Y, L, H_U=None, H_Y=None, rank_H_U=None, rank_H_Y=None, gpe_satisfied=None):
    """Summarize Hankel dimensions and explain why the GPE check can or cannot pass."""
    N = int(U.shape[1])
    m = int(U.shape[0])
    p = int(Y.shape[0])
    L = int(L)
    K = int(N - L + 1)

    h_u_rows = int(m * L)
    h_y_rows = int(p * L)
    h_u_cols = max(K, 0)
    h_y_cols = max(K, 0)
    required_rank = h_u_rows
    max_possible_rank = int(min(h_u_rows, h_u_cols))
    dimensionally_possible = (L <= N) and (h_u_cols >= h_u_rows)

    lines = [
        "Hankel diagnostics:",
        f"  Inputs m = {m}, outputs p = {p}, samples N = {N}, depth L = {L}",
        f"  Hankel columns K = N - L + 1 = {K}",
        f"  Expected H_U shape = ({h_u_rows}, {h_u_cols}) = (mL, K)",
        f"  Expected H_Y shape = ({h_y_rows}, {h_y_cols}) = (pL, K)",
        f"  GPE target: rank(H_U) must equal mL = {required_rank}",
        f"  Best possible rank from dimensions: min(mL, K) = {max_possible_rank}",
    ]

    if L > N:
        summary = (
            f"Failure before construction: L={L} is larger than N={N}, "
            f"so K=N-L+1={K} and the Hankel matrices cannot be formed."
        )
    elif not dimensionally_possible:
        summary = (
            f"Failure from dimensions: H_U needs full row rank mL={required_rank}, "
            f"but it only has K={h_u_cols} columns, so rank(H_U) <= {max_possible_rank} < {required_rank}."
        )
    elif rank_H_U is None:
        summary = (
            f"Dimensions are acceptable: K={h_u_cols} >= mL={required_rank}, "
            "so full row rank is possible if the input data is sufficiently exciting."
        )
    elif rank_H_U == required_rank or bool(gpe_satisfied):
        summary = (
            f"Success: rank(H_U)={int(rank_H_U)} equals mL={required_rank}, "
            "so the input Hankel matrix has full row rank and the GPE test passes."
        )
    else:
        deficit = int(required_rank - rank_H_U)
        summary = (
            f"Failure after construction: rank(H_U)={int(rank_H_U)} is below mL={required_rank} by {deficit}. "
            "The Hankel matrix was built, but the input data does not span enough independent directions for this depth."
        )

    lines.append(f"  {summary}")

    if H_U is not None:
        lines.append(f"  Actual H_U shape = {tuple(int(v) for v in H_U.shape)}")
    if H_Y is not None:
        lines.append(f"  Actual H_Y shape = {tuple(int(v) for v in H_Y.shape)}")
    if rank_H_U is not None:
        lines.append(f"  Observed rank(H_U) = {int(rank_H_U)}")
    if rank_H_Y is not None:
        lines.append(f"  Observed rank(H_Y) = {int(rank_H_Y)}")

    return {
        "num_inputs": m,
        "num_outputs": p,
        "num_samples": N,
        "hankel_depth": L,
        "num_hankel_columns": K,
        "expected_H_U_shape": [h_u_rows, h_u_cols],
        "expected_H_Y_shape": [h_y_rows, h_y_cols],
        "required_rank_H_U": required_rank,
        "max_possible_rank_H_U": max_possible_rank,
        "dimensionally_possible": bool(dimensionally_possible),
        "summary": summary,
        "lines": lines,
    }
