import os
import sys
import numpy as np

# Make src/ importable
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from linalg_interp import gauss_iter_solve  # type: ignore


def load_mauna_loa_txt(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads Midterm2-Data.txt.
    Expected columns: Year, CO2, (maybe uncertainty).
    Handles NOAA-style comment lines starting with '#'.
    """
    data = np.loadtxt(path, comments="#")
    # If there are 3 columns: Year, CO2, Unc
    year = data[:, 0].astype(float)
    co2 = data[:, 1].astype(float)
    return year, co2


def build_not_a_knot_cubic_system(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds A and rhs for cubic spline with not-a-knot boundary conditions.
    Unknowns are c_0..c_N where N = len(x)-1.
    """
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()

    N = len(x) - 1
    h = np.diff(x)  # h_i = x_{i+1}-x_i

    A = np.zeros((N + 1, N + 1), dtype=float)
    rhs = np.zeros(N + 1, dtype=float)

    # --- Not-a-knot boundary conditions ---
    A[0, 0] = h[1]
    A[0, 1] = -(h[0] + h[1])
    A[0, 2] = h[0]
    rhs[0] = 0.0

    A[-1, -3] = h[-1]
    A[-1, -2] = -(h[-2] + h[-1])
    A[-1, -1] = h[-2]
    rhs[-1] = 0.0

    # --- Interior equations ---
    for i in range(1, N):
        A[i, i - 1] = h[i]                 # note: matches your existing spline_function
        A[i, i] = 2 * (h[i] + h[i - 1])
        A[i, i + 1] = h[i - 1]
        rhs[i] = 3.0 * (
            (y[i + 1] - y[i]) / h[i]
            - (y[i] - y[i - 1]) / h[i - 1]
        )

    return A, rhs


def main() -> None:
    data_path = os.path.join(HERE, "Midterm2-Data.txt")
    year, co2 = load_mauna_loa_txt(data_path)

    # Filter 2010–2020 inclusive
    mask = (year >= 2010) & (year <= 2020)
    x = year[mask]
    y = co2[mask]

    if len(x) != 11:
        raise ValueError(f"Expected 11 points (2010–2020), got {len(x)}. Check the file.")

    A, rhs = build_not_a_knot_cubic_system(x, y)

    # Solve using Gauss-Seidel (or set alg="jacobi")
    c = gauss_iter_solve(A, rhs, tol=1e-12, alg="seidel").flatten()

    # Check residual
    resid_vec = A @ c - rhs
    resid_norm = np.linalg.norm(resid_vec)
    resid_inf = np.max(np.abs(resid_vec))

    print("\n=== Q1(c): Not-a-knot cubic spline system ===")
    print(f"Points used: {x[0]:.0f} to {x[-1]:.0f} (n={len(x)})")
    print("\nSolution c (c0..c10):")
    for i, val in enumerate(c):
        print(f"c{i:02d} = {val:.12f}")

    print("\nResidual checks:")
    print(f"||A c - rhs||_2   = {resid_norm:.3e}")
    print(f"||A c - rhs||_inf = {resid_inf:.3e}")


if __name__ == "__main__":
    main()
