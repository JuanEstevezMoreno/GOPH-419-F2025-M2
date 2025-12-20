import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from linalg_interp import gauss_iter_solve  # type: ignore


def load_mauna_loa_txt(path: str):
    data = np.loadtxt(path, comments="#")
    year = data[:, 0].astype(float)
    co2  = data[:, 1].astype(float)
    return year, co2


def build_not_a_knot_cubic_system(x, y):
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    N = len(x) - 1
    h = np.diff(x)

    A = np.zeros((N+1, N+1), dtype=float)
    rhs = np.zeros(N+1, dtype=float)

    # not-a-knot
    A[0, 0] = h[1]
    A[0, 1] = -(h[0] + h[1])
    A[0, 2] = h[0]
    rhs[0] = 0.0

    A[-1, -3] = h[-1]
    A[-1, -2] = -(h[-2] + h[-1])
    A[-1, -1] = h[-2]
    rhs[-1] = 0.0

    # interior
    for i in range(1, N):
        A[i, i-1] = h[i]
        A[i, i]   = 2*(h[i] + h[i-1])
        A[i, i+1] = h[i-1]
        rhs[i] = 3.0 * ((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])

    return A, rhs


def main():
    data_path = os.path.join(HERE, "Midterm2-Data.txt")
    year, co2 = load_mauna_loa_txt(data_path)

    # 2010–2020
    mask = (year >= 2010) & (year <= 2020)
    x = year[mask]
    y = co2[mask]
    N = len(x) - 1
    h = np.diff(x)

    # Solve for c (same as part c)
    A, rhs = build_not_a_knot_cubic_system(x, y)
    c = gauss_iter_solve(A, rhs, tol=1e-12, alg="seidel").flatten()

    # ---- Part (d): compute a_i, b_i, d_i ----
    a = y[:-1]
    b = (y[1:] - y[:-1]) / h - (h/3.0) * (2*c[:-1] + c[1:])
    d = (c[1:] - c[:-1]) / (3.0*h)

    # Interpolate at xq = 2015.25
    xq = 2015.25
    i = np.searchsorted(x, xq) - 1   # interval index
    t = xq - x[i]

    yq = a[i] + b[i]*t + c[i]*t**2 + d[i]*t**3

    print("\n=== Q1(d): spline coefficients and interpolation ===")
    print(f"Interval used: [{x[i]:.0f}, {x[i+1]:.0f}] (i={i})")
    print(f"t = x - x_i = {t:.2f}")
    print(f"a_i = {a[i]:.6f}")
    print(f"b_i = {b[i]:.12f}")
    print(f"c_i = {c[i]:.12f}")
    print(f"d_i = {d[i]:.12f}")
    print(f"\nCO2({xq}) = {yq:.12f} ppm")


if __name__ == "__main__":
    main()
