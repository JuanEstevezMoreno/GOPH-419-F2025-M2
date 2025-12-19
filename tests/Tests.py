import os
import sys
import numpy as np
from scipy.interpolate import UnivariateSpline
from typing import Callable, Any, cast

# Make src/ importable so we can get linalg_interp
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
from linalg_interp import gauss_iter_solve, spline_function #type: ignore[import]


def _print_result(name: str, passed: bool) -> None:
    status = "PASSED" if passed else "FAILED"
    print(f"[{status}] {name}")

try:
    from scipy.interpolate import UnivariateSpline
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


SplineFunc = Callable[[Any], Any]

def test_gauss_single_rhs_seidel() -> None:
    """
    Solve a small diagonally dominant system with a single RHS and
    compare against numpy.linalg.solve using Gauss–Seidel.
    """
    A = np.array([[4.0, 1.0, 0.0],
                  [1.0, 3.0, 1.0],
                  [0.0, 1.0, 2.0]])
    b = np.array([1.0, 2.0, 3.0])

    x_gs = gauss_iter_solve(A, b, alg="seidel").flatten()
    x_np = np.linalg.solve(A, b)

    passed = np.allclose(x_gs, x_np, rtol=1e-8, atol=1e-10)
    _print_result("gauss_single_rhs_seidel", passed)


def test_gauss_single_rhs_jacobi() -> None:
    """
    Same as above but using Jacobi iteration 
    """
    A = np.array([[4.0, 1.0, 0.0],
                  [1.0, 3.0, 1.0],
                  [0.0, 1.0, 2.0]])
    b = np.array([1.0, 2.0, 3.0])

    x_jac = gauss_iter_solve (A, b, alg="jacobi").flatten()
    x_np = np.linalg.solve(A,b)

    passed = np.allclose(x_jac, x_np, rtol=1e-8, atol=1e-10)
    _print_result("gauss_single_rhs_jacobi", passed)


def test_gauss_inverse_seidel() -> None:
    """
    Use the solver with the identity matrix on the RHS so that result it A^(-1)
    Check that A A^(-1) = I using Gauss-Seidel.
    """
    A = np.array([[4.0, 1.0, 0.0],
                  [1.0, 3.0, 1.0],
                  [0.0, 1.0, 2.0]])
    
    I = np.eye(A.shape[0])
    A_inv_approx = gauss_iter_solve(A, I, alg="seidel")

    prod = A @ A_inv_approx
    passed = np.allclose(prod, I,rtol=1e-8, atol=1e-10)
    _print_result ("gauss_inverse_seidel", passed)


def test_gauss_inverse_jacobi() -> None:
    """
    Same inverse test but using Jacobi iteration.
    """
    A = np.array([[4.0, 1.0, 0.0],
                  [1.0, 3.0, 1.0],
                  [0.0, 1.0, 2.0]])
    
    I = np.eye(A.shape[0])
    A_inv_approx = gauss_iter_solve(A, I, alg="jacobi")

    prod = A @ A_inv_approx
    passed = np.allclose(prod, I,rtol=1e-8, atol=1e-10)
    _print_result ("gauss_inverse_jacobi", passed)


def test_spline_polynomial_recovery() -> None:
    """
    Check that spline_function recovers polynomials where expected.

    -Linear data: orders 1, 2, 3 should all match exactly.
    -Quadratic data: orders 2 and 3 should match exactly, order 1 not expected to.
    -Cubic data: order 3 should match exactly, order 1 and 2 not expected. 
    """
    xd = np.linspace(0.0, 5.0, 6)  # 6 points
    xs = np.linspace(xd[0], xd[-1], 50)

    # ----- Linear: y = 2x + 1 -----
    yd_lin = 2.0 * xd + 1.0
    true_lin = 2.0 * xs + 1.0

    for order in (1, 2, 3):
        f_any = spline_function(xd, yd_lin, order=order)
        f = cast(SplineFunc, f_any)  # tells Pylance: this is callable
        ys = f(xs)
        passed = np.allclose(ys, true_lin, atol=1e-10, rtol=1e-10)
        _print_result(f"spline_linear_recovery_order_{order}", passed)

    # ----- Quadratic: y = x^2 - 3x + 2 -----
    yd_quad = xd**2 - 3.0 * xd + 2.0
    true_quad = xs**2 - 3.0 * xs + 2.0

    f1 = cast(SplineFunc, spline_function(xd, yd_quad, order=1))
    f2 = cast(SplineFunc, spline_function(xd, yd_quad, order=2))
    f3 = cast(SplineFunc, spline_function(xd, yd_quad, order=3))

    ys1 = f1(xs)
    ys2 = f2(xs)
    ys3 = f3(xs)

    # Compares errors rather than demanding exact equality 
    err1 = np.max(np.abs(ys1 -true_quad))
    err2 = np.max(np.abs(ys2 - true_quad))
    err3 = np.max(np.abs(ys3 - true_quad))

    passed2 = err2 <= err1 #order 2 no worse than linear
    passed3 = err3 <= err1 #order 3 no worse than linear
    not_exact1 = err1 > 1e-6

    _print_result("spline_quadratic_order_2_not_worse_than_linear", passed2)
    _print_result("spline_quadratic_order_3_not_worse_than_linear", passed3)
    _print_result("spline_quadratic_not_exact_order_1", not_exact1)

    # ----- Cubic: y = x^3 - x + 1 -----
    yd_cub = xd**3 - xd + 1.0
    true_cub = xs**3 - xs + 1.0

    f1 = cast(SplineFunc, spline_function(xd, yd_cub, order=1))
    f2 = cast(SplineFunc, spline_function(xd, yd_cub, order=2))
    f3 = cast(SplineFunc, spline_function(xd, yd_cub, order=3))

    ys1 = f1(xs)
    ys2 = f2(xs)
    ys3 = f3(xs)

    passed3_cub = np.allclose(ys3, true_cub, atol=1e-10, rtol=1e-10)
    not_exact1_cub = not np.allclose(ys1, true_cub, atol=1e-10, rtol=1e-10)
    not_exact2_cub = not np.allclose(ys2, true_cub, atol=1e-10, rtol=1e-10)

    _print_result("spline_cubic_recovery_order_3", passed3_cub)
    _print_result("spline_cubic_not_exact_order_1", not_exact1_cub)
    _print_result("spline_cubic_not_exact_order_2", not_exact2_cub)


def test_spline_vs_scipy_cubic() -> None:
    """
    Compare spline_function(order=3) to SciPy's UnivariateSpline
    with k=3, s=0 on a smooth nonlinear function.

    Note: Some SciPy stubs expect `ext` as int, so we use `ext=3`
    (which corresponds to "raise" behaviour in older APIs).
    """
    if not HAVE_SCIPY:
        _print_result("spline_vs_scipy_cubic (skipped, no SciPy)", True)
        return

    xd = np.linspace(0.0, 5.0, 20)
    yd = np.exp(-xd) * np.cos(2 * xd)  # something not polynomial

    f_custom = cast(SplineFunc, spline_function(xd, yd, order=3))
    # ext=3 ≈ 'raise' : raise ValueError on extrapolation
    uspline = UnivariateSpline(xd, yd, k=3, s=0, ext=3)

    xs = np.linspace(xd[0], xd[-1], 200)
    y_custom = f_custom(xs)
    y_scipy = uspline(xs)

    passed = np.allclose(y_custom, y_scipy, rtol=1e-5, atol=1e-6)
    _print_result("spline_vs_scipy_cubic", passed)


if __name__ == "__main__":
    # Linear system tests
    test_gauss_single_rhs_seidel()
    test_gauss_single_rhs_jacobi()
    test_gauss_inverse_seidel()
    test_gauss_inverse_jacobi()

    # Spline tests
    test_spline_polynomial_recovery()
    test_spline_vs_scipy_cubic()