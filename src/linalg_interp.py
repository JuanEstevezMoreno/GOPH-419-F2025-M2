import numpy as np
import warnings

def gauss_iter_solve(A, b, x0=None, tol=1e-12, alg='seidel'):
    """
    Solve Ax = b using Gauss-Seidel or Jacobi iteration.

    Parameters
    ----------
    A : array_like (n,n)
    b : array_like (n,) or (n,m)
    x0 : array_like (n,) or (n,m), optional
    tol : float
    alg : str {'seidel', 'jacobi'}

    Returns
    -------
    x : numpy.ndarray
    """

    # Convert input to arrays
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # Check shapes
    if A.ndim != 2:
        raise ValueError("A must be 2D.")
    n, mA = A.shape
    if n != mA:
        raise ValueError("A must be square.")

    if b.ndim == 1:
        b = b.reshape(n, 1)
    if b.shape[0] != n:
        raise ValueError("b must have same number of rows as A.")

    # Process x0
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x0 = np.array(x0, dtype=float)
        if x0.ndim == 1:
            if x0.shape[0] != n:
                raise ValueError("x0 has wrong row count.")
            x = np.tile(x0.reshape(n, 1), (1, b.shape[1]))
        elif x0.ndim == 2:
            if x0.shape != b.shape:
                raise ValueError("x0 shape mismatch.")
            x = x0.copy()
        else:
            raise ValueError("x0 must be 1D or 2D.")

    # Normalize algorithm flag
    alg = alg.strip().lower()
    if alg not in ('seidel', 'jacobi'):
        raise ValueError("alg must be 'seidel' or 'jacobi'.")

    # Extract diagonal
    diag = np.diag(A)
    if np.any(diag == 0):
        raise ValueError("Zero diagonal entry — cannot iterate.")

    # Normalize A for efficiency: A = I - (offdiag/diag)
    Ad = np.diag(1.0 / diag)
    A_norm = Ad @ A
    b_norm = Ad @ b

    # Split normalized A = I - As
    I = np.eye(n)
    As = I - A_norm

    max_iter = 100000

    for k in range(max_iter):
        x_old = x.copy()

        if alg == 'jacobi':
            # Jacobi update uses ONLY x_old
            x_new = np.empty_like(x_old)
            for i in range(n):
                sigma = np.dot(A[i, :], x_old) - A[i, i] * x_old[i, :]
                x_new[i, :] = (b[i, :] - sigma) / A[i, i]
            x = x_new

        else:  # Gauss–Seidel
            for i in range(n):
                sigma = A[i, :] @ x - A[i, i] * x[i, :]
                x[i, :] = (b[i, :] - sigma) / A[i, i]

        # Convergence check
        residual = np.linalg.norm(A @ x - b)
        if residual < tol:
            return x
        
        # Checl relative change
        change = np.linalg.norm(x - x_old)
        if change < tol * (np.linalg.norm(x) + 1e-14):
            return x

    warnings.warn("Gauss-Seidel/Jacobi did not converge.", RuntimeWarning)
    return x


def spline_function(xd, yd, order=3):
    """
    Returns a spline interpolation function of requested order.
    """

    xd = np.array(xd, dtype=float).flatten()
    yd = np.array(yd, dtype=float).flatten()

    # Basic checks
    if xd.shape != yd.shape:
        raise ValueError("xd and yd must have same length.")
    if len(np.unique(xd)) != len(xd):
        raise ValueError("xd contains repeated values.")
    if not np.all(np.sort(xd) == xd):
        raise ValueError("xd must be in increasing order.")
    if order not in (1, 2, 3):
        raise ValueError("order must be 1, 2, or 3.")

    N = len(xd) - 1
    h = np.diff(xd)

    # --- ORDER 1: Linear spline ---
    if order == 1:
        slopes = np.diff(yd) / h

        def f_linear(x):
            x = np.array(x, dtype=float)
            if np.any((x < xd[0]) | (x > xd[-1])):
                raise ValueError(f"x outside range [{xd[0]}, {xd[-1]}].")
            i = np.searchsorted(xd, x, side='right') - 1
            i = np.clip(i, 0, N-1)
            return yd[i] + slopes[i] * (x - xd[i])

        return f_linear

    # --- ORDER 2: Quadratic spline ---
    if order == 2:
        # Solve for c_i using the system in eq. (34)
        A = np.zeros((N, N))
        rhs = np.zeros(N)

        # Build system
        for i in range(N-1):
            A[i, i] = h[i]
            A[i, i+1] = h[i+1]
            rhs[i] = 2 * (
                (yd[i+2] - yd[i+1]) / h[i+1]
                - (yd[i+1]-yd[i]) / h[i]
            )

        # Not-a-knot boundary condition at i = 1
        A[-1, 0] = h[0]
        A[-1, -1] = -h[1]
        rhs[-1] = 0

        # USE DIRECT SOLVER INSTEAD
        try:
            c = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            # Fallback to least squares if matrix is singular
            c = np.linalg.lstsq(A, rhs, rcond=None)[0]

        # Compute b_i and a_i
        a = yd[:-1]
        b = (yd[1:] - yd[:-1]) / h - c * h

        def f_quadratic(x):
            x = np.array(x, dtype=float)
            if np.any((x < xd[0]) | (x > xd[-1])):
                raise ValueError(f"x outside range [{xd[0]}, {xd[-1]}].")
            i = np.searchsorted(xd, x, side='right') - 1
            i = np.clip(i, 0, N-1)
            dx = x - xd[i]
            return a[i] + b[i] * dx + c[i] * dx**2

        return f_quadratic

    # --- ORDER 3: Cubic spline ---
    if order == 3:
        # Build tridiagonal matrix from eq. (46)
        A = np.zeros((N+1, N+1))
        rhs = np.zeros(N+1)

        # Boundary conditions (not-a-knot)
        A[0, 0] = h[1]
        A[0, 1] = -(h[0] + h[1])
        A[0, 2] = h[0]

        A[-1, -3] = h[-1]
        A[-1, -2] = -(h[-2] + h[-1])
        A[-1, -1] = h[-2]

        # Interior equations
        for i in range(1, N):
            A[i, i-1] = h[i]
            A[i, i] = 2 * (h[i] + h[i-1])
            A[i, i+1] = h[i-1]
            rhs[i] = 3 * (
                (yd[i+1] - yd[i]) / h[i]
                - (yd[i] - yd[i-1]) / h[i-1]
            )

        c = gauss_iter_solve(A, rhs).flatten()

        # Compute coefficients
        a = yd[:-1]
        b = (np.diff(yd) / h) - (2*c[:-1] + c[1:]) * h / 3
        d = (c[1:] - c[:-1]) / (3*h)

        def f_cubic(x):
            x = np.array(x, dtype=float)
            if np.any((x < xd[0]) | (x > xd[-1])):
                raise ValueError(f"x outside range [{xd[0]}, {xd[-1]}].")
            i = np.searchsorted(xd, x, side='right') - 1
            i = np.clip(i, 0, N-1)
            dx = x - xd[i]
            return a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3

        return f_cubic