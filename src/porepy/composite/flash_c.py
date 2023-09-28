"""Module containing implementation of the unified flash using (parallel) compiled
functions created with numba.

The flash system, including a non-parametric interior point method, is assembled and
compiled using :func:`numba.njit`, to enable an efficient solution of the equilibrium
problem.

The compiled functions are such that they can be used to solve multiple flash problems
in parallel.

Parallelization is achieved by applying Newton in parallel for multiple input.
The intended use is for larg compositional flow problems, where an efficient solution
to the local equilibrium problem is required.

References:
    [1]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_
    [2]: `Vu et al. (2021) <https://doi.org/10.1016/j.matcom.2021.07.015>`_


"""
from __future__ import annotations

import abc
import logging
import time
from typing import Callable, Literal, Optional, Sequence

import numba
import numpy as np

from .composite_utils import safe_sum
from .flash import del_log, logger
from .mixture import ThermodynamicState

__all__ = [
    "parse_xyz",
    "parse_pT",
    "normalize_fractions",
    "EoSCompiler",
    "Flash_c",
]


# region Helper methods


@numba.njit(
    "Tuple((float64[:,:], float64[:], float64[:]))(float64[:], UniTuple(int32, 2))",
    fastmath=True,
    cache=True,
)
def parse_xyz(
    X_gen: np.ndarray, npnc: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to parse the fractions from a generic argument.

    NJIT-ed function with signature
    ``(float64[:], UniTuple(int32, 2)) -> Tuple(float64[:,:], float64[:], float64[:])``.

    The feed fractions are always the first ``num_comp - 1`` entries
    (feed per component except reference component).

    The phase compositions are always the last ``num_phase * num_comp`` entries,
    orderered per phase per component (phase-major order),
    with phase and component order as given by the mixture model.

    the ``num_phase - 1`` entries before the phase compositions, are always
    the independent molar phase fractions.

    Important:
        This method also computes the feed fraction of the reference component and the
        molar fraction of the reference phase.
        Hence it returns 2 values not found in ``X_gen``.

        The computed values are always the first ones in the respective vector.

    Parameters:
        X_gen: Generic argument for a flash system.
        npnc: 2-tuple containing information about number of phases and components
            (``num_phase`` and ``num_comp``).
            This information is required for pre-compilation of a mixture-independent
            function.

    Returns:
        A 3-tuple containing

        1. Phase compositions as a matrix with shape ``(num_phase, num_comp)``
        2. Molar phase fractions as an array with shape ``(num_phase,)``
        3. Feed fractions as an array with shape ``(num_comp,)``

    """
    nphase, ncomp = npnc
    # feed fraction per component, except reference component
    Z = np.empty(ncomp, dtype=np.float64)
    Z[1:] = X_gen[: ncomp - 1]
    Z[0] = 1.0 - np.sum(Z[1:])
    # phase compositions
    X = X_gen[-ncomp * nphase :].copy()  # must copy to be able to reshape
    # matrix:
    # rows have compositions per phase,
    # columns have compositions related to a component
    X = X.reshape((nphase, ncomp))
    # phase fractions, -1 because fraction of ref phase is eliminated and not part of
    # the generic argument
    Y = np.empty(nphase, dtype=np.float64)
    Y[1:] = X_gen[-(ncomp * nphase + nphase - 1) : -ncomp * nphase]
    Y[0] = 1.0 - np.sum(Y[1:])

    return X, Y, Z


@numba.njit(
    "float64[:](float64[:], UniTuple(int32, 2))",
    fastmath=True,
    cache=True,
)
def parse_pT(X_gen: np.ndarray, npnc: tuple[int, int]) -> np.ndarray:
    """Helper function extracing pressure and temperature from a generic
    argument.

    NJIT-ed function with signature
    ``(float64[:], UniTuple(int32, 2)) -> float64[:]``.

    Pressure and temperature are the last two values before the independent molar phase
    fractions (``num_phase - 1``) and the phase compositions (``num_phase * num_comp``).

    Parameters:
        X_gen: Generic argument for a flash system.
        npnc: 2-tuple containing information about number of phases and components
            (``num_phase`` and ``num_comp``).
            This information is required for pre-compilation of a mixture-independent
            function.

    Returns:
        An array with shape ``(2,)``.

    """
    nphase, ncomp = npnc
    return X_gen[-(ncomp * nphase + nphase - 1) - 2 : -(ncomp * nphase + nphase - 1)]


@numba.njit("float64[:,:](float64[:,:])", fastmath=True, cache=True)
def normalize_fractions(X: np.ndarray) -> np.ndarray:
    """Takes a matrix of phase compositions (rows - phase, columns - component)
    and normalizes the fractions.

    Meaning it divides each matrix element by the sum of respective row.

    NJIT-ed function with signature ``(float64[:,:]) -> float64[:,:]``.

    Parameters:
        X: ``shape=(num_phases, num_components)``

            A matrix of phase compositions, containing per row the (extended)
            fractions per component.

    Returns:
        A normalized version of ``X``, with the normalization performed row-wise
        (phase-wise).

    """
    return (X.T / X.sum(axis=1)).T


# endregion
# region General flash equation independent of flash type and EoS


@numba.njit(
    "float64[:](float64[:,:], float64[:], float64[:])",
    fastmath=True,
    cache=True,
)
def mass_conservation_res(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    r"""Assembles the residual of the mass conservation equations.

    For each component ``i``, except reference component, it holds

    ... math::

        z\left[i\right] - \sum_j y\left[j\right] x\left[j, i\right] = 0

    Number of phases and components is determined from the chape of ``x``.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:], float64[:]) -> float64[:]``.

    Note:
        See :func:`parse_xyz` for obtaining the properly formatted arguments ``x,y,z``.

    Parameters:
        x: ``shape=(num_phase, num_comp)``

            Phase compositions
        y: ``shape=(num_phase,)``

            Molar phase fractions.
        z: ``shape=(num_comp,)``

            Overall fractions per component.

    Returns:
        An array with ``shape=(num_comp - 1,)`` containg the residual of the mass
        conservation equation (left-hand side of above equation) for each component,
        except the first one (in ``z``).

    """
    res = np.empty(z.shape[0] - 1, dtype=np.float64)

    for i in range(1, z.shape[0]):
        res[i - 1] = z[i] - np.sum(y * x[:, i])

    return res


@numba.njit(
    "float64[:,:](float64[:,:], float64[:])",
    fastmath=True,
    cache=True,
)
def mass_conservation_jac(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns the Jacobian of the residual described in
    :func:`mass_conservation_res`

    The Jacobian is of shape ``(num_comp - 1, num_phase - 1 + num_phase * num_comp)``.
    The derivatives (columns) are taken w.r.t. to each independent molar fraction,
    except feed fractions.

    The order of derivatives w.r.t. phase compositions is in phase-major order.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:]) -> float64[:,:]``.

    Note:
        The Jacobian does not depend on the overall fractions ``z``, since they are
        assumed given and constant, hence only relevant for residual.

    """
    nphase, ncomp = x.shape

    # must fill with zeros, since slightly sparse and below fill-up does not cover
    # elements which are zero
    jac = np.zeros((ncomp - 1, nphase - 1 + nphase * ncomp), dtype=np.float64)

    for i in range(ncomp - 1):
        # (1 - sum_j y_j) x_ir + y_j x_ij is there, per phase
        # hence d mass_i / d y_j = x_ij - x_ir
        jac[i, : nphase - 1] = x[1:, i + 1] - x[0, i + 1]  # i + 1 to skip ref component

        # d.r.t. w.r.t x_ij is always y_j for all j per mass conv.
        jac[i, nphase + i :: nphase] = y  # nphase -1 + i + 1 to skip ref component

    # -1 because of z - z(x,y) = 0
    # and above is d z(x,y) / d[y, x]
    return (-1) * jac


@numba.njit("float64[:](float64[:,:], float64[:])", fastmath=True, cache=True)
def complementary_conditions_res(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Assembles the residual of the complementary conditions.

    For each phase ``j`` it holds

    ... math::

        y\left[j\right] \cdot \left(1 - \sum_i x\left[j, i\right]\right) = 0

    Number of phases and components is determined from the chape of ``x``.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:]) -> float64[:]``.

    Note:
        See :func:`parse_xyz` for obtaining the properly formatted arguments ``x,y``.

    Parameters:
        x: ``shape=(num_phase, num_comp)``

            Phase compositions
        y: ``shape=(num_phase,)``

            Molar phase fractions.

    Returns:
        An array with ``shape=(num_phase,)`` containg the residual of the complementary
        condition per phase.

    """
    nphase = y.shape[0]
    ccs = np.empty(nphase, dtype=np.float64)
    for j in range(nphase):
        ccs[j] = y[j] * (1.0 - np.sum(x[j]))

    return ccs


@numba.njit("float64[:,:](float64[:,:], float64[:])", fastmath=True, cache=True)
def complementary_conditions_jac(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns the Jacobian of the residual described in
    :func:`complementary_conditions_res`

    The Jacobian is of shape ``(num_phase, num_phase - 1 + num_phase * num_comp)``.
    The derivatives (columns) are taken w.r.t. to each independent molar fraction,
    except feed fractions.

    The order of derivatives w.r.t. phase compositions is in phase-major order.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:]) -> float64[:,:]``.

    """
    nphase, ncomp = x.shape
    # must fill with zeros, since matrix sparsely populated.
    d_ccs = np.zeros((nphase, nphase - 1 + nphase * ncomp), dtype=np.float64)

    unities = 1 - np.sum(x, axis=1)

    # first complementary condition is w.r.t. to reference phase
    # (1 - sum_j y_j) * (1 - sum_i x_i0)
    d_ccs[0, : nphase - 1] = (-1) * unities[0]
    d_ccs[0, nphase - 1 : nphase - 1 + ncomp] = y[0] * (-1)
    for j in range(1, nphase):
        # for the other phases, its slight easier since y_j * (1 - sum_i x_ij)
        d_ccs[j, j - 1] = unities[j]
        d_ccs[j, nphase - 1 + j * ncomp : nphase - 1 + (j + 1) * ncomp] = y[j] * (-1)

    return d_ccs


# endregion
# region NPIPM related functions


@numba.njit(
    "float64(float64[:], float64[:], float64, float64, float64)",
    fastmath=True,
    cache=True,
)
def slack_equation_res(
    v: np.ndarray, w: np.ndarray, nu: float, u: float, eta: float
) -> float:
    r"""Implementation of the residual of the slack equation for the non-parametric
    interior point method.

    .. math::

        \frac{1}{2}\left( \lVert v^{-}\rVert^2 + \lVert w^{-}\rVert^2 +
        \frac{u}{n_p}\left(\langle v, w \rangle^{+}\right)^2 \right) +
        \eta\nu + \nu^2 = 0

    Parameters:
        v: ``shape=(num_phase,)``

            Vector containing phase fractions.
        w: ``shape=(num_phase,)``

            Vector containing the unity of phase compositions per phase.
        nu: Value of slack variable.
        eta: Parameter for steepness of decline of slack variable.
        u: Parameter to tune the penalty for violation of complementarity.

    Returns:
        The evaluation of above equation.

    """

    nphase = v.shape[0]
    # dot = np.dot(v, w)  # numba performance warning
    dot = np.sum(v * w)

    # copy because of modifications for negative and positive
    v = v.copy()
    w = w.copy()
    v[v > 0.0] = 0.0
    v[w > 0.0] = 0.0

    # penalization of negativity
    res = 0.5 * (np.sum(v**2) + np.sum(w**2))

    # penalization of violation of complementarity
    dot = 0.0 if dot < 0.0 else dot
    res += 0.5 * dot**2 * u / nphase

    # decline of slack variable
    res += eta * nu + nu**2

    return res


@numba.njit(
    "float64[:](float64[:], float64[:], float64, float64, float64)",
    fastmath=True,
    cache=True,
)
def slack_equation_jac(
    v: np.ndarray, w: np.ndarray, nu: float, u: float, eta: float
) -> float:
    """Implementation of the gradient of the slack equation for the non-parametric
    interior point method (see :func:`slack_equation_res`).

    Parameters:
        v: ``shape=(num_phase,)``

            Vector containing phase fractions.
        w: ``shape=(num_phase,)``

            Vector containing the unity of phase compositions per phase.
        nu: Value of slack variable.
        eta: Parameter for steepness of decline of slack variable.
        u: Parameter to tune the penalty for violation of complementarity.

    Returns:
        The gradient of the slcak equation with derivatives w.r.t. all elements in
        ``v``, ``w`` and ``nu``, with ``shape=(2 * num_phase + 1,)``.

    """

    nphase = v.shape[0]

    jac = np.zeros(2 * nphase + 1, dtype=np.float64)

    # dot = np.dot(v, w)  # numba performance warning
    dot = np.sum(v * w)

    # derivatives of pos() and neg()
    dirac_dot = 1.0 if dot > 0.0 else 0.0  # dirac for positivity of dotproduct
    dirac_v = (v < 0.0).astype(np.float64)  # dirac for negativity in v, elementwise
    dirac_w = (w < 0.0).astype(np.float64)  # same for w

    d_dot_outer = 2 * u / nphase**2 * dot * dirac_dot

    # derivatives w.r.t. to elements in v
    jac[:nphase] = dirac_v * v + d_dot_outer * w
    jac[nphase : 2 * nphase] = dirac_w * v + d_dot_outer * v

    # derivative w.r.t. nu
    jac[-1] = eta + 2 * nu

    return jac


@numba.njit(
    "Tuple((float64[:,:], float64[:]))"
    + "(float64[:], float64[:,:], float64[:], float64, UniTuple(int32, 2))",
    fastmath=True,
    cache=True,
)
def npipm_regularization(
    X: np.ndarray, A: np.ndarray, b: np.ndarray, u: float, npnc: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Regularization of a linearized flash system assembled in the unified setting
    using the non-parametric interior point method.

    Parameters:
        X: Generic argument for the flash.
        A: Linearized flash system (including NPIPM).
        b: Residual corresponding to ``A``.
        u: see :func:`slack_equation_res`
        npnc: ``len=2``

            2-tuple containing the number of phases and number of components

    Returns:
        A regularization according to Vu et al. 2021.

    """
    nphase, ncomp = npnc
    x, y, _ = parse_xyz(X[:-1], npnc)

    reg = 0.0
    for j in range(nphase):
        # summation of complementarity conditions
        reg += y[j] * (1 - np.sum(x[j]))

    reg = 0.0 if reg < 0 else reg
    reg *= u / ncomp**2

    # subtract all relaxed complementary conditions multiplied with reg from the slack equation
    b[-1] = b[-1] - reg * np.sum(b[-(nphase + 1) : -1])
    # do the same for respective rows in the Jacobian
    for j in range(nphase):
        # +2 to skip slack equation and because j start with 0
        v = A[-(j + 2)] * reg
        A[-1] = A[-1] - v

    return A, b


# endregion
# region Methods related to the numerical solution strategy


@numba.njit("float64(float64[:])", fastmath=True, cache=True)
def l2_potential(vec: np.ndarray) -> float:
    return np.sum(vec * vec) / 2.0


@numba.njit(  # TODO type signature once typing for functions is available
    fastmath=True,
    cache=True,
)
def Armijo_line_search(
    Xk: np.ndarray,
    dXk: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    rho_0: float,
    kappa: float,
    j_max: int,
) -> float:
    r"""Armijo line search to be used inside Newton.

    Uses the L2-potential to find a minimizing step size.

    Parameters:
        Xk: Current Newton iterate.
        dXk: New update obtained from Newton.
        F: Callable to evaluate the residual.
            Must be compatible with ``Xk + dXk`` as argument.
        rho_0: ``(0, 1)``

            First step-size.
        kappa: ``(0, 0.5)``

            Slope for the line search.
        j_max: Maximal number of line search iterations.


    Returns:
        A step size :math:`\rho` minimizing
        :math:`\frac{1}{2}\lVert F(X_k + \rho dX_k)\rVert^2`.

    """

    # evaluation of the potential as is
    fk = F(Xk)
    potk = l2_potential(fk)
    rho_j = rho_0
    # kappa = 0.4
    # j_max = 150

    for j in range(1, j_max + 1):
        rho_j = rho_j**j

        try:
            fk_j = F(Xk + rho_j * dXk)
        except:
            continue

        potk_j = l2_potential(fk_j)

        if potk_j <= (1 - 2 * kappa * rho_j) * potk:
            return rho_j

    # return max
    return rho_j


@numba.njit(  # TODO same as for Armijo signature
    cache=True,
)
def newton(
    X_0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    tol: float,
    max_iter: int,
    npnc: tuple[int, int],
    u: float,
    rho_0: float,
    kappa: float,
    j_max: int,
) -> tuple[np.ndarray, int, int]:
    """Compiled Newton with Armijo line search and NPIPM regularization.

    Intended use is for the unified flash problem.

    Parameters:
        X_0: Initial guess.
        F: Callable representing the residual. Must be callable with ``X0``.
        DF: Callable representing the Jacobian. Must be callable with ``X0``.
        tol: Residual tolerance as stopping criterion.
        max_iter: Maximal number of iterations.
        npnc: ``len=2``

            2-tuple containing the number of phases and number of flashes in the flash
            problem.
        u: See :func:`slack_equation_res`. Required for regularization.
        rho_0: See :func:`Armijo_line_search`.
        kappa: See :func:`Armijo_line_search`.
        j_max: See :func:`Armijo_line_search`.

    Returns:
        A 3-tuple containing

        1. the result of the Newton algorithm,
        2. a success flag
            - 0: success
            - 1: max iter reached
            - 2: failure in the evaluation of the residual
            - 3: failure in the evaluation of the Jacobian
        3. final number of perfored iterations

        If the success flag indicates failure, the last iterate state of the unknown
        is returned.

    """
    # default return values
    num_iter = 0
    success = 1

    X = X_0.copy()
    DX = np.zeros_like(X_0)

    try:
        f_i = F(X)
    except:
        return X, 2, num_iter

    if np.linalg.norm(f_i) <= tol:
        success = 0  # root already found
    else:
        for i in range(1, max_iter + 1):
            num_iter += 1

            df_i = DF(X)
            try:
                df_i = DF(X)
            except:
                success = 3
                break

            A, b = npipm_regularization(X, df_i, -f_i, u, npnc)

            dx = np.linalg.solve(A, b)

            # X contains also parameters (p, T, z_i, ...)
            # exactly ncomp - 1 feed fractions and 2 state definitions (p-T, p-h, ...)
            # for broadcasting insert solution into new vector
            DX[npnc[1] - 1 + 2 :] = dx

            s = Armijo_line_search(X, DX, F, rho_0, kappa, j_max)

            X = X + s * DX

            try:
                f_i = F(X)
            except:
                success = 2
                break

            if np.linalg.norm(f_i) <= tol:
                success = 0
                break

    return X, success, num_iter


@numba.njit(
    parallel=True,
    cache=True,
)
def parallel_newton(
    X0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    F_dim: int,
    tol: float,
    max_iter: int,
    npnc: tuple[int, int],
    u: float,
    rho_0: float,
    kappa: float,
    j_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel Newton, assuming each row in ``X0`` is a starting point to find a root
    of ``F``.

    Numba-parallelized loop over all rows in ``X0``, calling
    :func:`newton` for each row.

    For an explanation of all parameters, see :func:`newton`.

    Note:
        ``X0`` can contain parameters for the evaluation of ``F``.
        Therefore the dimension of the image of ``F`` must be defined by passing
        ``F_dim``.
        I.e., ``len(F(X0[i])) == F_dim`` and ``DF(X0[i]).shape == (F_dim, F_dim)``.

    Returns:
        The same as :func:`newton` in vectorized form, containing the results per
        row in ``X0``.

        Note however, that the returned results contain only the actual results,
        not the whole, generic flash argument given in ``X0``.
        More precisely, the first ``num_comp - 1 + 2`` elements per row are assumed to
        contain flash specifications in terms of feed fractions and thermodynamic state.
        Hence they are not duplicated and returned to safe memory.

    """

    N = X0.shape[0]

    result = np.empty((N, F_dim))
    num_iter = np.empty(N, dtype=np.int32)
    converged = np.empty(N, dtype=np.int32)

    for n in numba.prange(N):
        res_i, conv_i, n_i = newton(
            X0[n], F, DF, tol, max_iter, npnc, u, rho_0, kappa, j_max
        )
        converged[n] = conv_i
        num_iter[n] = n_i
        result[n] = res_i[-F_dim:]

    return result, converged, num_iter


@numba.njit(
    cache=True,
)
def linear_newton(
    X0: np.ndarray,
    F: Callable[[np.ndarray], np.ndarray],
    DF: Callable[[np.ndarray], np.ndarray],
    F_dim: int,
    tol: float,
    max_iter: int,
    npnc: tuple[int, int],
    u: float,
    rho_0: float,
    kappa: float,
    j_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Does the same as :func:`parallel_newton`, only the loop over the rows of ``X0``
    is not parallelized, but executed in a classical loop.

    Intended use is for smaller amount of flash problems, where the parallelization
    would produce a certain overhead in the initialization.

    """

    N = X0.shape[0]

    result = np.empty((N, F_dim))
    num_iter = np.empty(N, dtype=np.int32)
    converged = np.empty(N, dtype=np.int32)

    for n in range(N):
        res_i, conv_i, n_i = newton(
            X0[n], F, DF, tol, max_iter, npnc, u, rho_0, kappa, j_max
        )
        converged[n] = conv_i
        num_iter[n] = n_i
        result[n] = res_i[-F_dim:]

    return result, converged, num_iter


# endregion


class EoSCompiler(abc.ABC):
    """Abstract base class for EoS specific compilation using numba.

    The :class:`FlashCompiler` needs functions computing

    - fugacity coefficients
    - enthalpies
    - densities
    - the derivatives of above w.r.t. pressure, temperature and phase compositions

    Respective functions must be assembled and compiled by a child class with a specific
    EoS.

    The compiled functions are expected to have the following signature:

    ``(prearg: np.ndarray, phase_index: int, p: float, T: float, xn: numpy.ndarray)``

    1. ``prearg``: A 2-dimensional array containing the results of the pre-computation
       using the function returned by :meth:`get_pre_arg_computation`.
    2. ``phase_index``, the index of the phase, for which the quantities should be
       computed (``j = 0 ... num_phase - 1``), assuming 0 is the reference phase
    3. ``p``: The pressure value.
    4. ``T``: The temperature value.
    5 ``xn``: An array with ``shape=(num_comp,)`` containing the normalized fractions
      per component in phase ``phase_index``.

    The purpose of the ``prearg`` is efficiency.
    Many EoS have computions of some coterms or compressibility factors f.e.,
    which must only be computed once for all remaining thermodynamic quantities.

    The function for the ``prearg`` computation must have the signature:

    ``(p: float, T: float, XN: np.ndarray)``

    where ``XN`` contains **all** normalized compositions,
    stored row-wose per phase as a matrix.

    There are two ``prearg`` computations: One for the residual, one for the Jacobian
    of the flash system.

    The ``prearg`` for the Jacobian will be fed to the functions representing
    derivatives of thermodynamic quantities
    (e.g. derivative fugacity coefficients w.r.t. p, T, X).

    The ``prearg`` for the residual will be passed to thermodynamic quantities without
    derivation.

    """

    @abc.abstractmethod
    def get_pre_arg_computation_res(
        self,
    ) -> Callable[[float, float, np.ndarray], np.ndarray]:
        """Abstract function for obtaining the compiled computation of the pre-argument
        for the residual.

        Returns:
            A NJIT-ed function taking values for pressure, temperature and normalized
            phase compositions for all phases as a matrix, and returning any matrix,
            which will be passed to other functions computing thermodynamic quantities.

        """
        pass

    @abc.abstractmethod
    def get_pre_arg_computation_jac(
        self,
    ) -> Callable[[float, float, np.ndarray], np.ndarray]:
        """Abstract function for obtaining the compiled computation of the pre-argument
        for the Jacobian.

        Returns:
            A NJIT-ed function taking values for pressure, temperature and normalized
            phase compositions for all phases as a matrix, and returning any matrix,
            which will be passed to other functions computing derivatives of
            thermodynamic quantities.

        """
        pass

    @abc.abstractmethod
    def get_fugacity_computation(
        self,
    ) -> Callable[[np.ndarray, int, float, float, np.ndarray], np.ndarray]:
        """Abstract assembler for compiled computations of the fugacity coefficients.

        Returns:
            A NJIT-ed function taking
            - the pre-argument for the residual,
            - the phase index,
            - pressure value,
            - temperature value,
            - an array normalized fractions of componentn in phase ``phase_index``

            and returning an array of fugacity coefficients with ``shape=(num_comp,)``.

        """
        pass

    @abc.abstractmethod
    def get_dpTX_fugacity_computation(
        self,
    ) -> Callable[[np.ndarray, int, float, float, np.ndarray], np.ndarray]:
        """Abstract assembler for compiled computations of the derivative of fugacity
        coefficients.

        The functions should return the derivative fugacities for each component
        row-wise in a matrix.
        It must contain the derivatives w.r.t. pressure, temperature and each fraction
        in a specified phase.
        I.e. the return value must be an array with ``shape=(num_comp, 2 + num_comp)``.

        Returns:
            A NJIT-ed function taking
            - the pre-argument for the Jacobian,
            - the phase index,
            - pressure value,
            - temperature value,
            - an array normalized fractions of componentn in phase ``phase_index``

            and returning an array of derivatives offugacity coefficients with
            ``shape=(num_comp, 2 + num_comp)``., where the columns indicate the
            derivatives w.r.t. to pressure, temperature and fractions.

        """
        pass


class Flash_c:
    """A class providing efficient unified flash calculations using numba-compiled
    functions.

    It uses the no-python mode of numba to produce highly efficient, compiled code.

    Flash equations are represented by callable residuals and Jacobians. Various
    flash types are assembled in a modular way by combining required, compiled equations
    into a solvable system.

    Since each system depends on the modelled phases and components, significant
    parts of the equilibrium problem must be compiled on the fly.

    This is a one-time action once the modelling process is completed.

    The supported flash types are than available until destruction.

    Supported flash types/specifications:

    1. ``'p-T'``: state definition in terms of pressure and temperature
    2. ``'p-h'``: state definition in terms of pressure and specific mixture enthalpy
    3. ``'v-h'``: state definition in terms of specific volume and enthalpy of the
       mixture

    Multiple flash problems can be solved in parallel by passing vectorized state
    definitions.

    The NPIPM approach is parametrizable. Each re-confugration requires a re-compilation
    since the system of equations must be presented as a vector-valued function
    taking a single vector (thermodynmaic input state).

    Parameters:
        npnc: ``len=2``

            2-tuple containing the number of phases and number of components.
        eos_compiler: An EoS compiler instance required to create a
            :class:`~porepy.composite.flash_compiler.FlashCompiler`.

    """

    def __init__(
        self,
        npnc: tuple[int, int],
        eos_compiler: EoSCompiler,
    ) -> None:
        self.npnc: tuple[int, int] = npnc
        """Number of phases and components, passed at instantiation."""

        # currently only 2-phase flash is supported
        assert npnc[0] == 2, "Currently supports only 2-phase mixtures."

        self.eos_compiler: EoSCompiler = eos_compiler
        """Assembler and compiler of EoS-related expressions equation.
        passed at instantiation."""

        self.residuals: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the respective residuals as a callable."""

        self.jacobians: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the respective Jacobian as a callable."""

        self.npipm_res: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the residual of extended flash system.

        The extended flash system included the NPIPM slack equation, and hence an
        additional dependency on the slack variable :math:`\\nu`.

        The evaluation calls internally respective functions from :attr:`residuals`.
        The resulting array has one element more.

        """

        self.npipm_jac: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the Jacobian of extended flash system.

        The extended flash system included the NPIPM slack equation, and hence an
        additional dependency on the slack variable :math:`\\nu`.

        The evaluation calls internally respective functions from :attr:`jacobians`.
        The resulting matrix hase one row and one column more.

        """

        self.npipm_parameters: dict[str, float] = {
            "eta": 0.5,
            "u": 1.0,
        }
        """A dictionary containing per parameter name (str, key) the respective
        parameter for the NPIPM:

        - ``'eta': 0.5``
        - ``'u': 1.``

        Values can be set directly by modifying the values of this dictionary.

        """

        self.armijo_parameters: dict[str, float] = {
            "kappa": 0.4,
            "rho": 0.99,
            "j_max": 150,
        }
        """A dictionary containing per parameter name (str, key) the respective
        parameter for the Armijo line-search:

        - ``'kappa': 0.4``
        - ``'rho_0': 0.99``
        - ``'j_max': 150`` (maximal number of Armijo iterations)

        Values can be set directly by modifying the values of this dictionary.

        """

        self.tolerance: float = 1e-7
        """Convergence criterion for the flash algorithm. Defaults to ``1e-7``."""

        self.max_iter: int = 100
        """Maximal number of iterations for the flash algorithms. Defaults to 100."""

    def compile(self, verbosity: int = 1) -> None:
        """Triggers the assembly and compilation of equilibrium equations, including
        the NPIPM approach.

        Important:
            This takes a considerable amount of time.
            The compilation is therefore separated from the instantiation of this class.

        Parameters:
            verbosity: ``default=1``

                Enable progress logs. Set to zero to disable.

        """
        # setting logging verbosity
        if verbosity == 1:
            logger.setLevel(logging.INFO)
        elif verbosity >= 2:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

        nphase, ncomp = self.npnc

        ## dimension of flash systems, excluding NPIPM
        # number of equations for the pT system
        # ncomp - 1 mass constraints
        # (nphase - 1) * ncomp fugacity constraints (w.r.t. ref phase formulated)
        # nphase complementary conditions
        pT_dim = ncomp - 1 + (nphase - 1) * ncomp + nphase
        # p-h flash: additional var T, additional equ enthalpy constraint
        ph_dim = pT_dim + 1
        # v-h flash: additional vars p, s_j j!= ref
        # additional equations volume constraint and density constraints
        vh_dim = ph_dim + 1 + (nphase - 1)

        ## Compilation start
        logger.info(f"{del_log}Starting compilation of EoS-related functions\n")
        logger.info(f"{del_log}Compiling residual pre-argument ..")
        prearg_res_c = self.eos_compiler.get_pre_arg_computation_res()
        logger.info(f"{del_log}Compiling Jacobian pre-argument ..")
        prearg_jac_c = self.eos_compiler.get_pre_arg_computation_jac()
        logger.info(f"{del_log}Compiling fugacity coefficient function ..")
        phi_c = self.eos_compiler.get_fugacity_computation()
        logger.info(f"{del_log}Compiling derivatives of fugacity coefficients ..")
        d_phi_c = self.eos_compiler.get_dpTX_fugacity_computation()

        logger.info(f"{del_log}Starting compilation isofugacity constraints\n")
        logger.info(f"{del_log}Compiling residual of isogucacity constraints ..")

        @numba.njit("float64[:](float64[:,:], float64, float64, float64[:,:])")
        def isofug_constr_c(
            prearg_res: np.ndarray,
            p: float,
            T: float,
            Xn: np.ndarray,
        ):
            """Helper function to assemble the isofugacity constraint.

            Formulation is always w.r.t. the reference phase r, assumed to be r=0.

            """
            isofug = np.empty(ncomp * (nphase - 1), dtype=np.float64)

            phi_r = phi_c(prearg_res, 0, p, T, Xn[0])

            for j in range(1, nphase):
                phi_j = phi_c(prearg_res, j, p, T, Xn[j])
                # isofugacity constraint between phase j and phase r
                isofug[(j - 1) * ncomp : j * ncomp] = Xn[j] * phi_j - Xn[0] * phi_r

            return isofug

        logger.info(f"{del_log}Compiling Jacobian of isogucacity constraints ..")

        @numba.njit(
            "float64[:,:]"
            + "(float64[:,:], float64[:,:], int32, float64, float64, float64[:])",
        )
        def d_isofug_block_j(
            prearg_res: np.ndarray,
            prearg_jac: np.ndarray,
            j: int,
            p: float,
            T: float,
            Xn: np.ndarray,
        ):
            """Helper function to construct a block representing the derivative
            of x_ij * phi_ij for all i as a matrix, with i row index.
            This is constructed for a given phase j.
            """
            # derivatives w.r.t. p, T, all compositions, A, B, Z
            dx_phi_j = np.zeros((ncomp, 2 + ncomp))

            phi_j = phi_c(prearg_res, j, p, T, Xn)
            d_phi_j = d_phi_c(prearg_jac, j, p, T, Xn)

            # product rule d(x * phi) = dx * phi + x * dphi
            # dx is is identity
            dx_phi_j[:, 2:] = np.diag(phi_j)
            d_xphi_j = dx_phi_j + (d_phi_j.T * Xn).T

            return d_xphi_j

        @numba.njit(
            "float64[:,:](float64[:,:], float64[:, :], float64, float64, float64[:,:])"
        )
        def d_isofug_constr_c(
            prearg_res: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            Xn: np.ndarray,
        ):
            """Helper function to assemble the derivative of the isofugacity constraints

            Formulation is always w.r.t. the reference phase r, assumed to be zero 0.

            Important:
                The derivative is taken w.r.t. to A, B, Z (among others).
                An forward expansion must be done after a call to this function.

            """
            d_iso = np.zeros((ncomp * (nphase - 1), 2 + ncomp * nphase))

            # creating derivative parts involving the reference phase
            d_xphi_r = d_isofug_block_j(prearg_res, prearg_jac, 0, p, T, Xn[0])

            for j in range(1, nphase):
                # construct the same as above for other phases
                d_xphi_j = d_isofug_block_j(prearg_res, prearg_jac, 1, p, T, Xn[j])

                # filling in the relevant blocks
                # remember: d(x_ij * phi_ij - x_ir * phi_ir)
                # hence every row-block contains (-1)* d_xphi_r
                # p, T derivative
                d_iso[(j - 1) * ncomp : j * ncomp, :2] = (
                    d_xphi_j[:, :2] - d_xphi_r[:, :2]
                )
                # derivative w.r.t. fractions in reference phase
                d_iso[(j - 1) * ncomp : j * ncomp, 2 : 2 + ncomp] = (-1) * d_xphi_r[
                    :, 2:
                ]
                # derivatives w.r.t. fractions in independent phase j
                d_iso[
                    (j - 1) * ncomp : j * ncomp, 2 + j * ncomp : 2 + (j + 1) * ncomp
                ] = d_xphi_j[:, 2:]

            return d_iso

        logger.info(f"{del_log}Starting compilation of flash systems\n")
        logger.info(f"{del_log}Compiling p-T flash equations ..")

        @numba.njit("float64[:](float64[:])")
        def F_pT(X_gen: np.ndarray) -> np.ndarray:
            x, y, z = parse_xyz(X_gen, (nphase, ncomp))
            p, T = parse_pT(X_gen, (nphase, ncomp))

            # declare residual array of proper dimension
            res = np.empty(pT_dim, dtype=np.float64)

            res[: ncomp - 1] = mass_conservation_res(x, y, z)
            # last nphase equations are always complementary conditions
            res[-nphase:] = complementary_conditions_res(x, y)

            # EoS specific computations
            xn = normalize_fractions(x)
            prearg = prearg_res_c(p, T, xn)

            res[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1)] = isofug_constr_c(
                prearg, p, T, xn
            )

            return res

        @numba.njit("float64[:,:](float64[:])")
        def DF_pT(X_gen: np.ndarray) -> np.ndarray:
            x, y, _ = parse_xyz(X_gen, (nphase, ncomp))
            p, T = parse_pT(X_gen, (nphase, ncomp))

            # declare Jacobian or proper dimension
            jac = np.zeros((pT_dim, pT_dim), dtype=np.float64)

            jac[: ncomp - 1] = mass_conservation_jac(x, y)
            # last nphase equations are always complementary conditions
            jac[-nphase:] = complementary_conditions_jac(x, y)

            # EoS specific computations
            xn = normalize_fractions(x)
            prearg_res = prearg_res_c(p, T, xn)
            prearg_jac = prearg_jac_c(p, T, xn)

            jac[
                ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1), nphase - 1 :
            ] = d_isofug_constr_c(prearg_res, prearg_jac, p, T, xn)[:, 2:]

            return jac

        logger.info(f"{del_log}Storing compiled flash equations ..")
        self.residuals.update(
            {
                "p-T": F_pT,
            }
        )

        self.jacobians.update(
            {
                "p-T": DF_pT,
            }
        )

        logger.info(f"{del_log}Compilation compleded.\n")
        u = self.npipm_parameters["u"]
        eta = self.npipm_parameters["eta"]

        self.reconfigure_npipm(u, eta, verbosity)

    def reconfigure_npipm(
        self, u: float = 1.0, eta: float = 0.5, verbosity: int = 1
    ) -> None:
        """Re-compiles the NPIPM slack equation using updated parameters ``u, eta``.

        For more information on ``u`` and ``eta``, see :func:`slack_equation_res`.

        Default values ``u=1, eta=0.5`` are used in the first compilation
        while calling :meth:`compile`.

        Note:
            The user must recompile the system with ``u,eta`` being quasi-static in
            order to be able to call the system with a single argument
            (slack variable appended thermodynamic state) for the numerical
            methods to work.
            Passing ``u`` and ``eta`` as separate arguments is quite cumbersome
            with the current state of numba and the approach of having a single-argument
            callables which represents residuals and Jacobian.

        """
        # setting logging verbosity
        if verbosity == 1:
            logger.setLevel(logging.INFO)
        elif verbosity >= 2:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

        npnc = self.npnc
        nphase, ncomp = npnc
        u = float(u)
        eta = float(eta)

        logger.info(f"{del_log}Starting compilation of NPIPM systems\n")
        logger.info(f"{del_log}Compiling NPIPM p-T flash ..")

        F_pT = self.residuals["p-T"]
        DF_pT = self.jacobians["p-T"]

        @numba.njit("float64[:](float64[:])")
        def F_npipm_pT(X: np.ndarray) -> np.ndarray:
            X_thd = X[:-1]
            nu = X[-1]
            x, y, _ = parse_xyz(X_thd, npnc)

            f_flash = F_pT(X_thd)

            # couple complementary conditions with nu
            f_flash[-nphase:] -= nu

            # NPIPM equation
            unity_j = np.zeros(nphase)
            for j in range(nphase):
                unity_j[j] = 1.0 - np.sum(x[j])

            # complete vector of fractions
            slack = slack_equation_res(y, unity_j, nu, u, eta)

            # NPIPM system has one equation more at end
            f_npipm = np.zeros(f_flash.shape[0] + 1)
            f_npipm[:-1] = f_flash
            f_npipm[-1] = slack

            return f_npipm

        @numba.njit("float64[:,:](float64[:])")
        def DF_npipm_pT(X: np.ndarray) -> np.ndarray:
            X_thd = X[:-1]
            nu = X[-1]
            x, y, _ = parse_xyz(X_thd, npnc)

            df_flash = DF_pT(X_thd)

            # NPIPM matrix has one row and one column more
            df_npipm = np.zeros((df_flash.shape[0] + 1, df_flash.shape[1] + 1))
            df_npipm[:-1, :-1] = df_flash
            # relaxed complementary conditions read as y * (1 - sum x) - nu
            # add the -1 for the derivative w.r.t. nu
            df_npipm[-(nphase + 1) : -1, -1] = np.ones(nphase) * (-1)

            # derivative NPIPM equation
            unity_j = np.zeros(nphase)
            for j in range(nphase):
                unity_j[j] = 1.0 - np.sum(x[j])

            # complete vector of fractions
            d_slack = slack_equation_jac(y, unity_j, nu, u, eta)
            # d slack has derivatives w.r.t. y_j and w_j
            # d w_j must be expanded since w_j = 1 - sum x_j
            # d y_0 must be expanded since reference phase is eliminated by unity
            expand_yr = np.ones(nphase - 1) * (-1)
            expand_x_in_j = np.ones(ncomp) * (-1)

            # expansion of y_0 and cut of redundant value
            d_slack[1 : nphase - 1] += d_slack[0] * expand_yr
            d_slack = d_slack[1:]

            # expand it also to include possibly other derivatives
            d_slack_expanded = np.zeros(df_npipm.shape[1])
            # last derivative is w.r.t. nu
            d_slack_expanded[-1] = d_slack[-1]

            for j in range(nphase):
                # derivatives w.r.t. x_ij, +2 because nu must be skipped and j starts with 0
                vec = expand_x_in_j * d_slack[-(j + 2)]
                d_slack_expanded[-(1 + (j + 1) * ncomp) : -(1 + j * ncomp)] = vec

            # derivatives w.r.t y_j. j != r
            d_slack_expanded[
                -(1 + nphase * ncomp + nphase - 1) : -(1 + nphase * ncomp)
            ] = d_slack[: nphase - 1]

            df_npipm[-1] = d_slack_expanded

            return df_npipm

        logger.info(f"{del_log}Storing compiled flash equations ..")

        self.npipm_res.update(
            {
                "p-T": F_npipm_pT,
            }
        )

        self.npipm_jac.update(
            {
                "p-T": DF_npipm_pT,
            }
        )

        logger.info(f"{del_log}Compilation completed.\n")

    def flash(
        self,
        z: Sequence[np.ndarray],
        p: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
        initial_state: Optional[ThermodynamicState] = None,
        mode: Literal["linear", "parallel"] = "linear",
        verbosity: int = 0,
    ) -> tuple[ThermodynamicState, np.ndarray, np.ndarray]:
        """Performes the flash for given feed fractions and state definition.

        Exactly 2 thermodynamic state must be defined in terms of ``p, T, h`` or ``v``
        for an equilibrium problem to be well-defined.

        One state must relate to pressure or volume.
        The other to temperature or energy.

        Supported combination:

        - p-T
        - p-h
        - v-h

        Parameters:
            z: ``len=num_comp - 1``

                A squence of feed fractions per component, except reference component.
            p: Pressure at equilibrium.
            T: Temperature at equilibrium.
            h: Specific enthalpy of the mixture at equilibrium,
            v: Specific volume of the mixture at equilibrium,
            initial_state: ``default=None``

                If not given, an initial guess is computed for the unknowns of the flash
                type.

                If given, it must have at least values for phase fractions and
                compositions.
                Molar phase fraction for reference phase **must not** be provided.

                It must have additionally values for temperature, for
                a state definition where temperature is not known at equilibrium.

                It must have additionally values for pressure and saturations, for
                state definitions where pressure is not known at equilibrium.
                Saturation for reference phase **must not** be provided.
            mode: ``default='linear'``

                Mode of solving the equilibrium problems for multiple state definitions
                given by arrays.

                - ``'linear'``: A classical loop over state defintions (row-wise).
                - ``'parallel'``: A parallelized loop, intended for larger amounts of
                  problems.

            verbosity: ``default=0``

                For logging information about progress. Note that as of now, there is
                no support for logs during solution procedures in the loop since
                compiled code is exectuded.

        Raises:
            ValueError: If an insufficient amount of feed fractions is passed or they
                violate the unity constraint.
            NotImplementedError: If an unsupported combination of insufficient number of
                of thermodynamic states is passed.

        Returns:
            A 3-tuple containing the results, success flags and number of iterations as
            returned by :func:`newton`.
            The results are stored in a thermodynamic state structure.

        """
        # setting logging verbosity
        if verbosity == 1:
            logger.setLevel(logging.INFO)
        elif verbosity >= 2:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

        nphase, ncomp = self.npnc
        z_sum = safe_sum(z)
        if len(z) != ncomp - 1:
            raise ValueError(f"Need {ncomp - 1} feed fractions. {len(z)} Given.")
        if not np.all(z_sum <= 1.0):
            raise ValueError(f"Feed fractions violate unity constraint.")

        flash_type: Literal["p-T", "p-h", "v-h"]
        F_dim: int
        NF: int  # number of vectorized target states
        X0: np.ndarray  # vectorized, generic flash argument
        gen_arg_dim: int  # number of required values for a flash
        result_state = ThermodynamicState()

        if p is not None and T is not None and (h is None and v is None):
            flash_type = "p-T"
            F_dim = nphase - 1 + nphase * ncomp
            NF = (z_sum + p + T).shape[0]
            gen_arg_dim = ncomp - 1 + 2 + F_dim + 1
            state_1 = p
            state_2 = T
            result_state.p = p
            result_state.T = T
        elif p is not None and h is not None and (T is None and v is None):
            flash_type = "p-h"
            F_dim = nphase - 1 + nphase * ncomp + 1
            NF = (z_sum + p + h).shape[0]
            gen_arg_dim = ncomp - 1 + 2 + 1 + F_dim + 1
            state_1 = p
            state_2 = h
            result_state.p = p
            result_state.h = h
        elif v is not None and h is not None and (T is None and v is None):
            flash_type = "v-h"
            F_dim = nphase - 1 + nphase * ncomp + 2 + nphase - 1
            NF = (z_sum + p + h).shape[0]
            gen_arg_dim = ncomp - 1 + 2 + nphase - 1 + 2 + F_dim + 1
            state_1 = v
            state_2 = h
            result_state.v = v
            result_state.h = h
        else:
            raise NotImplementedError(
                f"Unsupported flash with state definitions {p, T, h, v}"
            )

        logger.info(f"{del_log}Determined flash type: {flash_type}\n")

        logger.debug(f"{del_log}Assembling generic flash arguments ..")
        X0 = np.zeros((NF, gen_arg_dim))
        for i, z_i in enumerate(z):
            X0[:, i] = z_i
        X0[:, ncomp - 1] = state_1
        X0[:, ncomp] = state_2

        if initial_state is None:
            logger.info(f"{del_log}Computing initial state ..")
            start = time.time()

            end = time.time()
            logger.info(f"{del_log}Initial state computed.\n")
            t = end - start
            logger.debug(
                f"Elapsed time (min): {t / 60.}\n"
                if t > 60.0
                else f"Elapsed time (s): {t}\n"
            )
        else:
            logger.debug(f"{del_log}Parsing initial state ..")
            # parsing phase compositions and molar fractions
            for j in range(nphase):
                # values for molar phase fractions except for reference phase
                if j < nphase - 1:
                    X0[:, -(1 + nphase * ncomp + nphase - 1 + j)] = initial_state.y[j]
                # composition of phase j
                for i in range(ncomp):
                    X0[:, -(1 + (nphase - j) * ncomp + i)] = initial_state.X[j][i]

            # If T is unknown, get provided guess for T
            if "T" not in flash_type:
                X0[:, -(1 + ncomp * nphase + nphase - 1 + 1)] = initial_state.T
            # If p is unknown, get provided guess for p and saturations
            if "p" not in flash_type:
                X0[:, -(1 + ncomp * nphase + nphase - 1 + 2)] = initial_state.p
                for j in range(nphase - 1):
                    X0[
                        :, -(1 + ncomp * nphase + nphase - 1 + 2 + nphase - 1 + j)
                    ] = initial_state.s[j]

            # parsing molar phsae fractions

        F = self.npipm_res[flash_type]
        DF = self.npipm_jac[flash_type]
        solver_args = (
            F_dim,
            self.tolerance,
            self.max_iter,
            self.npnc,
            self.npipm_parameters["u"],
            self.armijo_parameters["rho"],
            self.armijo_parameters["kappa"],
            self.armijo_parameters["j_max"],
        )

        logger.info(f"{del_log}Solving ..")
        start = time.time()
        if mode == "linear":
            results, success, num_iter = linear_newton(X0, F, DF, *solver_args)
        elif mode == "parallel":
            results, success, num_iter = parallel_newton(X0, F, DF, *solver_args)
        else:
            raise ValueError(f"Unknown mode of compuation {mode}")
        end = time.time()
        logger.info(f"{del_log}Flash computations done.\n")
        t = end - start
        logger.debug(
            f"Elapsed time (min): {t / 60.}\n"
            if t > 60.0
            else f"Elapsed time (s): {t}\n"
        )

        logger.debug(f"{del_log}Parsing and returning results.\n")
        return (
            self._parse_and_complete_results(results, result_state, flash_type),
            success,
            num_iter,
        )

    def _parse_and_complete_results(
        self, results: np.ndarray, result_state: ThermodynamicState, flash_type: str
    ) -> ThermodynamicState:
        """Helper function to fill a result state with the results from the flash.

        Modifies and returns the passed result state structur containing flash
        specifications.

        Also, fills up secondary expressions for respective flash type.

        """
