from __future__ import annotations
from typing import Sequence, Callable
from numpy.typing import NDArray
from numpy.linalg import eig, norm
import numpy as np

ETA = 1
TAU = 1


def xy_projection(k: NDArray) -> float:
    return k[0] ** 2 + k[1] ** 2


def f_1(k: NDArray, lattice: float) -> complex:
    return lattice * (TAU * k[0] - k[1] * 1j)


def f_2(k: NDArray, lattice: float) -> float:
    return lattice**2 * xy_projection(k)


def f_3(k: NDArray, lattice: float) -> complex:
    return lattice**2 * (TAU * k[0] + k[1] * 1j)


def f_4(k: NDArray, lattice: float) -> float:
    return lattice**3 * TAU * k[0] * (k[0] ** 2 - 3 * k[1] ** 2)


def f_5(k: NDArray, lattice: float) -> complex:
    return lattice**3 * xy_projection(k) * (TAU * k[0] - k[1] * 1j)


def h_0(energy: float, delta: float, lambda_c: float, lambda_v: float) -> NDArray:
    return np.array(
        (
            (energy + delta - TAU * lambda_c, 0, 0, 0),
            (0, energy + TAU * ETA * lambda_v, 0, 0),
            (0, 0, energy + delta + TAU * lambda_c, 0),
            (0, 0, 0, energy - TAU * ETA * lambda_v),
        ),
        dtype=np.complex128,
    )


def h_1(k: NDArray, lattice: float, gamma_0: float) -> NDArray:
    f_1_value = f_1(k, lattice)
    f_1_conjugate = f_1_value.conjugate()
    return np.array(
        (
            (0, gamma_0 * f_1_value, 0, 0),
            (gamma_0 * f_1_conjugate, 0, 0, 0),
            (0, 0, 0, gamma_0 * f_1_value),
            (0, 0, gamma_0 * f_1_conjugate, 0),
        ),
        dtype=np.complex128,
    )


def h_2(
    k: NDArray, lattice: float, gamma_1: float, gamma_2: float, gamma_3: float
) -> NDArray:
    f_2_value = f_2(k, lattice)
    f_3_value = f_3(k, lattice)
    f_3_conjugate = f_3_value.conjugate()
    return np.array(
        (
            (gamma_1 * f_2_value, gamma_3 * f_3_value, 0, 0),
            (gamma_3 * f_3_conjugate, gamma_2 * f_2_value, 0, 0),
            (0, 0, gamma_1 * f_2_value, gamma_3 * f_3_value),
            (0, 0, gamma_3 * f_3_conjugate, gamma_2 * f_2_value),
        ),
        dtype=np.complex128,
    )


def h_3(
    k: NDArray, lattice: float, gamma_4: float, gamma_5: float, gamma_6: float
) -> NDArray:
    f_4_value = f_4(k, lattice)
    f_5_value = f_5(k, lattice)
    f_5_conjugate = f_5_value.conjugate()
    return np.array(
        (
            (gamma_4 * f_4_value, gamma_6 * f_5_value, 0, 0),
            (gamma_6 * f_5_conjugate, gamma_5 * f_4_value, 0, 0),
            (0, 0, gamma_4 * f_4_value, gamma_6 * f_5_value),
            (0, 0, gamma_6 * f_5_conjugate, gamma_5 * f_4_value),
        ),
        dtype=np.complex128,
    )


def first_order_ham_factory(
    k: NDArray,
    lattice: float,
    energy: float,
    delta: float,
    lambda_c: float,
    lambda_v: float,
    gamma_0: float,
) -> NDArray:
    return h_0(energy, delta, lambda_c, lambda_v) + h_1(k, lattice, gamma_0)


def second_order_ham_factory(
    k: NDArray,
    lattice: float,
    energy: float,
    delta: float,
    lambda_c: float,
    lambda_v: float,
    gamma_0: float,
    gamma_1: float,
    gamma_2: float,
    gamma_3: float,
) -> NDArray:
    return first_order_ham_factory(
        k, lattice, energy, delta, lambda_c, lambda_v, gamma_0
    ) + h_2(k, lattice, gamma_1, gamma_2, gamma_3)


def third_order_ham_factory(
    k: NDArray,
    lattice: float,
    energy: float,
    delta: float,
    lambda_c: float,
    lambda_v: float,
    gamma_0: float,
    gamma_1: float,
    gamma_2: float,
    gamma_3: float,
    gamma_4: float,
    gamma_5: float,
    gamma_6: float,
) -> NDArray:
    return second_order_ham_factory(
        k,
        lattice,
        energy,
        delta,
        lambda_c,
        lambda_v,
        gamma_0,
        gamma_1,
        gamma_2,
        gamma_3,
    ) + h_3(k, lattice, gamma_4, gamma_5, gamma_6)


def get_eigenvalues(hamiltonian: NDArray) -> NDArray:
    return np.real(eig(hamiltonian)[0])


def get_energies(
    ks: NDArray, ham_factory: Callable, params: Sequence[float]
) -> NDArray:
    """
    Return the sorted energy eigenvalues of the given hamiltonian for the
    respective parameters. Each row of the resultant NDArray corresponds to a
    row of ks.

    Parameters
    ----------
    ks : NDArray
        A sequence of k vectors, passed as a matrix, each row being a vector.
    ham_factory : Callable
        A function that returns a hamiltonian, given a k vector as first arg,
        followed by the unpacked params sequence.
    params : Sequence[float]
        The params to create the hamiltonian. This sequence will be unpacked and
        passed to the ham_factory.

    Returns
    -------
    NDArray
        An array with 4 columns and the same amount of rows as ks. This
        reepresents the 4 sorted hamiltonian eigenvalues for each vector k, as a
        row of the supplied ks matrix.
    """
    hams = np.apply_along_axis(ham_factory, 1, ks, *params)
    eigenvalues = np.array([get_eigenvalues(hams[i]) for i in range(hams.shape[0])])
    return np.sort(eigenvalues)


def avg_squared_diff(
    ks: NDArray,
    sorted_energies: NDArray,
    ham_factory: Callable,
    params: Sequence[float],
) -> float:
    """
    Average of the squared differences between all the energies and the
    respective eigenvalues of the hamiltonian of the correspondent k vector.
    This function is meant to be used to construct an objective function for the
    genetic algorithm. Its minimization will lead to the parameters that best
    describes the molecule by means of the kp model.

    Parameters
    ----------
    ks : NDArray
        A sequence of k vectors, passed as a matrix, each row being a vector.
    sorted_energies : NDArray
        A sequence of energy vectors, passed as a matrix, each row being a
        vector. This array need to be sorted along the axis=1.
    ham_factory : Callable
        A function that returns a hamiltonian, given a k vector as first arg,
        followed by the unpacked params sequence.
    params : Sequence[float]
        The params to create the hamiltonian. This sequence will be unpacked and
        passed to the ham_factory.

    Returns
    -------
    float
        The average of the squared differences between all the energies and the
        respective eigenvalues of the hamiltonian of the correspondent k vector.
    """
    sorted_eigenvalues = get_energies(ks, ham_factory, params)
    squared_diffs = (sorted_eigenvalues - sorted_energies) ** 2
    return np.sum(squared_diffs) / (squared_diffs.shape[0] * squared_diffs.shape[1])


def get_fitting_region(
    ks: NDArray,
    lower: float = -0.2,
    upper: float = 0.2,
) -> NDArray:
    """
    Given a ks array, return the row indices between which lower < kx < upper
    """
    (indices,) = np.where((ks[:, 0] >= lower) & (ks[:, 0] <= upper))
    return indices[0], indices[-1]


def get_k_k_index(ks: NDArray) -> int:
    (k_k_index,) = np.where(ks[:, 0] == 0.0)
    return k_k_index[0]


def get_plot_domain(ks: NDArray) -> NDArray:
    k_k_index = get_k_k_index(ks)
    k_gamma = ks[0]
    k_k = ks[k_k_index]
    k_m = ks[-1]
    norms = np.apply_along_axis(norm, 1, ks - k_k)
    first_region_norms = norms[:k_k_index]
    second_region_norms = norms[k_k_index:]
    first_region_xs = -first_region_norms / norm(k_gamma - k_k)
    second_region_xs = second_region_norms / norm(k_m - k_k)
    return np.concatenate((first_region_xs, second_region_xs))
