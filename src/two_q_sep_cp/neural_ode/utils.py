import re
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import ClassVar

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import joblib
import optax
import orbax.checkpoint as ocp
from jax.typing import ArrayLike
from jaxtyping import Array, Complex, Float

from two_q_sep_cp.neural_ode.cp_projection import ChoiProjection
from two_q_sep_cp.SPAM_estimation.misc import (
    array_two_qubits_measurements_gates,
    array_two_qubits_states_gates,
    make_all_density_matrices,
    make_complete_povms,
)

_norm_hermitian_basis = jnp.array(
    [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
) / jnp.sqrt(2)


# ---------------------------------------------------------------------------- #
#                                GENERATOR STUFF                               #
# ---------------------------------------------------------------------------- #
def make_hermitian_basis(n_qubits):
    n_qubits = int(n_qubits)

    if n_qubits == 1:
        hermitian_basis = _norm_hermitian_basis
    else:
        list_of_prods = list(_norm_hermitian_basis)
        for extra_qubit in range(1, n_qubits):
            aux_list = []
            for elem_prod_i in list_of_prods:
                for pauli_i in _norm_hermitian_basis:
                    aux_list.append(
                        jnp.kron(
                            elem_prod_i,
                            pauli_i,
                        )
                    )
            list_of_prods = aux_list
        hermitian_basis = jnp.array(list_of_prods)

    return hermitian_basis


def make_unnormalized_hermitian_basis(n_qubits):
    unnormalized_hermitian_basis = jnp.array(
        [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
    )

    n_qubits = int(n_qubits)
    if n_qubits == 1:
        hermitian_basis = unnormalized_hermitian_basis
    else:
        list_of_prods = list(unnormalized_hermitian_basis)
        for extra_qubit in range(1, n_qubits):
            aux_list = []
            for elem_prod_i in list_of_prods:
                for pauli_i in unnormalized_hermitian_basis:
                    aux_list.append(
                        jnp.kron(
                            elem_prod_i,
                            pauli_i,
                        )
                    )
            list_of_prods = aux_list
        hermitian_basis = jnp.array(list_of_prods)

    return hermitian_basis


# ------------------------------- BLOCH VECTOR ------------------------------- #


# class Bloch(eqx.Module):
#     d: int
#     hermitian_basis: jnp.ndarray

#     def __init__(self, d):
#         self.d = d
#         self.hermitian_basis = make_hermitian_basis(jnp.log2(d))

#     def from_matrix_to_bloch(self, rho):
#         chex.assert_shape(rho, (self.d, self.d))
#         return jnp.array([jnp.trace(G_i @ rho) for G_i in self.hermitian_basis]).real

#     def from_bloch_to_matrix(self, v):
#         chex.assert_shape(v, (self.d**2,))
#         return jnp.sum(self.hermitian_basis * v[:, None, None], axis=0)

#     @jax.jit
#     def compute_lkl_ij(self, array_states_bloch, array_povm_bloch):
#         """Computes Trace[rho_i povm^j]

#         Args:
#             array_states_bloch (_type_): _description_
#             array_povm_bloch (_type_): _description_

#         Returns:
#             _type_: _description_
#         """
#         # TODO FIX THIS WITH THE SHAPE OF D NOT THE VALUE
#         # chex.assert_shape(
#         #     array_states_bloch,
#         #     (
#         #         ...,
#         #         self.d**2,
#         #     ),
#         # )
#         # chex.assert_shape(
#         #     array_povm_bloch,
#         #     (
#         #         ...,
#         #         self.d**2,
#         #     ),
#         # )
#         probs = []
#         for i, rho_i in enumerate(array_states_bloch):
#             aux_list = []
#             for j, povm_j in enumerate(array_povm_bloch):
#                 pij = jnp.dot(povm_j, rho_i)
#                 aux_list.append(pij)
#             probs.append(aux_list)
#         probs = jnp.array(probs)
#         return probs.real.flatten()

#     @staticmethod
#     @jax.jit
#     def apply_map_kij_to_bloch(parameters: jnp.ndarray, map_kij, bloch_vector):
#         aux = jnp.einsum("kij,j-> ki", map_kij, bloch_vector)
#         return (parameters[:, None] * aux[:, :]).sum(0)


class Bloch2D(eqx.Module):
    # Hermitian basis for the Bloch representation
    hermitian_basis: jnp.ndarray

    def __init__(self, hermitian_basis: jnp.ndarray):
        # Initialize the Bloch2D class with the given Hermitian basis
        # Args:
        #     hermitian_basis: Array of Hermitian basis elements
        self.hermitian_basis = hermitian_basis

    @jax.jit
    def from_matrix_to_bloch(self, matrix: jnp.ndarray) -> jnp.ndarray:
        # Convert a density matrix to a Bloch vector using the Hermitian basis
        # Args:
        #     matrix: Density matrix of shape (4, 4)
        # Returns:
        #     Bloch vector of shape (4,)
        # The einsum operation contracts the indices of the Hermitian basis and the matrix
        return jnp.einsum("iab,ba->i", self.hermitian_basis, matrix).real

    @jax.jit
    def from_bloch_to_matrix(self, bloch_vector: jnp.ndarray) -> jnp.ndarray:
        # Convert a Bloch vector to a density matrix using the Hermitian basis
        # Args:
        #     bloch_vector: Bloch vector of shape (4,)
        # Returns:
        #     Density matrix of shape (4, 4)
        # The sum operation combines the Hermitian basis elements weighted by the Bloch vector components
        return jnp.sum(self.hermitian_basis * bloch_vector[:, None, None], axis=0)


# ----------------------- ENCAPSULATION STATES AND POVM ---------------------- #


class InitialStates(eqx.Module):
    # Array of initial states, possibly noisy
    initial_states: jnp.ndarray

    def __init__(self, initial_states: jnp.ndarray):
        # Initialize the InitialStates class with the given initial states
        # Args:
        #     initial_states: Array of initial states, possibly noisy
        self.initial_states = initial_states

    @jax.jit
    def get_rho_from_index(self, index: int) -> jnp.ndarray:
        # Return the density matrix at the given index
        # Args:
        #     index: Index of the desired density matrix
        # Returns:
        #     Array of shape (4, 4), i.e., the density matrix
        return self.initial_states[index]


class POVMS(eqx.Module):
    # Array of POVMs, possibly noisy
    povms: jnp.ndarray

    def __init__(self, povms: jnp.ndarray):
        # Initialize the POVMS class with the given POVMs
        # Args:
        #     povms: Array of POVMs, possibly noisy
        self.povms = povms

    @jax.jit
    def get_povm_from_index(self, index: int) -> jnp.ndarray:
        # Return the POVM at the given index
        # Args:
        #     index: Index of the desired POVM
        # Returns:
        #     Array of shape (4, 4, 4), i.e., 4 operators of dimension 4
        return self.povms[index]


# ---------------------------- DATASET MANAGEMENT ---------------------------- #


def split_dataset(
    X: Float[Array, "3"], initial_states: InitialStates, povms: POVMS
) -> tuple[Complex[Array, "4 4"], Complex[Array, " d_povm 4 4"], Float]:
    # Extract state index, basis index, and time from the input array X
    # Args:
    #     X: Input array of shape (3,) containing state index, basis index, and time
    #     initial_states: InitialStates object containing the initial states
    #     povms: POVMS object containing the POVMs
    #     bloch2d: Bloch2D object containing the Hermitian basis for the Bloch representation
    # Returns:
    #     rho_bloch: Initial state
    #     povms_vecs: Array of POVMS
    #     time: Time value

    # Convert state index and basis index to int32
    state_index = jnp.int32(X[0])
    basis_index = jnp.int32(X[1])
    time = X[2]

    # Get the density matrix of the initial state
    rho_matrices = initial_states.get_rho_from_index(state_index)

    # Get the POVM matrices
    povms_matrices = povms.get_povm_from_index(basis_index)

    return (
        rho_matrices,
        povms_matrices,
        time,
    )  # state_index, basis_index


@jax.jit
def compute_probability_from_rho_bloch_and_povms_bloch(
    rho_bloch: jnp.ndarray, povms_bloch: jnp.ndarray
) -> jnp.ndarray:
    # Compute the scalar product between the Bloch vector of the initial state and the Bloch vectors of the POVMs
    # Args:
    #     rho_bloch: Array of shape (16,) containing the Bloch vector of the initial state
    #     povms_bloch: Array of shape (4, 16) containing the Bloch vectors of the POVMs
    # Returns:
    #     Array of shape (4,) containing the probabilities of the POVMs

    result = jnp.clip(jnp.sum(rho_bloch * povms_bloch, axis=1), 0, 1)

    return result


@jax.jit
def prepare_batch_for_ode(
    rhos_v: jnp.ndarray, povms_v: jnp.ndarray, times: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Prepare the batch of data for the ODE solver by sorting the data points by time
    # Args:
    #     rhos_v: Array of shape (n, 4) containing the Bloch vectors of the initial states
    #     povms_v: Array of shape (n, 4, 4) containing the Bloch vectors of the POVMs
    #     times: Array of shape (n,) containing the time values
    # Returns:
    #     rhos_v: Array of shape (n, 4) containing the sorted Bloch vectors of the initial states
    #     povms_v: Array of shape (n, 4, 4) containing the sorted Bloch vectors of the POVMs
    #     times: Array of shape (n,) containing the sorted time values

    # Get the indices that would sort the times array
    sorted_idx = jnp.argsort(times)

    # Now we do the inverse sorting
    inv_sorted_idx = jnp.argsort(sorted_idx)

    # Sort the rhos_v, povms_v, and times arrays using the time_ordering indices
    rhos_v = rhos_v[sorted_idx]
    povms_v = povms_v[sorted_idx]
    times = times[sorted_idx]

    return rhos_v, povms_v, times, sorted_idx, inv_sorted_idx


# ---------------------------------------------------------------------------- #
#                                  PARAMETERS                                  #
# ---------------------------------------------------------------------------- #


class Parameters(eqx.Module):
    # d: int
    dimension_matrix: ArrayLike
    H: Array  # shape: (m,)
    S: Array  # shape: (m,)
    C: Array  # shape: (comb,)
    A: Array  # shape: (comb,)
    # class_attributes_str: list[str]

    class_attributes_str: ClassVar[list[str]] = ["H", "S", "C", "A"]

    def __init__(
        self, d: int, h_pars: Array, s_pars: Array, c_pars: Array, a_pars: Array
    ):
        """
        Args:
            d: Hilbert‐space dimension.
            H, S: 1D arrays of length m = d**2 - 1
            C, A: 1D arrays of length comb = C(m, 2) = m*(m-1)//2
        """

        self.dimension_matrix = jnp.identity(d)

        # d = self.dimension
        # compute expected sizes
        m = d**2 - 1
        comb = m * (m - 1) // 2

        # sanity checks on shapes
        chex.assert_rank(h_pars, 1)
        chex.assert_shape(
            h_pars,
            (m,),
        )
        chex.assert_rank(s_pars, 1)
        chex.assert_shape(
            s_pars,
            (m,),
        )

        chex.assert_rank(c_pars, 1)
        chex.assert_shape(
            c_pars,
            (comb,),
        )
        chex.assert_rank(a_pars, 1)
        chex.assert_shape(a_pars, (comb,))

        # assign fields
        # self.d = d
        self.H = h_pars
        self.S = s_pars
        self.C = c_pars
        self.A = a_pars
        # self.class_attributes_str = ["H", "S", "C", "A"]

    @property
    def dimension(self):
        return self.dimension_matrix.shape[0]

    def flatten_pars(self) -> jnp.ndarray:
        """Return a single 1D array concatenating [H, S, C, A]."""
        return jnp.concatenate(
            [
                self.H,
                self.S,
                self.C,
                self.A,
            ]
        )


# ---------------------------------------------------------------------------- #
#                                GENERATOR STUFF                               #
# ---------------------------------------------------------------------------- #


# IMPORTANT
# WE ARE GONNA USE NOW COLUMN-ORDER VECTORIZATION BC I AM TIRED OF CONVERTING
# AND TO MAKE IT MORE COMPATIBLE WITH QUTIP
# AND ALSO WE ARE NOT GONNA USE ORTHONORMAL BASIS


def sprepost(a, b):
    # return jnp.kron(a, b.T)
    return jnp.kron(b.T, a)


def spre(a):
    return sprepost(a, jnp.identity(a.shape[0]))


def spost(a):
    return sprepost(jnp.identity(a.shape[0]), a)


def adjoint_op(h):
    return -1j * (spre(h) - spost(h))


def commutator(a, b):
    return a @ b - b @ a


def anticommutator(a, b):
    return a @ b + b @ a


def make_two_qubit_generators(
    unnormalized_hermitian_basis_one_qubit: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Generate the generators for the Lie algebra of the quantum operations
    # Args:
    #     unnormalized_hermitian_basis: Array of shape (4, 2, 2) containing the unnormalized Hermitian basis
    # Returns:
    #     H_generators: Array of shape (3, 2, 2) containing the generators for the Hamiltonian
    #     S_generators: Array of shape (3, 4, 4) containing the generators for the stochastic part
    #     C_generators: Array of shape (3, 4, 4) containing the generators for the coherent part
    #     A_generators: Array of shape (3, 4, 4) containing the generators for the anti-commutator part

    # Extract the identity and Pauli matrices from the unnormalized Hermitian basis
    id = unnormalized_hermitian_basis_one_qubit[0]
    pauli_basis_nonid = unnormalized_hermitian_basis_one_qubit[1:]

    # Generate the generators for the Hamiltonian
    H_generators = []
    for P in pauli_basis_nonid:
        H_generators.append(adjoint_op(P))
        # H_generators.append(
        #     -1j*
        # )
    H_generators = jnp.array(H_generators)

    # Generate the generators for the stochastic part
    S_generators = []
    for P in pauli_basis_nonid:
        S_generators.append(sprepost(P, P) - sprepost(id, id))
    S_generators = jnp.array(S_generators)

    # Generate the generators for the coherent part
    C_generators = []
    for P, Q in combinations(pauli_basis_nonid, 2):
        C_generators.append(
            sprepost(P, Q)
            + sprepost(Q, P)
            - 0.5 * (spre(anticommutator(P, Q)) + spost(anticommutator(P, Q)))
        )
    C_generators = jnp.array(C_generators)

    # Generate the generators for the anti-commutator part
    A_generators = []
    for P, Q in combinations(pauli_basis_nonid, 2):
        A_generators.append(
            1j
            * (
                sprepost(P, Q)
                - sprepost(Q, P)
                + 0.5 * (spre(commutator(P, Q)) + spost(commutator(P, Q)))
            )
        )
    A_generators = jnp.array(A_generators)

    return H_generators, S_generators, C_generators, A_generators


def make_two_qubit_dual_generators(
    unnormalized_hermitian_basis: jnp.ndarray, dim: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Generate the dual generators for the Lie algebra of the quantum operations
    # Args:
    #     unnormalized_hermitian_basis: Array of shape (4, 2, 2) containing the unnormalized Hermitian basis
    #     dim: Dimension of the Hilbert space
    # Returns:
    #     H_dual_generators: Array of shape (3, 2, 2) containing the dual generators for the Hamiltonian
    #     S_dual_generators: Array of shape (3, 4, 4) containing the dual generators for the stochastic part
    #     C_dual_generators: Array of shape (3, 4, 4) containing the dual generators for the coherent part
    #     A_dual_generators: Array of shape (3, 4, 4) containing the dual generators for the anti-commutator part

    # Extract the Pauli matrices from the normalized Hermitian basis
    unnormalized_pauli_basis = unnormalized_hermitian_basis[1:]

    # Generate the dual generators for the Hamiltonian
    H_dual_generators = []
    for P in unnormalized_pauli_basis:
        H_dual_generators.append(adjoint_op(P))
    H_dual_generators = jnp.array(H_dual_generators) / 2 / dim**2

    # Generate the dual generators for the stochastic part
    S_dual_generators = []
    for P in unnormalized_pauli_basis:
        S_dual_generators.append(sprepost(P, P))
    S_dual_generators = jnp.array(S_dual_generators) / dim**2

    # Generate the dual generators for the coherent part
    C_dual_generators = []
    for P, Q in combinations(unnormalized_pauli_basis, 2):
        C_dual_generators.append(sprepost(P, Q) + sprepost(Q, P))
    C_dual_generators = jnp.array(C_dual_generators) / 2 / dim**2

    # Generate the dual generators for the anti-commutator part
    A_dual_generators = []
    for P, Q in combinations(unnormalized_pauli_basis, 2):
        A_dual_generators.append(1j * (sprepost(P, Q) - sprepost(Q, P)))
    A_dual_generators = jnp.array(A_dual_generators) / 2 / dim**2

    return H_dual_generators, S_dual_generators, C_dual_generators, A_dual_generators


class GeneratorsSet(eqx.Module):
    H: ArrayLike
    S: ArrayLike
    C: ArrayLike
    A: ArrayLike


class DualGeneratorsSet(eqx.Module):
    H: ArrayLike
    S: ArrayLike
    C: ArrayLike
    A: ArrayLike


class LindbladGenerators(eqx.Module):
    dimension_matrix: ArrayLike
    gens: GeneratorsSet
    dual_gens: DualGeneratorsSet

    def __init__(
        self, dimension, generators: GeneratorsSet, dual_generators: DualGeneratorsSet
    ):
        self.dimension_matrix = jnp.identity(dimension)
        self.gens = generators
        self.dual_gens = dual_generators
        self.check_shapes()

    @property
    def dimension(self):
        return self.dimension_matrix.shape[0]

    def check_shapes(self):
        m = int(self.dimension**2 - 1)
        combinations = m * (m - 1) // 2

        chex.assert_shape(self.gens.H, (m, self.dimension**2, self.dimension**2))
        chex.assert_shape(self.gens.S, (m, self.dimension**2, self.dimension**2))
        chex.assert_shape(
            self.gens.C, (combinations, self.dimension**2, self.dimension**2)
        )
        chex.assert_shape(
            self.gens.A, (combinations, self.dimension**2, self.dimension**2)
        )

        chex.assert_shape(self.dual_gens.H, (m, self.dimension**2, self.dimension**2))
        chex.assert_shape(self.dual_gens.S, (m, self.dimension**2, self.dimension**2))
        chex.assert_shape(
            self.dual_gens.C, (combinations, self.dimension**2, self.dimension**2)
        )
        chex.assert_shape(
            self.dual_gens.A, (combinations, self.dimension**2, self.dimension**2)
        )

    @jax.jit
    def make_lindbladian(self, parameters: Parameters) -> Array:
        L = (
            (self.gens.H * parameters.H[:, None, None]).sum(0)
            + (self.gens.S * parameters.S[:, None, None]).sum(0)
            + (self.gens.C * parameters.C[:, None, None]).sum(0)
            + (self.gens.A * parameters.A[:, None, None]).sum(0)
        )
        # identity_part = jnp.kron(self.dimension_matrix, self.dimension_matrix)
        # L = L + identity_part
        return L

    @jax.jit
    def from_lindbladian(self, lindbladian: ArrayLike) -> Parameters:
        """
        Given a Lindbladian matrix L (shape d^2 x d^2), recover the parameters
        by projecting onto the dual generators.

        Returns:
        Parameters: An instance of the Parameters class with recovered H, S, C, A.
        """
        d = self.dimension

        # Assert shape of L
        chex.assert_shape(lindbladian, (d**2, d**2))

        H_pars = jax.vmap(lambda g: jnp.vdot(lindbladian, g))(self.dual_gens.H)
        S_pars = jax.vmap(lambda g: jnp.vdot(lindbladian, g))(self.dual_gens.S)
        C_pars = jax.vmap(lambda g: jnp.vdot(lindbladian, g))(self.dual_gens.C)
        A_pars = jax.vmap(lambda g: jnp.vdot(lindbladian, g))(self.dual_gens.A)

        return Parameters(
            d=d, h_pars=H_pars, s_pars=S_pars, c_pars=C_pars, a_pars=A_pars
        )


# ---------------------------------------------------------------------------- #
#                              UPDATE BLOCH VECTOR                             #
# ---------------------------------------------------------------------------- #


# def make_update_matrix_bloch(l_gens: LindbladGenerators, normalized_basis):
#     def vec_col(A):
#         return A.flatten(order="F")

#     vec_norm_basis = jax.vmap(vec_col)(normalized_basis)

#     def f(G_beta_vec, L_tau, G_alpha_vec):
#         return jnp.dot(G_beta_vec.conj(), L_tau @ G_alpha_vec)
#         # return jnp.dot(G_alpha_vec.conj(), L_tau @ G_beta_vec)

#     f_vmap = jax.vmap(
#         jax.vmap(jax.vmap(f, in_axes=(None, None, 0)), in_axes=(0, None, None)),
#         in_axes=(None, 0, None),
#     )
#     re = []
#     for key_i in l_gens.gens.__dict__:
#         gens_arr = l_gens.gens.__dict__[key_i]
#         re.append(f_vmap(vec_norm_basis, gens_arr, vec_norm_basis))

#     update_matrix = jnp.vstack(re)
#     return update_matrix


@jax.jit
def update_bloch_vector_from_parameters(
    bloch_vector, update_matrix, parameters: Parameters
):
    """
    Update the Bloch vector based on the update matrix and parameters.

    Args:
        bloch_vector (jnp.ndarray): Current Bloch vector.
        update_matrix (jnp.ndarray): Update matrix.
        parameters (Parameters): System parameters.

    Returns:
        jnp.ndarray: Updated Bloch vector.
    """
    # Compute the update to the Bloch vector using einsum
    bloch_vector_update = jnp.einsum(
        "tab,t,b->a",
        update_matrix,
        parameters.flatten_pars(),
        bloch_vector,
    )

    # Add the identity matrix contribution and the update <- this is wrong, this is true if u work directly
    # at the map level, but  we are working with lindbladians:
    # $ \tr (rho^{dot}) = 0 = sum_ij d_{ij} G_j^{dag} G_i  -> sum_{i} d_{ii} = 0$
    # bloch_vector_update = jnp.identity(16) @ bloch_vector + bloch_vector_update

    # Return the real part of the updated Bloch vector
    return bloch_vector_update.real


def make_update_matrix_bloch(l_gens: LindbladGenerators, normalized_basis):
    """
    Create an update matrix for the Bloch vector based on Lindblad generators and a normalized basis.

    Args:
        l_gens (LindbladGenerators): Lindblad generators.
        normalized_basis: A normalized basis for the system.

    Returns:
        jnp.ndarray: The update matrix for the Bloch vector.
    """

    def vec_col(A):
        """
        Flatten a matrix column-wise.

        Args:
            A (jnp.ndarray): Input matrix.

        Returns:
            jnp.ndarray: Flattened matrix.
        """
        return A.flatten(order="F")

    # Vectorize the normalized basis using vmap
    vec_norm_basis = jax.vmap(vec_col)(normalized_basis)

    def f(G_beta_vec, L_tau, G_alpha_vec):
        """
        Compute the dot product of G_beta_vec.conj(), L_tau @ G_alpha_vec.

        Args:
            G_beta_vec (jnp.ndarray): Vector G_beta.
            L_tau (jnp.ndarray): Matrix L_tau.
            G_alpha_vec (jnp.ndarray): Vector G_alpha.

        Returns:
            jnp.ndarray: Result of the dot product.
        """
        return jnp.dot(G_beta_vec.conj(), L_tau @ G_alpha_vec)

    # Vectorize the function f using vmap
    f_vmap = jax.vmap(
        jax.vmap(jax.vmap(f, in_axes=(None, None, 0)), in_axes=(0, None, None)),
        in_axes=(None, 0, None),
    )

    re = []
    for key_i in l_gens.gens.__dict__:
        gens_arr = l_gens.gens.__dict__[key_i]
        re.append(f_vmap(vec_norm_basis, gens_arr, vec_norm_basis))

    # Stack the results vertically to form the update matrix
    update_matrix = jnp.vstack(re)
    return update_matrix


# ---------------------------------------------------------------------------- #
#                                  DATALOADER                                  #
# ---------------------------------------------------------------------------- #


def make_data_loader(arrays, batch_size, subkey):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        key, subkey = jax.random.split(subkey)
        perm = jax.random.permutation(subkey, indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def get_abstract_structure(model):
    """Returns the abstract structure for checkpointing of the array_like part

    Args:
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    array_part, _ = eqx.partition(model, eqx.is_array)
    abstract_structure = jax.tree_util.tree_map(
        ocp.utils.to_shape_dtype_struct, array_part
    )
    return abstract_structure


def _sanitize_comment(s: str, max_len: int = 40) -> str:
    """
    Keep only safe characters for filenames: letters, numbers, - and _.
    Replace whitespace with '_'. Truncate to max_len.
    """
    if s is None:
        return ""
    s = str(s).strip()
    # replace whitespace with underscore
    s = re.sub(r"\s+", "_", s)
    # keep only alnum, dash, underscore
    s = re.sub(r"[^A-Za-z0-9\-_]", "", s)
    if len(s) > max_len:
        s = s[:max_len]
    return s


def make_ckpt_path(config):
    """Generate checkpoint folder path from config.

    Behavior:
      - Uses `checkpoint_parent` (default "./checkpoints").
      - If `checkpoint_name` present -> use it as base folder name.
      - Otherwise builds name from architecture, depth, width, num_bands and date.
      - If `comment` (or alternate keys) is present in config, append a sanitized
        short comment to the folder name.
    """
    parent = Path(config.get("checkpoint_parent", "./checkpoints"))
    override_name = config.get("checkpoint_name")

    # support a few possible keys the user might use for comments
    comment_raw = (
        config.get("comment")
        or config.get("checkpoint_comment")
        or config.get("ckpt_comment")
        or None
    )
    comment = _sanitize_comment(comment_raw) if comment_raw else ""

    if override_name:
        # If user explicitly provided checkpoint_name, respect it but optionally add comment
        folder_name = str(override_name)
        if comment:
            folder_name = f"{folder_name}-{comment}"
    else:
        # Build descriptive name
        arch = config.get("architecture", "model")
        depth = config.get("depth", 0)
        width = config.get("width", 0)
        num_bands = config.get("num_bands", None)
        timestamp = datetime.now().strftime("%Y_%m_%d")
        folder_name = f"{arch}-depth={depth}-width={width}"
        if num_bands:
            folder_name += f"-bands={num_bands}"
        if comment:
            folder_name += f"-{comment}"
        folder_name += f"-{timestamp}"

    path = (parent / folder_name).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def make_scheduler(init_lr, warmup_steps, total_steps):
    warmup = optax.linear_schedule(
        init_value=0.0, end_value=init_lr, transition_steps=warmup_steps
    )
    cosine = optax.cosine_decay_schedule(
        init_value=init_lr, decay_steps=max(1, total_steps - warmup_steps)
    )
    return optax.join_schedules([warmup, cosine], [warmup_steps])


def build_restored_model(full_model, loaded_array_leaves):
    _, non_array_leaves = eqx.partition(full_model, eqx.is_array)
    return eqx.combine(non_array_leaves, loaded_array_leaves)


@dataclass
class Setup:
    initial_states: InitialStates
    povms: POVMS
    bloch2d: Bloch2D
    lindblad_gens: LindbladGenerators
    update_matrix: Array
    choi_projection: None


def initialize_setup(config, spam_estimator="mean") -> Setup:
    spam_estimation = joblib.load(config["spam_estimation_path"])
    if spam_estimation == "mean":
        rho_00 = spam_estimation["rhos"].mean(0)
        povm_z = spam_estimation["povms"].mean(0)
    if spam_estimation == "mle":
        rho_00 = spam_estimation["rho_mle"]
        povm_z = spam_estimation["povms_mle"]

    array_povms = make_complete_povms(povm_z, array_two_qubits_measurements_gates)
    array_rhos = make_all_density_matrices(rho_00, array_two_qubits_states_gates)
    normalized_hermitian_basis = make_hermitian_basis(n_qubits=2)
    bloch2d = Bloch2D(normalized_hermitian_basis)
    initial_states = InitialStates(array_rhos)
    povms = POVMS(array_povms)
    unnorm_herm_basis = make_unnormalized_hermitian_basis(n_qubits=2)
    dimension = 4
    H, S, C, A = make_two_qubit_generators(unnorm_herm_basis)
    H_dual, S_dual, C_dual, A_dual = make_two_qubit_dual_generators(
        unnorm_herm_basis, dim=dimension
    )
    gens_set = GeneratorsSet(jnp.array(H), jnp.array(S), jnp.array(C), jnp.array(A))
    dual_gens_set = DualGeneratorsSet(
        jnp.array(H_dual), jnp.array(S_dual), jnp.array(C_dual), jnp.array(A_dual)
    )
    lindblad_gens = LindbladGenerators(
        dimension=4, generators=gens_set, dual_generators=dual_gens_set
    )
    update_matrix = make_update_matrix_bloch(lindblad_gens, normalized_hermitian_basis)
    choi_projection = ChoiProjection(system_dimension=4)
    args_loss = {
        "initial_states": initial_states,
        "povms": povms,
        "bloch2d": bloch2d,
        "update_matrix": update_matrix,
        "lindblad_generators": lindblad_gens,
        "short_time_gamma": 2,  # originally 3
        "choi_projection": choi_projection,
    }
    return Setup(
        initial_states, povms, bloch2d, lindblad_gens, update_matrix, choi_projection
    ), args_loss
