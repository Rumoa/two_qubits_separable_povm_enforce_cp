import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .utils import make_hermitian_basis


@jax.jit
def bipartite_swap(A: ArrayLike) -> Array:
    d = int(np.sqrt(A.shape[0]))
    return A.reshape(d, d, d, d).transpose(1, 0, 3, 2).reshape(d**2, d**2)


@jax.jit
def col_reshuffle(A: ArrayLike) -> Array:
    d = int(np.sqrt(A.shape[0]))
    return A.reshape(d, d, d, d).transpose(3, 1, 2, 0).reshape(d**2, d**2)


@jax.jit
def row_reshuffle(A: ArrayLike) -> Array:
    d = int(np.sqrt(A.shape[0]))
    return A.reshape(d, d, d, d).transpose(0, 2, 1, 3).reshape(d**2, d**2)


def convert_col_row_super(A: ArrayLike) -> Array:
    return bipartite_swap(A)


def convert_col_row(A: ArrayLike) -> Array:
    """Converts a superoperator representation from col<->row.
    It can be applied to superoperators and choi matrices.
    """
    return bipartite_swap(A)


# @partial(jax.jit, static_argnames="order")
@eqx.filter_jit
def from_super_to_choi(superoperator: ArrayLike, order: str = "col") -> Array:
    """Constructs the Choi matrix associated to a superoperator.
    The output choi follows the convention where the unnormalized maximally
    entangled stated is used.

    Args:
        superoperator (ArrayLike)
        order (str, optional): The vectorization ordering the superoperator follows.
        Defaults to "col".

    Returns:
        Array: Choi matrix associated to the superoperator.

    If order == "col", the choi matrix is constructed as (Id \otimes \Lambda)(\Omega)
    If order == "row", the choi matrix is constructed as (\Lambda \otimes Id)(\Omega)
    where \Omega is the unnormalized maximally entangled state.

    """
    if order == "col":
        return col_reshuffle(superoperator)
    if order == "row":
        return row_reshuffle(superoperator)


def from_choi_to_chi(choi_matrix: jax.Array, order="col") -> jax.Array:
    """Transforms a choi matrix to the chi representation.
    The chi representation is normalized such that the [0, 0] element is 1.

    Args:
        choi_matrix: Choi matrix to convert

    Returns:
        Chi matrix (4, 4)
    """
    _order = "C" if order == "row" else "F" if order == "col" else None
    if _order is None:
        raise ValueError("order must be either 'row' or 'col'")
    d = int(np.sqrt(choi_matrix.shape[0]))
    normalized_hermitian_basis = make_hermitian_basis(jnp.log2(d))
    aux_list = []
    for i in normalized_hermitian_basis:
        aux_list_2 = []
        for j in normalized_hermitian_basis:
            aux_list_2.append(
                (i.flatten(order=_order)) @ choi_matrix @ j.flatten(order=_order).conj()
            )
        aux_list.append(aux_list_2)

    chi_aux = jnp.array(aux_list)
    return chi_aux


# def remove_global_phase(jump_op):
#     # Find the matrix entry with largest magnitude
#     idx = jnp.unravel_index(jnp.abs(jump_op).argmax(), jump_op.shape)
#     max_entry = jump_op[idx]
#     phase = jnp.angle(max_entry)
#     return jump_op * jnp.exp(-1j * phase)


# def from_super_to_chi(A, ordering="col"):
#     return from_choi_to_chi(from_super_to_choi(A, order=ordering))


# def compute_decay_rates_jump_ops(lindbladian, normalized_pauli_basis, ordering="col"):
#     # chi = lindbladian[1:, 1:]
#     chi = from_super_to_chi(lindbladian, ordering)[1:, 1:]
#     decay_rates, evecs = jnp.linalg.eigh(chi)
#     jump_ops = (
#         evecs.T[
#             :,
#             :,
#             None,
#             None,
#         ]
#         * normalized_pauli_basis[None, :]
#     ).sum(1)
#     jump_ops = jax.vmap(remove_global_phase)(jump_ops)
#     return decay_rates, jump_ops
