from functools import partial

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax.typing import ArrayLike


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


# @partial(jax.jit, static_argnames="order")
@eqx.filter_jit
def from_super_to_choi(superoperator: ArrayLike, order: str = "col") -> Array:
    if order == "col":
        return col_reshuffle(superoperator)
    if order == "row":
        return row_reshuffle(superoperator)


def convert_col_row_super(A: ArrayLike) -> Array:
    return bipartite_swap(A)
