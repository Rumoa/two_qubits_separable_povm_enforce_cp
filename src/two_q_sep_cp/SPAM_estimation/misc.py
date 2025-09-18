import itertools
from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import qutip as qt
from .gates import (
    array_two_qubits_measurements_gates,
    array_two_qubits_states_gates,
)
from jax import jit
from scipy.stats import entropy

dim = 4
n_povm_ops = 4


n_pars_density_matrix = dim**2

qt.rand_kraus_map(dimensions=4)
rng = np.random.default_rng(seed=0)

pars_kraus = jnp.array(
    [
        qt.rand_unitary(dimensions=dim, density=0.5, seed=rng).full()
        / np.sqrt(n_povm_ops)
        for _ in range(n_povm_ops)
    ]
)

params_dm = rng.normal(size=n_pars_density_matrix)


@jit
def get_block(kops):
    """Get a block matrix from the operators

    Args:
        kops (array[compelx]): A (N, k, k) array of Kraus operators
    """
    return jnp.concatenate([*kops])


@partial(jit, static_argnums=1)
def get_unblock(kmat, num_kraus):
    """Get Kraus operators from a block form

    Args:
        kmat (array[compelx]): A (Nk, k) matrix
    """
    return jnp.array(jnp.vsplit(kmat, num_kraus))


@jax.jit
def make_povm_from_pars(array_kraus_ops):
    return jnp.einsum("iba, ibc->iac", array_kraus_ops.conj(), array_kraus_ops)


@jax.jit
def construct_dm(params_dm, dim=4):
    # params_dm should be a vector of length 16.

    L = jnp.zeros(shape=(dim, dim), dtype=jnp.complex64)
    n_diagonal = dim
    n_off_d_real = int(dim * (dim - 1) / 2)
    n_off_d_imag = n_off_d_real

    diag_vals = jnp.exp(params_dm[:n_diagonal])
    off_d_reals = params_dm[n_diagonal : (n_diagonal + n_off_d_real)]
    off_d_imag = params_dm[(n_diagonal + n_off_d_real) :]

    complex_off_diag = off_d_reals + 1j * off_d_imag

    L = L.at[jnp.diag_indices_from(L)].set(diag_vals)
    L = L.at[jnp.tril_indices_from(L, k=-1)].set(complex_off_diag)

    rho = L @ L.conj().T
    rho = rho / jnp.trace(rho)
    return rho


@jax.jit
def make_complete_povms(array_povms, array_unitaries):
    return jnp.einsum(
        "iab,jbc,idc->ijad", array_unitaries, array_povms, array_unitaries.conj()
    )


@jax.jit
def make_all_density_matrices(rho_00, array_unitaries):
    return jnp.einsum(
        "iab,bc,idc->iad", array_unitaries, rho_00, array_unitaries.conj()
    )


@jax.jit
def compute_probabilities(array_rhos, array_povms):
    # array_rhos: (n_states, d, d)
    # array_povms: (n_basis, n_outcomes, d, d)
    # want probs: (n_states, n_basis, n_outcomes)
    return jnp.clip(jnp.einsum("sij,boji->sbo", array_rhos, array_povms).real, 0, 1)


@jax.jit
def negative_log_lkl(predicted_probs, data_counts):
    # We expect the datasets to be of the shape (no_states, no_basis, no_outcomes) = (16, 9, 4)
    eps = 1e-12
    log_preds = jnp.log(predicted_probs + eps)
    # log_preds = log_preds + jnp.max(log_preds)

    return -1 * jnp.sum(log_preds * data_counts)


Params = namedtuple("Params", "pars_dm pars_kraus")


@jit
def stiefel_update(params, grads, step_size):
    """Updates params in the direction of gradients while staying on the
    Stiefel manifold

    Args:
        params (array[complex]): (n x m) array representing parameters to update
        grads (array[complex]): (n x m) array representing gradients (note)
        step_size (float): The step size for the update

    Returns:
        updated_params (array[complex]): Updated parameters
    """
    U = jnp.hstack([grads, params])
    V = jnp.hstack([params, -grads])

    prod_term = V.T.conj() @ U
    invterm = jnp.eye(prod_term.shape[0]) + (step_size / 2.0) * prod_term
    A = step_size * (U @ jnp.linalg.inv(invterm))

    B = V.T.conj() @ params

    updated_params = params - A @ B
    return updated_params


def learning_step(
    params: Params, optimizer_state_rho, optimizer_for_rho, curried_loss_function
):
    lr = 0.05
    grads = jax.grad(curried_loss_function)(params)
    grads_rhos = grads.pars_dm
    grads_kraus = grads.pars_kraus

    ## RHO UPDATE ###
    params_rho = params.pars_dm
    updates, optimizer_state_rho = optimizer_for_rho.update(
        grads_rhos, optimizer_state_rho, params_rho
    )

    params_rho = optax.apply_updates(params_rho, updates)

    ## POVM UPDATE ###
    params_kraus = params.pars_kraus
    params_kraus_stacked = get_block(params_kraus)

    grads_kraus_stacked = get_block(grads_kraus)

    grads_kraus_stacked = jnp.conj(grads_kraus_stacked)

    grads_kraus_stacked = grads_kraus_stacked / jnp.linalg.norm(grads_kraus_stacked)

    new_kraus_params_stacked = stiefel_update(
        params_kraus_stacked, grads_kraus_stacked, lr
    )

    new_kraus_params = get_unblock(new_kraus_params_stacked, 4)

    new_params = Params(params_rho, new_kraus_params)
    return new_params, optimizer_state_rho, optimizer_for_rho


@jax.jit
def compute_probs_from_pars(params: Params):
    pars_dm = params.pars_dm
    pars_kraus = params.pars_kraus

    # rho part
    rho_00 = construct_dm(pars_dm)
    array_all_rhos = make_all_density_matrices(rho_00, array_two_qubits_states_gates)

    # povm_part
    povm_z = make_povm_from_pars(pars_kraus)
    array_all_povms = make_complete_povms(povm_z, array_two_qubits_measurements_gates)

    # compute probs
    probs = compute_probabilities(array_all_rhos, array_all_povms)
    return probs


@jax.jit
def loss_function(params: Params, data):
    # we expect datasets to be of the shape ijk where i is the number of states, j is the number
    # of bases and k is the number of outcomes. The values of the array are the counts of the observed
    # datasets. In our case, the shape is (16, 9, 4)
    pars_dm = params.pars_dm
    pars_kraus = params.pars_kraus

    # rho part
    rho_00 = construct_dm(pars_dm)
    array_all_rhos = make_all_density_matrices(rho_00, array_two_qubits_states_gates)

    # povm_part
    povm_z = make_povm_from_pars(pars_kraus)
    array_all_povms = make_complete_povms(povm_z, array_two_qubits_measurements_gates)

    # compute probs
    probs = compute_probabilities(array_all_rhos, array_all_povms)

    loss = negative_log_lkl(probs, data)

    return loss

    print(array_all_rhos.shape)
    print(array_all_povms.shape)

    # return (jnp.sum(pars_dm) + jnp.sum(pars_kraus)).real


def generate_random_pars(rng, dim=4, n_povm_ops=4, n_pars_density_matrix=16):
    pars_kraus = jnp.array(
        [
            qt.rand_unitary(dimensions=dim, density=0.5, seed=rng).full()
            / np.sqrt(n_povm_ops)
            for _ in range(n_povm_ops)
        ]
    )

    params_dm = rng.normal(size=n_pars_density_matrix)

    aux_params = Params(params_dm, pars_kraus)
    return aux_params


def make_curried_loss(data):
    # returns a function params -> scalar loss suitable for grad/value_and_grad
    return lambda params: loss_function(params, data)


# @partial(jax.jit, static_argnames=("n_steps", "kraus_ops"))
def train_using_scan(
    init_params: Params,
    optimizer_for_rho: optax.GradientTransformation,
    optimizer_state_rho,
    data,
    n_steps: int,
    kraus_ops: int = 4,
    stiefel_step_size: float = 0.05,
    eps_norm: float = 1e-12,
):
    """
    Runs n_steps of gradient-based training using jax.lax.scan.
    Args:
      init_params: initial Params object
      optimizer_for_rho: optax optimizer (GradientTransformation) used for rho params
      optimizer_state_rho: initial state for optimizer_for_rho
      data: training datasets passed to loss_function
      n_steps: number of optimization steps
      kraus_ops: number of POVM/Kraus blocks (used by get_unblock)
      stiefel_step_size: step size used by stiefel_update for Kraus update
      eps_norm: small epsilon to prevent div-by-zero when normalizing gradients
    Returns:
      final_params, final_optimizer_state_rho, losses (array length n_steps)
    """

    curried_loss = make_curried_loss(data)

    # scan carry: (params, optimizer_state_rho)
    carry_init = (init_params, optimizer_state_rho)

    # value_and_grad of the curried loss
    value_and_grad_loss = jax.value_and_grad(curried_loss)

    def scan_step(carry, _step_index):
        params, opt_state = carry

        # compute loss and grads
        loss_val, grads = value_and_grad_loss(params)

        # rho updates via optax
        grads_rhos = grads.pars_dm
        params_rho = params.pars_dm
        updates, new_opt_state = optimizer_for_rho.update(
            grads_rhos, opt_state, params_rho
        )
        new_params_rho = optax.apply_updates(params_rho, updates)

        # povm/kraus update via Stiefel update
        params_kraus = params.pars_kraus
        # stack/unstack helpers provided by the user
        params_kraus_stacked = get_block(params_kraus)  # user function
        grads_kraus = grads.pars_kraus
        grads_kraus_stacked = get_block(grads_kraus)  # user function

        # conjugate & normalize gradient (matching your original code)
        grads_kraus_stacked = jnp.conj(grads_kraus_stacked)
        norm_grad = jnp.linalg.norm(grads_kraus_stacked)
        grads_kraus_stacked = grads_kraus_stacked / (norm_grad + eps_norm)

        new_kraus_params_stacked = stiefel_update(
            params_kraus_stacked, grads_kraus_stacked, stiefel_step_size
        )

        new_kraus_params = get_unblock(
            new_kraus_params_stacked, kraus_ops
        )  # user function

        new_params = Params(new_params_rho, new_kraus_params)

        new_carry = (new_params, new_opt_state)
        aux_out = loss_val
        return new_carry, aux_out

    # run scan across n_steps (xs array only used to set number of iterations)
    xs = jnp.arange(n_steps)
    (final_params, final_opt_state), losses = jax.lax.scan(scan_step, carry_init, xs)

    # losses shape (n_steps,)
    return final_params, final_opt_state, losses


# ----------------------------- PERMUTATIONS PART ---------------------------- #


def trace_real(A, B):
    """Real part of Hilbert-Schmidt inner product: Re[Tr(A @ B^†)]"""
    return jnp.real(jnp.trace(A @ B.conj().T))


def compute_cost_matrix(povms_learned, povms_ideal):
    """C[i,j] = overlap( learned_i, ideal_j )"""
    n = povms_learned.shape[0]
    C = jnp.zeros((n, n), dtype=jnp.float32)
    for i in range(n):
        for j in range(n):
            C = C.at[i, j].set(trace_real(povms_learned[i], povms_ideal[j]))
    return C


def best_perm_from_cost(C):
    """Brute force over permutations (OK for n=4). Returns sigma: learned_index -> canonical_index."""
    n = C.shape[0]
    best_score = -1e9
    best_sigma = None
    # permutations of canonical indices assigned to learned indices
    for perm in itertools.permutations(range(n)):
        s = 0.0
        for i in range(n):
            s += float(C[i, perm[i]])
        if s > best_score:
            best_score = s
            best_sigma = jnp.array(perm, dtype=jnp.int32)
    return best_sigma, float(best_score)


def permutation_index_vector_from_sigma(sigma):
    """
    Convert sigma (learned_index -> canonical_index) into `perm` so that:
       P = I[perm]  (i.e. perm[new] = old),
    which yields P @ rho @ P^† that places old basis -> new basis.
    """
    n = int(sigma.shape[0])
    perm = -jnp.ones(n, dtype=jnp.int32)
    for old in range(n):
        new = int(sigma[old])
        perm = perm.at[new].set(old)
    return perm


def permutation_matrix_from_permvec(permvec, dtype=jnp.complex64):
    """Return permutation matrix P = I[permvec] (rows permuted)."""
    return jnp.eye(len(permvec), dtype=dtype)[permvec]


def order_rho_povms(rho_i, povms_i, ideal_povm):
    C = compute_cost_matrix(povms_i, ideal_povm)
    sigma, score = best_perm_from_cost(C)
    sigma

    permvec = permutation_index_vector_from_sigma(sigma)
    # print(sigma)
    P = permutation_matrix_from_permvec(permvec, dtype=rho_i.dtype)
    P

    # print(rho_i.round(2))
    new_rho_i = P @ rho_i @ P.conj().T

    new_povms_i = povms_i[sigma, :, :]
    return new_rho_i, new_povms_i


# ------------------------------- VISUALIZATION ------------------------------ #


def visualize_discrepancy_with_labels(counts_exp, counts_ideal, labels, title=None):
    i_dim, j_dim, k_dim = counts_exp.shape
    eps = 1e-10

    # normalize safely
    p_exp = (counts_exp + eps) / (counts_exp.sum(axis=-1, keepdims=True) + k_dim * eps)
    p_ideal = (counts_ideal + eps) / (
        counts_ideal.sum(axis=-1, keepdims=True) + k_dim * eps
    )

    # KL divergence
    kl = np.zeros((i_dim, j_dim))
    for i in range(i_dim):
        for j in range(j_dim):
            kl[i, j] = entropy(p_exp[i, j], p_ideal[i, j])

    # Plot
    plt.figure(figsize=(10, 8))
    im = plt.imshow(kl, cmap="viridis", aspect="auto")

    # set ticks with your dict labels
    plt.xticks(
        ticks=np.arange(j_dim), labels=labels["measurements"], rotation=45, ha="right"
    )
    plt.yticks(ticks=np.arange(i_dim), labels=labels["states"])

    plt.colorbar(im, label="KL divergence")
    plt.xlabel("Measurement basis")
    plt.ylabel("Initial state")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()
