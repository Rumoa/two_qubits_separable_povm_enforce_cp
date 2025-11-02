import operator
from typing import Any, Callable, Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from two_q_sep_cp.neural_ode.model import (
    compute_choi_state_from_super_col,
    compute_probability_from_choi_array_povms,
    make_physical_choi,
    weighted_multinomial_nll_from_probs,
)
from two_q_sep_cp.neural_ode.utils import (
    POVMS,
    ChoiProjection,
    InitialStates,
    LindbladGenerators,
    Parameters,
    prepare_batch_for_ode,
    split_dataset,
)

# class LossWeights(eqx.Module):
#     weights: jnp.array

#     def __init__(self, weights=jnp.array([1.0, 1.0], dtype=jnp.float64)):
#         self.weights = jnp.array(weights, dtype=jnp.float64)

#     @property
#     def w_nll(self):
#         return self.weights[0]

#     @property
#     def w_cp(self):
#         return self.weights[1]


# class SeparateLosses(eqx.Module):
#     loss_nll: Array
#     loss_choi: Array

#     def _binary_op(self, other: Any, op):
#         # other is either the same pytree type, or a scalar/array
#         if isinstance(other, SeparateLosses):
#             return jax.tree_util.tree_map(lambda a, b: op(a, b), self, other)
#         # special-case for sum([...]) which starts with 0
#         if other == 0 and op is operator.add:
#             return self
#         return jax.tree_util.tree_map(lambda a: op(a, other), self)

#     # arithmetic operators
#     def __add__(self, other):
#         return self._binary_op(other, operator.add)

#     def __radd__(self, other):
#         if other == 0:  # support sum([...])
#             return self
#         return self._binary_op(other, operator.add)

#     def __sub__(self, other):
#         return self._binary_op(other, operator.sub)

#     def __rsub__(self, other):
#         # other - self  (if other is scalar/array)
#         if isinstance(other, SeparateLosses):
#             return other._binary_op(self, operator.sub)
#         return jax.tree_util.tree_map(lambda a: operator.sub(other, a), self)

#     def __mul__(self, other):
#         return self._binary_op(other, operator.mul)

#     def __rmul__(self, other):
#         return self._binary_op(other, operator.mul)

#     def __truediv__(self, other):
#         return self._binary_op(other, operator.truediv)

#     def __neg__(self):
#         return jax.tree_util.tree_map(lambda a: -a, self)

#     # optional convenience: sum of all fields
#     def total(self):
#         return sum(
#             jax.tree_leaves(self)
#         )  # sums all leaves (works if leaves are scalars/arrays)


class AuxSetup(eqx.Module):
    latent_dimension: int
    sigma: int
    num_fourier_features: int


class LatentEncoder(eqx.Module):
    mlp: eqx.nn.MLP
    latent_dimension: int
    num_fourier_features: int
    sigma: float
    _rff_B: jnp.ndarray

    def __init__(
        self, key, latent_dimension, width_size, depth, num_fourier_features, sigma
    ):
        self.num_fourier_features = int(num_fourier_features)
        self.sigma = float(sigma)
        self.latent_dimension = int(latent_dimension)

        key, subkey = jax.random.split(key)

        self._rff_B = (
            jax.random.normal(subkey, shape=(self.num_fourier_features,)) * self.sigma
        )

        # input dimension = rff_features + latent_dimension if we use rff
        # if we don´t use rff, input_dimension = 1 + latent_dimension
        mlp_input_dimension = 2 * self.num_fourier_features + self.latent_dimension
        output_mlp_dimension = self.latent_dimension

        self.mlp = eqx.nn.MLP(
            key=key,
            in_size=mlp_input_dimension,
            out_size=output_mlp_dimension,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.silu,
        )

    def __call__(self, t, a):
        """

        Args:
            t (_type_): time (scalar or 1-element array)
            a (_type_): Latent variable of shape (latent_dimension)
        """
        t = jnp.asarray(t, dtype=jnp.float64)

        rff_features = jnp.hstack(
            [
                jnp.cos(2 * jnp.pi * jax.lax.stop_gradient(self._rff_B) * t),
                jnp.sin(2 * jnp.pi * jax.lax.stop_gradient(self._rff_B) * t),
            ]
        )
        input_mlp = jnp.hstack([a, rff_features])
        output_mlp = self.mlp(input_mlp)
        return output_mlp


class AugmentedLindbladNetTwoQubits(eqx.Module):
    dimension: int
    mlp: eqx.nn.MLP
    total_number_parameters: int
    num_fourier_features: int
    sigma: float
    latent_dimension: int

    _rff_B: jnp.ndarray

    def __init__(
        self, key, latent_dimension, width_size, depth, num_fourier_features, sigma
    ):
        self.dimension = 4
        self.total_number_parameters = self.dimension**2 * (self.dimension**2 - 1)
        self.latent_dimension = latent_dimension
        self.sigma = sigma

        self.num_fourier_features = int(num_fourier_features)

        key, subkey = jax.random.split(key)

        self._rff_B = (
            jax.random.normal(subkey, shape=(self.num_fourier_features,)) * self.sigma
        )

        # input dimension = rff_features + latent_dimension if we use rff
        # if we don´t use rff, input_dimension = 1 + latent_dimension
        mlp_input_dimension = 2 * self.num_fourier_features + self.latent_dimension
        output_mlp_dimension = self.total_number_parameters

        self.mlp = eqx.nn.MLP(
            key=key,
            in_size=mlp_input_dimension,
            out_size=output_mlp_dimension,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.silu,
        )

    def __call__(self, t, a) -> Parameters:
        """

        Args:
            t (_type_): time (scalar or 1-element array)
            a (_type_): Latent variable of shape (latent_dimension)
        """
        t = jnp.asarray(t, dtype=jnp.float64)

        rff_features = jnp.hstack(
            [
                jnp.cos(2 * jnp.pi * jax.lax.stop_gradient(self._rff_B) * t),
                jnp.sin(2 * jnp.pi * jax.lax.stop_gradient(self._rff_B) * t),
            ]
        )
        input_mlp = jnp.hstack([a, rff_features])
        bare_output_parameters = self.mlp(input_mlp)

        # slicing into parameter blocks (same as your original code)
        m = self.dimension**2 - 1
        combinations = m * (m - 1) // 2

        h_pars = bare_output_parameters[0:m]
        s_pars = bare_output_parameters[m : 2 * m]
        c_pars = bare_output_parameters[2 * m : 2 * m + combinations]
        a_pars = bare_output_parameters[2 * m + combinations :]

        wrapped_parameters = Parameters(self.dimension, h_pars, s_pars, c_pars, a_pars)

        return wrapped_parameters


class JointLindbladNetEncoder(eqx.Module):
    lindblad_net: AugmentedLindbladNetTwoQubits
    latent_net: LatentEncoder


class AugmentedOdeY(eqx.Module):
    latent_variable: jnp.ndarray
    map_superoperator: jnp.ndarray


class ArgsODE(eqx.Module):
    model: JointLindbladNetEncoder
    lindblad_generators: LindbladGenerators


def ode_eqn_augmented(t, y: AugmentedOdeY, args) -> AugmentedOdeY:
    model = args.model
    l_gens = args.lindblad_generators

    latent_variable = jnp.array(y.latent_variable, dtype=jnp.float64)
    map_superoperator = y.map_superoperator

    parameters_at_t: Parameters = model.lindblad_net(t, latent_variable)
    lindbladian = l_gens.make_lindbladian(parameters_at_t)

    map_superoperator_prime = lindbladian @ map_superoperator

    # -------------------------------- LATENT PART ------------------------------- #
    latent_variable_prime = jnp.array(
        model.latent_net(t, latent_variable), dtype=jnp.complex128
    )

    y_prime = AugmentedOdeY(
        latent_variable=latent_variable_prime, map_superoperator=map_superoperator_prime
    )
    return y_prime


GLOBAL_AUGMENTED_ODE_SOLVER = diffrax.Tsit5()
GLOBAL_AUGMENTED_ODE_TERM_MAP = diffrax.ODETerm(ode_eqn_augmented)


def evolve_map(
    model, ts: Float[Array, " dim_batch"], lindblad_generators
) -> Complex[Array, " dim_batch 16 16"]:
    ode_args = ArgsODE(model=model, lindblad_generators=lindblad_generators)

    solver = GLOBAL_AUGMENTED_ODE_SOLVER
    term = GLOBAL_AUGMENTED_ODE_TERM_MAP

    # y0 is the identity map. for the case of two qubits, the superoperator has dimension 16x16

    y0 = AugmentedOdeY(
        latent_variable=jnp.zeros(
            ode_args.model.latent_net.latent_dimension, jnp.complex128
        ),
        map_superoperator=jnp.identity(16, dtype=jnp.complex128),
    )

    # stepsize_controller = PIDController(rtol=1e-9, atol=1e-12)
    dt0 = 0.001

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=jnp.array(0.0),
        t1=ts[-1],
        y0=y0,
        args=ode_args,
        dt0=dt0,
        saveat=diffrax.SaveAt(ts=ts),
    )
    array_maps_superop_col_order = solution.ys.map_superoperator
    return (
        array_maps_superop_col_order  # shape (n_times, 16, 16) dtype = jnp.complex128
    )


class StaticSetup(eqx.Module):
    initial_states: InitialStates
    povms: POVMS
    lindblad_gens: LindbladGenerators
    choi_projection: ChoiProjection
    evolve_map_fn: Callable


# @eqx.filter_jit
# def compute_separate_losses(
#     model,
#     X: Float[Array, "d_batch 3"],
#     Y: Float[Array, "d_batch 4"],
#     static_objects: StaticSetup,
#     arg_loss,
# ):
#     initial_states: InitialStates = static_objects.initial_states
#     povms: POVMS = static_objects.povms

#     weight_floor = arg_loss.get("short_time_weight_floor", 0.1)
#     gamma_override = arg_loss.get("short_time_gamma", jnp.nan)

#     # Get array of rhos, povms, and times

#     rhos, povms, times = jax.vmap(
#         lambda state: split_dataset(state, initial_states, povms)
#     )(X)

#     # sort them by time
#     rhos_sorted, povms_sorted, times_sorted, idx_sort, inv_sort = prepare_batch_for_ode(
#         rhos, povms, times
#     )
#     # Now we compute the evolved superoperator maps

#     array_superop_hat_col = evolve_map(
#         model, times_sorted, lindblad_generators=static_objects.lindblad_gens
#     )

#     # Now we compute the choi matrices of each map

#     array_unphysical_choi_row = jax.vmap(compute_choi_state_from_super_col)(
#         array_superop_hat_col
#     )

#     # array_projected_choi_row = jax.vmap(lambda choi_r: project_choi_row(choi_r, args))(
#     #     array_unphysical_choi_row
#     # )

#     # We check if the choi nees to be projected or not
#     array_physical_choi_row = jax.vmap(
#         lambda choi_r: make_physical_choi(choi_r, static_objects.choi_projection)
#     )(array_unphysical_choi_row)

#     # Now we need to compute the probabilities for each state and povm for the given times
#     # We need to be careful with the combinations of initial state povms and times

#     # the good thing is that we have the tuple (rhos_sorted, povms_sorted and array_projected_choi_row)
#     # where the order or the choi matrices is the same as the times, so we can just vmap
#     # the only problem is that each element of the povm array has 4 elements, so we need to compute it accordingly.

#     array_prob_hats_sorted = jax.vmap(
#         compute_probability_from_choi_array_povms, in_axes=(0, 0, 0)
#     )(array_physical_choi_row, rhos_sorted, povms_sorted)

#     # array_probs_hats_wrong_choi_sorted = jax.vmap(
#     #     compute_probability_from_choi_array_povms, in_axes=(0, 0, 0)
#     # )(array_unphysical_choi_row, rhos_sorted, povms_sorted)

#     # We need to give the original ordering back
#     array_prob_hats = array_prob_hats_sorted[inv_sort]

#     times_original_order = times_sorted[inv_sort]

#     # array_probs_hats_wrong_choi = array_probs_hats_wrong_choi_sorted[inv_sort]

#     # ---------------------------------- LOSS L1 --------------------------------- #

#     # network_output = jax.vmap(model)(times_sorted)

#     # --------------------------------- LOSS NLL --------------------------------- #
#     negative_ll_array, weights = weighted_multinomial_nll_from_probs(
#         array_prob_hats,
#         Y,
#         times_original_order,  # original-order times
#         weight_floor=weight_floor,
#         gamma_override=gamma_override,
#     )
#     # weights = weights + 1

#     loss_nll = (negative_ll_array * weights).mean()
#     # loss_nll = jnp.sum(negative_ll_array * weights) / jnp.sum(weights)

#     # loss_nll = multinomial_nll_from_probs(array_prob_hats, Y).mean()

#     # ---------------------------- LOSS CHOI MATRICES ---------------------------- #

#     # array_distance_choi = jax.vmap(
#     #     compute_distance_array_choi_matrices, in_axes=(0, 0)
#     # )(array_unphysical_choi_row, array_projected_choi_row)
#     # loss_cp = jnp.mean(array_distance_choi)

#     def distance_squared(a, b):
#         c = a - b
#         return jnp.trace(c @ c.conj().T).real

#     loss_cp = jax.vmap(distance_squared, in_axes=(0, 0))(
#         array_unphysical_choi_row, array_physical_choi_row
#     ).mean()
#     return (
#         loss_nll,
#         loss_cp,
#     )


class Intermediates(eqx.Module):
    times: jax.Array
    probs_hat: jax.Array
    unphysical_choi: jax.Array
    physical_choi: jax.Array


def compute_forward_and_intermediates(
    model,
    X,
    static_objects: StaticSetup,
) -> Intermediates:
    initial_states: InitialStates = static_objects.initial_states
    povms: POVMS = static_objects.povms

    # Get array of rhos, povms, and times
    rhos, povms, times = jax.vmap(
        lambda state: split_dataset(state, initial_states, povms)
    )(X)

    # sort them by time
    rhos_sorted, povms_sorted, times_sorted, idx_sort, inv_sort = prepare_batch_for_ode(
        rhos, povms, times
    )

    # Now we compute the evolved superoperator maps

    array_superop_hat_col = static_objects.evolve_map_fn(
        model, times_sorted, lindblad_generators=static_objects.lindblad_gens
    )

    # Now we compute the choi matrices of each map

    array_unphysical_choi_row = jax.vmap(compute_choi_state_from_super_col)(
        array_superop_hat_col
    )

    # array_projected_choi_row = jax.vmap(lambda choi_r: project_choi_row(choi_r, args))(
    #     array_unphysical_choi_row
    # )

    # We check if the choi nees to be projected or not
    array_physical_choi_row = jax.vmap(
        lambda choi_r: make_physical_choi(choi_r, static_objects.choi_projection)
    )(array_unphysical_choi_row)

    # Now we need to compute the probabilities for each state and povm for the given times
    # We need to be careful with the combinations of initial state povms and times

    # the good thing is that we have the tuple (rhos_sorted, povms_sorted and array_projected_choi_row)
    # where the order or the choi matrices is the same as the times, so we can just vmap
    # the only problem is that each element of the povm array has 4 elements, so we need to compute it accordingly.

    array_prob_hats_sorted = jax.vmap(
        compute_probability_from_choi_array_povms, in_axes=(0, 0, 0)
    )(array_physical_choi_row, rhos_sorted, povms_sorted)

    # array_probs_hats_wrong_choi_sorted = jax.vmap(
    #     compute_probability_from_choi_array_povms, in_axes=(0, 0, 0)
    # )(array_unphysical_choi_row, rhos_sorted, povms_sorted)

    # We need to give the original ordering back
    array_probs_hat = array_prob_hats_sorted[inv_sort]

    times_original_order = times_sorted[inv_sort]

    intermediates = Intermediates(
        times=times_original_order,
        probs_hat=array_probs_hat,
        unphysical_choi=array_unphysical_choi_row,
        physical_choi=array_physical_choi_row,
    )
    return intermediates


def loss_nll_from_probs(
    prob_hats: jnp.ndarray, Y: jnp.ndarray, times: jnp.ndarray, arg_loss: dict
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return scalar NLL (mean over batch) and per-example negative log-likelihoods"""
    weight_floor = arg_loss.get("short_time_weight_floor", 0.1)
    gamma_override = arg_loss.get("short_time_gamma", jnp.nan)
    negative_ll_array, weights = weighted_multinomial_nll_from_probs(
        prob_hats, Y, times, weight_floor=weight_floor, gamma_override=gamma_override
    )
    weighted_mean = (negative_ll_array * weights).mean()
    return weighted_mean, negative_ll_array


def loss_choi_distance(
    unphys_choi: jnp.ndarray, phys_choi: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Mean squared distance between unphysical and projected choi (scalar)"""

    def distance_squared(a, b):
        c = a - b
        return jnp.trace(c @ c.conj().T).real

    arr = jax.vmap(distance_squared, in_axes=(0, 0))(unphys_choi, phys_choi)
    return arr.mean(), arr


def compute_separate_losses(model, X, Y, static_objects: StaticSetup, arg_loss: dict):
    intermediate = compute_forward_and_intermediates(model, X, static_objects)

    loss_nll, _ = loss_nll_from_probs(
        intermediate.probs_hat, Y, intermediate.times, arg_loss
    )
    loss_choi, _ = loss_choi_distance(
        intermediate.unphysical_choi, intermediate.physical_choi
    )

    losses = [loss_nll, loss_choi]
    return jnp.array(losses)


def compute_loss(
    model, X, Y, loss_weights, static_objects: StaticSetup, arg_loss: dict
) -> tuple[Array, Array]:
    losses_unweighted = compute_separate_losses(
        model, X, Y, static_objects=static_objects, arg_loss=arg_loss
    )

    loss = jnp.dot(loss_weights, losses_unweighted)

    return loss, losses_unweighted


# compute_grad_loss = eqx.filter_grad(
#     compute_loss, has_aux=True
# )  # -> this returns the grad and the aux
compute_loss_and_grad = eqx.filter_value_and_grad(
    compute_loss, has_aux=True
)  # -> this returns (loss, separate_losses), and the gradient
# compute_grads_separate_losses = eqx.filter_jacrev(
#     compute_separate_losses,
# )


def filter_vgrad(f, x):
    y, vjp_fn = eqx.filter_vjp(f, x)
    return vjp_fn(jnp.ones(y.shape))[0]


@eqx.filter_jit
def update_loss_weights(model, x, y, loss_weights, static_objects, arg_loss):
    y, vjp_fn = eqx.filter_vjp(
        lambda m: compute_separate_losses(m, x, y, static_objects, arg_loss), model
    )
    n_losses = jnp.shape(y)[0]

    grads_losses = [vjp_fn(e_i) for e_i in jnp.identity(n_losses)]

    f_norm = lambda tree: jnp.linalg.norm(jax.flatten_util.ravel_pytree(tree)[0])

    norms_of_grads = jnp.array([f_norm(grad_i) for grad_i in grads_losses])

    sum_of_norms = norms_of_grads.sum()
    eps = 1e-10
    new_weights_iteration = sum_of_norms / (norms_of_grads + eps)

    alpha = arg_loss.get("loss_alpha", 0.9)
    old_weights = loss_weights
    new_weights = alpha * old_weights + (1 - alpha) * new_weights_iteration

    return new_weights


def compute_loss_in_batches(
    model, X, Y, loss_weights, static_objects, arg_loss, batch_size=1024
):
    n = X.shape[0]
    loss = []
    losses_unweighted = []

    for start in range(0, n, batch_size):
        end = start + batch_size
        X_batch = X[start:end]
        Y_batch = Y[start:end]

        loss_batch_i, losses_unweighted_batch_i = compute_loss(
            model, X_batch, Y_batch, loss_weights, static_objects, arg_loss
        )
        loss.append(loss_batch_i)
        losses_unweighted.append(losses_unweighted_batch_i)

    loss = jnp.array(loss).mean(0)
    losses_unweighted = jnp.array(losses_unweighted).mean(0)
    return loss, losses_unweighted


@eqx.filter_jit
def make_train_step(
    model, x, y, loss_weights, opt_state, optimizer, static_objects, arg_loss
):
    (loss_value, separate_losses), grad_loss_value = compute_loss_and_grad(
        model, x, y, loss_weights, static_objects, arg_loss
    )
    updates, opt_state = optimizer.update(
        grad_loss_value, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, (loss_value, separate_losses)
