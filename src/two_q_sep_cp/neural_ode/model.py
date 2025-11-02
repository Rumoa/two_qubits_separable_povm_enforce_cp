import abc
import math
from functools import partial
from typing import Callable, Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Complex, Float

from two_q_sep_cp.neural_ode.cp_projection import ChoiProjection
from two_q_sep_cp.neural_ode.representations import (
    convert_col_row_super,
    from_super_to_choi,
)
from two_q_sep_cp.neural_ode.utils import (
    POVMS,
    InitialStates,
    LindbladGenerators,
    Parameters,
    prepare_batch_for_ode,
    split_dataset,
)


def get_params(model):
    return eqx.filter(model, eqx.is_array)


class AbstractLindbladModel(eqx.Module):
    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, x) -> Parameters:
        raise NotImplementedError


class LindbladNetTwoQubits(AbstractLindbladModel):
    dimension: int
    mlp: eqx.nn.MLP
    total_number_of_parameters: int

    def __init__(
        self,
        key,
        width_size=256,
        depth=10,
    ):
        self.dimension = 4
        self.total_number_of_parameters = self.dimension**2 * (self.dimension**2 - 1)
        self.mlp = eqx.nn.MLP(
            in_size=1,
            out_size=self.total_number_of_parameters,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(self, t) -> Parameters:
        m = self.dimension**2 - 1
        combinations = m * (m - 1) // 2
        # Ensure t is an array of shape (1,) for the MLP.
        scale_pars = 1e0
        scaled_t = 1e0
        t = t * scaled_t
        bare_output_parameters = self.mlp(jnp.array([t])) * scale_pars

        h_pars = bare_output_parameters[0:m]
        s_pars = bare_output_parameters[m : 2 * m]
        c_pars = bare_output_parameters[2 * m : 2 * m + combinations]
        a_pars = bare_output_parameters[2 * m + combinations :]
        wrapped_parameters = Parameters(self.dimension, h_pars, s_pars, c_pars, a_pars)

        return wrapped_parameters


class FourierLindbladNetTwoQubits(AbstractLindbladModel):
    """MLP that takes time t (assumed normalized to (0,1]) and uses Fourier features."""

    dimension: int
    mlp: eqx.nn.MLP
    total_number_of_parameters: int
    num_bands: int
    max_freq: float
    include_input: bool
    scale_pars: float
    scaled_t: float

    def __init__(
        self,
        key,
        width_size: int = 256,
        depth: int = 6,
        num_bands: int = 32,
        max_freq: float = 100.0,
        include_input: bool = True,
        scale_pars: float = 1.0,
        scaled_t: float = 1.0,
    ):
        # physics dims
        self.dimension = 4
        self.total_number_of_parameters = self.dimension**2 * (self.dimension**2 - 1)

        # embedding params
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.include_input = include_input
        self.scale_pars = float(scale_pars)
        self.scaled_t = float(scaled_t)

        # compute embedding size: optionally raw t + 2*num_bands (sin+cos)
        embed_dim = (1 if include_input else 0) + 2 * num_bands

        # create MLP with input size = embedding dim, output size = number of parameters
        self.mlp = eqx.nn.MLP(
            in_size=embed_dim,
            out_size=self.total_number_of_parameters,
            width_size=width_size,
            depth=depth,
            key=key,
            # activation=jax.nn.tanh,
            activation=jax.nn.silu,
        )

    @staticmethod
    def _fourier_time_embedding(
        t: jnp.ndarray, num_bands: int, max_freq: float, include_input: bool
    ):
        """
        t: shape (N,) or (,) values assumed in (0,1] (but function is robust).
        returns: embeddings of shape (N, embed_dim)
        """
        t = jnp.reshape(t, (-1, 1))  # shape (N,1)

        # geometric/log spaced cycle frequencies in [1, max_freq]
        freqs = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(max_freq), num_bands))
        omegas = 2.0 * jnp.pi * freqs  # angular frequencies

        # compute arguments (N, num_bands)
        arg = t * omegas  # broadcasting

        sin_feats = jnp.sin(arg)
        cos_feats = jnp.cos(arg)
        feats = jnp.concatenate([sin_feats, cos_feats], axis=-1)  # (N, 2*num_bands)

        # scale variance similar to NeRF / Tancik
        feats = feats * jnp.sqrt(2.0 / num_bands)

        if include_input:
            feats = jnp.concatenate([t, feats], axis=-1)  # (N, 1 + 2*num_bands)

        return feats  # shape (N, embed_dim)

    def __call__(self, t) -> "Parameters":
        """
        t may be a scalar or a 1-element array. This function returns the Parameters object
        corresponding to the network output for that single time. (If you need batched
        evaluation, wrap this call with vmap externally.)
        """
        # ensure numeric JAX array (no python branching) and apply user scaling of t
        t_arr = jnp.asarray(t, dtype=jnp.float32) * self.scaled_t

        # create embedding (returns shape (1, embed_dim)); then take first row
        emb = self._fourier_time_embedding(
            t_arr, self.num_bands, self.max_freq, self.include_input
        )[0]

        # forward through MLP and scale outputs
        bare_output_parameters = (
            self.mlp(emb) * self.scale_pars
        )  # shape (total_number_of_parameters,)

        # slicing into parameter blocks (same as your original code)
        m = self.dimension**2 - 1
        combinations = m * (m - 1) // 2

        # slices (all are jnp arrays)
        h_pars = bare_output_parameters[0:m]
        s_pars = bare_output_parameters[m : 2 * m]
        c_pars = bare_output_parameters[2 * m : 2 * m + combinations]
        a_pars = bare_output_parameters[2 * m + combinations :]

        wrapped_parameters = Parameters(self.dimension, h_pars, s_pars, c_pars, a_pars)

        return wrapped_parameters


class SineMLP(eqx.Module):
    """SIREN-style MLP implemented with explicit weight & bias arrays so that we
    can use SIREN initializers without depending on eqx.nn.Linear's constructor API.
    - layer_sizes: list[int], e.g. [1, 128, 128, out]
    - omega_0: frequency scaling applied to first-layer pre-activation.
    """

    weights: list  # list of arrays shaped (out_dim, in_dim)
    biases: list  # list of arrays shaped (out_dim,)
    omega_0: float

    def __init__(self, key, layer_sizes, omega_0: float = 30.0, dtype=jnp.float64):
        keys = jr.split(key, len(layer_sizes) - 1)
        weights = []
        biases = []
        for i in range(len(layer_sizes) - 1):
            in_dim = int(layer_sizes[i])
            out_dim = int(layer_sizes[i + 1])
            k = keys[i]

            # SIREN init rules:
            if i == 0:
                # first layer: uniform(-1/in_dim, 1/in_dim)
                bound = 1.0 / in_dim
            else:
                # hidden layers: uniform(-sqrt(6/fan_in)/omega_0, sqrt(6/fan_in)/omega_0)
                bound = math.sqrt(6.0 / in_dim) / omega_0

            w = jr.uniform(
                k, shape=(out_dim, in_dim), minval=-bound, maxval=bound, dtype=dtype
            )
            # tiny bias init near zero
            b = jr.uniform(k, shape=(out_dim,), minval=-1e-6, maxval=1e-6, dtype=dtype)

            # cast to proper dtype and append
            weights.append(jnp.array(w, dtype=dtype))
            biases.append(jnp.array(b, dtype=dtype))

        self.weights = weights
        self.biases = biases
        self.omega_0 = float(omega_0)

    def __call__(self, x):
        # x expected scalar or 1D array of length input_dim (input_dim == 1 for your case)
        h = jnp.asarray(x)
        if h.ndim == 0:
            h = jnp.reshape(h, (1,))  # shape (1,)

        # iterate all but last layer with sine activations
        for i in range(len(self.weights) - 1):
            W = self.weights[i]  # (out, in)
            b = self.biases[i]  # (out,)
            pre = W @ h + b  # shape (out,)
            if i == 0:
                # scale first layer pre-activation by omega_0
                h = jnp.sin(self.omega_0 * pre)
            else:
                h = jnp.sin(pre)

        # final linear layer (no activation)
        W_last = self.weights[-1]
        b_last = self.biases[-1]
        out = W_last @ h + b_last
        return out


class RFLindbladNetTwoQubits(AbstractLindbladModel):
    """
    Random Fourier Feature embedding for 1D time t:
      z(t) = sqrt(2/D) * cos(w * t + b)
    where w ~ Normal(0, sigma^2) and b ~ Uniform(0, 2*pi).
    Optionally include raw t as an extra input dimension (stacked).
    """

    dimension: int
    mlp: eqx.nn.MLP
    total_number_of_parameters: int
    num_features: int
    sigma: float
    include_input: bool
    # train-time-constant random features (sampled at init)
    w: jnp.ndarray  # shape (num_features,)
    b: jnp.ndarray  # shape (num_features,)
    scale_pars: float
    scaled_t: float
    trainable_features: bool

    def __init__(
        self,
        key,
        width_size: int = 256,
        depth: int = 6,
        num_features: int = 128,
        sigma: float = 10.0,
        include_input: bool = True,
        scale_pars: float = 1.0,
        scaled_t: float = 1.0,
        trainable_features: bool = False,
    ):
        # physics dims
        self.dimension = 4
        self.total_number_of_parameters = self.dimension**2 * (self.dimension**2 - 1)

        # embedding params
        self.num_features = int(num_features)
        self.sigma = float(sigma)
        self.include_input = bool(include_input)
        self.scale_pars = float(scale_pars)
        self.scaled_t = float(scaled_t)
        self.trainable_features = trainable_features

        (lambda a: jax.lax.stop_gradient(a) if self.trainable_features else lambda a: a)

        # sample random features once at init (deterministic thereafter)
        k_w, k_b, k_mlp = jax.random.split(key, 3)
        # w ~ Normal(0, sigma^2)
        self.w = jax.random.normal(k_w, (self.num_features,)) * self.sigma
        # random phase b ~ Uniform(0, 2pi)
        self.b = jax.random.uniform(k_b, (self.num_features,)) * (2.0 * jnp.pi)

        # embedding dimension: optionally raw t + num_features
        embed_dim = (1 if self.include_input else 0) + self.num_features

        # create MLP with input size = embedding dim, output size = number of parameters
        self.mlp = eqx.nn.MLP(
            in_size=embed_dim,
            out_size=self.total_number_of_parameters,
            width_size=width_size,
            depth=depth,
            key=k_mlp,
            activation=jax.nn.silu,
        )

    @staticmethod
    def _rff_time_embedding(
        t: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray, include_input: bool
    ):
        """
        t: shape (N,) or (,) values.
        w: shape (num_features,)
        b: shape (num_features,)
        returns: embeddings shape (N, embed_dim)
        """
        t = jnp.reshape(t, (-1, 1))  # (N,1)
        # Compute (N, num_features): broadcasting multiplication
        arg = t * w + b  # (N, num_features)
        # RFF single-cosine version (random-phase)
        feats = jnp.cos(arg)  # (N, num_features)
        # scale like sqrt(2/D) to keep variance ~1 for kernel approx
        feats = feats * jnp.sqrt(2.0 / w.shape[0])

        if include_input:
            feats = jnp.concatenate([t, feats], axis=-1)  # (N, 1 + num_features)
        return feats

    def __call__(self, t) -> Parameters:
        """
        t may be a scalar or a 1-element array. Returns Parameters for that single time.
        (If you want batched evaluation, vmap over this call.)
        """
        # ensure JAX array and apply optional scaling
        t_arr = jnp.asarray(t, dtype=jnp.float32) * self.scaled_t

        # embedding returns shape (N, embed_dim); take first (and only) row

        w_features = jax.lax.cond(
            self.trainable_features,
            lambda a: a,
            lambda a: jax.lax.stop_gradient(a),
            self.w,
        )
        b_features = jax.lax.cond(
            self.trainable_features,
            lambda a: a,
            lambda a: jax.lax.stop_gradient(a),
            self.b,
        )

        emb = self._rff_time_embedding(
            t_arr,
            w_features,  # <- because we don't want to train the rff embedding
            b_features,  # <- because we don't want to train the rff embedding
            self.include_input,
        )[0]

        # forward through MLP and scale outputs
        bare_output_parameters = self.mlp(emb) * self.scale_pars

        # slicing into parameter blocks (same as your original code)
        m = self.dimension**2 - 1
        combinations = m * (m - 1) // 2

        h_pars = bare_output_parameters[0:m]
        s_pars = bare_output_parameters[m : 2 * m]
        c_pars = bare_output_parameters[2 * m : 2 * m + combinations]
        a_pars = bare_output_parameters[2 * m + combinations :]

        wrapped_parameters = Parameters(self.dimension, h_pars, s_pars, c_pars, a_pars)

        return wrapped_parameters


class SineLindbladNetTwoQubits(AbstractLindbladModel):
    """SIREN-based Lindblad net compatible with your Parameters slicing."""

    dimension: int
    mlp: SineMLP
    total_number_of_parameters: int
    omega_0: float
    scale_pars: float
    scaled_t: float

    def __init__(
        self,
        key,
        width_size: int = 128,
        depth: int = 4,
        omega_0: float = 8.0,
        scale_pars: float = 0.1,
        scaled_t: float = 1.0,
    ):
        self.dimension = 4
        self.total_number_of_parameters = self.dimension**2 * (self.dimension**2 - 1)
        self.omega_0 = float(omega_0)
        self.scale_pars = float(scale_pars)
        self.scaled_t = float(scaled_t)

        # build MLP layer sizes: input 1, depth hidden layers, final output
        layer_sizes = [1] + [width_size] * depth + [self.total_number_of_parameters]
        self.mlp = SineMLP(key, layer_sizes=layer_sizes, omega_0=self.omega_0)

    def __call__(self, t) -> Parameters:
        # t -> numeric, scale and forward
        t_arr = jnp.asarray(t, dtype=jnp.float64) * self.scaled_t
        emb = (
            jnp.reshape(t_arr, (-1,))[0] if jnp.ndim(t_arr) == 1 else t_arr
        )  # scalar as array->scalar
        bare_output_parameters = (self.mlp(emb) * self.scale_pars).reshape(-1)

        # slicing into parameter blocks (same as your other models)
        m = self.dimension**2 - 1
        combinations = m * (m - 1) // 2

        h_pars = bare_output_parameters[0:m]
        s_pars = bare_output_parameters[m : 2 * m]
        c_pars = bare_output_parameters[2 * m : 2 * m + combinations]
        a_pars = bare_output_parameters[2 * m + combinations :]

        wrapped_parameters = Parameters(self.dimension, h_pars, s_pars, c_pars, a_pars)
        return wrapped_parameters


def make_model(config, key="architecture"):
    # Check that the key exists
    if key not in config:
        raise KeyError(f"Config is missing required key '{key}'")

    arch = config[key]

    depth = config.get("depth", 6)
    width = config.get("width", 128)
    num_bands = config.get("num_bands", 6)
    seed = config.get("model_seed", 0)
    num_features = config.get("rf_num_features", 128)
    sigma = config.get("rf_sigma", 10.0)
    trainable_fourier_features = config.get("trainable_fourier_features", True)

    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)

    architectures = {
        "simple_mlp": lambda key: LindbladNetTwoQubits(
            key, width_size=width, depth=depth
        ),
        "fourier_mlp": lambda key: FourierLindbladNetTwoQubits(
            key, width_size=width, depth=depth, num_bands=num_bands
        ),
        "sinemlp": lambda key: SineLindbladNetTwoQubits(
            key,
            width_size=width,
            depth=depth,
        ),
        "random_fourier_mlp": lambda key: RFLindbladNetTwoQubits(
            key,
            width_size=width,
            depth=depth,
            num_features=num_features,
            sigma=sigma,
            trainable_features=trainable_fourier_features,
        ),
    }

    return key, architectures[arch](subkey)


# ---------------------------------------------------------------------------- #
#                                   ODE PART                                   #
# ---------------------------------------------------------------------------- #


class ArgsODE(eqx.Module):
    model: AbstractLindbladModel
    lindblad_generators: LindbladGenerators


# def ode_master_eqn_map(
#     t: Float, superoperator: Complex[Array, " 16 16"], args
# ) -> Complex[Array, " 16 16"]:
#     model = args["model"]
#     l_gens: LindbladGenerators = args["lindblad_generators"]

#     parameters_at_t: Parameters = model(t)  # parameters

#     # construct the lindbladian

#     lindbladian = l_gens.make_lindbladian(parameters_at_t)

#     superoperator_prime = lindbladian @ superoperator

#     return superoperator_prime


# GLOBAL_ODE_SOLVER = diffrax.Tsit5()
# GLOBAL_ODE_TERM_MAP = diffrax.ODETerm(ode_master_eqn_map)


@eqx.filter_jit
def compute_choi_state_from_super_col(
    superoperator_col: Complex[Array, " 16 16"],
) -> Complex[Array, " 16 16"]:
    # The superoperator col needs to be converted to superoperator row convention

    superoperator_row = convert_col_row_super(superoperator_col)

    choi_unnormalized_state = from_super_to_choi(
        superoperator_row, order="row"
    )  # CAUTION. This computes the choi using the unnormalized maximally mixed state. We need to divide by the dimension

    dimension_subsystem = int(np.sqrt(choi_unnormalized_state.shape[0]))
    choi_normalized = choi_unnormalized_state / dimension_subsystem
    return choi_normalized


# @eqx.filter_jit
@partial(jax.jit, static_argnames="args")
def project_choi_row(
    unphysical_choi_row: Complex[Array, " 16 16"], args: dict
) -> Complex[Array, " 16 16"]:
    choi_proj: ChoiProjection = args["choi_projection"]

    projected_choi_row = choi_proj.dykstraCBA(
        unphysical_choi_row, max_iter=4, tol=1e-4
    )  # original one, we fixed max_iter = 3, tol=1e-3
    return projected_choi_row


@eqx.filter_jit
def compute_probability_from_choi(
    normalized_choi_state_row: Complex[Array, " d^2 d^2"],
    rho: Complex[Array, " d d"],
    povm_element: Complex[Array, " d d"],
) -> Float:
    # We assume that we are working in the following convention.
    # The choi state has been created as (map tensor id) 1/d (unnormalized_max_entangled)
    # which means that the choi is a valid state.
    # this is important bc it affects how we plug rho and povm when computing the trace
    d = rho.shape[0]

    prob = (
        jnp.trace(jnp.kron(povm_element, rho.T) @ normalized_choi_state_row) * d
    ).real

    # prob = jnp.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)

    # We clip now the probabilities
    prob = jnp.clip(prob, min=1e-12, max=1)
    return prob


@eqx.filter_jit
def compute_probability_from_choi_array_povms(
    choi: Complex[Array, " d^2 d^2"],
    rho: Complex[Array, " d d"],
    array_povms: Complex[Array, " a d d"],
) -> Float[Array, " a"]:
    def f_scan(carry, x):
        return carry, compute_probability_from_choi(choi, rho, x)

    return jax.lax.scan(f_scan, init=jnp.array(0), xs=array_povms)[1]


def make_physical_choi(choi, choi_projection):
    """Checks if we need to project the choi matrix or it is already a valid
    physical one

    Args:
        choi (_type_): _description_
        choi_projection (_type_): _description_
    """

    def is_valid_choi(choi):
        evals = jnp.linalg.eigh(choi)[0]
        return ~((evals < 0).any())

    def f_project(choi):
        return choi_projection.dykstraCBA(choi, max_iter=3, tol=1e-3)

    def f_dont_project(choi):
        return choi

    return jax.lax.cond(is_valid_choi(choi), f_dont_project, f_project, choi)


# ---------------------------------------------------------------------------- #
#                                LOSS FUNCTIONS                                #
# ---------------------------------------------------------------------------- #


def compute_distance_array_choi_matrices(
    array_unphysical_choi: Complex[Array, "d_batch 16 16"],
    array_projected_choi: Complex[Array, "d_batch 16 16"],
) -> Float[Array, " d_batch"]:
    D = array_unphysical_choi - array_projected_choi
    # per-sample squared Frobenius norm (real)
    per_sample_sq = jnp.sum(jnp.abs(D) ** 2, axis=(1, 2)).real
    # Return per-sample distances OR a single loss:
    # return per_sample_sq           # per-sample
    return per_sample_sq  # scalar loss


def multinomial_nll_from_probs(predicted_probs, counts):
    """Use if your model returns probabilities (not logits)."""
    eps = 1e-12
    predicted_probs = jnp.clip(predicted_probs, a_min=eps, a_max=1.0)
    nll = -jnp.sum(counts * jnp.log(predicted_probs), axis=-1)
    return nll


def weighted_multinomial_nll_from_probs(
    predicted_probs, counts, times, weight_floor=0.1, gamma_override=jnp.nan, eps=1e-12
):
    """Compute per-sample NLL and exponential time-weights w(t)=exp(-gamma t).

    - `weight_floor` is the desired weight at t = max(times), i.e. w(t_max)=weight_floor.
    - If `gamma_override` is finite (not NaN), it is used directly; otherwise gamma is computed
      so that w(t_max)=weight_floor.
      gamma_override=0 disables weighting
    All branching is done with JAX ops (jnp.where / jnp.isfinite) so this is jittable.
    """
    # per-sample NLL
    nll = multinomial_nll_from_probs(
        predicted_probs,
        counts,
    )

    # normalize times shape to (N,)

    times = jnp.reshape(jnp.asarray(times), (-1,))

    def f_to_weight(operand):
        times, _ = operand
        tmax = jnp.max(times)

        # avoid division-by-zero: if tmax == 0 -> set denom to 1 (we then set gamma=0 via where)
        denom = jnp.where(tmax <= 0.0, 1.0, tmax)
        gamma_calc = -jnp.log(jnp.asarray(weight_floor)) / denom
        # if all times are zero, set gamma to 0
        gamma_calc = jnp.where(tmax <= 0.0, 0.0, gamma_calc)
        return jnp.array(gamma_calc, dtype=jnp.float64)

    def f_to_use_gamma(operand):
        _, gamma_override = operand
        gamma = gamma_override
        return jnp.array(gamma, dtype=jnp.float64)

    gamma = jax.lax.cond(
        jnp.isfinite(jnp.asarray(gamma_override)),
        f_to_use_gamma,
        f_to_weight,
        operand=(times, gamma_override),
    )

    # # compute gamma that gives w(tmax)=weight_floor: gamma = -ln(weight_floor)/tmax

    # # use override if provided (must pass a numeric; pass jnp.nan to skip)
    # gamma = jnp.where(
    #     jnp.isfinite(jnp.asarray(gamma_override)), gamma_override, gamma_calc
    # )

    weights = jnp.exp(-gamma * times)  # shape (N,)

    return nll, weights


class StaticSetup(eqx.Module):
    initial_states: InitialStates
    povms: POVMS
    lindblad_gens: LindbladGenerators
    choi_projection: ChoiProjection
    evolve_map_fn: Callable


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


# def compute_forward_and_intermediates(
#     model,
#     X,
#     static_objects: StaticSetup,
# ) -> Intermediates:
#     initial_states: InitialStates = static_objects.initial_states
#     povms: POVMS = static_objects.povms

#     # Get array of rhos, povms, and times
#     rhos, povms, times = jax.vmap(
#         lambda state: split_dataset(state, initial_states, povms)
#     )(X)

#     # sort them by time
#     rhos_sorted, povms_sorted, times_sorted, idx_sort, inv_sort = prepare_batch_for_ode(
#         rhos, povms, times
#     )

#     # Now we compute the evolved superoperator maps

#     array_superop_hat_col = static_objects.evolve_map_fn(
#         model, times_sorted, lindblad_generators=static_objects.lindblad_gens
#     )

#     # Now we compute the choi matrices of each map

#     array_unphysical_choi_row = jax.vmap(compute_choi_state_from_super_col)(
#         array_superop_hat_col
#     )

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
#     )(array_physical_choi_row, rhos_sorted, povms_sorted)  # <- original one

#     # We need to give the original ordering back
#     array_probs_hat = array_prob_hats_sorted[inv_sort]

#     times_original_order = times_sorted[inv_sort]

#     intermediates = Intermediates(
#         times=times_original_order,
#         probs_hat=array_probs_hat,
#         unphysical_choi=array_unphysical_choi_row,
#         physical_choi=array_physical_choi_row,
#     )
#     return intermediates


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


# # ---------------------------- EXAMPLE UPDATE STEP --------------------------- #


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
