import math
import operator
from typing import Any, Callable

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

from .utils import (
    POVMS,
    InitialStates,
    LindbladGenerators,
    Parameters,
    prepare_batch_for_ode,
    split_dataset,
)


def get_params(model):
    return eqx.filter(model, eqx.is_array)


class LindbladNetTwoQubits(eqx.Module):
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


class FourierLindbladNetTwoQubits(eqx.Module):
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


# class RFLindbladNetTwoQubits(eqx.Module):
#     """
#     Random Fourier Feature embedding for 1D time t:
#       z(t) = sqrt(2/D) * cos(w * t + b)
#     where w ~ Normal(0, sigma^2) and b ~ Uniform(0, 2*pi).
#     Optionally include raw t as an extra input dimension (stacked).
#     """

#     dimension: int
#     mlp: eqx.nn.MLP
#     total_number_of_parameters: int
#     num_features: int
#     sigma: float
#     include_input: bool
#     # train-time-constant random features (sampled at init)
#     w: jnp.ndarray  # shape (num_features,)
#     b: jnp.ndarray  # shape (num_features,)
#     scale_pars: float
#     scaled_t: float

#     def __init__(
#         self,
#         key,
#         width_size: int = 256,
#         depth: int = 6,
#         num_features: int = 128,
#         sigma: float = 10.0,
#         include_input: bool = True,
#         scale_pars: float = 1.0,
#         scaled_t: float = 1.0,
#     ):
#         # physics dims
#         self.dimension = 4
#         self.total_number_of_parameters = self.dimension**2 * (self.dimension**2 - 1)

#         # embedding params
#         self.num_features = int(num_features)
#         self.sigma = float(sigma)
#         self.include_input = bool(include_input)
#         self.scale_pars = float(scale_pars)
#         self.scaled_t = float(scaled_t)

#         # sample random features once at init (deterministic thereafter)
#         k_w, k_b, k_mlp = jax.random.split(key, 3)
#         # w ~ Normal(0, sigma^2)
#         self.w = jax.random.normal(k_w, (self.num_features,)) * self.sigma
#         # random phase b ~ Uniform(0, 2pi)
#         self.b = jax.random.uniform(k_b, (self.num_features,)) * (2.0 * jnp.pi)

#         # embedding dimension: optionally raw t + num_features
#         embed_dim = (1 if self.include_input else 0) + self.num_features

#         # create MLP with input size = embedding dim, output size = number of parameters
#         self.mlp = eqx.nn.MLP(
#             in_size=embed_dim,
#             out_size=self.total_number_of_parameters,
#             width_size=width_size,
#             depth=depth,
#             key=k_mlp,
#             activation=jax.nn.silu,
#         )

#     @staticmethod
#     def _rff_time_embedding(
#         t: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray, include_input: bool
#     ):
#         """
#         t: shape (N,) or (,) values.
#         w: shape (num_features,)
#         b: shape (num_features,)
#         returns: embeddings shape (N, embed_dim)
#         """
#         t = jnp.reshape(t, (-1, 1))  # (N,1)
#         # Compute (N, num_features): broadcasting multiplication
#         arg = t * w + b  # (N, num_features)
#         # RFF single-cosine version (random-phase)
#         feats = jnp.cos(arg)  # (N, num_features)
#         # scale like sqrt(2/D) to keep variance ~1 for kernel approx
#         feats = feats * jnp.sqrt(2.0 / w.shape[0])

#         if include_input:
#             feats = jnp.concatenate([t, feats], axis=-1)  # (N, 1 + num_features)
#         return feats

#     def __call__(self, t) -> "Parameters":
#         """
#         t may be a scalar or a 1-element array. Returns Parameters for that single time.
#         (If you want batched evaluation, vmap over this call.)
#         """
#         # ensure JAX array and apply optional scaling
#         t_arr = jnp.asarray(t, dtype=jnp.float32) * self.scaled_t

#         # embedding returns shape (N, embed_dim); take first (and only) row
#         # emb = self._rff_time_embedding(
#         #     t_arr,
#         #     jax.lax.stop_gradient(
#         #         self.w
#         #     ),  # <- because we don't want to train the rff embedding
#         #     jax.lax.stop_gradient(
#         #         self.b
#         #     ),  # <- because we don't want to train the rff embedding
#         #     self.include_input,
#         # )[0]

#         emb = self._rff_time_embedding(
#             t_arr,
#             (self.w),  # <- because we don't want to train the rff embedding
#             (self.b),  # <- because we don't want to train the rff embedding
#             self.include_input,
#         )[0]

#         # forward through MLP and scale outputs
#         bare_output_parameters = self.mlp(emb) * self.scale_pars

#         # slicing into parameter blocks (same as your original code)
#         m = self.dimension**2 - 1
#         combinations = m * (m - 1) // 2

#         h_pars = bare_output_parameters[0:m]
#         s_pars = bare_output_parameters[m : 2 * m]
#         c_pars = bare_output_parameters[2 * m : 2 * m + combinations]
#         a_pars = bare_output_parameters[2 * m + combinations :]

#         wrapped_parameters = Parameters(self.dimension, h_pars, s_pars, c_pars, a_pars)

#         return wrapped_parameters


class RFLindbladNetTwoQubits(eqx.Module):
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

    def __call__(self, t) -> "Parameters":
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


class SineLindbladNetTwoQubits(eqx.Module):
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

    def __call__(self, t) -> "Parameters":
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


def ode_master_eqn_map(
    t: Float, superoperator: Complex[Array, " 16 16"], args
) -> Complex[Array, " 16 16"]:
    model = args["model"]
    l_gens: LindbladGenerators = args["lindblad_generators"]

    parameters_at_t: Parameters = model(t)  # parameters

    # construct the lindbladian

    lindbladian = l_gens.make_lindbladian(parameters_at_t)

    superoperator_prime = lindbladian @ superoperator

    return superoperator_prime


GLOBAL_ODE_SOLVER = diffrax.Tsit5()
GLOBAL_ODE_TERM_MAP = diffrax.ODETerm(ode_master_eqn_map)


@eqx.filter_jit
def evolve_map(
    model, ts: Float[Array, " dim_batch"], ode_args: dict
) -> Complex[Array, " dim_batch 16 16"]:
    # Include the model in the args dictionary
    ode_args["model"] = model
    solver = GLOBAL_ODE_SOLVER
    term = GLOBAL_ODE_TERM_MAP

    # y0 is the identity map. for the case of two qubits, the superoperator has dimension 16x16

    y0 = jnp.identity(16, dtype=jnp.complex128)

    # stepsize_controller = PIDController(rtol=1e-9, atol=1e-12)

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=jnp.array(0),
        t1=ts[-1],
        y0=y0,
        args=ode_args,
        dt0=0.001,
        saveat=diffrax.SaveAt(ts=ts),
    )
    array_maps_superop_col_order = solution.ys
    return (
        array_maps_superop_col_order  # shape (n_times, 16, 16) dtype = jnp.complex128
    )


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


@eqx.filter_jit
def project_choi_row(
    unphysical_choi_row: Complex[Array, " 16 16"], args: dict
) -> Complex[Array, " 16 16"]:
    choi_proj: ChoiProjection = args["choi_projection"]

    projected_choi_row = choi_proj.dykstraCBA(unphysical_choi_row, max_iter=3, tol=1e-3)
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


# ---------------------------------------------------------------------------- #
#                                LOSS FUNCTIONS                                #
# ---------------------------------------------------------------------------- #


def compute_distance_array_choi_matrices(
    array_unphysical_choi: Complex[Array, "d_batch 16 16"],
    array_projected_choi: Complex[Array, "d_batch 16 16"],
) -> Float[Array, " d_batch"]:
    array_norms = jax.vmap(jnp.linalg.norm)(
        array_unphysical_choi - array_projected_choi
    )
    loss = jnp.mean(array_norms)
    return loss


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
    All branching is done with JAX ops (jnp.where / jnp.isfinite) so this is jittable.
    """
    # per-sample NLL
    nll = multinomial_nll_from_probs(
        predicted_probs,
        counts,
    )

    # normalize times shape to (N,)
    times = jnp.reshape(jnp.asarray(times), (-1,))

    # compute gamma that gives w(tmax)=weight_floor: gamma = -ln(weight_floor)/tmax
    tmax = jnp.max(times)
    # avoid division-by-zero: if tmax == 0 -> set denom to 1 (we then set gamma=0 via where)
    denom = jnp.where(tmax <= 0.0, 1.0, tmax)
    gamma_calc = -jnp.log(jnp.asarray(weight_floor)) / denom
    # if all times are zero, set gamma to 0
    gamma_calc = jnp.where(tmax <= 0.0, 0.0, gamma_calc)

    # use override if provided (must pass a numeric; pass jnp.nan to skip)
    gamma = jnp.where(
        jnp.isfinite(jnp.asarray(gamma_override)), gamma_override, gamma_calc
    )

    weights = jnp.exp(-gamma * times)  # shape (N,)

    return nll, weights


# compute_loss_and_grad = eqx.filter_value_and_grad(compute_loss)


@eqx.filter_jit
def compute_separate_losses(
    model, X: Float[Array, "d_batch 3"], Y: Float[Array, "d_batch 4"], args
):
    initial_states: InitialStates = args["initial_states"]
    povms: POVMS = args["povms"]

    # weight_floor = args.get("short_time_weight_floor", 0.1)
    # gamma_override = args.get("short_time_gamma", jnp.nan)

    # Get array of rhos, povms, and times

    rhos, povms, times = jax.vmap(
        lambda state: split_dataset(state, initial_states, povms)
    )(X)

    # sort them by time
    rhos_sorted, povms_sorted, times_sorted, idx_sort, inv_sort = prepare_batch_for_ode(
        rhos, povms, times
    )

    # Now we compute the evolved superoperator maps

    array_superop_hat_col = evolve_map(model, times_sorted, ode_args=args)

    # Now we compute the choi matrices of each map

    array_unphysical_choi_row = jax.vmap(compute_choi_state_from_super_col)(
        array_superop_hat_col
    )

    array_projected_choi_row = jax.vmap(lambda choi_r: project_choi_row(choi_r, args))(
        array_unphysical_choi_row
    )

    # Now we need to compute the probabilities for each state and povm for the given times
    # We need to be careful with the combinations of initial state povms and times

    # the good thing is that we have the tuple (rhos_sorted, povms_sorted and array_projected_choi_row)
    # where the order or the choi matrices is the same as the times, so we can just vmap
    # the only problem is that each element of the povm array has 4 elements, so we need to compute it accordingly.

    array_prob_hats_sorted = jax.vmap(
        compute_probability_from_choi_array_povms, in_axes=(0, 0, 0)
    )(array_projected_choi_row, rhos_sorted, povms_sorted)

    # array_probs_hats_wrong_choi_sorted = jax.vmap(
    #     compute_probability_from_choi_array_povms, in_axes=(0, 0, 0)
    # )(array_unphysical_choi_row, rhos_sorted, povms_sorted)

    # We need to give the original ordering back
    array_prob_hats = array_prob_hats_sorted[inv_sort]

    # array_probs_hats_wrong_choi = array_probs_hats_wrong_choi_sorted[inv_sort]

    # --------------------------------- LOSS NLL --------------------------------- #
    # negative_ll_array, weights = weighted_multinomial_nll_from_probs(
    #     array_prob_hats,
    #     Y,
    #     times,  # original-order times
    #     weight_floor=weight_floor,
    #     gamma_override=gamma_override,
    # )

    # loss_nll = jnp.sum(negative_ll_array * weights) / jnp.sum(weights)

    loss_nll = multinomial_nll_from_probs(array_prob_hats, Y).mean()

    # ---------------------------- LOSS CHOI MATRICES ---------------------------- #

    array_distance_choi = jax.vmap(
        compute_distance_array_choi_matrices, in_axes=(0, 0)
    )(array_unphysical_choi_row, array_projected_choi_row)
    loss_cp = jnp.mean(array_distance_choi)

    return (
        loss_nll,
        loss_cp,
    )


# class LossWeights(eqx.Module):
#     weight_nll: jnp.array
#     weight_choi: jnp.array


class LossWeights(eqx.Module):
    weights: jnp.array

    def __init__(self, weights=jnp.array([1.0, 1.0], dtype=jnp.float64)):
        self.weights = jnp.array(weights, dtype=jnp.float64)

    @property
    def w_nll(self):
        return self.weights[0]

    @property
    def w_cp(self):
        return self.weights[1]


# class SeparateLosses(eqx.Module):
#     loss_nll: Array
#     loss_choi: Array

#     def __add__(self, other):
#         if isinstance(other, SeparateLosses):
#             return SeparateLosses(
#                 self.loss_nll + other.loss_nll,
#                 self.loss_choi + other.loss_choi,
#             )
#         else:  # scalar or array
#             return SeparateLosses(
#                 self.loss_nll + other,
#                 self.loss_choi + other,
#             )

#     def __radd__(self, other):
#         return self.__add__(other)

#     def __mul__(self, other):
#         if isinstance(other, SeparateLosses):
#             return SeparateLosses(
#                 self.loss_nll * other.loss_nll,
#                 self.loss_choi * other.loss_choi,
#             )
#         else:  # scalar or array
#             return SeparateLosses(
#                 self.loss_nll * other,
#                 self.loss_choi * other,
#             )

#     def __rmul__(self, other):
#         return self.__mul__(other)

#     def __truediv__(self, other):
#         if isinstance(other, SeparateLosses):
#             return SeparateLosses(
#                 self.loss_nll / other.loss_nll,
#                 self.loss_choi / other.loss_choi,
#             )
#         else:
#             return SeparateLosses(
#                 self.loss_nll / other,
#                 self.loss_choi / other,
#             )


class SeparateLosses(eqx.Module):
    loss_nll: Array
    loss_choi: Array

    def _binary_op(self, other: Any, op):
        # other is either the same pytree type, or a scalar/array
        if isinstance(other, SeparateLosses):
            return jax.tree_util.tree_map(lambda a, b: op(a, b), self, other)
        # special-case for sum([...]) which starts with 0
        if other == 0 and op is operator.add:
            return self
        return jax.tree_util.tree_map(lambda a: op(a, other), self)

    # arithmetic operators
    def __add__(self, other):
        return self._binary_op(other, operator.add)

    def __radd__(self, other):
        if other == 0:  # support sum([...])
            return self
        return self._binary_op(other, operator.add)

    def __sub__(self, other):
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other):
        # other - self  (if other is scalar/array)
        if isinstance(other, SeparateLosses):
            return other._binary_op(self, operator.sub)
        return jax.tree_util.tree_map(lambda a: operator.sub(other, a), self)

    def __mul__(self, other):
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other):
        return self._binary_op(other, operator.mul)

    def __truediv__(self, other):
        return self._binary_op(other, operator.truediv)

    def __neg__(self):
        return jax.tree_util.tree_map(lambda a: -a, self)

    # optional convenience: sum of all fields
    def total(self):
        return sum(
            jax.tree_leaves(self)
        )  # sums all leaves (works if leaves are scalars/arrays)


def compute_loss(model, x, y, loss_weights: LossWeights, arg_loss):
    loss_nll, loss_choi = compute_separate_losses(model, x, y, arg_loss)

    loss = jnp.dot(loss_weights.weights, jnp.array([loss_nll, loss_choi]))
    # loss_aux_info = {}
    # loss_aux_info["loss_nll"] = loss_nll
    # loss_aux_info["loss_choi"] = loss_choi

    loss_aux_info = SeparateLosses(loss_nll, loss_choi)

    return loss, loss_aux_info


compute_grad_loss = eqx.filter_grad(
    compute_loss, has_aux=True
)  # -> this returns the grad and the aux
compute_loss_and_grad = eqx.filter_value_and_grad(
    compute_loss, has_aux=True
)  # -> this returns (loss, separate_losses), and the gradient
compute_grads_separate_losses = eqx.filter_jacrev(
    compute_separate_losses,
)


@eqx.filter_jit
def update_loss_weights(model, x, y, loss_weights: LossWeights, args_loss):
    grads_nll, grads_choi = compute_grads_separate_losses(model, x, y, args_loss)

    norm_grads_nll = jnp.linalg.norm(jax.flatten_util.ravel_pytree(grads_nll)[0])
    norm_grads_choi = jnp.linalg.norm(jax.flatten_util.ravel_pytree(grads_choi)[0])

    l_nll = (norm_grads_nll + norm_grads_choi) / norm_grads_nll
    l_choi = (norm_grads_nll + norm_grads_choi) / norm_grads_choi

    alpha = args_loss.get("loss_alpha", 0.9)

    old_weights = loss_weights.weights

    new_weights = alpha * old_weights + (1 - alpha) * jnp.array([l_nll, l_choi])

    return LossWeights(new_weights)


# def compute_loss_in_batches(model, X, Y, loss_weights, args, batch_size=1024):
#     n = X.shape[0]
#     total_loss = 0.0
#     total_count = 0

#     for start in range(0, n, batch_size):
#         end = start + batch_size
#         X_batch = X[start:end]
#         Y_batch = Y[start:end]

#         # Compute loss for this batch
#         loss, _ = compute_loss(model, X_batch, Y_batch, loss_weights, args=args)
#         total_loss += loss * X_batch.shape[0]  # weight by batch size
#         total_count += X_batch.shape[0]

#     return total_loss / total_count


def compute_loss_in_batches(
    model, X, Y, loss_weights, args, batch_size=1024
) -> tuple[float, SeparateLosses]:
    n = X.shape[0]
    total_loss = 0.0
    total_count = 0

    total_separate_losses = SeparateLosses(jnp.array(0), jnp.array(0))

    for start in range(0, n, batch_size):
        end = start + batch_size
        X_batch = X[start:end]
        Y_batch = Y[start:end]

        # Compute loss for this batch
        loss, separate_losses = compute_loss(
            model, X_batch, Y_batch, loss_weights, args
        )
        total_loss += loss * X_batch.shape[0]  # weight by batch size
        total_count += X_batch.shape[0]

        total_separate_losses = (
            total_separate_losses + separate_losses * X_batch.shape[0]
        )

    return total_loss / total_count, total_separate_losses / total_count


# # ---------------------------- EXAMPLE UPDATE STEP --------------------------- #


# @eqx.filter_jit
# def simple_update_step(model, opt_state, x, y, optimizer, args_loss):
#     loss_value, grad_value = compute_loss_and_grad(model, x, y, args_loss)
#     updates, opt_state = optimizer.update(
#         grad_value, opt_state, eqx.filter(model, eqx.is_array)
#     )
#     model = eqx.apply_updates(model, updates)
#     return model, opt_state, loss_value


@eqx.filter_jit
def simple_update_step(
    model, opt_state, x, y, optimizer, loss_weights: LossWeights, args_loss
):
    (loss_value, separate_losses), grad_loss_value = compute_loss_and_grad(
        model, x, y, loss_weights, args_loss
    )
    updates, opt_state = optimizer.update(
        grad_loss_value, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, (loss_value, separate_losses)
