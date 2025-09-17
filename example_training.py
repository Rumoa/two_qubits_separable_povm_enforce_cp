import jax

jax.config.update("jax_enable_x64", True)
import json

import equinox as eqx

# import jax
import jax.numpy as jnp
import joblib
import numpy as np
import optax
import qutip as qt
from sklearn.model_selection import train_test_split

from two_q_sep_cp.neural_ode.model import (
    FourierLindbladNetTwoQubits,
    LindbladNetTwoQubits,
    compute_loss,
    get_params,
    simple_update_step,
)
from two_q_sep_cp.neural_ode.utils import (
    POVMS,
    Bloch2D,
    DualGeneratorsSet,
    GeneratorsSet,
    InitialStates,
    LindbladGenerators,
    Parameters,
    compute_probability_from_rho_bloch_and_povms_bloch,
    make_data_loader,
    make_hermitian_basis,
    make_two_qubit_dual_generators,
    make_two_qubit_generators,
    make_unnormalized_hermitian_basis,
    make_update_matrix_bloch,
    prepare_batch_for_ode,
    split_dataset_vector,
    update_bloch_vector_from_parameters,
)
from two_q_sep_cp.SPAM_estimation.misc import (
    array_two_qubits_measurements_gates,
    array_two_qubits_states_gates,
    make_all_density_matrices,
    make_complete_povms,
)

# ---------------------------------------------------------------------------- #
#                                DATASET LOADING                               #
# ---------------------------------------------------------------------------- #

dataset = joblib.load("datasets/q15_q16_250ns_2025-08-05_FILTERED_DATASET.job")

X = dataset["X"]  # THIS IN THEORY DOES NOT INCLUDE THE 0 TIME
Y = dataset["Y"]

# random_ordering = jax.random.permutation(subkey, len(X))
# X = X[random_ordering]
# Y = Y[random_ordering]

max_time = X[:, 2].max()

X[:, 2] = X[:, 2] / max_time


mask = X[:, 2] <= 0.4  # We are gonna learn only short times
X = X[mask]
Y = Y[mask]

seed_train_test = 0


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=seed_train_test)
X_train, X_test, Y_train, Y_test = [
    jnp.array(arr) for arr in [X_train, X_test, Y_train, Y_test]
]


# ---------------------------------------------------------------------------- #
#                         LOADING GENERATORS AND STUFF                         #
# ---------------------------------------------------------------------------- #

key = jax.random.key(seed=0)
key, subkey = jax.random.split(key)


spam_estimation = joblib.load(
    "results_spam_estimation/q15_q16_250ns_2025-08-05_SPAM_ESTIMATION.job"
)
rho_00_mean = spam_estimation["rhos"].mean(0)
povm_z_mean = spam_estimation["povms"].mean(0)

array_povms = make_complete_povms(povm_z_mean, array_two_qubits_measurements_gates)
array_rhos = make_all_density_matrices(rho_00_mean, array_two_qubits_states_gates)


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


# ----------------------------- SOME DEFINITIONS ----------------------------- #

args_loss = {
    "initial_states": initial_states,
    "povms": povms,
    "bloch2d": bloch2d,
    "update_matrix": update_matrix,
    "short_time_gamma": 2,  # originally 3
}
key, subkey = jax.random.split(key)

# ------------------------------- MODEL CHOICE ------------------------------- #
# model = LindbladNetTwoQubits(subkey, depth=6)
model = FourierLindbladNetTwoQubits(subkey, depth=3, width_size=128, num_bands=16)


# model = eqx.tree_deserialise_leaves(
#     "saved_models/lindblad_two_qubits_depth_6_width_256.eqx", model
# )


optimizer = optax.adamw(learning_rate=0.001)
opt_state = optimizer.init(get_params(model))

key, subkey = jax.random.split(key)

# BATCH_SIZE = 512
BATCH_SIZE = 2048

N_EPOCHS = 250

max_iterations_per_epoch = X_train.shape[0] // BATCH_SIZE

# train_loader = DataLoader([X_train, Y_train], batch_size=BATCH_SIZE, subkey=subkey)
train_loader = make_data_loader(
    [X_train, Y_train], batch_size=BATCH_SIZE, subkey=subkey
)


train_loss_history = []
test_loss_history = []
lr_scale_history = []


# def compute_loss_in_batches(model, X, Y, args, batch_size=256):
#     n = X.shape[0]
#     total_loss = 0.0
#     total_count = 0

#     for start in range(0, n, batch_size):
#         end = start + batch_size
#         X_batch = X[start:end]
#         Y_batch = Y[start:end]

#         # Compute loss for this batch
#         loss = compute_loss(model, X_batch, Y_batch, args=args)
#         total_loss += loss * X_batch.shape[0]  # weight by batch size
#         total_count += X_batch.shape[0]

#     return total_loss / total_count


for epoch in range(N_EPOCHS):
    epoch_train_loss = 0.0
    epoch_test_loss = 0.0

    # Iterate over batches in the epoch
    for step in range(max_iterations_per_epoch):
        # Get the next batch of training data
        x, y = next(train_loader)

        # Perform a training step
        model, opt_state, train_loss = simple_update_step(
            model, opt_state, x, y, optimizer=optimizer, args_loss=args_loss
        )

        print(train_loss)

        # Compute test loss after each batch
        # test_loss = compute_loss(model, X_test, Y_test, args=args_loss)
        # test_loss = compute_loss_in_batches(
        #     model, X_test, Y_test, args_loss, batch_size=128
        # )

        # Accumulate losses
        epoch_train_loss += float(train_loss)
        # epoch_test_loss += float(test_loss)

        # # Optionally, track learning rate scale
        # lr_scale = optax.tree_utils.tree_get(opt_state, "scale")
        # lr_scale_history.append(lr_scale)

    # Average the losses for the entire epoch
    epoch_train_loss /= max_iterations_per_epoch
    epoch_test_loss /= max_iterations_per_epoch

    train_loss_history.append(epoch_train_loss)
    test_loss_history.append(epoch_test_loss)

    # print(
    #     f"Epoch {epoch + 1}/{N_EPOCHS} | Train Loss: {epoch_train_loss:.4e} | Test Loss: {epoch_test_loss:.4e}"
    # )
    print(f"Epoch {epoch + 1}/{N_EPOCHS} | Train Loss: {epoch_train_loss:.4e}")
