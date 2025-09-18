import os

import jax
import jax.numpy as jnp
import joblib

jax.config.update("jax_enable_x64", True)


import qutip as qt

from two_q_sep_cp.neural_ode.utils import (
    Bloch2D,
    DualGeneratorsSet,
    GeneratorsSet,
    LindbladGenerators,
    Parameters,
    make_hermitian_basis,
    make_two_qubit_dual_generators,
    make_two_qubit_generators,
    make_unnormalized_hermitian_basis,
    make_update_matrix_bloch,
    update_bloch_vector_from_parameters,
)


def wrap_superop(A):
    return qt.Qobj(A, dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]])


def wrap_rho(A):
    return qt.Qobj(A, dims=[[2, 2], [2, 2]])


# Get the directory where this test file lives
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to this test file's directory
GENS_PATH = os.path.join(THIS_DIR, "gens_set.joblib")
DUAL_GENS_PATH = os.path.join(THIS_DIR, "dual_gens_set.joblib")

# Compute from scratch
normalized_hermitian_basis = make_hermitian_basis(2)
unnorm_herm_basis = make_unnormalized_hermitian_basis(n_qubits=2)


if os.path.exists(GENS_PATH) and os.path.exists(DUAL_GENS_PATH):
    # Load precomputed objects
    gens_set = joblib.load(GENS_PATH)
    dual_gens_set = joblib.load(DUAL_GENS_PATH)
    print("Loaded gens_set and dual_gens_set from cache.")
else:
    H, S, C, A = make_two_qubit_generators(unnorm_herm_basis)
    H_dual, S_dual, C_dual, A_dual = make_two_qubit_dual_generators(
        unnorm_herm_basis, dim=4
    )

    gens_set = GeneratorsSet(jnp.array(H), jnp.array(S), jnp.array(C), jnp.array(A))

    dual_gens_set = DualGeneratorsSet(
        jnp.array(H_dual), jnp.array(S_dual), jnp.array(C_dual), jnp.array(A_dual)
    )

    # Save for future use
    joblib.dump(gens_set, GENS_PATH)
    joblib.dump(dual_gens_set, DUAL_GENS_PATH)
    print("Computed and saved gens_set and dual_gens_set.")


l_gens = LindbladGenerators(
    dimension=4, generators=gens_set, dual_generators=dual_gens_set
)

bloch2d = Bloch2D(normalized_hermitian_basis)
update_matrix = make_update_matrix_bloch(l_gens, normalized_hermitian_basis)


def test_lindbladian_reconstruction():
    """Test the reconstruction of a Lindbladian."""
    example_lindbladian = qt.rand_super(
        dimensions=[2, 2], seed=5
    ).logm()  # beware that qt.rand_super creates the map, not the lindbladian, that's why we need to take the logarithm
    pars_from_example = l_gens.from_lindbladian(example_lindbladian.full())
    example_lindbladian_reconstructed = l_gens.make_lindbladian(pars_from_example)
    assert jnp.allclose(example_lindbladian_reconstructed, example_lindbladian.full())


def test_update_bloch_vector():
    """Test the update of a Bloch vector."""
    rho = qt.rand_dm(dimensions=[2, 2], seed=5)
    lindblad = qt.rand_super(
        dimensions=[2, 2], seed=8
    ).logm()  # beware that qt.rand_super creates the map, not the lindbladian, that's why we need to take the logarithm
    rho_prime = qt.vector_to_operator(lindblad * qt.operator_to_vector(rho))
    rho_bloch = bloch2d.from_matrix_to_bloch(rho.full())
    pars = l_gens.from_lindbladian(lindblad.full())
    rho_prime_bloch_hat = update_bloch_vector_from_parameters(
        rho_bloch, update_matrix, pars
    )
    rho_prime_hat = bloch2d.from_bloch_to_matrix(rho_prime_bloch_hat)
    assert jnp.allclose(rho_prime_hat, rho_prime.full())
