import json
from importlib import resources

import jax
import jax.numpy as jnp
import numpy as np
import qutip as qt
from qutip import gates

hgate = gates.hadamard_transform()
sgate = gates.s_gate()

single_qubit_states_str = ["0", "1", "+", "i+"]
single_qubit_measurements_str = ["X", "Y", "Z"]


config_file = resources.files("two_q_sep.config_dict").joinpath(
    "config_dict_states_measurements.json"
)
with config_file.open("r", encoding="utf-8") as f:
    config_dict_states_measurements = json.load(f)


# with open("config_dict_states_measurements.json", "r") as f:
#     config_dict_states_measurements = json.load(f)


states_str = config_dict_states_measurements["states"]
measurements_str = config_dict_states_measurements["measurements"]


list_unitaries_states_single_qubit = jnp.array(
    [qt.identity(2).full(), qt.sigmax().full(), (hgate.full()), (sgate * hgate).full()]
)

list_unitaries_measure_single_qubit = jnp.array(
    [(hgate.full()), (sgate * hgate).full(), qt.identity(2).full()]
)


dict_one_qubit_states_gates = dict(
    zip(single_qubit_states_str, list_unitaries_states_single_qubit)
)

dict_one_qubit_measurements_gates = dict(
    zip(single_qubit_measurements_str, list_unitaries_measure_single_qubit)
)


list_two_qubits_states_gates = []
for state_i in states_str:
    st_1, st_2 = state_i.split(",")

    gate = jnp.kron(
        dict_one_qubit_states_gates[st_1], dict_one_qubit_states_gates[st_2]
    )
    list_two_qubits_states_gates.append(gate)


array_two_qubits_states_gates = jnp.array(list_two_qubits_states_gates)


list_two_qubits_measurements_gates = []

for measurement_i in measurements_str:
    basis_1, basis_2 = measurement_i.split(",")
    gate = jnp.kron(
        dict_one_qubit_measurements_gates[basis_1],
        dict_one_qubit_measurements_gates[basis_2],
    )
    list_two_qubits_measurements_gates.append(gate)


array_two_qubits_measurements_gates = jnp.array(list_two_qubits_measurements_gates)


dict_two_qubits_states = dict(zip(states_str, list_two_qubits_states_gates))
dict_two_qubits_measurements = dict(
    zip(measurements_str, list_two_qubits_measurements_gates)
)
