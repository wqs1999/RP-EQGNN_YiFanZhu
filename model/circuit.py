
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, execute, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RealAmplitudes, PauliEvolutionGate, ZFeatureMap, TwoLocal, NLocal

# Graph libraries
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx



# Function to encode coordinate information (without parameters)
def encode_coor(num_qubits):
    # Create a quantum circuit with `num_qubits` qubits
    rot = QuantumCircuit(num_qubits)
    params = ParameterVector('r', length=num_qubits * 3)# Create a vector of parameters
    # Apply rotation gates (ry, rx) on each qubit
    for i in range(num_qubits):
        rot.ry(params[i * 3], i)
        rot.rx(params[1 + i * 3], i)
        rot.ry(params[2 + i * 3], i)
    # Create a NLocal quantum circuit with rotations, 1 repetition
    qc_nlocal = NLocal(num_qubits=num_qubits, rotation_blocks=rot,
                       reps=1,
                       parameter_prefix='r',
                       skip_final_rotation_layer=True, insert_barriers=True)

    return qc_nlocal


# # Function to create entanglement between qubits based on coordinate information
def entanglement_edge(num_qubits, edges):
    num_qubits = num_qubits
    ent = QuantumCircuit(num_qubits)

    # param+inputs
    params = ParameterVector('e', length=1 + len(edges))


    for i in range(int(len(edges) / 2)):
        ent.evo(params[i + 1] * params[0], edges[2 * i], edges[2 * i + 1])
        # ent.barrier()

    ent.barrier()


    return ent
# Function to encode features (including convolution and entanglement)
def encode_feature(num_qubits):

    rot = QuantumCircuit(num_qubits)
    # param+inputs
    params = ParameterVector('c', length=num_qubits * 11 + 11)

    for i in range(num_qubits):
        rot.ry(params[11 + i * 11] * params[0], i)
        rot.rx(params[12 + i * 11] * params[1], i)
        rot.rx(params[13 + i * 11] * params[2], i)
        rot.ry(params[14 + i * 11] * params[3], i)
        rot.rx(params[15 + i * 11] * params[4], i)
        rot.rx(params[16 + i * 11] * params[5], i)
        rot.ry(params[17 + i * 11] * params[6], i)
        rot.rx(params[18 + i * 11] * params[7], i)
        rot.rx(params[19 + i * 11] * params[8], i)
        rot.ry(params[20 + i * 11] * params[9], i)
        rot.rx(params[21 + i * 11] * params[10], i)


    qc_nlocal = NLocal(num_qubits=num_qubits, rotation_blocks=rot,
                       reps=1,
                       parameter_prefix='c',
                       skip_final_rotation_layer=True, insert_barriers=True)

    return qc_nlocal


# Function to create entanglement between qubits (using rxx gates)
def entanglement_edge(num_qubits, edges):
    num_qubits = num_qubits
    ent = QuantumCircuit(num_qubits)

    # param+inputs
    params = ParameterVector('e', length=1 + len(edges))

    for i in range(int(len(edges) / 2)):
        ent.rxx(params[i + 1] * params[0], edges[2 * i], edges[2 * i + 1])
        # ent.barrier()

    ent.barrier()


    return ent

# Function to create entangling circuit with specified edges
def entangle_circuit(num_qubits, edges):

    num_qubits = num_qubits
    ent = QuantumCircuit(num_qubits)

    # param+inputs
    params = ParameterVector('e', length=1 + len(edges))



    for i in range(int(len(edges) / 2)):
        ent.rxx(params[i + 1] * params[0], edges[2 * i], edges[2 * i + 1])
        # ent.barrier()

    ent.barrier()
    return ent
