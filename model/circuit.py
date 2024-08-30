
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, execute, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RealAmplitudes, PauliEvolutionGate, ZFeatureMap, TwoLocal, NLocal

# Graph libraries
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


# def encode_feature(input, params):
#     circuit = QuantumCircuit(1)
#
#     circuit.rx(input[0] * params[0], 0)
#     circuit.ry(input[1] * params[1], 0)
#
#     return circuit
# 编码坐标信息 不含参数
def encode_coor(num_qubits):

    rot = QuantumCircuit(num_qubits)
    params = ParameterVector('r', length=num_qubits * 3)

    for i in range(num_qubits):
        rot.ry(params[i * 3], i)
        rot.rx(params[1 + i * 3], i)
        rot.ry(params[2 + i * 3], i)

    qc_nlocal = NLocal(num_qubits=num_qubits, rotation_blocks=rot,
                       reps=1,
                       parameter_prefix='r',
                       skip_final_rotation_layer=True, insert_barriers=True)

    return qc_nlocal

# circuit = encode_coor(5)
# print(circuit.parameters)
#
# circuit.decompose().draw("mpl", style="clifford")
# plt.show()

# 纠缠坐标信息
def entanglement_edge(num_qubits, edges):
    num_qubits = num_qubits
    ent = QuantumCircuit(num_qubits)

    # param+inputs
    params = ParameterVector('e', length=1 + len(edges))

    # operator = (2 ^ 2) - 0.1 * (X ^ I)
    # evo = PauliEvolutionGate(operator, time=0.2)

    for i in range(int(len(edges) / 2)):
        ent.evo(params[i + 1] * params[0], edges[2 * i], edges[2 * i + 1])
        # ent.barrier()

    ent.barrier()


    return ent
# 针对非几何信息 编码卷积 + 纠缠
def encode_feature(num_qubits):

    rot = QuantumCircuit(num_qubits)
    # 参数+输入信息
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

# circuit = encode_feature(3)
# print(circuit.parameters)
#
# circuit.decompose().draw("mpl", style="clifford")
# plt.show()

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
# circuit = entanglement_edge(4, [0,1,0,2,1,3])
# circuit.draw("mpl", style="clifford")
# plt.show()

# circuit = encode_feature(3)
# print(circuit.parameters)
#
# circuit.decompose().draw("mpl", style="clifford")
# plt.show()

# def conv_circuit(num_qubits):
#     rot = QuantumCircuit(num_qubits)
#
#     params = ParameterVector('c', length=3)
#     for i in range(num_qubits):
#         rot.ry(params[0], i)
#         rot.rx(params[1], i)
#         rot.ry(params[2], i)
#
#     qc_nlocal = NLocal(num_qubits=num_qubits, rotation_blocks=rot,
#                        reps=1,
#                        parameter_prefix='c',
#                        skip_final_rotation_layer=True, insert_barriers=True)
#
#     return qc_nlocal

# circuit = conv_circuit(14)
# print(circuit.parameters)
#
# circuit.decompose().draw("mpl", style="clifford")
# plt.show()

# def conv_circuit(params):
#     circuit = QuantumCircuit(1)
#
#     circuit.rx(params[0], 0)
#     circuit.ry(params[1], 0)
#     circuit.rz(params[2], 0)
#
#     return circuit
#
#
# def conv_layer(num_qubits, params):
#     qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
#     qubits = list(range(num_qubits))
#     # param_index = 0
#     # params = ParameterVector(param_prefix, length=3)  # num_qubits * 3
#     for q1 in range(num_qubits):
#         # qc = qc.compose(conv_circuit(params[param_index: (param_index + 3)]), q1)
#         qc = qc.compose(conv_circuit(params), q1)
#         # qc.barrier()
#         # param_index += 3
#     # for i in range(num_qubits):
#     #     qc = qc.compose(conv_circuit(params[param_index: (param_index + 3)]), i)
#     # qc.barrier()
#     # param_index += 3
#
#     qc_inst = qc.to_instruction()
#
#     qc = QuantumCircuit(num_qubits)
#     qc.append(qc_inst, qubits)
#     return qc


# Let's draw this circuit and see what it looks like
# params = ParameterVector("θ", length=3)
# circuit = conv_circuit(3, params)
# circuit.draw("mpl", style="clifford")
# plt.show()

# circuit = conv_layer(4, [1,2,3])
# circuit.decompose().draw("mpl", style="clifford")
# plt.show()

# num_qubits 输入（相似度）+参数 放到一起 作为整个参数

def entangle_circuit(num_qubits, edges):

    num_qubits = num_qubits
    ent = QuantumCircuit(num_qubits)

    # param+inputs
    params = ParameterVector('e', length=1 + len(edges))


    # inputs = ParameterVector(inputs, length=len(edges))
    # weights = ParameterVector(weights, length=len(edges))

    # edges[i][0], edges[i][1]

    for i in range(int(len(edges) / 2)):
        ent.rxx(params[i + 1] * params[0], edges[2 * i], edges[2 * i + 1])
        # ent.barrier()

    ent.barrier()

    # random entanglement # 全部增加辅助qubit

    # for i in range(len(edges)):
    #     ent.ryy(inputs[i] * weights[1], edges[i][0], edges[i][1])

    return ent


# circuit = entangle_circuit(4, [0,1,0,2,1,3])
# circuit.draw("mpl", style="clifford")
# plt.show()


# sources, sinks, param_prefix
# def entangle_layer(num_qubits, edges, inputs):
#     # num_qubits = np.max(edges) + 1
#     params = ParameterVector('e', length=1)
#
#     qc = QuantumCircuit(num_qubits)
#     qc = qc.compose(entangle_circuit(edges, inputs))
#
#
#     return qc

# def random_entanglement_layer(random_q, current_layer, num_qubits):
#
#     current_qubit = random_q[current_layer - 1]
#     qc = QuantumCircuit(num_qubits)
#     qml.CNOT(wires=[current_qubit, num_qubits + current_layer])
#
#     current_layer = current_layer + 1
#     return qc

# edges = [[0,1],[0,2],[1,3]]
# circuit = entangle_layer(edges, "θ","γ")
# circuit.draw("mpl", style="clifford")
# plt.show()

# edge: np.array(nx.edges(graphs[0].g))
# get_node: np.array(nx.edges(graphs[0].g))[0][1]
# def QEGNN(num_qubits, edges):
#     rot = QuantumCircuit(1)
#
#     inputs = ParameterVector("input", length= len(edges))
#     weights = ParameterVector("weight", length= 2 + len(edges))
#     # params = ParameterVector('r', length=2 + len(edges))
#
#     # rot.ry(params[0], 0)
#     # rot.rx(params[1], 0)
#     rot.ry(weights[0], 0)
#     rot.rx(weights[1], 0)
#
#     ent = QuantumCircuit(num_qubits)
#     for i in range(len(edges)):
#         # ent.cx(edges[i][0], edges[i][1])
#         ent.rxx(inputs[2 + i] * weights[2 + i], edges[i][0], edges[i][1])
#
#     qc_nlocal = NLocal(num_qubits=num_qubits, rotation_blocks=rot,
#                        entanglement_blocks=ent,
#                        # entanglement='linear',
#                        reps=1,
#                        parameter_prefix='r',
#                        skip_final_rotation_layer=True, insert_barriers=True)
#
#     return qc_nlocal


# from qegnn.utils import get_dataset, get_single_data, num_latent, num_trash, Partial_Statevector
# from qegnn.preprocess import load_split_data
# graphs = load_split_data('MUTAG')

