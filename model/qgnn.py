from model.circuit import encode_coor, encode_feature, entanglement_edge
from utils.p_utils import get_similarity

from qiskit.circuit import Parameter, ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN

from qiskit_algorithms.utils import algorithm_globals

from qiskit import QuantumCircuit, transpile, Aer
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import Sampler
import torch.nn as nn
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.connectors import TorchConnector

import matplotlib.pyplot as plt



class QGNN(torch.nn.Module):
    def __init__(self, num_qubits=1, edges_index=[]):  # inputs, edges, num_qubits, params
        super(QGNN, self).__init__()

        self.num_qubits = num_qubits
        self.edges_index = edges_index
        # Initialize quantum circuit components: encoder, feature encoder, and entanglement
        self.encoder = encode_coor(self.num_qubits)
        self.conv_feature = encode_feature(self.num_qubits)
        self.entanglement = entanglement_edge(self.num_qubits, self.edges_index)
        # Build the quantum circuit by combining the components
        self.circuit = QuantumCircuit(self.num_qubits)

        self.circuit.compose(self.encoder, range(self.num_qubits), inplace=True)
        self.circuit.compose(self.conv_feature, range(self.num_qubits), inplace=True)
        self.circuit.compose(self.entanglement, range(self.num_qubits), inplace=True)

        # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
        # inputs：self.encoder.parameters, self.conv_feature.parameters[3:], self.entanglement[1:]
        # paras：self.conv_feature.parameters[:3]， self.entanglement[:1]
        # Create the quantum neural network (QNN)
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            input_params=self.encoder.parameters[:] + self.conv_feature.parameters[11:] + self.entanglement[1:],
            weight_params=self.conv_feature.parameters[:11] + self.entanglement[:1],
            input_gradients=True,
        )
        self.qnn = TorchConnector(self.qnn)

        self.lin = torch.nn.Linear(2, 2)

        self.post_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1, 3),
            nn.SiLU(),
            nn.Linear(3, 1))
    # graph.h, graph.x, graph.edges
    def forward(self, graph):
        # self.num_qubits = nx.number_of_nodes(graph.g)

        self.edges_index = np.array(graph.edges.cpu()).reshape(-1)
        edges_similarity = get_similarity(graph)

        x = torch.tensor(graph.x.flatten().numpy().tolist() + graph.h.flatten().numpy().tolist() + edges_similarity)
        # x = self.qnn.forward(x).clone().detach()  # apply QNN
        x = self.qnn.forward(x).clone().detach()
        x = torch.unsqueeze(x, 0)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.post_mlp(x)

        x = x.view(-1)
        return x
# from graph_regression_qm9.process.sample_graph import SampleGraph
# from graph_regression_qm9.process.get_dataloader import get_dataloader
#
# train_loader, val_loader, test_loader, charge_scale = get_dataloader(num_workers=0)
# batch = SampleGraph(iter(train_loader).next(), 'homo', False)
# edges = np.array(batch.edges).reshape(-1)
#
# model = QGNN(batch.num[-1].numpy(), edges)
# print(model(batch))
