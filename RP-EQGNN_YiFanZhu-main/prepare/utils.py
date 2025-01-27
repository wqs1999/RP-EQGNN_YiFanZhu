import os
from urllib.request import urlopen
from urllib import request
from six.moves import urllib
import sys
import numpy as np
import torch
from rdkit import Chem
import networkx as nx
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os.path as osp

fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


# Check if qm9 string can be converted to an int, without throwing an error.
def is_int(d_str):
    try:
        int(d_str)
        return True
    except:
        return False


# Cleanup. Use try-except to avoid race condition.
def cleanup_file(file, cleanup=True):
    if cleanup:
        try:
            os.remove(file)
        except OSError:
            pass


def _progress(block_num, block_size, total_size):
    """回调函数
       @block_num: 已经下载的数据块
       @block_size: 数据块的大小
       @total_size: 远程文件的大小
    """
    sys.stdout.write('\r>> Downloading %s %.1f%%' % ('进度->',
                                                     float(block_num * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()


# def symmetric_matrix_normalization(matrix):
#     # Ensure the matrix is symmetric
#
#     # Calculate the square root of the diagonal elements
#     diag_sqrt = torch.sqrt(torch.diag(matrix))
#
#     # Inverse of the square root of diagonal elements
#     diag_sqrt_inv = 1.0 / diag_sqrt
#
#     # Create a diagonal matrix from the inverse square root
#     diag_sqrt_inv_matrix = torch.diag(diag_sqrt_inv)
#
#     # Normalize the matrix
#     normalized_matrix = diag_sqrt_inv_matrix @ matrix @ diag_sqrt_inv_matrix
#
#     return normalized_matrix


# def get_atom_and_bond_features(mol_SMILE):
#     type_idx = []
#     atomic_number = []
#     aromatic = []
#     sp = []
#     sp2 = []
#     sp3 = []
#     num_hs = []
#
#     types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
#     bonds = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}
#
#     mol_SMILE = Chem.MolFromSmiles(mol_SMILE)
#     N = mol_SMILE.GetNumAtoms()
#
#     # range
#     for atom in mol_SMILE.GetAtoms():
#         type_idx.append(types[atom.GetSymbol()])
#         # numbers of protons 电荷数
#         atomic_number.append(atom.GetAtomicNum())
#         # In an aromatic system (binary) 芳香族
#         aromatic.append(1 if atom.GetIsAromatic() else 0)
#         # sp, sp2, sp3 (one-hot or null) 杂化
#         hybridization = atom.GetHybridization()
#         sp.append(1 if hybridization == HybridizationType.SP else 0)
#         sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
#         sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
#
#     z = torch.tensor(atomic_number, dtype=torch.long)
#
#     row, col, edge_type = [], [], []
#     for bond in mol_SMILE.GetBonds():
#         start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#         row += [start, end]
#         col += [end, start]
#         edge_type += 2 * [bonds[bond.GetBondType()]]
#
#     edge_index = torch.tensor([row, col], dtype=torch.long)
#     # edge_type = torch.tensor(edge_type, dtype=torch.long)
#     # edge_attr = one_hot(edge_type, num_classes=len(bonds))
#
#     perm = (edge_index[0] * N + edge_index[1]).argsort()
#     edge_index = edge_index[:, perm]
#     # edge_type = edge_type[perm]
#     # edge_attr = edge_attr[perm]
#
#     row, col = edge_index
#     hs = (z == 1).to(torch.float)
#
#     num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()
#
#     x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
#     x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
#                       dtype=torch.float).t().contiguous()
#     atom_feature = torch.cat([x1, x2], dim=-1)
#
#     return atom_feature, edge_index
#
#
# a, b = get_atom_and_bond_features('C1N2C3C4C5OC13C2C45')
# print(a)
# print(b)


# def get_nodes(g):
#     feat = []
#     for n, d in g.nodes(data=True):
#         h_t = []
#         h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F']]
#         h_t.append(d['a_num'])
#         h_t.append(d['acceptor'])
#         h_t.append(d['donor'])
#         h_t.append(int(d['aromatic']))
#         h_t += [int(d['hybridization'] == x)
#                 for x in (Chem.rdchem.HybridizationType.SP,
#                           Chem.rdchem.HybridizationType.SP2,
#                           Chem.rdchem.HybridizationType.SP3)]
#         h_t.append(d['num_h'])
#         # 5 more
#         h_t.append(d['ExplicitValence'])
#         h_t.append(d['FormalCharge'])
#         h_t.append(d['ImplicitValence'])
#         h_t.append(d['NumExplicitHs'])
#         h_t.append(d['NumRadicalElectrons'])
#         feat.append((n, h_t))
#     feat.sort(key=lambda item: item[0])
#     node_attr = torch.FloatTensor([item[1] for item in feat])
#     return node_attr
#
#
# def get_edges(g):
#     e = {}
#     for n1, n2, d in g.edges(data=True):
#         e_t = [int(d['b_type'] == x)
#                for x in (Chem.rdchem.BondType.SINGLE,
#                          Chem.rdchem.BondType.DOUBLE,
#                          Chem.rdchem.BondType.TRIPLE,
#                          Chem.rdchem.BondType.AROMATIC)]
#
#         e_t.append(int(d['IsConjugated'] == False))
#         e_t.append(int(d['IsConjugated'] == True))
#         e[(n1, n2)] = e_t
#     edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
#     edge_attr = torch.FloatTensor(list(e.values()))
#     return edge_index, edge_attr
#
#
# def get_atom_and_bond_features(mol_SMILE):
#     if mol_SMILE is None:
#         return None
#     mol = Chem.MolFromSmiles(mol_SMILE)
#
#     feats = chem_feature_factory.GetFeaturesForMol(mol)
#     g = nx.DiGraph()
#     for i in range(mol.GetNumAtoms()):
#         atom_i = mol.GetAtomWithIdx(i)
#         g.add_node(i,
#                    a_type=atom_i.GetSymbol(),
#                    a_num=atom_i.GetAtomicNum(),
#                    acceptor=0,
#                    donor=0,
#                    aromatic=atom_i.GetIsAromatic(),
#                    hybridization=atom_i.GetHybridization(),
#                    num_h=atom_i.GetTotalNumHs(),
#
#                    # 5 more node features
#                    ExplicitValence=atom_i.GetExplicitValence(),
#                    FormalCharge=atom_i.GetFormalCharge(),
#                    ImplicitValence=atom_i.GetImplicitValence(),
#                    NumExplicitHs=atom_i.GetNumExplicitHs(),
#                    NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
#                    )
#
#     for i in range(len(feats)):
#         if feats[i].GetFamily() == 'Donor':
#             node_list = feats[i].GetAtomIds()
#             for n in node_list:
#                 g.nodes[n]['donor'] = 1
#         elif feats[i].GetFamily() == 'Acceptor':
#             node_list = feats[i].GetAtomIds()
#             for n in node_list:
#                 g.nodes[n]['acceptor'] = 1
#
#     # Read Edges
#     for i in range(mol.GetNumAtoms()):
#         for j in range(mol.GetNumAtoms()):
#             e_ij = mol.GetBondBetweenAtoms(i, j)
#             if e_ij is not None:
#                 g.add_edge(i, j,
#                            b_type=e_ij.GetBondType(),
#                            # 1 more edge features 2 dim
#                            IsConjugated=int(e_ij.GetIsConjugated()),
#                            )
#
#     node_attr = get_nodes(g)
#     edge_index, edge_attr = get_edges(g)
#
#     return node_attr, edge_index


# a, b = get_atom_and_bond_features('CC12CC(C)(C1)OC2')
# print(a)
# print(b)
