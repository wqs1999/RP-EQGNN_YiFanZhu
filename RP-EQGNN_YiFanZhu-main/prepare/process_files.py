import logging
import os
import torch
import tarfile
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, HybridizationType
from torch_geometric.utils import scatter
from torch.nn.functional import one_hot

charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}


def process_xyz_files(data, process_file_fn, file_ext=None, file_idx_list=None):
    """
    Take qm9 set of datafiles and apply qm9 predefined data processing script to each
    one. Data can be stored in qm9 directory, tarfile, or zipfile. An optional
    file extension can be added.

    Parameters
    ----------
    data : str
        Complete path to datafiles. Files must be in qm9 directory, tarball, or zip archive.
    process_file_fn : callable
        Function to process files. Can be defined externally.
        Must input qm9 file, and output qm9 dictionary of properties, each of which
        is qm9 torch.tensor. Dictionary must contain at least three properties:
        {'num_elements', 'charges', 'positions'}
    file_ext : str, optional
        Optionally add qm9 file extension if multiple types of files exist.
    file_idx_list : ?????, optional
        Optionally add qm9 file filter to check qm9 file index is in qm9
        predefined list, for example, when constructing qm9 train/valid/test split. (stack)
    """
    logging.info('Processing data file: {}'.format(data))
    if tarfile.is_tarfile(data):
        tardata = tarfile.open(data, 'r')
        files = tardata.getmembers()

        readfile = lambda data_pt: tardata.extractfile(data_pt)

    elif os.is_dir(data):
        files = os.listdir(data)
        files = [os.path.join(data, file) for file in files]

        readfile = lambda data_pt: open(data_pt, 'r')

    else:
        raise ValueError('Can only read from directory or tarball archive!')

    # Use only files that end with specified extension.
    if file_ext is not None:
        files = [file for file in files if file.endswith(file_ext)]

    # Use only files that match desired filter.
    if file_idx_list is not None:
        files = [file for idx, file in enumerate(files) if idx in file_idx_list]

    # Now loop over files using readfile function defined above
    # Process each file accordingly using process_file_fn

    molecules = []

    for file in files:
        with readfile(file) as openfile:
            molecules.append(process_file_fn(openfile))

    # Check that all molecules have the same set of items in their dictionary:
    props = molecules[0].keys()
    assert all(props == mol.keys() for mol in molecules), 'All molecules must have same set of properties/keys!'

    # Convert list-of-dicts to dict-of-lists 字典
    # 再list转array，方便下一步转tensor
    # key: [list 包含所有molecules的对应特征值]

    molecules = {prop: [mol[prop] for mol in molecules] for prop in props}
    # for key, val in molecules.items():
    #     print(key, val)
    #     print('\n')
    #     print(val[0])

    # condition_func = lambda x: x.get('num_atoms') < 20
    # molecules = {k: v for k, v in molecules.items() if condition_func((k, v))}
    # print('------------', molecules)

    molecules = {
        # 横向填充, 默认batch_size在第一维度
        key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val)
        for key, val in molecules.items()}
    # print('2', molecules)

    c_values = molecules['num_atoms']

    # 检查 'aaa' 中的值是否低于20
    mask = c_values < 20

    # 创建一个新字典，只包含其他键中对应位置的元素
    molecules = {key: value[mask] for key, value in molecules.items()}
    # for key, val in molecules.items():
    #     print('--------------------')
    #     print(key)
    #     print(val.shape, val)

    # If stacking is desireable, pad and then stack.
    # if stack:
    #     molecules = {
    #         # 横向填充 val[0] torch.stack(val)
    #         key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val)
    #         for key, val in molecules.items()}

    # 以列表返回可遍历的(键, 值)元组数组 val[0] 所有molecules的对应特征值列表的第一个

    return molecules


def process_xyz_gdb9(datafile):
    """
    Read xyz file and return qm9 molecular dict with number of atoms, energy, forces, coordinates and atom-type for the gdb9 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.

    Notes
    -----
    TODO : Replace breakpoint with qm9 more informative failure?
    """
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]
    # 原子个数
    num_atoms = int(xyz_lines[0])
    # 分子特性
    mol_props = xyz_lines[1].split()
    # 坐标
    mol_xyz = xyz_lines[2:num_atoms + 2]
    # 谐波振动频率
    mol_freq = xyz_lines[num_atoms + 2]
    # 分子表示
    mol_SMILE = xyz_lines[num_atoms + 3].split()[0]

    atom_feature, edge_index = get_atom_and_bond_features(mol_SMILE)

    atom_charges, atom_positions = [], []
    for line in mol_xyz:
        atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
        atom_charges.append(charge_dict[atom])
        atom_positions.append([float(posx), float(posy), float(posz)])

    prop_strings = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H',
                    'G', 'Cv']
    prop_strings = prop_strings[1:]
    mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
    mol_props = dict(zip(prop_strings, mol_props))
    mol_props['omega1'] = max(float(omega) for omega in mol_freq.split())
    # 'atom_feature': atom_feature, 'bonds': edge_index
    molecule = {'num_atoms': num_atoms, 'atom_feature': atom_feature, 'bonds': edge_index, 'charges': atom_charges,
                'positions': atom_positions}

    molecule.update(mol_props)
    molecule = {key: torch.tensor(val) for key, val in molecule.items()}

    return molecule


def get_atom_and_bond_features(mol_SMILE):
    type_idx = []
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []

    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    bonds = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}

    mol_smile = Chem.MolFromSmiles(mol_SMILE)
    # print(mol_smile)
    mol_smile = Chem.AddHs(mol_smile)
    # print(mol_smile)
    N = mol_smile.GetNumAtoms()

    # range
    for atom in mol_smile.GetAtoms():
        type_idx.append(types[atom.GetSymbol()])
        # print(atom.GetSymbol())
        # numbers of protons 电荷数
        atomic_number.append(atom.GetAtomicNum())
        # In an aromatic system (binary) 芳香族
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        # sp, sp2, sp3 (one-hot or null) 杂化
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    # print('type_idx', type_idx)
    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol_smile.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # print(start, end)
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]
    # row, col
    edge_index = torch.tensor([row, col], dtype=torch.long)
    # edge_type = torch.tensor(edge_type, dtype=torch.long)
    # edge_attr = one_hot(edge_type, num_classes=len(bonds))

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    # edge_type = edge_type[perm]
    # edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    # 氢数
    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()
    # 原子类型
    x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
    # print("x1", x1)
    # 原子个数 芳香性 杂化 氢数
    x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                      dtype=torch.float).t().contiguous()
    # print(atomic_number, '--', aromatic, '--', sp, sp2, sp3, '--', num_hs)
    atom_feature = torch.cat([x1, x2], dim=-1)
    edge_index = edge_index.T
    return atom_feature, edge_index


# a, b = get_atom_and_bond_features('OCc1ccnoc1=O')
# print(a)
# print(b)

# f = open('E:\pycharmProject\QEGNN\data2\dsgdb9nsd_019332.xyz', 'r')
# a = process_xyz_gdb9(f)
# print(a)
