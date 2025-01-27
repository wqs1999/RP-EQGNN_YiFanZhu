import torch
from torch.utils.data import Dataset
import logging


class ProcessedDataset(Dataset):
    """
    Data structure for qm9 pre-processed cormorant dataset.

    Parameters
    ----------
    data : dict
        Dictionary of arrays containing molecular properties.
    included_species : tensor of scalars, optional
        Atomic species to include in ?????.  If None, uses all species. 种类
    num_pts : int, optional,使用多少数据
        Desired number of points to include in the dataset.
        Default value, -1, uses all of the datapoints.
    normalize : bool, optional
        ????? IS THIS USED?
    shuffle : bool, optional 随机排列
        If true, shuffle the points in the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.

    """

    def __init__(self, data, included_species=None, num_pts=-1, shuffle=True):

        self.data = data
        # num_pts 期望的数据量 == -1 使用所有数据
        if num_pts < 0:
            self.num_pts = len(data['charges'])
        else:
            if num_pts > len(data['charges']):
                logging.warning(
                    'Desired number of points ({}) is greater than the number of data points ({}) available in the '
                    'dataset!'.format(
                        num_pts, len(data['charges'])))
                self.num_pts = len(data['charges'])
            else:
                self.num_pts = num_pts

        # if max_nodes < 0:
        #     self.max_nodes = max(data['num_atoms'])
        # else:
        #     self.max_nodes = torch.tensor(max_nodes)

        # If included species is not specified 未指定,则包含所有
        if included_species is None:
            # torch.unique：类似于集合，给出tensor中的独立不重复元素
            # 对原子电荷排列
            included_species = torch.unique(self.data['charges'], sorted=True)
            if included_species[0] == 0:
                included_species = included_species[1:]

        self.included_species = included_species

        self.data['one_hot'] = self.data['charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)

        self.num_species = len(included_species)
        self.max_charge = max(included_species)

        self.parameters = {'num_species': self.num_species, 'max_charge': self.max_charge}

        # Get qm9 dictionary of statistics for all properties that are one-dimensional tensors.
        self.calc_stats()

        if shuffle:
            self.perm = torch.randperm(len(data['charges']))[:self.num_pts]
        else:
            self.perm = None

    def calc_stats(self):
        self.stats = {key: (val.mean(), val.std()) for key, val in self.data.items() if
                      type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}

    # 转换单位
    def convert_units(self, units_dict):
        for key in self.data.keys():
            if key in units_dict:
                self.data[key] *= units_dict[key]

        self.calc_stats()

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]
        return {key: val[idx] for key, val in self.data.items()}
