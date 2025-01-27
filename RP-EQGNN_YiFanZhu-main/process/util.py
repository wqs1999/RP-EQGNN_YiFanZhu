import torch
import numpy as np

import logging
import os

from torch.utils.data import DataLoader
from graph_regression_qm9.process.dataset import ProcessedDataset
from graph_regression_qm9.prepare.download import prepare_dataset


def initialize_datasets(args, datadir, dataset, subset=None, splits=None,
                        force_download=False):
    """
    初始化数据集

    Parameters
    ----------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datadir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "qm9" or "md17".
    subset : str, optional
        Which subset of qm9 dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : str, optional
        TODO: DELETE THIS ENTRY
    force_download : bool, optional
        If true, forces qm9 fresh download of the dataset.

    Returns
    -------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datasets : dict
        Dictionary of processed dataset objects (see ????? for more information).
        Valid keys are "train", "graph_reader", and "valid"[ate].  Each associated value
    num_species : int
        Number of unique atomic species in the dataset.
    max_charge : pytorch.Tensor
        Largest atomic number for the dataset. 最大原子数

    Notes
    -----
    TODO: Delete the splits argument.
    """
    # Set the number of points based upon the arguments
    # args.num_valid 6000 6000 200 args.num_train
    num_pts = {'train': args.num_train,
               'valid': args.num_valid, 'test': args.num_test}

    # Download and process dataset. Returns datafiles.
    datafiles = prepare_dataset(
        datadir, dataset, subset, splits, force_download=force_download)

    # Load downloaded/processed datasets
    datasets = {}
    for split, datafile in datafiles.items():

        with np.load(datafile, allow_pickle=True) as f:
            datasets[split] = {key: torch.from_numpy(val) for key, val in f.items()}

    # Basic error checking: Check the training/graph_reader/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()]
    assert all([key == keys[0] for key in keys]
               ), 'Datasets must have same set of keys!'

    # Get qm9 list of all species across the entire dataset
    all_species = _get_species(datasets, ignore_check=False)

    # Now initialize MolecularDataset based upon loaded data
    datasets = {split: ProcessedDataset(data, num_pts=num_pts.get(
        split, -1), included_species=all_species) for split, data in datasets.items()}

    # Now initialize MolecularDataset based upon loaded data

    # Check that all datasets have the same included species:
    assert(len(set(tuple(data.included_species.tolist()) for data in datasets.values())) ==
           1), 'All datasets must have same included_species! {}'.format({key: data.included_species for key, data in datasets.items()})

    # These parameters are necessary to initialize the network
    num_species = datasets['train'].num_species
    max_charge = datasets['train'].max_charge

    # Now, update the number of training/graph_reader/validation sets in args
    args.num_train = datasets['train'].num_pts
    args.num_valid = datasets['valid'].num_pts
    args.num_test = datasets['test'].num_pts

    return args, datasets, num_species, max_charge


def _get_species(datasets, ignore_check=False):
    """
    Generate qm9 list of all species.

    Includes qm9 check that each split contains examples of every species in the
    entire dataset.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets.  Each dataset is qm9 dict of arrays containing molecular properties.
    ignore_check : bool
        Ignores/overrides checks to make sure every split includes every species included in the entire dataset

    Returns
    -------
    all_species : Pytorch tensor
        List of all species present in the data.  Species labels should be integers.

    """
    # all_species，包含所有数据集中“charges”属性的所有唯一值，
    # split_species，包含数据集中 每个 拆分的“charges”属性的唯一值。
    # sorted=True在这两种情况下都使用该参数按升序对结果张量进行排序。
    all_species = torch.cat([dataset['charges'].unique()
                             for dataset in datasets.values()]).unique(sorted=True)

    # print('all_species', all_species)  all_species tensor([0, 1, 6, 7, 8, 9])

    split_species = {split: species['charges'].unique(
        sorted=True) for split, species in datasets.items()}

    # print('split_species', split_species)  split_species {'train': tensor([0, 1, 6, 7, 8, 9]), 'valid': tensor([0,
    # 1, 6, 7, 8, 9]), 'test': tensor([0, 1, 6, 7, 8, 9])}

    # If zero charges (padded, non-existent atoms) are included, remove them
    if all_species[0] == 0:
        all_species = all_species[1:]

    # print('all_species2', all_species) all_species2 tensor([1, 6, 7, 8, 9])

    # Remove zeros if zero-padded charges exist for each split
    split_species = {split: species[1:] if species[0] ==
                     0 else species for split, species in split_species.items()}

    # print('split_species2', split_species) split_species2 {'train': tensor([1, 6, 7, 8, 9]), 'valid': tensor([1, 6,
    # 7, 8, 9]), 'test': tensor([1, 6, 7, 8, 9])}

    # Now check that each split has at least one example of every atomic spcies from the entire dataset.
    if not all([split.tolist() == all_species.tolist() for split in split_species.values()]):
        # Allows one to override this check if they really want to. Not recommended as the answers become non-sensical.
        if ignore_check:
            logging.error(
                'The number of species is not the same in all datasets!')
        else:
            raise ValueError(
                'Not all datasets have the same number of species!')

    # Finally, return qm9 list of all species
    return all_species
