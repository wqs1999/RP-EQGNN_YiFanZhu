import numpy as np

import logging
import os
import urllib

from os.path import join as join
import urllib.request

from graph_regression_qm9.prepare.process_files import process_xyz_files, process_xyz_gdb9
from graph_regression_qm9.prepare.utils import is_int, cleanup_file, _progress

logging.getLogger().setLevel(logging.INFO)


# calculate_thermo=True,
def download_dataset_qm9(datadir, dataname, splits=None, cleanup=True):
    """
    Download and prepare the QM9 (GDB9) dataset.
    """
    # Define directory for which data will be output. 存放GDB9数据集
    gdb9dir = join(*[datadir, dataname])

    # 创建多层目录 【目录名，是否在目录存在时触发异常】 该方法无返回值
    os.makedirs(gdb9dir, exist_ok=True)

    logging.info(
        'Downloading and processing GDB9 dataset. Output will be in directory: {}.'.format(gdb9dir))

    logging.info('Beginning download of GDB9 dataset!')

    # urlretrieve()方法直接将远程数据下载到本地。
    # 如果URL指向本地文件，则对象将不会被复制，除非提供文件名。
    gdb9_url_data = 'https://springernature.figshare.com/ndownloader/files/3195389'
    gdb9_tar_data = join(gdb9dir, 'dsgdb9nsd.xyz.tar.bz2')

    urllib.request.urlretrieve(gdb9_url_data, filename=gdb9_tar_data, reporthook=_progress)  # filename指定要保存到的文件位置
    logging.info('GDB9 dataset downloaded successfully!')

    # If splits are not specified, automatically generate them. clean up 是否分割
    if splits is None:
        splits = gen_splits_gdb9(gdb9dir, cleanup)

    # Process GDB9 dataset, and return dictionary of splits
    gdb9_data = {}
    for split, split_idx in splits.items():
        gdb9_data[split] = process_xyz_files(
            gdb9_tar_data, process_xyz_gdb9, file_idx_list=split_idx)

    # # Subtract thermochemical energy if desired.
    # if calculate_thermo:
    #     # Download thermochemical energy from GDB9 dataset, and then process it into qm9 dictionary
    #     therm_energy = get_thermo_dict(gdb9dir, cleanup)
    #
    #     # For each of train/validation/test split, add the thermochemical energy
    #     for split_idx, split_data in gdb9_data.items():
    #         gdb9_data[split_idx] = add_thermo_targets(split_data, therm_energy)

    # Save processed GDB9 data into train/validation/test splits
    logging.info('Saving processed data:')
    for split, data in gdb9_data.items():
        savedir = join(gdb9dir, split + '.npz')
        np.savez_compressed(savedir, **data)

    logging.info('Processing/saving complete!')

def gen_splits_gdb9(gdb9dir, cleanup=True):
    """
    Generate GDB9 training/validation/test splits used.

    First, use the file 'uncharacterized.txt' in the GDB9 figshare to find qm9
    list of excluded molecules.

    Second, create qm9 list of molecule ids, and remove the excluded molecule
    indices.

    Third, assign 100k molecules to the training set, 10% to the test set,
    and the remaining to the validation set.

    Finally, generate torch.tensors which give the molecule ids for each
    set.
    """
    logging.info('Splits were not specified! Automatically generating.')
    gdb9_url_excluded = 'https://springernature.figshare.com/ndownloader/files/3195404'
    gdb9_txt_excluded = join(gdb9dir, 'uncharacterized.txt')
    urllib.request.urlretrieve(gdb9_url_excluded, filename=gdb9_txt_excluded)

    # First get list of excluded indices
    excluded_strings = []
    with open(gdb9_txt_excluded) as f:
        lines = f.readlines()
        # 每一行 单词数>0，取第一个单词，组成excluded_strings
        excluded_strings = [line.split()[0]
                            for line in lines if len(line.split()) > 0]

    excluded_idxs = [int(idx) - 1 for idx in excluded_strings if is_int(idx)]

    assert len(excluded_idxs) == 3054, 'There should be exactly 3054 excluded atoms. Found {}'.format(
        len(excluded_idxs))

    # 创建qm9索引列表
    Ngdb9 = 133885
    Nexcluded = 3054
    # 分子数 删去 uncharacterized(未知)中对应的index
    included_idxs = np.array(
        sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))

    # Now generate random permutations to assign molecules to training/validation/test sets.
    # 减去未知的，得到现有数据总量
    Nmols = Ngdb9 - Nexcluded

    Ntrain = 100000
    Ntest = int(0.1 * Nmols)
    Nvalid = Nmols - (Ntrain + Ntest)

    # 数据量 数据随机排列
    np.random.seed(0)
    data_perm = np.random.permutation(Nmols)

    # Now use the permutations to generate the indices of the dataset splits.
    # train, valid, test, extra = np.split(included_idxs[data_perm], [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])
    # 得到索引
    train, valid, test, extra = np.split(
        data_perm, [Ntrain, Ntrain + Nvalid, Ntrain + Nvalid + Ntest])

    assert (len(extra) == 0), 'Split was inexact {} {} {} {}'.format(
        len(train), len(valid), len(test), len(extra))

    train = included_idxs[train]
    valid = included_idxs[valid]
    test = included_idxs[test]

    splits = {'train': train, 'valid': valid, 'test': test}

    # Cleanup
    cleanup_file(gdb9_txt_excluded, cleanup)

    return splits


# download_dataset_qm9('..\data', 'qm9')
