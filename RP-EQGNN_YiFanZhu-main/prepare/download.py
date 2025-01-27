import logging
import os
from graph_regression_qm9.prepare.qm9 import download_dataset_qm9


# 检查数据集下载情况，返回处理好的数据文件
# 判断所有文件的路径是否存在
def prepare_dataset(datadir, dataset, subset=None, splits=None, cleanup=True, force_download=False):
    """
    Download and process dataset.

    Parameters
    ----------
    datadir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "qm9" or "md17".
    subset : str, optional
        Which subset of qm9 dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : dict, optional
        Dataset splits to use.
    cleanup : bool, optional
        Clean up files created while preparing the data.
    force_download : bool, optional
        If true, forces qm9 fresh download of the dataset.

    Returns
    -------
    datafiles : dict of strings
        Dictionary of strings pointing to the files containing the data. 

    Notes
    -----
    TODO: Delete the splits argument?
    """

    # If datasets have subsets
    if subset:
        dataset_dir = [datadir, dataset, subset]
    else:
        dataset_dir = [datadir, dataset]

    # Names of splits, based upon keys if split dictionary exists, elsewise default to train/valid/test.
    split_names = splits.keys() if splits is not None else [
        'train', 'valid', 'test']

    # Assume one data file for each split
    datafiles = {split: os.path.join(
        *(dataset_dir + [split + '.npz'])) for split in split_names}

    # Check datafiles exist 用于判断路径 path 是否存在，若存在返回True，不存在返回False
    datafiles_checks = [os.path.exists(datafile)
                        for datafile in datafiles.values()]

    # Check if prepared dataset exists, and if not set flag to download below.
    # Probably should add more consistency checks, such as number of datapoints, etc...
    # 检查文件下载情况
    new_download = False
    if all(datafiles_checks):
        logging.info('Dataset exists and is processed.')
    # all() 函数用于判断给定的可迭代参数 iterable 中的所有元素是否都为 TRUE，如果是返回 True，否则返回 False。
    elif all([not x for x in datafiles_checks]):
        # If checks are failed.
        new_download = True
    else:
        raise ValueError(
            'Dataset only partially processed. Try deleting {} and running again to download/process.'.format
            (os.path.join(dataset_dir)))

    # If need to download dataset, pass to appropriate downloader
    if new_download or force_download:
        logging.info('Dataset does not exist. Downloading!')
        if dataset.lower().startswith('qm9'):
            download_dataset_qm9(datadir, dataset, splits, cleanup=cleanup)
        else:
            raise ValueError(
                'Incorrect choice of dataset! Must chose qm9!')

    return datafiles
