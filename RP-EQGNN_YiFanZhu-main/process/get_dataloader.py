from torch.utils.data import DataLoader
from graph_regression_qm9.process.util import initialize_datasets
from graph_regression_qm9.utils.args import init_argparse
from graph_regression_qm9.process.collate import collate_fn


# 检索dataloader batch_size
def retrieve_dataloaders(num_workers=1):
    # Initialize dataloader
    args = init_argparse('qm9')
    args, datasets, num_species, charge_scale = initialize_datasets(args, args.datadir, 'qm9',
                                                                    force_download=args.force_download
                                                                    )

    qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114,
                 'homo': 27.2114,
                 'lumo': 27.2114}

    # 单位转换 1 a.u.（能量）= 1 Hartree = 27.2114 eV
    for dataset in datasets.values():
        dataset.convert_units(qm9_to_eV)

    # Construct PyTorch dataloaders from datasets
    # batch_size=batch_size
    # collate_fn=collate_fn
    dataloaders = {split: DataLoader(dataset,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=num_workers,
                                     collate_fn=collate_fn)
                   for split, dataset in datasets.items()}

    return dataloaders, charge_scale


def get_dataloader(num_workers=3):
    dataloaders, charge_scale = retrieve_dataloaders(num_workers)
    return dataloaders['train'], dataloaders['valid'], dataloaders['test'], charge_scale


if __name__ == '__main__':

    dataloader, _ = retrieve_dataloaders()

    for i, batch in enumerate(dataloader['train']):
        print(i)

