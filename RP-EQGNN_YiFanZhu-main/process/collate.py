import torch

def batch_stack(props):
    """
    Stack qm9 list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...] if to_keep.dim() == 1 else props[to_keep]


def collate_fn(batch):
    """
    Collation function that collates datapoints into the * batch * format for cormorant

    Parameters
    ----------
    batch : list of datapoints
        The data to be collated.

    Returns
    -------
    batch : dict of Pytorch tensors
        The collated data.
    """
    batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}
    to_keep_other = (batch['charges'].sum(0) > 0)

    to_keep_b = (batch['bonds'].sum(-1) > 0)  # [:, :, None].expand(-1, -1, 2).squeeze()

    batch = {key: drop_zeros(prop, to_keep_other) if key !='bonds' else drop_zeros(prop, to_keep_b)
             for key, prop in batch.items()}

    atom_mask = batch['charges'] > 0
    batch['atom_mask'] = atom_mask
    # print("batch['atom_mask']", batch['atom_mask'])

    edge_mask = batch['bonds'].sum(dim=1) > 0
    # print('edge_mask', edge_mask)

    batch['edge_mask'] = edge_mask
    # print("batch['edge_mask']", batch['edge_mask'])

    return batch
