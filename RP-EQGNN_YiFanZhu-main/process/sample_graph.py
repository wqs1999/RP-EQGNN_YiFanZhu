import numpy as np
import torch


class SampleGraph(object):
    def __init__(self, data, property, cuda):
        self.num = data['num_atoms']

        self.h = data['atom_feature'][data['atom_mask']]
        self.x = data['positions'][data['atom_mask']]

        batch_size, n_nodes, _ = data['positions'].size()

        self.edges = torch.LongTensor(np.vstack(data['bonds']))

        batch = []
        for i, n in enumerate(data['num_atoms']):
            batch.append(np.ones(n) * i)
        self.batch = np.hstack(batch).astype(int)
        self.nG = batch_size
        self.batch = torch.LongTensor(self.batch)

        self.y = data[property]

        if cuda:
            self.to(torch.device('cuda'))

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)
        return self

    def __repr__(self):
        return f"""In the batch: num_graphs {self.nG} num_nodes {len(self.h)}
> .h \t\t a tensor of nodes representations \t\tshape {' x '.join(map(str, self.h.shape))}
> .x \t\t a tensor of nodes positions \t\t\tshape {' x '.join(map(str, self.x.shape))}
> .edges \t a tensor of edges  \t\t\t\t\tshape {' x '.join(map(str, self.edges.shape))}
"""
