import pickle 

from libraries.strategies import deserialize
from torch.utils.data import Dataset

class VDSDataset(Dataset):
    def __init__(self, path2dgl_graph, begin_percentage=0.0, end_percentage=100.0):
        super().__init__()
        graph_and_label = deserialize(path2dgl_graph, pickle)
        self.graph_and_label = graph_and_label[int(begin_percentage*len(graph_and_label)):int(end_percentage*len(graph_and_label))]
        self.num_node_features = self.graph_and_label[0]['embedding'].shape[-1]
        self.num_classes = max([g.label for g in self.graph_and_label]) + 1

    def process(self):
        pass 

    def __getitem__(self, idx):
        return self.graph_and_label[idx]

    def __len__(self):
        return len(self.graph_and_label)
