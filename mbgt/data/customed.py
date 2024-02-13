import torch
from torch_geometric.data import Data, Dataset

class CustomGraphDataset(Dataset):
    def __init__(self, node_features_path,transform=None):
        self.node_features_and_labels = torch.load(node_features_path)
        self._indices = None
        self.transform=transform

    def get(self, idx):
        node_features = self.node_features_and_labels[idx][0]
        label = self.node_features_and_labels[idx][1]
        spatial_encoding=self.node_features_and_labels[idx][2]
        label=label.long()
        data = Data(x=node_features, edge_index=None, edge_attr=spatial_encoding, y=label)
        return data

    def len(self):
        return len(self.node_features_and_labels)