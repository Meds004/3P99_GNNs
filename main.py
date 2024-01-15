import os.path as osp
import pandas as pd
import torch
import matplotlib.pyplot as plt
import math
import networkx as nx
import numpy as np
from torch_geometric.data import InMemoryDataset

### Create dataframe from dataset ###
df = pd.read_csv("May 4th Dataset Car and Tower  - Sheet1 (1).csv")
df.reset_index(drop=True, inplace=True)
df['Car ID'] = pd.to_numeric(df['Car ID'], errors='coerce')
print(df)

### Create Graph ###
G = nx.Graph()
edge_index = []

### Add nodes to graph ###
for i in range(len(df)):
    G.add_node(i, x=df.loc[i, 'X'], y=df.loc[i, 'Y'])

car_ids = torch.tensor(df['Car ID'].values, dtype=torch.float)
vertex = car_ids

### Add edges to graph ###
for i in range(len(df)):
    # distances = []
    for j in range(len(df)):
        if i < j:
            dist = math.dist([df.loc[i, 'X'], df.loc[i, 'Y']], [df.loc[j, 'X'], df.loc[j, 'Y']])
            G.add_edge(i, j, weight=dist)
            # can add distance condition here, otherwise fully connected graph
            # distances.append((dist, j))

    # for k in range(len(distances)):
    #     G.add_edge(i, distances[k][1], weight=distances[k][0])

print(G)
fig, ax = plt.subplots(figsize=(12, 8))
X = df[['X', 'Y']].values
pos = dict(zip(range(len(df)), X))
nx.draw(G, pos, node_size=15, node_color='blue', edge_color='white')
plt.show()


class MyDataset(InMemoryDataset):
    def __init__(self, root, file_name, transform=None, pre_transform=None):
        self.filename = file_name
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.filename]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        from_path = osp.join(root, filename)
        to_path = osp.join(self.raw_dir, self.filename)
        df = pd.read_csv(from_path)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(to_path, index=False)

    def read_file(self):
        path = osp.join(self.raw_dir, self.filename)
        df = pd.read_csv(path)
        df.reset_index(drop=True, inplace=True)

        df['Type'] = np.where((df['Car ID'].str.contains('Tower')) | (df['Car ID'].str.contains('RSU')), 1, 0)
        df.drop(columns=['Car ID'], inplace=True)
        return df

    def process(self):
        df = self.read_file()
        x = torch.from_numpy(df.values).float()

        for i in range(len(df)):
            for j in range(len(df)):
                if i == j:

# root = './'
# filename = 'May 4th Dataset Car and Tower  - Sheet1 (1).csv'


