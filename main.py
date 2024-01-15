import os.path as osp
import pandas as pd
import torch
import matplotlib.pyplot as plt
import math
import networkx as nx
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.transforms as T

"""
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
"""

### Testing Parameters ###
root = './'
filename = 'May 4th Dataset Car and Tower  - Sheet1 (1).csv'

num_clusters = 20
n = 1000  # epochs?
clusters = []

max_dist = 500

# Transform Parameters
transform_set = False
value_num = 0.1
test_num = 0.2

# Optimizer Parameters (learning rate)
learn_rate = 0.9

# Number of iterations/generations of the training dataset
epochs = n

# Colour setup?


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
        from_path = osp.join(self.root, self.filename)
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

        edge_source = []
        edge_target = []

        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                dist = math.dist([df.loc[i, 'X'], df.loc[i, 'Y']], [df.loc[j, 'X'], df.loc[j, 'Y']])

                if 0 < dist <= max_dist:
                    edge_source.append(i)
                    edge_target.append(j)

        data = Data(
            x=x,
            edge_index=torch.tensor([edge_source, edge_target])
        )
        torch.save(data, self.processed_paths[0])


transform = T.RandomLinkSplit(
    num_val=value_num, num_test=test_num,
    is_undirected=True,
    split_labels=True
)

if transform_set:
    dataset = MyDataset(root, filename, transform=T.NormalizeFeatures)
else:
    dataset = MyDataset(root, filename)

data = dataset[0]

train_data, val_data, test_data = transform(data)

print('Dataset:', dataset)
print('Data:', data)
print('Train data:', train_data)
print('Test data:', test_data)
