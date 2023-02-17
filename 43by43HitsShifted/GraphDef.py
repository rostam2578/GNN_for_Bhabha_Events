print('\n\n\n\n', 'The graph ...', '\n\n')

#Defining the MDC wires as a grid with connected neighbours. 
#Each node on the border is connected vith 5 neighbores.
#Other vertices are connected with 8 neighbotrs.
import torch
from torch_geometric.data import Data
import numpy as np
import datetime
device = torch.device('cuda')

cell = 43
layer = 43
x = np.arange(0, cell * layer)
edge_index = []

x = torch.tensor(x, dtype=torch.float).view(cell * layer, 1)#.to(device)
t1 = datetime.datetime.now()

def wire(i, j):
  return layer * j + i
for i in range(cell):
  for j in range(layer):
    if i == 0:
      edge_index1 = [wire(i, j), wire(i + 1, j)]
      edge_index.extend([edge_index1])
      edge_index2 = [wire(i, j), wire(cell-1, j)]
      edge_index.extend([edge_index2])
      if j != layer - 1:
        edge_index3 = [wire(i, j), wire(cell-1, j + 1)]
        edge_index.extend([edge_index3])
        edge_index4 = [wire(i, j), wire(i + 1, j + 1)]
        edge_index.extend([edge_index4])
        edge_index13 = [wire(i, j), wire(i, j + 1)]
        edge_index.extend([edge_index13])        
      if j != 0:
        edge_index5 = [wire(i, j), wire(cell-1, j - 1)]
        edge_index.extend([edge_index5])
        edge_index6 = [wire(i, j), wire(i + 1, j - 1)]
        edge_index.extend([edge_index6])        
        edge_index14 = [wire(i, j), wire(i, j - 1)]
        edge_index.extend([edge_index14])
    else:
      edge_index7 = [wire(i, j), wire(i - 1, j)]
      edge_index.extend([edge_index7])
      edge_index8 = [wire(i, j), wire(((i + 1) % cell), j)]
      edge_index.extend([edge_index8])
      if j != layer - 1:
        edge_index9 = [wire(i, j), wire(i-1, j + 1)]
        edge_index.extend([edge_index9])
        edge_index10 = [wire(i, j), wire(((i + 1) % cell), j + 1)]
        edge_index.extend([edge_index10])
        edge_index15 = [wire(i, j), wire(i, j + 1)]
        edge_index.extend([edge_index15])
      if j != 0:
        edge_index11 = [wire(i, j), wire(i-1, j - 1)]
        edge_index.extend([edge_index11])
        edge_index12 = [wire(i, j), wire(((i + 1) % cell), j - 1)]
        edge_index.extend([edge_index12])
        edge_index16 = [wire(i, j), wire(i, j - 1)]
        edge_index.extend([edge_index16])

set_edge_index = {(i, j) for i, j in edge_index}
edge_index = list(set_edge_index)
edge_index = torch.tensor(edge_index, dtype=torch.long).view( (8 * cell * layer - 3 * (cell + layer)), 2).t().contiguous()#.to(device)
#edge_index = torch.tensor(edge_index, dtype=torch.long).view( 270, 2).t().contiguous()
data = Data(x=x, edge_index=edge_index)
data = data.to(device)
data
print('graph forming time:', datetime.datetime.now() - t1)

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
print(f'contains self/isolated loop: {data.contains_self_loops()}, {data.contains_isolated_nodes()}')
print(f'is coalesced: {data.is_coalesced()}') #returns True, if edge indices are ordered and do not contain duplicate entries. 
print(f'number of features, per nodes, per edges: {data.num_features, data.num_node_features, data.num_edge_features}')

print('x:', x, '\n', 'edge_index:', edge_index)
x.shape
