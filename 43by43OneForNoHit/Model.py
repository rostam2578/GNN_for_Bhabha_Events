print('\n\n\n\n', 'The Network ...', '\n\n')
from GraphDef import *
from DataLoad import *

# Defining the network
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
writer = SummaryWriter()
device = torch.device('cuda')

modelname = 'bhabaGCN04b1'
class Net(torch.nn.Module):
    def __init__(self):        
        super(Net, self).__init__()
        self.conv1 = GCNConv(1, 256)
        self.conv2 = GCNConv(256, 256)
        self.conv3 = GCNConv(256, 128)
        self.conv4 = GCNConv(128, 64)
        self.conv5 = GCNConv(64, 32)
        self.conv6 = GCNConv(32, 1)
        #self.Linear = Linear(2, 1)
        #self.classifier = Linear(2, 1)
        #self.conv4 = GCNConv(64, 1849)
        #self.Linear = Linear(16, 1600)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.conv6(x, edge_index)
        #x = F.relu(x)
        #x = self.Linear(x)
        x = torch.sigmoid(x)
        return x

vars()[modelname] = Net().double().to(device)
net = vars()[modelname]
print(net)

# Passing two sample data from the network before training
EvBTr = 20
data1 = Data(x=TraTen[EvBTr : EvBTr + 2].reshape(-1, 43*43, 1), edge_index=edge_index).to(device)
result1 = net(data1.x, data1.edge_index)
print('\nPassing two sample data from the network before training', '\nresult1:', result1, '\ndata1:',  data1, result1.shape, '\nx.shape:', x.shape)

result1 

data1

result1.shape

plt.matshow((torch.Tensor.cpu(result1[0]).detach().numpy()).reshape(1849).reshape(43, 43))
plt.title(f'passing event number {EvBTr} through the network \n before training.png')
plt.savefig(f'passing event number {EvBTr} through the network before training.png', bbox_inches='tight')
print(result1[0].reshape(1849).reshape(43, 43))
