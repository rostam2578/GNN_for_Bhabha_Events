print('\n\n\n\n', 'Results ...', '\n\n')
from GraphDef import *
from DataLoad import *
from Model import *
#from Training import *

# Results
# Results of the GCN for some events
import matplotlib.pyplot as plt
import os
from os import path
EvNum = 73230
load_model =True

log_dir = "/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/bhabha-files/" + modelname #+ time.time()
checkpoint_dir = "/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/bhabha-files/" + modelname
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=5e-4)  # Define optimizer.

def checkpoint_load(loadedcheckpoint):
  print("=> loading checkpoint from", F"{checkpoint_dir}/saved_checkpoint.pth.tar")
  net.load_state_dict(loadedcheckpoint['state_dict'])
  optimizer.load_state_dict(loadedcheckpoint['optimizer'])

if load_model:
  checkpoint_load(torch.load(F"{checkpoint_dir}/saved_checkpoint.pth.tar"))
else:
  from Training import *

data1 = Data(x=TraTen[EvNum].reshape(-1,43*43, 1), edge_index=edge_index).to(device)
result2 = net(data1.x, data1.edge_index)

fig = plt.figure(figsize=(25, 20))
ax1 = fig.add_subplot(141)
ax1.matshow((torch.Tensor.cpu(result2).detach().numpy()).reshape(1849).reshape(43, 43))
plt.title('GNN results')

ax2 = fig.add_subplot(142)
ax2.matshow((torch.Tensor.cpu(result2).detach().numpy()).reshape(1849).reshape(43, 43) > 0.20)
plt.title('GNN results. Threshold is applied.')

ax3 = fig.add_subplot(143)
ax3.matshow(torch.Tensor.cpu(TrvTen[EvNum].reshape(43, 43)))
plt.title('data without noise')

ax4 = fig.add_subplot(144)
ax4.matshow(torch.Tensor.cpu(TraTen[EvNum].reshape(43, 43)))
plt.title('data with noise')

plt.savefig(f'GNN result for event number {EvNum}.png', bbox_inches='tight')

print(datetime.datetime.now() - t1)
