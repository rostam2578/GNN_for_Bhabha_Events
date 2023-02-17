print('\n\n\n\n', 'Results ...', '\n\n')
#from GraphDef import *
#from DataLoad import *
#from Model import modelname
#from Training import *

# Results
# Results of the GCN for some events
import matplotlib.pyplot as plt
import os
from os import path
EvNum = 72601
load_model = True
Thrd = 0.79

if load_model:
  from Model import *
  from Training import optimizer
else:
  from Training import *

log_dir = "/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/bhabha-files/" + modelname #+ time.time()
checkpoint_dir = "/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/bhabha-files/" + modelname

def checkpoint_load(loadedcheckpoint):
  print("=> loading checkpoint from", F"{checkpoint_dir}/saved_checkpoint.pth.tar")
  net.load_state_dict(loadedcheckpoint['state_dict'])
  optimizer.load_state_dict(loadedcheckpoint['optimizer'])

if load_model:
  checkpoint_load(torch.load(F"{checkpoint_dir}/saved_checkpoint.pth.tar"))

data1 = Data(x=TraTen[EvNum].reshape(-1,43*43, 1), edge_index=edge_index).to(device)
result2 = net(data1.x, data1.edge_index)

fig = plt.figure(figsize=(25, 20))
ax1 = fig.add_subplot(141)
ax1.matshow((torch.Tensor.cpu(result2).detach().numpy()).reshape(1849).reshape(43, 43))
plt.title('GNN results')

ax2 = fig.add_subplot(142)
ax2.matshow((torch.Tensor.cpu(result2).detach().numpy()).reshape(1849).reshape(43, 43) < Thrd)
plt.title(f'GNN results. Threshold is <{Thrd}.')

ax3 = fig.add_subplot(143)
ax3.matshow(torch.Tensor.cpu(TrvTen[EvNum].reshape(43, 43)))
plt.title('data without noise')

ax4 = fig.add_subplot(144)
ax4.matshow(torch.Tensor.cpu(TraTen[EvNum].reshape(43, 43)))
plt.title(f'data with noise, event {EvNum}')

plt.savefig(f'GNN result for event number {EvNum} and threshold {Thrd}.png', bbox_inches='tight')

purres = np.zeros(shape=20)
effres = np.zeros(shape=20)
purity = np.zeros(shape=(10000, 20))
efficiency = np.zeros(shape=(10000, 20))
for i in range(0, 10000):
  data = Data(x=TraTen[i + 10000].reshape(-1,43*43, 1), edge_index=edge_index).to(device)
  result = net(data.x, data.edge_index)
  gnnres = (torch.Tensor.cpu(result).detach().numpy()).reshape(1849)
  for j in range(0, 20):
     thrres = gnnres < 0.3 + j * 0.03
     datnoi = torch.Tensor.cpu(TraTen[i + 10000].reshape(1849)).numpy() < 1
     datnon = torch.Tensor.cpu(TrvTen[i + 10000].reshape(1849)).numpy() < 1
     outres = thrres & datnoi 
     comres = outres & datnon
     purity[i, j] = comres.sum() / outres.sum()
     efficiency[i, j] = comres.sum() / datnon.sum()
purres = purity.mean(axis = 0)
effres = efficiency.mean(axis = 0)

plt.figure(figsize=(25, 20))
plt.scatter(np.arange(0.3, 0.89, 0.03), purres, label='purity')
plt.scatter(np.arange(0.3, 0.89, 0.03), effres, label='efficiency')
plt.legend()
plt.xlabel('threshold')
plt.xlim([0, 1])
plt.savefig(f'Purity and efficiency.png', bbox_inches='tight')

print(datetime.datetime.now() - t1)
