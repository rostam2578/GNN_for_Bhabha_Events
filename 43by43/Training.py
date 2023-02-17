print('\n\n\n\n', 'Training ...', '\n\n')
from GraphDef import *
from DataLoad import *
from Model import *

# Training
import os
from os import path
load_model = False
TraEvN = 70001
BatchSize = 50
EpochNum = 5
epoch_save = 3

log_dir = "/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/bhabha-files/" + modelname #+ time.time()
checkpoint_dir = "/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/bhabha-files/" + modelname 
if not os.path.exists('checkpoint_dir'):
  os.mkdir('checkpoint_dir')

def checkpoint_save(state, epoch):
  print("=> saveing checkpoint at epoch", epoch)
  torch.save(state, F"{checkpoint_dir}/saved_checkpoint.pth.tar")

def checkpoint_load(loadedcheckpoint):
  print("=> loading checkpoint from", F"{checkpoint_dir}/saved_checkpoint.pth.tar")
  net.load_state_dict(loadedcheckpoint['state_dict'])
  optimizer.load_state_dict(loadedcheckpoint['optimizer'])

loss_function = torch.nn.BCELoss()#CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.

if load_model:
  checkpoint_load(torch.load(F"{checkpoint_dir}/saved_checkpoint.pth.tar"))

#Clear gradients.
t1 = datetime.datetime.now()
for epoch in range(EpochNum):
    mloss = 0
    for i in np.arange(0, TraEvN, BatchSize):
        optimizer.zero_grad()
        xi = (TraTen[i : i + BatchSize]).reshape(-1, 43*43, 1)
        datai = Data(x=xi, edge_index=edge_index).to(device)
        outi = net(datai.x, datai.edge_index)#.type(torch.LongTensor)  # Perform a single forward pass.
        truevaluebach = ((TrvTen[i : i + BatchSize])>0).type(torch.double).to(device)
        loss = loss_function((outi.reshape(BatchSize, 43*43)), truevaluebach)  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        mloss += loss
        #print the loss function for the event which comes after every 50 batches.
        if i % (50 * BatchSize) == 0:
            print('epoch:', epoch, 'batch', i / (50 * BatchSize), 'event:', i, 'loss:', loss)
    print('time passed so far:', datetime.datetime.now() - t1)
    print('epoch:', epoch, 'mean loss:', mloss/(TraEvN//(BatchSize)))

    if (epoch % epoch_save == 0) or epoch == EpochNum - 1:
      checkpoint = {'state_dict' : net.state_dict(), 'optimizer' : optimizer.state_dict()}
      checkpoint_save(checkpoint, epoch)
      print('checkpoint is saved at:', checkpoint_dir)
    writer.add_scalar(log_dir, loss, BatchSize)
print(datetime.datetime.now() - t1)
