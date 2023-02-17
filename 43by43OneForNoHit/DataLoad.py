print('\n\n\n\n\n', 'Loading data ...', '\n\n')

import numpy as np
import torch
device = torch.device('cuda')

# Upload input files.
trainmatrix80d2tnb1mt = np.load('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/bhabha-files/trainmatrix80d2tnb1mt.npy')
trvalmatrix80d2tnb1mt = np.load('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/bhabha-files/trvalmatrix80d2tnb1mt.npy')

trvalmatrix80d2tnb1mt.shape
trainmatrix80d2tnb1mt.shape
(trvalmatrix80d2tnb1mt > 0).sum()
(trainmatrix80d2tnb1mt>0).sum()

# Increasing the layers to 43 by adding zeros because the above graph definition works if dimensions are equal
a = trainmatrix80d2tnb1mt
b = trvalmatrix80d2tnb1mt
a_tra = np.ones(shape=(80000, 43, 43))#, dtype=torch.double)
b_trv = np.ones(shape=(80000, 43, 43))
a_tra[:a.shape[0], :a.shape[1], :a.shape[2]] = a
b_trv[:b.shape[0], :b.shape[1], :b.shape[2]] = b

(a_tra>0).sum()
(b_trv>0).sum()
a_tra.shape
b_trv.shape

TraTen = torch.tensor(a_tra.reshape(80000, 43 * 43)).to(device)
TrvTen = torch.tensor(b_trv.reshape(80000, 43 * 43)).to(device)

TraTen.shape
