import os
os.system('echo ''; echo modinfo: ; modinfo nvidia')
os.system('echo ''; echo nvidia-smi: ; nvidia-smi')
os.system('echo ''; echo nvcc --version: ; nvcc --version')


import torch
print('\n', 'torch version:', torch.__version__)
print('\n', 'cuda version:', torch.version.cuda)
print('\n', 'is cuda available:', torch.cuda.is_available())
print('\n', 'CUDNN VERSION:', torch.backends.cudnn.version())
print('\n', 'Number CUDA Devices:', torch.cuda.device_count())
print('\n', 'CUDA Device Name:',torch.cuda.get_device_name(device=None))
print('\n', 'CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(device=None).total_memory/1e9)
print(
'\n', 'Device capability:',  torch.cuda.get_device_capability(device=None), \
'\n\n', 'Cuda deviice:',  torch.cuda.device(device=None), \
'\n\n', 'Is cuda initialized:', torch.cuda.is_initialized(), \
 )

from torch.utils.cpp_extension import CUDA_HOME
print('\n', 'CUDA_HOME:', CUDA_HOME)

'''import numpy as np
traingnn80 = np.load('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/bhabha-files/traingnn80.npy')
print(traingnn80)'''
