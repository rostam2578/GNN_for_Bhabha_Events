#! /bin/bash

######## Part 1 #########
# Script parameters     #
#########################

# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu

# Specify the QOS, mandatory option
#SBATCH --qos=debug

# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
# write the experiment which will pay for your resource consumption
#SBATCH --account=mlgpu

# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=gpu_test

# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1

# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/GNN/RealisticGrid/TestCuda.out

# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --mem-per-cpu=8192

# Specify how many GPU cards to use
#SBATCH --gres=help

######## Part 2 ######
# Script workload    #
######################

# Replace the following lines with your real workload

# list the allocated hosts
srun -l hostname

# list the GPU cards of the host
/usr/bin/nvidia-smi -L

echo "Allocate GPU cards : ${CUDA_VISIBLE_DEVICES}"

# Cuda info
python ./TestCuda.py


#python -c "import torch; print('torch version', torch.__version__)"
#python -c "import torch; print('cuda version', torch.version.cuda)"
#python -c "import torch; print('is cuda available', torch.cuda.is_available())"
#python -c "import torch; print('CUDNN VERSION', torch.backends.cudnn.version())"
#python -c "import torch; print('Number CUDA Devices:', torch.cuda.device_count())"
#python -c "import torch; print('CUDA Device Name:',torch.cuda.get_device_name())"
#python -c "import torch; print('CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)"

