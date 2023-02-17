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
#SBATCH --output=/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/GNN/RealisticGrid/GraphDef.out

# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --mem-per-cpu=8192

# Specify how many GPU cards to use
#SBATCH --gres=gpu:v100:3

######## Part 2 ######
# Script workload    #
######################
# list the allocated hosts
srun -l hostname

# list the GPU cards of the host
/usr/bin/nvidia-smi -L
echo "Allocate GPU cards : ${CUDA_VISIBLE_DEVICES}"

# Cuda info
time(python ./TestCuda.py)

# Graph Formation
time(python ./GraphDef.py)
