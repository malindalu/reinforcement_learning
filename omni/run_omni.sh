#!/bin/bash
#SBATCH -p mit_normal_gpu      # partition name
#SBATCH --job-name=omni_cpo_costlim10_adjr           # name for your job
#SBATCH --gres=gpu:1                 # if you need GPUs
#SBATCH --ntasks=1                   # number of tasks (often 1 for serial jobs)
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mem=16G                    # memory per node
#SBATCH --time=05:00:00              # max walltime (HH:MM:SS)
#SBATCH --output=slurm-omni-cpo-costlim10-adjr-%j.out        # output file (%j = job ID) to capture logs for debugging

#!/bin/bash
#SBATCH -p mit_normal_gpu      # partition name
#SBATCH --job-name=omni_cpo            # name for your job
#SBATCH --gres=gpu:1                 # if you need GPUs
#SBATCH --ntasks=1                   # number of tasks (often 1 for serial jobs)
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mem=16G                    # memory per node
#SBATCH --time=05:00:00              # max walltime (HH:MM:SS)
#SBATCH --output=slurm-omni-cpo-%j.out        # output file (%j = job ID) to capture logs for debu$

# Load your shell environment to activate your Conda environment
module load miniforge
# source /home/user/.bashrc
conda activate rl_omni
# export PYTHONNOUSERSITE=1

# conda install pytorch numpy -c pytorch
# pip install -r requirements_work.txt

# Run your command or script
python cpo.py --cost-limit 10