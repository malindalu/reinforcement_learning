#!/bin/bash
#SBATCH -p mit_normal_gpu      # partition name
#SBATCH --job-name=omni_cpo            # name for your job
#SBATCH --gres=gpu:1                 # if you need GPUs
#SBATCH --ntasks=1                   # number of tasks (often 1 for serial jobs)
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mem=16G                    # memory per node
#SBATCH --time=05:00:00              # max walltime (HH:MM:SS)
#SBATCH --output=slurm-omni-cpo-%j.out        # output file (%j = job ID) to capture logs for debugging

# Load your shell environment to activate your Conda environment
source /home/user/.bashrc
conda activate rl_omni
pip install -r omni/equirements_cluster.txt

# Run your command or script
python omni/cpo.py