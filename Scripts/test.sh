#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --output=./outputs/vasp.out
#SBATCH --error=./outputs/vasp.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=somasundaramv@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --distribution=cyclic:cyclic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=70000mb
#SBATCH --time=24:00:00

cd /blue/cap4773/somasundaramv/Few-shot-NL2SQL-with-prompting/
module purge
module load conda cuda/11.4.3 nvhpc/23.7 openmpi/4.1.5 vasp/6.4.1 
source .venv/bin/activate
srun --mpi=${HPC_PMIX} python3 -u llama-test-hf.py