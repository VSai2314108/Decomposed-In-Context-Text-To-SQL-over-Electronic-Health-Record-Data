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
#SBATCH --gres=gpu:a100:2
#SBATCH --mem-per-gpu=70000mb
#SBATCH --time=24:00:00

cd /blue/cap4773/somasundaramv/Few-shot-NL2SQL-with-prompting/
module purge
module load python/3.10 cuda/12.2.2 nvhpc/23.7 openmpi/4.1.5 vasp/6.4.1
# rm -rf .long
# python3 -m venv .long
source .long/bin/activate
# pip install --upgrade pip
# pip install wheel setuptools
# pip install torch torchvision torchaudio
# pip install accelerate
# pip install transformers==4.31.0
# pip install sentencepiece
# pip install ninja
# pip install flash-attn --no-build-isolation
# pip install pandas
srun --mpi=${HPC_PMIX} python3 -u ./MainScript/DIN-SQL-LLAMA.py --dataset ./data/ --output predicted_sql.txt