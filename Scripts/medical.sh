#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --output=./outputs/vasp.out
#SBATCH --error=./outputs/vasp.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=somasundaramv@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --distribution=cyclic:cyclic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=70000
#SBATCH --time=48:00:00

export HF_HOME=/blue/daisyw/somasundaramv/mis
export HF_DATASETS_CACHE=/blue/daisyw/somasundaramv/datasets
export TRANSFORMERS_CACHE=/blue/daisyw/somasundaramv/models
cd /blue/daisyw/somasundaramv/Few-shot-NL2SQL-with-prompting/
module purge
module load python/3.10 cuda/11.4.3 nvhpc/23.7 openmpi/4.1.5 vasp/6.4.1
# rm -rf .autogptq
# python3 -m venv .autogptq
source .autogptq/bin/activate
# pip install --upgrade pip
# pip install wheel setuptools
# pip install torch torchvision torchaudio
# pip install accelerate
# pip install transformers
# pip install optimum
# pip install pandas
# pip3 install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu117/  # Use cu117 if on CUDA 11.7
srun --mpi=${HPC_PMIX} python3 -u ./MainScript/DIN-SQL-LLAMA-MEDICAL.py --dataset /blue/daisyw/somasundaramv/Few-shot-NL2SQL-with-prompting/TREQS/mimicsql_data/mimicsql_natural_v2/ --output predicted_sql.txt