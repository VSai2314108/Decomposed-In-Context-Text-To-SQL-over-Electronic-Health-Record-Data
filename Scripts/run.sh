#!/bin/bash
#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --output=./outputs/vasp.out
#SBATCH --error=./outputs/vasp.err
#SBATCH --mail-type=NONE
#SBATCH --mail-user=email@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=8
#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=7000mb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:4
#SBATCH --time=00:30:00

module purge
module load cuda/11.4.3 nvhpc/23.7 openmpi/4.1.5 vasp/6.4.1 python/3.10

cd /blue/cap4773/somasundaramv/Few-shot-NL2SQL-with-prompting/
source .venv/bin/activate
srun --mpi=${HPC_PMIX} python3 -u DIN-SQL-LLAMA-CPP.py --dataset ./data/ --output predicted_sql.txt