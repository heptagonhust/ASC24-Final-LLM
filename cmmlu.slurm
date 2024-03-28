#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --job-name=run_cmmlu
#SBATCH --partition=Compute
#SBATCH --output=outputs/dp_%j.out
#SBATCH --error=outputs/dp_%j.err
#SBATCH --export=ALL

srun -w hepnode1 --mpi=pmix -J run_cmmlu ./run_cmmlu.sh
