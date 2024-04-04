#!/bin/bash
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --job-name=TRT-multinodes-DP
#SBATCH --partition=Compute
#SBATCH --output=outputs/dp_%j.out
#SBATCH --error=outputs/dp_%j.err
#SBATCH --export=ALL
#SBATCH --exclusive

export headnode=`hostname -s`
export headaddr="$(ip addr | awk '/^[0-9]+: / {}; /inet.*global/ {print gensub(/(.*)\/(.*)/, "\\1", "g",$2)}' | grep "192.168.250.*")"
export port=12340


srun --mpi=pmix --overlap -J run_dp_server ./run_server.sh &
srun --mpi=pmix --overlap -J run_inference ./run_client.sh &
wait