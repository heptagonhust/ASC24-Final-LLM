#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=TRT-multinodes-DP
#SBATCH --partition=Compute
#SBATCH --output=./outputs/dp_%j.out
#SBATCH --error=./outputs/dp_%j.err
#SBATCH --export=ALL
#SBATCH --exclusive

export headnode=`hostname -s`
export headaddr="$(ip addr | awk '/^[0-9]+: / {}; /inet.*global/ {print gensub(/(.*)\/(.*)/, "\\1", "g",$2)}' | grep "192.168.250.*")"
export port=12345
# export A100_engine_dir=$1
# export H800_engine_dir=$2
export NVIDIA_VISIBLE_DEVICES=all

echo "head node: $headnode"

srun --mpi=pmix --overlap -J run_dp_server ./run_server.sh &
srun --mpi=pmix --overlap -J run_inference ./run_client.sh &
wait