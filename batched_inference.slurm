#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:H800:2
#SBATCH --time=24:00:00
#SBATCH --job-name=TRT-multinodes-DP
#SBATCH --partition=Compute
#SBATCH --output=outputs/vllm_%j.out
#SBATCH --error=outputs/vllm_%j.err
#SBATCH --export=ALL


export headnode=`hostname -s`
export headaddr="$(ip addr | awk '/^[0-9]+: / {}; /inet.*global/ {print gensub(/(.*)\/(.*)/, "\\1", "g",$2)}' | grep "192.168.250.*")"
export port=12345

echo $headaddr

#srun --mpi=pmix -J run_dp_server ./run_server.sh

srun --mpi=pmix -J run_inference ./run_client.sh
