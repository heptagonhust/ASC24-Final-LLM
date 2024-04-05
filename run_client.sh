#!/bin/bash

host=`hostname -s`
if [ "$host" == "hepnode3" ] || [ "$host" == "hepnode4" ]; then
if [ "$SLURM_LOCALID" == "0" ];then
    singularity exec --bind /data:/mnt --writable -e ~/container \
        bash -c "CUDA_VISIBLE_DEVICES=$SLURM_LOCALID ./build/client --engine_dir $H800_engine_dir --eos_id 100007 --pad_id 0 --rpc_address $headaddr --rpc_port $port --rpc_batch_size 500 --rpc_seqs_threshold 300"
else
    singularity exec --bind /data:/mnt --writable -e ~/container \
        bash -c "CUDA_VISIBLE_DEVICES=$SLURM_LOCALID ./build/client --engine_dir $A100_engine_dir --eos_id 100007 --pad_id 0 --rpc_address $headaddr --rpc_port $port --rpc_batch_size 300 --rpc_seqs_threshold 200"
fi
elif [ "$host" == "hepnode2" ]; then
    singularity exec --bind /data:/mnt --writable -e ~/container \
        bash -c "CUDA_VISIBLE_DEVICES=$SLURM_LOCALID ./build/client --engine_dir $A100_engine_dir --eos_id 100007 --pad_id 0 --rpc_address $headaddr --rpc_port $port --rpc_batch_size 300 --rpc_seqs_threshold 200"    
else
    singularity exec --bind /data:/mnt --writable -e ~/container \
        bash -c "CUDA_VISIBLE_DEVICES=$SLURM_LOCALID ./build/client --engine_dir $H800_engine_dir --eos_id 100007 --pad_id 0 --rpc_address $headaddr --rpc_port $port --rpc_batch_size 500 --rpc_seqs_threshold 300"
fi