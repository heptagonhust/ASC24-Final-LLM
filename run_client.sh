#!/bin/bash

host=`hostname -s`
if [ "$host" == "hepnode2" ] || [ "$host" == "hepnode3" ] || [ "$host" == "hepnode4" ]; then
singularity exec --bind /data:/mnt --writable -e ~/container \
    bash -c "CUDA_VISIBLE_DEVICES=$SLURM_LOCALID ./build/client --engine_dir ~/trt-llm-engines/bs64-A100-awq-int4-int8 --eos_id 100007 --pad_id 0 --rpc_address $headaddr --rpc_port $port --rpc_seqs_threshold 300"
else
singularity exec --bind /data:/mnt --writable -e ~/container \
    bash -c "CUDA_VISIBLE_DEVICES=$SLURM_LOCALID ./build/client --engine_dir ~/trt-llm-engines/bs256-h800-awq_int4_fp8 --eos_id 100007 --pad_id 0 --rpc_address $headaddr --rpc_port $port --rpc_seqs_threshold 300"
fi
