#!/bin/bash

singularity exec --bind /data:/mnt --writable -e ~/containers/trtllm-sandbox/ \
    bash -c "CUDA_VISIBLE_DEVICES=$SLURM_LOCALID ./build/client --engine_dir ~/trt-llm-engines/bs256-h800 --eos_id 100007 --pad_id 0 --rpc_address $headaddr --rpc_port $port --rpc_seqs_threshold 500"
