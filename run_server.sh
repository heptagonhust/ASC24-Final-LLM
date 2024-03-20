#!/bin/bash

host=`hostname -s`
if [ "$host" == "$headnode" ] && [ $SLURM_LOCALID == "0" ]; then
    echo $headaddr
    echo $port
    singularity exec --bind /data:/mnt --writable -e  ~/containers/trtllm-sandbox ./build/server --server_addr $headaddr --server_port $port &
fi