#!/bin/bash

host=`hostname -s`
echo "server host: $host"
if [ "$host" == "$headnode" ] && [ $SLURM_LOCALID == "0" ]; then
    ./build/server --dataset ./cmmlu_data/agronomy.csv --server_addr $headaddr --server_port $port
fi