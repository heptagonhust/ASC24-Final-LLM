#!/bin/bash

host=`hostname -s`
echo "server host: $host"
if [ "$host" == "$headnode" ] && [ $SLURM_LOCALID == "0" ]; then
    ./build/server --dataset ./mmlu_data/abstract_algebra_test.csv --server_addr $headaddr --server_port $port --test_choice=mmlu
fi