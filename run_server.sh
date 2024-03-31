#!/bin/bash

host=`hostname -s`
if [ "$host" == "$headnode" ] && [ $SLURM_LOCALID == "0" ]; then
    ./build/server --server_addr $headaddr --server_port $port --batch_size 50
fi