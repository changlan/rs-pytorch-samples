#!/bin/sh

# Change the following variables as needed:
NUM_GPUS_PER_WORKER=8
NUM_WORKERS=`expr $WORLD_SIZE / $NUM_GPUS_PER_WORKER`
WORKER_RANK=`expr $RANK / $NUM_GPUS_PER_WORKER`

echo "World size: $WORLD_SIZE"
echo "GPUs per worker: $NUM_GPUS_PER_WORKER"
echo "Number of workers: $NUM_WORKERS"
echo "Worker rank: $WORKER_RANK"

python -m torch.distributed.launch \
  --nproc_per_node $NUM_GPUS_PER_WORKER --nnodes $NUM_WORKERS --node_rank $WORKER_RANK \
  --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  mnist_trainer.py