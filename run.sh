#!/bin/sh

# Change the following variables as needed:
NUM_GPUS_PER_WORKER=${NUM_GPUS_PER_WORKER:-8}

echo "Number of workers: $WORLD_SIZE"
echo "Worker rank: $RANK"
echo "GPUs per worker: $NUM_GPUS_PER_WORKER"

python -m torch.distributed.launch \
  --nproc_per_node $NUM_GPUS_PER_WORKER --nnodes $WORLD_SIZE --node_rank $RANK \
  --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  mnist_trainer.py