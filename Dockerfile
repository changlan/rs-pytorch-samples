FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-7
WORKDIR /root

# Install Reduction Server NCCL Plugin
RUN apt-get update && \
    apt-get remove -y google-fast-socket && \
    apt-get install -y libcupti-dev google-reduction-server

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
ENV NCCL_DEBUG=INFO

COPY mnist_trainer.py mnist_trainer.py

# Change the following variables as needed:
# --nproc_per_node: number of GPUs per worker
# --nnodes: number of workers (including the master node)
ENTRYPOINT python -m torch.distributed.launch \
  --nproc_per_node 8 --nnodes 2 \
  --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  mnist_trainer.py
