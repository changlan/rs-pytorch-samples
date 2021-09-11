FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-8
WORKDIR /root

# Install Reduction Server NCCL Plugin
RUN apt-get update && \
    apt-get remove -y google-fast-socket && \
    apt-get install -y libcupti-dev google-reduction-server

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
ENV NCCL_DEBUG=INFO

COPY mnist_trainer.py mnist_trainer.py
COPY run.sh run.sh

ENTRYPOINT ["/root/run.sh"]
