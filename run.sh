
export GPU_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export OMP_NUM_THREADS=1

torchrun \
--nnodes=${NNODES} \
--nproc_per_node=${GPU_PER_NODE} \
--rdzv_id=1 \
--rdzv_backend=c10d distribute.py