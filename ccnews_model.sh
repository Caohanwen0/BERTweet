
export GPU_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export OMP_NUM_THREADS=1

CONFIG="roberta-base"
TRAIN_CORPUS="twitter-reddit-ccnews"
DATASET_NAME="Article-Bias"
SAVE_PATH="/root/BERTweet/save"

OPTS=""
OPTS+=" --base-path /root/BERTweet"
OPTS+=" --save ${SAVE_PATH}/${CONFIG}-${TRAIN_CORPUS}_${DATASET_NAME}"

OPTS+=" --model-config /root/bm_train_codes/config/${CONFIG}.json"
#OPTS+=" --checkpoint roberta-base"
OPTS+=" --tokenizer /root/BERTweet/tokenizer/tokenizer_twitter_reddit_ccnews.json"
OPTS+=" --load /data0/private/caohanwen/OpenSoCo/save/roberta-mini-from-scratch_twitter_char/checkpoints/checkpoint-37999.pt"

mkdir -p ${SAVE_PATH}/${CONFIG}-${TRAIN_CORPUS}_${DATASET_NAME}
CMD="torchrun \
--nnodes=${NNODES} \
--nproc_per_node=${GPU_PER_NODE} \
--rdzv_id=1 \
--rdzv_backend=c10d distribute.py ${OPTS}"


if [[ $NODE_RANK == 0 ]]; then
    ${CMD} 2>&1 | tee ${SAVE_PATH}/${CONFIG}-${TRAIN_CORPUS}_${DATASET_NAME}/logs_$(date +"%Y_%m_%d_%H_%M_%S").log
else
    ${CMD}
fi
