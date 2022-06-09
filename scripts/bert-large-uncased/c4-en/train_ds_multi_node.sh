#!/bin/bash

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=10.146.113.163
# MASTER_PORT=6000
NNODES=4

MICRO_BATCH_SIZE=24
GLOBAL_BATCH_SIZE=8064

RUN_NAME=bert_no_binary_345m_ds_multi_run001

CHECKPOINT_PATH=/mnt/ssd-1/Megatron-DeepSpeed/checkpoints/$RUN_NAME
TENSORBOARD_PATH=/mnt/ssd-1/Megatron-DeepSpeed/checkpoints/$RUN_NAME/tensorboard
VOCAB_FILE=/home/mchorse/marius/Megatron-DeepSpeed/data/bert-large-uncased-vocab.txt
DATA_PATH=/mnt/ssd-1/data/c4/c4-train_text_document
DS_CONFIG=/home/mchorse/marius/Megatron-DeepSpeed/ds_config.json

# use --load $RESUME_PATH  to start from checkpoint
# RESUME_PATH=/mnt/Megatron-DeepSpeed/checkpoints/bert_no_binary_345m_dist_run000/iter_0001000


BERT_ARGS="--bert-no-binary-head \
           --num-layers 24 \
           --hidden-size 1024 \
           --num-attention-heads 16 \
           --seq-length 512 \
           --max-position-embeddings 512 \
           --mask-prob 0.15 \
           --lr 0.0001 \
           --lr-decay-iters 100 \
           --train-iters 100 \
           --min-lr 0.00001 \
           --lr-warmup-fraction 0.1 \
	       --micro-batch-size $MICRO_BATCH_SIZE \
           --global-batch-size $GLOBAL_BATCH_SIZE \
           --tokenizer-type BertWordPieceLowerCase \
           --vocab-file $VOCAB_FILE \
           --split 949,50,1 \
           --weight-decay 0.01 \
           --clip-grad 1.0 \
           --adam-beta1 0.9 \
           --adam-beta2 0.98 \
           --adam-eps 1e-6 \
           --fp16 \
           --seed 123"

OUTPUT_ARGS="--log-interval 10 \
             --tensorboard-dir $TENSORBOARD_PATH \
             --tensorboard-log-interval 10 \
             --log-params-norm \
             --log-num-zeros-in-grad \
             --log-timers-to-tensorboard \
             --log-batch-size-to-tensorboard \
             --log-validation-ppl-to-tensorboard \
             --save-interval 100 \
             --eval-interval 1000 \
             --eval-iters 10"


DEEPSPEED_ARGS=" \
        --deepspeed \
        --deepspeed_config $DS_CONFIG \
        --zero-stage 2
"

deepspeed --num_nodes ${NNODES} --num_gpus ${GPUS_PER_NODE} \
       pretrain_bert.py \
       $BERT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       $DEEPSPEED_ARGS
