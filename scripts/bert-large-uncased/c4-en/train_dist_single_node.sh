#!/bin/bash

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

MICRO_BATCH_SIZE=24
GLOBAL_BATCH_SIZE=192

RUN_NAME=bert_no_binary_345m_dist_run005

CHECKPOINT_PATH=/mnt/Megatron-DeepSpeed/checkpoints/$RUN_NAME
TENSORBOARD_PATH=/mnt/Megatron-DeepSpeed/checkpoints/$RUN_NAME/tensorboard
VOCAB_FILE=/Megatron-DeepSpeed/data/bert-large-uncased-vocab.txt
DATA_PATH=/mnt/datasets/c4/en/preprocessed/c4-train_text_document

# use --load $RESUME_PATH  to start from checkpoint
RESUME_PATH=/mnt/Megatron-DeepSpeed/checkpoints/bert_no_binary_345m_dist_run003
# use --override-lr-scheduler to change lr schedule


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

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
             --save-interval 5 \
             --eval-interval 1000 \
             --eval-iters 10"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       $BERT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --load $RESUME_PATH