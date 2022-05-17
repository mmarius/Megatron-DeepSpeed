#!/bin/bash

RUN_NAME=bert_no_binary_345m_run000
CHECKPOINT_PATH=/mnt/Megatron-DeepSpeed/checkpoints/$RUN_NAME
TENSORBOARD_PATH=/mnt/Megatron-DeepSpeed/checkpoints/$RUN_NAME/tensorboard
VOCAB_FILE=/Megatron-DeepSpeed/data/bert-large-uncased-vocab.txt
DATA_PATH=/mnt/datasets/c4/en/preprocessed/c4-train_text_document

BERT_ARGS="--bert-no-binary-head \
           --num-layers 24 \
           --hidden-size 1024 \
           --num-attention-heads 16 \
           --seq-length 512 \
           --max-position-embeddings 512 \
           --mask-prob 0.15 \
           --lr 0.0001 \
           --lr-decay-iters 5 \
           --train-iters 20 \
           --min-lr 0.00001 \
           --lr-warmup-fraction 0.1 \
	       --micro-batch-size 4 \
           --global-batch-size 8 \
           --tokenizer-type BertWordPieceLowerCase \
           --vocab-file $VOCAB_FILE \
           --split 949,50,1 \
           --fp16"

OUTPUT_ARGS="--log-interval 2 \
             --tensorboard-dir $TENSORBOARD_PATH \
             --tensorboard-log-interval 2 \
             --log-params-norm \
             --log-num-zeros-in-grad \
             --log-timers-to-tensorboard \
             --log-batch-size-to-tensorboard \
             --log-validation-ppl-to-tensorboard \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 5 \
             --checkpoint-activations"

python pretrain_bert.py \
       $BERT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --data-path $DATA_PATH