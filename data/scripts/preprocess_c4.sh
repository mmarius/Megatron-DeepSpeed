#!/bin/bash

INPUT_FILE="/mnt/datasets/c4/en/c4-train.json"
OUTPUT_PATH="/mnt/datasets/c4/en/preprocessed/c4-train"

python /home/ubuntu/Megatron-DeepSpeed/tools/preprocess_data.py \
       --input $INPUT_FILE \
       --output-prefix $OUTPUT_PATH \
       --vocab /home/ubuntu/Megatron-DeepSpeed/data/bert-large-uncased-vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --workers 100