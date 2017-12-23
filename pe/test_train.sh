#!/bin/bash

# warning: this test is useful to check if training fails, and what speed you can achieve
# the toy datasets are too small to obtain useful translation results,
# and hyperparameters are chosen for speed, not for quality.
# For a setup that preprocesses and trains a larger data set,
# check https://github.com/rsennrich/wmt16-scripts/tree/master/sample

mkdir -p models

THEANO_FLAGS=optimizer=None python ../nematus/nmt_copy.py \
  --model models/model.npz \
  --datasets data/train.src,data/train1.src data/train.trg \
  --dictionaries data/train.src.json data/train.src.json data/train.trg.json \
  --multi_src \
  --dim_word 256 \
  --dim 512 \
  --n_words_src 30 \
  --n_words 30 \
  --maxlen 50 \
  --optimizer adam \
  --lrate 0.0001 \
  --batch_size 40 \
  --no_shuffle \
  --dispFreq 1 \
  --valid_datasets data/test.src,data/test1.src data/test.trg \
  --valid_batch_size 2 \
  --validFreq 2 \
  --decoder gru_double_cond \
  --sampleFreq 2 \
  --finish_after 10
