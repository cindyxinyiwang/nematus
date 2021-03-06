#!/bin/bash

# warning: this test is useful to check if training fails, and what speed you can achieve
# the toy datasets are too small to obtain useful translation results,
# and hyperparameters are chosen for speed, not for quality.
# For a setup that preprocesses and trains a larger data set,
# check https://github.com/rsennrich/wmt16-scripts/tree/master/sample

mkdir -p models

THEANO_FLAGS=optimizer=None python ../nematus/nmt_double_enc.py \
  --model models/model_double_enc.npz \
  --datasets data/3train.src,data/3train1.src data/3train.trg \
  --dictionaries data/train.trg.json data/train1.src.json data/train.trg.json \
  --multi_src \
  --dim_word 256 \
  --dim 512 \
  --n_words_src 30 \
  --n_words 30 \
  --maxlen 50 \
  --optimizer adam \
  --lrate 0.0001 \
  --batch_size 1 \
  --no_shuffle \
  --dispFreq 1 \
  --valid_datasets data/test.src,data/test1.src data/test.trg \
  --valid_batch_size 2 \
  --validFreq 2 \
  --decoder gru_double_cond \
  --finish_after 30 \
  --maxibatch_size 4 \
  --no_shuffle \
  --bleu \
  --postprocess bpe
