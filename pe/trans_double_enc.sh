#! /bin/bash

python translate_script.py \
	--input_file data/test.src,data/test1.src \
	--model models/model.npz \
	--beam_size 12 \
	--output_file double_enc.trans \
	--model_type double_enc \
	--postprocess bpe \
	--reference data/test.trg
