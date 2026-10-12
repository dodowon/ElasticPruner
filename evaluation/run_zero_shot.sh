# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

lm_eval --model hf \
	--model_args pretrained=/path/to/hf_model \
	--tasks winogrande,piqa,sciq,arc_easy,logiqa,lambada \
	--device cuda:0 \
	--output_path results \
	--batch_size auto \
	
