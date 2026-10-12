# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

python -m elasticpruner.data.merge_data \
	--input_dir /path/to/mds_data/eval \
	--output_dir /path/to/mds_data/for_ft \
	--output_split eval_merge \
	--split_names arxiv c4 cc github stackexchange 
