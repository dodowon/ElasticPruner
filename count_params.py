# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

import sys
import torch
from pathlib import Path

path = sys.argv[1]

if path.endswith('.bin') or path.endswith('.pt'):
    # Load pruned model checkpoint
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = ckpt['model']
    print(f"[Loaded from checkpoint: {path}]")
else:
    # Load HuggingFace model (local dir or hub name)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16)
    print(f"[Loaded HF model: {path}]")

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total params:     {total:,}")
print(f"Trainable params: {trainable:,}")
print(f"Size (approx):    {total * 2 / 1024**3:.2f} GB (fp16)")
