# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

import os
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from streaming import StreamingDataset, MDSWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

# ==========================================
# 1. Configuration
# ==========================================
SOURCE_ROOT = '/path/to/mds_data'  # original top-level folder
TARGET_ROOT = '/path/to/mds_data_with_ref_loss'  # top-level folder to save to
MODEL_NAME = 'meta-llama/Llama-2-7b-hf'   # reference model path or name
BATCH_SIZE = 8                        # adjust to fit GPU memory
NUM_WORKERS = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. Load the model
# ==========================================
print(f"Loading Reference Model: {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
).to(device)
model.eval()

# ==========================================
# 3. Define the dataset class (with the conversion logic built in)
# ==========================================
class RefLossDataset(StreamingDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        
        # Check for the 'tokens' or 'input_ids' key
        input_ids = sample.get('tokens') or sample.get('input_ids')
        
        # [Important] Bytes -> Numpy(int) -> Tensor(long) conversion
        if isinstance(input_ids, bytes):
            # dtype must match the type used when saving (usually int32, int64, int16, etc.)
            # Assumed int16 here following the user's code, but it must be int32 if the vocab size exceeds 32k
            # If an error occurs, try changing to np.int32 or np.int64.
            input_ids_np = np.frombuffer(input_ids, dtype=np.int16) 
            input_ids = torch.from_numpy(input_ids_np.copy()).to(torch.long)
        elif isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids).to(torch.long)
        elif isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            
        return {'input_ids': input_ids}

# ==========================================
# 4. Loss computation and processing function
# ==========================================
def compute_ref_loss(batch_input_ids):
    """
    batch_input_ids: LongTensor of shape (Batch, Seq_Len)
    """
    batch_input_ids = batch_input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=batch_input_ids)
        logits = outputs.logits

        # Shift Logits and Labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch_input_ids[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1), 
            reduction='none'
        )
        
        # Reshape back to (Batch, Seq_Len-1)
        loss = loss.view(shift_labels.size(0), -1)
        
        # Per-sample mean Loss (Batch Size, )
        sample_loss = loss.mean(dim=1)
        
    return sample_loss.to(torch.float32).cpu().numpy()

def process_mds_folder(src_path, dst_path):
    print(f"Processing: {src_path} -> {dst_path}")
    
    # Use the custom dataset
    try:
        dataset = RefLossDataset(local=src_path, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"Skipping {src_path}: {e}")
        return

    if len(dataset) == 0: return

    # Get the existing column information (based on the first sample)
    # Note: StreamingDataset internally returns decoded samples,
    # so accessing the raw sample may be needed to keep the original bytes form.
    # Here, simply define 'tokens' as bytes and 'ref_loss' as float32
    
    columns = {
        'tokens': 'bytes',  # saved as is (bytes)
        'ref_loss': 'float32'
    }

    # Custom Collate Fn (padding handling etc. when building batches)
    def collate_fn(batch):
        # Match the input_ids length (Padding)
        input_ids = [item['input_ids'] for item in batch]
        # pad_token_id differs by model (here, 0 or eos_token_id is used)
        pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
        
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )
        return padded_input_ids

    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    if os.path.exists(dst_path): shutil.rmtree(dst_path)
    os.makedirs(dst_path, exist_ok=True)

    with MDSWriter(out=dst_path, columns=columns) as out:
        for batch_input_ids in tqdm(dataloader):
            # 1. Compute the Loss
            ref_losses = compute_ref_loss(batch_input_ids)
            
            # 2. Write to MDS
            # The original data must be saved again, so convert batch_input_ids back to bytes
            # (Passing the original data through as is would be better, but it is re-converted considering memory/complexity)
            
            cpu_ids = batch_input_ids.cpu().numpy()
            
            for i in range(len(ref_losses)):
                # Padding removal logic (optional: if the original length is unknown, padding is saved too)
                # Here, simply save only the non-zero part or save everything
                # raw_ids = cpu_ids[i] # including padding
                
                # Convert to the original data type (int32/int16) and then to bytes
                # Here, int32 is used for safety
                token_bytes = cpu_ids[i].astype(np.int16).tobytes()
                
                sample = {
                    'tokens': token_bytes,
                    'ref_loss': float(ref_losses[i])
                }
                out.write(sample)

def main():
    for root, dirs, files in os.walk(SOURCE_ROOT):
        if 'index.json' in files:
            rel_path = os.path.relpath(root, SOURCE_ROOT)
            target_path = os.path.join(TARGET_ROOT, rel_path)
            process_mds_folder(root, target_path)

if __name__ == "__main__":
    main()