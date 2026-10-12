# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

import os
import shutil
import warnings
from typing import Any, Dict, List
import numpy as np

import torch
# Additional import of the Stream class
from streaming import MDSWriter, StreamingDataset, Stream
from tqdm import tqdm

# ================= Configuration =================
# Top-level path where the original data is located (where cc, github, etc. are)
SRC_ROOT = "/path/to/mds_data_with_ref_loss/for_ft"

# New path where the merged data will be saved
DST_ROOT = "/path/to/mds_data_merged/for_ft"

# List of domains to process (modify as needed)
# DOMAINS = ['cc', 'github', 'stackexchange', 'wiki', 'arxiv', 'c4']
DOMAINS = ['eval_merge']
# ============================================

def infer_mds_types(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    Infers the MDS column types based on the sample data.
    """
    columns = {}
    for k, v in sample.items():
        if isinstance(v, (int, np.integer)):
            columns[k] = 'int16'
        elif isinstance(v, (float, np.floating)):
            columns[k] = 'float32'
        elif isinstance(v, str):
            columns[k] = 'str'
        elif isinstance(v, bytes):
            columns[k] = 'bytes'
        elif isinstance(v, (list, dict)):
            # Lists and dictionaries are usually saved as json
            columns[k] = 'json'
        elif isinstance(v, np.ndarray):
            # Numpy arrays are generally converted to bytes for saving
            # (MDSWriter can also support ndarray directly, but bytes has the best compatibility)
            columns[k] = 'bytes'
        else:
            # Other types are safely handled as pickle (pkl) or bytes
            print(f"Warning: Unknown type for column '{k}': {type(v)}. Defaulting to 'bytes'.")
            columns[k] = 'bytes'
    return columns

def get_part_streams(domain_path: str) -> List[Dict[str, str]]:
    """
    Finds the part_* folders inside the given domain folder and returns them as a list of streams.
    """
    if not os.path.exists(domain_path):
        print(f"Warning: {domain_path} does not exist. Skipping.")
        return []

    # Search the subdirectories in the folder for those containing index.json or starting with part_
    subdirs = sorted([
        d for d in os.listdir(domain_path) 
        if os.path.isdir(os.path.join(domain_path, d))
    ])
    
    streams = []
    print(f"  Found parts in {os.path.basename(domain_path)}: {subdirs}")
    
    for subdir in subdirs:
        # Add each part folder as a local stream
        streams.append(Stream(local=os.path.join(domain_path, subdir)))
        
    return streams

def merge_domain(domain_name: str):
    src_domain_path = os.path.join(SRC_ROOT, domain_name)
    dst_domain_path = os.path.join(DST_ROOT, domain_name)
    
    print(f"\nProcessing Domain: {domain_name}")
    print(f"  Source: {src_domain_path}")
    print(f"  Dest:   {dst_domain_path}")

    # 1. Build the streams (read multiple parts as if they were one dataset)
    streams = get_part_streams(src_domain_path)
    if not streams:
        print(f"  No parts found for {domain_name}. Skipping.")
        return

    # 2. Read with StreamingDataset (parts are concatenated automatically)
    try:
        dataset = StreamingDataset(
            streams=streams,
            batch_size=1,
            shuffle=False  # shuffle is off since the purpose is merging
        )
    except Exception as e:
        print(f"  Error loading dataset parts: {e}")
        return

    # 3. Check the data schema (columns)
    # Get the first sample and identify the column structure
    try:
        first_sample = dataset[0]
        columns = infer_mds_types(first_sample)
        # Note: most LLM data is bytes (tokens), but an encoding type may need to be specified if required
        # e.g.: {'tokens': 'bytes', 'attention_mask': 'bytes'}, etc.
        # MosaicML MDS usually stores raw bytes and decodes them.
        print(f"  Detected columns: {list(columns.keys())}")
    except Exception as e:
        print(f"  Error reading first sample: {e}")
        return

    # 4. Write to the new path with MDSWriter
    if os.path.exists(dst_domain_path):
        print(f"  Destination {dst_domain_path} already exists. Deleting to overwrite...")
        shutil.rmtree(dst_domain_path)
    
    os.makedirs(dst_domain_path, exist_ok=True)

    # Compression settings (keep the same as the original or zstd is recommended)

    
    print("  Starting merge (Writing new MDS files)...")
    with MDSWriter(out=dst_domain_path, columns=columns) as out:
        for i, sample in tqdm(enumerate(dataset), total=dataset.num_samples, desc=f"  Merging {domain_name}"):
            out.write(sample)
            
    print(f"  Done! Merged data saved to {dst_domain_path}")

if __name__ == "__main__":
    # Create the output folder
    os.makedirs(DST_ROOT, exist_ok=True)
    
    for domain in DOMAINS:
        merge_domain(domain)
        
    print("\nAll tasks finished.")