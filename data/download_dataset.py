# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

from datasets import load_dataset
import os
import time

os.environ["FSSPEC_TIMEOUT"] = "300"

MAX_RETRIES = 100  # Maximum number of attempts
WAIT_SECONDS = 30  # Wait time on failure

#ds = load_dataset("togethercomputer/RedPajama-Data-1T",'default',trust_remote_code=True)
#exit()

for attempt in range(1, MAX_RETRIES + 1):
    try:
        ds = load_dataset("togethercomputer/RedPajama-Data-1T",'default',trust_remote_code=True)
        # ds.save_to_dist("./redpajama")
        print("Download sucessful.")
    except Exception as e:
        print(f"❌ Download failed on attempt {attempt}: {e}")
        if attempt == MAX_RETRIES:
            print("🛑 Max retries reached. Exiting.")
            raise
        else:
            print(f"⏳ Waiting {WAIT_SECONDS} seconds before retrying...")
            time.sleep(WAIT_SECONDS)
