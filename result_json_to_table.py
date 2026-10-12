# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

import json
import argparse

def json_to_markdown_table(json_data):
    # Convert to a dictionary if it is a JSON string
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data

    # Set the table header
    headers = ["Tasks", "Version", "Filter", "n-shot", "Metric", "Value", "Stderr"]
    md_table = "| " + " | ".join(headers) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    # Skip the top-level model path and process (run_mmlu, etc.) to access the task list
    for model_path in data.values():
        for process_name, task_list in model_path.items():
            for task in task_list:
                # Extract the values matching the header from each task dictionary
                row = []
                for header in headers:
                    # Handle missing or empty-string values as '-'
                    val = task.get(header, "-")
                    if val == "" or val is None:
                        val = "-"
                    row.append(str(val))
                md_table += "| " + " | ".join(row) + " |\n"
    
    return md_table

# --- Usage example ---
# user_json = { ... } # the JSON data you provided above
# print(json_to_markdown_table(user_json))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that converts JSON data to a Markdown table")
    parser.add_argument("json_file", type=str, help="Path to the JSON file")
    args = parser.parse_args()

    with open(args.json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    md_table = json_to_markdown_table(json_data)
    print(md_table)