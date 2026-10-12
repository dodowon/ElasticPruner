# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

import subprocess
import re
import json
import os
import sys

class InteractiveBenchmarker:
    def __init__(self, model_paths, benchmark_scripts, seeds=None, output_json="bench_results.json"):
        self.model_paths = model_paths
        self.benchmark_scripts = benchmark_scripts
        self.seeds = seeds if seeds else [None]  # if None, run once without a seed
        self.output_json = output_json
        self.all_results = {}

    def modify_command(self, script_path, new_model_path, seed=None):
        if not os.path.exists(script_path): return None
        with open(script_path, 'r') as f:
            content = f.read()
        modified_content = re.sub(r'(pretrained=)[^,\s\\]+', r'\1' + new_model_path, content)
        lines = [line.strip() for line in modified_content.split('\n') if line.strip() and not line.strip().startswith('#')]
        command = " ".join(lines).replace('\\', ' ')
        
        # Add the --seed option if a seed is specified
        if seed is not None:
            command += f" --seed {seed}"
        
        return command

    def get_output_json_path(self, seed):
        """Generate a different output file path for each seed"""
        base, ext = os.path.splitext(self.output_json)
        if seed is not None:
            return f"{base}_seed{seed}{ext}"
        return self.output_json

    def parse_table(self, output_text):
        """
        1. Fill empty task names with the previous task (Forward Fill)
        2. Store all Metrics such as acc, acc_norm
        """
        results = []
        lines = output_text.split('\n')
        is_table = False
        table_raw_text = ""
        
        last_task_name = "" # variable to remember the previous task name

        for line in lines:
            line_strip = line.strip()
            if '|' in line and 'Tasks' in line and 'Metric' in line:
                is_table = True
                table_raw_text += line + "\n"
                continue
            
            if is_table:
                if line_strip == "" and len(results) > 0:
                    is_table = False
                    continue
                
                if line_strip.startswith('|'):
                    table_raw_text += line + "\n"
                    if line_strip.startswith('|--'): continue
                    
                    # Split on pipes, then strip whitespace from both ends
                    raw_cols = [c.strip() for c in line.split('|')][1:-1]
                    # Remove special symbols
                    clean_cols = [c for c in raw_cols if c not in ['↑', '↓', '±']]
                    
                    if len(clean_cols) >= 6:
                        current_task = clean_cols[0]
                        
                        # [Key fix] If the task name is empty, use the previous task name
                        if current_task == "" or current_task.isspace():
                            current_task = last_task_name
                        else:
                            last_task_name = current_task # store the new task name
                        
                        results.append({
                            "Tasks": current_task,
                            "Version": clean_cols[1],
                            "Filter": clean_cols[2],
                            "n-shot": clean_cols[3],
                            "Metric": clean_cols[4], # stores acc, acc_norm, etc.
                            "Value": clean_cols[5],
                            "Stderr": clean_cols[6] if len(clean_cols) > 6 else ""
                        })
        return results, table_raw_text

    def run(self):
        for seed in self.seeds:
            seed_label = f"seed={seed}" if seed is not None else "no_seed"
            print(f"\n{'='*72}")
            print(f"[Seed] {seed_label}")
            print(f"{'='*72}")
            
            seed_results = {}

            for model_path in self.model_paths:
                print(f"\n[Model] {model_path}")
                seed_results[model_path] = {}

                for script in self.benchmark_scripts:
                    print(f"\n>>> Running {script} ({seed_label})...")
                    command = self.modify_command(script, model_path, seed=seed)
                    if not command: continue

                    print(f"[CMD] {command}")

                    captured_output = []
                    process = subprocess.Popen(
                        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                    )

                    for line in process.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        captured_output.append(line)
                    process.wait()

                    full_output = "".join(captured_output)
                    parsed_data, table_text = self.parse_table(full_output)
                    
                    task_key = os.path.splitext(script)[0]
                    seed_results[model_path][task_key] = parsed_data

                    print("\n" + "-"*30 + f" [Result summary - {seed_label}] " + "-"*30)
                    if table_text:
                        print(table_text)
                        print(f"Status: extracted {len(parsed_data)} Metric rows in total.")
                    else:
                        print("Could not find a table.")
                    print("-" * 72)

            # Save the results for each seed
            output_path = self.get_output_json_path(seed)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(seed_results, f, indent=4, ensure_ascii=False)
            print(f"\n[Saved] {seed_label} -> {output_path}")
        
        print(f"\n[All done] Benchmarks for all seeds have completed.")

if __name__ == "__main__":
    target_models = [
        "/path/to/output/after_train/hf"
        
    ]


    # List of benchmark scripts to run
    target_benchmarks = [
        "run_boolq.sh", 
        "run_mmlu.sh", 
        "run_arc_c.sh", 
        "run_hellaswag.sh", 
        "run_nq.sh",
        "run_zero_shot.sh"
    ]

    # Seed list (run and save separately for each seed)
    seed_list = [0, 1234, 42]
    save_path="save_path.json"

    runner = InteractiveBenchmarker(
        target_models, 
        target_benchmarks, 
        seeds=seed_list,
        output_json=save_path
    )
    runner.run()
