# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

import json

def process_evaluation_data(json_data_list):
    """
    json_data_list: List of JSON file paths (e.g., ['result1.json', 'result2.json'])
                    or a list of already-parsed dictionary objects
    """
    
    # Dictionary to store the results (each task holds a list)
    final_result = {}
    
    # Iterate over all files
    for data in json_data_list:
        
        # If data is a file path (string), read the file.
        if isinstance(data, str):
            with open(data, 'r', encoding='utf-8') as f:
                parsed_data = json.load(f)
        else:
            parsed_data = data
            
        # The model path is usually the top-level key, so extract it
        for model_path, runs in parsed_data.items():
            
            file_task_scores = {}
            print(model_path)
            # Iterate over keys such as run_boolq, run_mmlu
            for run_key, tasks_list in runs.items():
                for task_info in tasks_list:
                    task_name = task_info.get("Tasks", "").strip()
                    metric = task_info.get("Metric", "").strip()
                    value = float(task_info.get("Value", 0.0))
                    
                    # 1. Discard tasks that start with '-'
                    if task_name.startswith("-"):
                        continue
                        
                    # 2. Keep only entries whose metric is acc or exact_match
                    if metric not in ["acc", "exact_match"]:
                        continue
                        
                    # 3. For lambada, record only lambada_openai
                    if "lambada" in task_name and task_name != "lambada_openai":
                        continue
                        
                    # Store in a temporary dictionary when the conditions are met
                    file_task_scores[task_name] = value
            
            # 4. Collect everything and compute the average
            if file_task_scores:
                avg_score = sum(file_task_scores.values()) / len(file_task_scores)
            else:
                avg_score = 0.0
                
            # 5. Store in a list for each task (builds the final result dictionary)
            for t_name, score in file_task_scores.items():
                if t_name not in final_result:
                    final_result[t_name] = []
                final_result[t_name].append(score)
                
            # Also append the average to the list under a separate key
            if "average" not in final_result:
                final_result["average"] = []
            final_result["average"].append(avg_score)
            
    return final_result

import statistics

def calculate_paper_statistics(data_dict, idx=0):
    """
    Takes benchmark results in dictionary form and computes the mean and sample standard deviation.
    """
    stats_result = {}
    
    for task, scores in data_dict.items():
        # 1. Flatten the nested list (e.g., [[0.58], [0.59]] -> [0.58, 0.59])



        flat_scores = [x[idx] if isinstance(x, list) else x for x in scores]
        
            # 2. Compute the mean
        mean_val = statistics.mean(flat_scores)
        
        # 3. Compute the sample standard deviation (only when there are 2 or more values; 0.0 if there is 1)
        # Papers usually use the sample standard deviation (n-1) rather than the population one.
        if len(flat_scores) > 1:
            std_val = statistics.stdev(flat_scores)
        else:
            std_val = 0.0
            
        # 4. Paper-ready format (shown up to 4 decimal places)
        formatted_str = f"{mean_val:.4f} ± {std_val:.4f}"
        
        stats_result[task] = {
            "mean": mean_val,
            "std": std_val,
            "formatted": formatted_str
        }
    
    return stats_result

def run(json_list):
    total_result = {}
    # Function call (pass as a list)
    for path in json_list:
        with open(path, 'r', encoding='utf-8') as f:
            dummy_json_data = json.load(f)
            result_dict = process_evaluation_data([dummy_json_data])
            for key, value in result_dict.items():
                if key not in total_result:
                    total_result[key] = []
                total_result[key].append(value)
            print(total_result)
    return total_result

def calc_t_test(baseline, proposed):
    import numpy as np
    from scipy import stats
    import pandas as pd
    print("k")
    print(baseline)
    print(proposed)


    results = []

# Get the keys (task names) of the two dictionaries.
    tasks = baseline.keys()
    
    for task in tasks:
        try:
            # 1. Extract the data as a list and force-convert it to numbers (float).
            # Use float(item[0]) in case item[0] is a string.
            b_scores = [float(item[0]) for item in baseline[task]]
            p_scores = [float(item[0]) for item in proposed[task]]
            
            # 2. Convert to a numpy array (check the type once more)
            b_scores = np.array(b_scores, dtype=float)
            p_scores = np.array(p_scores, dtype=float)
            
            # 3. Run the t-test
            # nan_policy can be set in case all values are identical and the variance is 0.
            t_stat, p_val = stats.ttest_ind(b_scores, p_scores, equal_var=False) # Welch's t-test recommended
            
            is_significant = "Yes" if p_val < 0.05 else "No"
            
            results.append({
                'Task': task,
                'Base Mean': np.mean(b_scores),
                'Prop Mean': np.mean(p_scores),
                'p-value': p_val,
                'Significant': is_significant
            })
        except Exception as e:
            print(f"Error processing task '{task}': {e}")
            continue
    # Print the results
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_name="ElasticPruner-1.3B-seed"

    json_list = []
    seed_keys = [0, 1234, 42]
    for key in seed_keys:
        tail = "{}.json".format(key)
        json_list.append(run_name + tail)
    total_result = run(json_list)
    
    stats_result = calculate_paper_statistics(total_result, 0)
    print(stats_result)
