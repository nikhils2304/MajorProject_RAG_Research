
import re
from datetime import datetime
from pathlib import Path
import pandas as pd

def parse_log_for_latency(log_file):
    if not Path(log_file).exists():
        print(f"Log file not found: {log_file}")
        return []

    print(f"Processing {log_file}...")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    experiments = []
    current_exp = {}
    
    # Regex patterns
    start_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - --- Starting .*Evaluation for (\w+)")
    count_pattern = re.compile(r"Loaded (\d+) QA pairs")
    end_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - .*Evaluation Complete for (\w+)")
    
    for line in lines:
        if "Starting" in line and "Evaluation" in line:
             # print(f"DEBUG found start line: {line.strip()}")
             pass

        # Check start
        start_match = start_pattern.search(line)
        if start_match:
            # If we were already in an experiment that didn't finish, discard it or close it?
            # We assume sequential execution per language
            timestamp_str, language = start_match.groups()
            current_exp = {
                "Language": language,
                "Start": datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f"),
                "File": log_file
            }
            continue

        # Check count
        count_match = count_pattern.search(line)
        if count_match and current_exp:
            current_exp["Count"] = int(count_match.group(1))
            continue
            
        # Check end
        end_match = end_pattern.search(line)
        if end_match:
            timestamp_str, language = end_match.groups()
            if current_exp and current_exp.get("Language") == language:
                current_exp["End"] = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                
                # Calculate
                duration = (current_exp["End"] - current_exp["Start"]).total_seconds()
                count = current_exp.get("Count", 50) # Default/Fallback
                avg_latency = (duration / count) * 1000 # ms
                
                # Extract experiment name
                if "/" in log_file:
                    exp_name = log_file.split('/')[1].split('_')[-1] # e.g. p1, p2
                else:
                    exp_name = "p1" # Default for root log

                experiments.append({
                    "Experiment": exp_name,
                    "Language": language,
                    "Total_Time_s": duration,
                    "Queries": count,
                    "Avg_Latency_ms": round(avg_latency, 2)
                })
                current_exp = {}

    return experiments

def main():
    logs = [
        "evaluation_p1.log",
        "pipelines/4_experiment_p2/experiment_p1.log",
        "pipelines/5_experiment_p3/experiment_p1.log"
    ]
    
    all_data = []
    for log in logs:
        try:
            data = parse_log_for_latency(log)
            all_data.extend(data)
        except Exception as e:
            print(f"Error parsing {log}: {e}")
            
    if not all_data:
        print("No latency data found.")
        return

    df = pd.DataFrame(all_data)
    print("\n--- Average Latency per Experiment (ms) ---")
    print(df.to_markdown(index=False))
    
    # Pivot for cleaner view
    pivot_df = df.pivot(index="Language", columns="Experiment", values="Avg_Latency_ms")
    print("\n--- Latency Comparison Matrix (ms) ---")
    print(pivot_df.to_markdown())

if __name__ == "__main__":
    main()
