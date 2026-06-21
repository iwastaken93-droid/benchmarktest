import sys
import os
import json
import time
import urllib.request
import urllib.error
from datetime import datetime

# Reconfigure stdout to use UTF-8, avoiding CP1252/UnicodeEncodeError on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add the repository to python path dynamically
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

import nim_server

def test_model(model_id, api_key):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Write a 500-word story about a speed unicorn."}
        ],
        "temperature": 0.7,
        "max_tokens": 1000,
        "stream": True
    }
    
    req_data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=req_data, method='POST')
    req.add_header('Content-Type', 'application/json')
    req.add_header('Authorization', f"Bearer {api_key}")
    req.add_header('Accept', '*/*')
    
    start_time = time.time()
    ttft = None
    generated_text = ""
    error_occurred = None
    
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            first_chunk = True
            for line in response:
                if first_chunk:
                    ttft = (time.time() - start_time) * 1000.0
                    first_chunk = False
                
                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str.startswith("data:"):
                    data_str = line_str[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        if "error" in data_json:
                            error_occurred = f"API Error: {data_json['error'].get('message', 'Unknown error')}"
                            break
                        choices = data_json.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content") or ""
                            reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
                            generated_text += content + reasoning
                    except Exception:
                        pass
            total_time = (time.time() - start_time) * 1000.0
    except urllib.error.HTTPError as e:
        try:
            error_occurred = f"HTTP Error {e.code}: {e.read().decode('utf-8')}"
        except Exception:
            error_occurred = f"HTTP Error {e.code}"
    except Exception as e:
        error_occurred = str(e)
        
    success = error_occurred is None and len(generated_text.strip()) > 0
    
    # Calculate stats
    token_count = 0
    tps = 0.0
    tpot = 0.0
    if success:
        token_count = nim_server.count_completion_tokens(generated_text)
        latency_delta_s = (total_time - ttft) / 1000.0 if total_time and ttft else 0
        if latency_delta_s > 0 and token_count > 1:
            tps = (token_count - 1) / latency_delta_s
            tpot = (total_time - ttft) / (token_count - 1)
            
    if not success:
        print(f"[{model_id}] Error: {error_occurred or 'Empty response'}\n")
    else:
        print(f"\n==================================================")
        print(f"MODEL: {model_id}")
        print(f"==================================================")
        print(f"Response: {generated_text.strip()}")
        print(f"--------------------------------------------------")
        print(f"TTFT: {ttft:.2f} ms")
        print(f"TPS: {tps:.2f} tokens/sec")
        print(f"Tokens Generated: {token_count}")
        print(f"Total Time: {total_time:.2f} ms")
        print(f"==================================================\n")
        
    return {
        "success": success,
        "error": error_occurred or ("Empty response" if not success else None),
        "ttft_ms": ttft or 0.0,
        "tps": tps,
        "tpot_ms": tpot,
        "tokens": token_count,
        "total_time_ms": total_time or 0.0
    }

def main():
    api_key = nim_server.get_api_key()
    if not api_key:
        print("API Key not found!")
        return
        
    models = [
        'z-ai/glm-5.1', 
        'minimaxai/minimax-m3', 
        'deepseek-ai/deepseek-v4-flash', 
        'deepseek-ai/deepseek-v4-pro', 
        'qwen/qwen3.5-122b-a10b', 
        'qwen/qwen3.5-397b-a17b'
    ]
    
    print(f"Starting simple quick test on {len(models)} models sequentially...\n")
    
    trial_results = {model_id: [] for model_id in models}
    total_models = len(models)
    
    # Pass 1: Run 1 trial for each model (6 runs total)
    print("--- Starting Pass 1 (1 trial per model) ---")
    for idx, model_id in enumerate(models):
        print(f"Testing {model_id} (Pass 1, trial 1/3)...")
        res = test_model(model_id, api_key)
        trial_results[model_id].append(res)
        
        # Sleep 2 seconds between models
        if idx < total_models - 1:
            time.sleep(2.0)
            
    # Sleep 2 seconds before Pass 2
    time.sleep(2.0)
    
    # Pass 2: Run another 2 trials for each model (12 runs total)
    print("\n--- Starting Pass 2 (2 additional trials per model) ---")
    for idx, model_id in enumerate(models):
        for trial_num in range(2):
            print(f"Testing {model_id} (Pass 2, trial {trial_num+2}/3)...")
            res = test_model(model_id, api_key)
            trial_results[model_id].append(res)
            
            # Sleep 2 seconds between trials/models
            time.sleep(2.0)

    # Compile the results and average them
    new_results_dict = {}
    for model_id in models:
        trials = trial_results[model_id]
        success_trials = [t for t in trials if t.get("success")]
        success_rate = len(success_trials) / len(trials) if trials else 0.0
        
        if success_trials:
            avg_ttft = sum(t["ttft_ms"] for t in success_trials) / len(success_trials)
            avg_tps = sum(t["tps"] for t in success_trials) / len(success_trials)
            avg_tokens = sum(t["tokens"] for t in success_trials) / len(success_trials)
            avg_tpot = sum(t.get("tpot_ms", 0.0) for t in success_trials) / len(success_trials)
        else:
            avg_ttft = 0.0
            avg_tps = 0.0
            avg_tokens = 0.0
            avg_tpot = 0.0
            
        json_trials = []
        for t in trials:
            trial_data = {
                "success": t["success"],
                "ttft_ms": t["ttft_ms"],
                "tokens": t["tokens"],
                "tps": t["tps"],
                "tpot_ms": t["tpot_ms"]
            }
            if not t["success"]:
                trial_data["error"] = t["error"]
            json_trials.append(trial_data)
            
        new_results_dict[model_id] = {
            "model": model_id,
            "avg_ttft_ms": avg_ttft,
            "avg_tps": avg_tps,
            "avg_tpot_ms": avg_tpot,
            "avg_tokens": avg_tokens,
            "success_rate": success_rate,
            "trials": json_trials
        }

    # Load existing benchmark results to merge
    results_path = os.path.join(SCRIPT_DIR, "public", "benchmark_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    history = []
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        except Exception:
            pass

    if history:
        # Merge into the latest run
        latest_run = history[-1]
        print(f"\n[Merge] Found latest run from {latest_run.get('timestamp')}. Merging results...")
        
        current_results = latest_run.get("results", [])
        updated_results = []
        merged_models = set()
        
        for item in current_results:
            model_id = item.get("model")
            if model_id in new_results_dict:
                updated_results.append(new_results_dict[model_id])
                merged_models.add(model_id)
                print(f"  -> Replaced/Updated results for {model_id}")
            else:
                updated_results.append(item)
                
        for model_id, result in new_results_dict.items():
            if model_id not in merged_models:
                updated_results.append(result)
                print(f"  -> Added new results for {model_id}")
                
        latest_run["results"] = updated_results
    else:
        # Create a new run
        print("\n[Merge] No existing run history found. Creating a new run...")
        new_run = {
            "timestamp": datetime.now().isoformat(),
            "results": list(new_results_dict.values()),
            "status": "completed"
        }
        history.append(new_run)
        
    try:
        with open(results_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"[Benchmark] Successfully saved consolidated results to {results_path}")
    except Exception as e:
        print(f"[Benchmark] Error saving consolidated results: {e}")

if __name__ == '__main__':
    main()
