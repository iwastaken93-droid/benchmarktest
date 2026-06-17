import sys
import os
import json
import time
import urllib.request
import urllib.error
import threading
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

# Results tracking
results_lock = threading.Lock()
trial_results = {}  # model_id -> list of trial dictionaries
completed_tasks = 0

def fetch_trial_worker(model_id, trial_idx, api_key, messages, total_tasks):
    global completed_tasks
    
    # Copy messages to avoid side-effects
    messages = list(messages)
    
    # For models that don't support system instructions (e.g. gemma-2-), merge system prompt into first user message
    if any(kw in model_id.lower() for kw in ["gemma-2-"]):
        new_messages = []
        system_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                if system_content:
                    new_messages.append({"role": "user", "content": f"{system_content}\n\n{msg['content']}"})
                    system_content = ""
                else:
                    new_messages.append(msg)
            else:
                new_messages.append(msg)
        messages = new_messages

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.7,
        "max_tokens": 2000,
        "stream": True
    }
    
    req_data = json.dumps(payload).encode('utf-8')
    
    success = False
    generated_text = ""
    ttft = None
    total_time = None
    error_occurred = None
    
    # Retry transient errors up to 3 times
    for attempt in range(3):
        req = urllib.request.Request(url, data=req_data, method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('Authorization', f"Bearer {api_key}")
        req.add_header('Accept', '*/*')
        req.add_header('Cache-Control', 'no-cache')
        req.add_header('Connection', 'keep-alive')
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        start_time = time.time()
        ttft = None
        total_time = None
        generated_text = ""
        error_occurred = None
        actual_token_count = None
        
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
                            if "usage" in data_json:
                                actual_token_count = data_json["usage"].get("completion_tokens")
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
        if success:
            break
            
        # If it's a structural failure (404, 401, 403), do not retry
        err_msg = error_occurred or ""
        if any(h in err_msg.lower() for h in ["404", "401", "403"]):
            break
            
        # Wait 2 seconds before retrying transient issues (e.g. timeouts, 503s, 429s)
        if attempt < 2:
            time.sleep(2.0)
            
    # Process stats
    tps = 0.0
    tpot = 0.0
    token_count = 0
    if success:
        if actual_token_count is not None and actual_token_count > 0:
            token_count = actual_token_count
        else:
            token_count = nim_server.count_completion_tokens(generated_text)
        latency_delta_s = (total_time - ttft) / 1000.0 if total_time and ttft else 0
        if latency_delta_s > 0 and token_count > 1:
            tps = (token_count - 1) / latency_delta_s
            tpot = (total_time - ttft) / (token_count - 1)
            
    trial_res = {
        "trial_idx": trial_idx,
        "success": success,
        "text": generated_text,
        "error": error_occurred or ("Empty response" if len(generated_text.strip()) == 0 else None),
        "ttft_ms": ttft or 0.0,
        "tps": tps,
        "tpot_ms": tpot,
        "tokens": token_count,
        "total_time_ms": total_time or 0.0
    }
    
    with results_lock:
        if model_id not in trial_results:
            trial_results[model_id] = []
        trial_results[model_id].append(trial_res)
        completed_tasks += 1
        print(f"[Progress] Completed {completed_tasks}/{total_tasks} trials. ({model_id} trial {trial_idx+1}/3 finished)")

def main():
    global completed_tasks
    api_key = nim_server.get_api_key()
    if not api_key:
        print("API Key not found!")
        return

    # Fetch models dynamically from NVIDIA API
    print("Fetching active models from NVIDIA API...")
    try:
        all_models = nim_server.fetch_models(api_key)
        models = [m for m in all_models if nim_server.is_chat_model(m)]
        print(f"Successfully fetched chat models to test ({len(models)}): {models}")
    except Exception as e:
        print(f"Error fetching active models: {e}. Falling back to default list.")
        models = [
            "google/gemma-2-2b-it",
            "deepseek-ai/deepseek-v4-flash",
            "deepseek-ai/deepseek-v4-pro",
            "nvidia/nemotron-mini-4b-instruct",
            "qwen/qwen3.5-122b-a10b",
            "z-ai/glm-5.1",
            "qwen/qwen3.5-397b-a17b",
            "abacusai/dracarys-llama-3.1-70b-instruct",
            "minimaxai/minimax-m3"
        ]
        models = [m for m in models if nim_server.is_chat_model(m)]
        
    # Reorder models so priority models run first
    priority_models = [
        "z-ai/glm-5.1",
        "minimaxai/minimax-m3",
        "deepseek-ai/deepseek-v4-flash",
        "deepseek-ai/deepseek-v4-pro"
    ]
    priority_found = [m for m in models if m in priority_models]
    other_models = [m for m in models if m not in priority_models]
    priority_found.sort(key=lambda x: priority_models.index(x))
    models = priority_found + other_models
        
    # Interleave tasks across different models to distribute load
    all_tasks = []
    for trial_idx in range(3):
        for model_id in models:
            all_tasks.append((model_id, trial_idx))
            
    total_tasks = len(all_tasks)
    print(f"Starting Multi-Model Concurrency Test: {len(models)} models, 3 trials each = {total_tasks} total tasks.")
    print("Spawning trial threads spaced 2.0 seconds apart...")
    
    # Spawn background threads
    for idx, (model_id, trial_idx) in enumerate(all_tasks):
        # Generate a fresh randomized context for each trial run
        messages = nim_server.generate_random_messages()
        
        t = threading.Thread(
            target=fetch_trial_worker,
            args=(model_id, trial_idx, api_key, messages, total_tasks),
            daemon=True
        )
        t.start()
        
        # Wait 2.0 seconds before launching the next task
        time.sleep(2.0)
        
    print("All tasks spawned. Waiting for completions in the background...")
    while True:
        with results_lock:
            done = completed_tasks == total_tasks
        if done:
            break
        time.sleep(1.0)
        
    print("\n==================================================")
    print("ALL CONCURRENT BATCH TRIALS COMPLETED. PRINTING RESULTS:")
    print("==================================================")
    
    for model_id in sorted(models):
        print(f"\n##################################################")
        print(f"MODEL: {model_id}")
        print(f"##################################################")
        
        trials = trial_results.get(model_id, [])
        # Sort trials by trial index
        trials = sorted(trials, key=lambda x: x["trial_idx"])
        
        for trial in trials:
            print(f"\n--- TRIAL {trial['trial_idx'] + 1} ---")
            if trial["success"]:
                print(f"Success: True")
                print(f"TTFT: {trial['ttft_ms']:.2f} ms")
                print(f"TPS: {trial['tps']:.2f} tokens/sec")
                print(f"Tokens: {trial['tokens']}")
                print(f"Total Time: {trial['total_time_ms']:.2f} ms")
                print(f"\n--- Response Text ---")
                print(trial["text"])
            else:
                print(f"Success: False")
                print(f"Error: {trial['error']}")
                
    # Save results to public/benchmark_results.json
    results_path = os.path.join(SCRIPT_DIR, "public", "benchmark_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    run_timestamp = datetime.now().isoformat()
    
    run_results = []
    for model_id in models:
        trials = trial_results.get(model_id, [])
        trials = sorted(trials, key=lambda x: x["trial_idx"])
        
        json_trials = []
        success_trials = []
        for t in trials:
            cleaned = dict(t)
            cleaned.pop("text", None)  # Remove response text to keep JSON size small
            json_trials.append(cleaned)
            if t.get("success"):
                success_trials.append(t)
                
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
            
        is_filtered_guardrail = False
        if not success_trials:
            for t in trials:
                if "too few tokens" in str(t.get("error", "")):
                    is_filtered_guardrail = True
                    break
        if is_filtered_guardrail:
            print(f"[Benchmark] Skipping saving results for filtered guardrail model {model_id}")
            continue
            
        run_results.append({
            "model": model_id,
            "avg_ttft_ms": avg_ttft,
            "avg_tps": avg_tps,
            "avg_tpot_ms": avg_tpot,
            "avg_tokens": avg_tokens,
            "success_rate": success_rate,
            "trials": json_trials
        })
        
    history = []
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        except Exception:
            pass
            
    new_run = {
        "timestamp": run_timestamp,
        "results": run_results,
        "status": "completed"
    }
    history.append(new_run)
    if len(history) > 336:
        history = history[-336:]
        
    try:
        with open(results_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\n[Benchmark] Successfully saved consolidated results to {results_path}")
    except Exception as e:
        print(f"\n[Benchmark] Error saving consolidated results: {e}")

if __name__ == '__main__':
    main()
