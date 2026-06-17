import sys
import os

# Reconfigure stdout to use UTF-8, avoiding CP1252/UnicodeEncodeError on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import json
import time
import urllib.request
import urllib.error
import threading

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
                        choices = data_json.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content") or ""
                            generated_text += content
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
        
    # Process stats
    success = error_occurred is None and len(generated_text.strip()) > 0
    tps = 0.0
    token_count = 0
    if success:
        token_count = nim_server.count_completion_tokens(generated_text)
        latency_delta_s = (total_time - ttft) / 1000.0 if total_time and ttft else 0
        tps = (token_count - 1) / latency_delta_s if latency_delta_s > 0 and token_count > 1 else 0
        
    trial_res = {
        "trial_idx": trial_idx,
        "success": success,
        "text": generated_text,
        "error": error_occurred or ("Empty response" if len(generated_text.strip()) == 0 else None),
        "ttft_ms": ttft,
        "tps": tps,
        "tokens": token_count,
        "total_time_ms": total_time
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
    
    for model_id in models:
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
                
if __name__ == '__main__':
    main()
