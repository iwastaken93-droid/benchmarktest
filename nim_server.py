import json
import time
import urllib.request
import urllib.error
import os
import threading
import random
import sys
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# Reconfigure stdout to use UTF-8, avoiding CP1252/UnicodeEncodeError on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import tiktoken
    tokenizer_encoder = tiktoken.get_encoding("cl100k_base")
except ImportError:
    tokenizer_encoder = None

def count_completion_tokens(text):
    if tokenizer_encoder:
        try:
            return len(tokenizer_encoder.encode(text))
        except Exception:
            pass
    # Fallback to statistical estimator
    word_count = len(text.split())
    char_count = len(text)
    if word_count > 0:
        val = int(word_count * 1.33)
    else:
        val = int(char_count / 4)
    return max(1, val)

# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML_PATH = os.path.join(SCRIPT_DIR, "public", "index.html")
PLAYGROUND_HTML_PATH = os.path.join(SCRIPT_DIR, "public", "playground.html")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "public", "benchmark_results.json")

# Threading locks and states
benchmark_lock = threading.Lock()
benchmark_running = False
benchmark_status = "Idle"
active_run_results = {}
completed_tasks_count = 0
active_run_timestamp = ""

def get_api_key():
    env_key = os.environ.get("NVIDIA_API_KEY")
    if env_key:
        return env_key
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
                return data.get("api_key", "")
    except Exception as e:
        print(f"[Server] Error reading config: {e}")
    return ""

def resolve_auth_header(auth_header):
    # Inject API key from config.json if not provided, is null/undefined/placeholder,
    # or doesn't contain a real NVIDIA API key starting with 'nvapi-'
    has_real_key = False
    if auth_header:
        parts = auth_header.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
            if token.startswith("nvapi-"):
                has_real_key = True
                
    if not has_real_key:
        key = get_api_key()
        if key:
            return f"Bearer {key}"
    return auth_header

def save_api_key(api_key):
    try:
        data = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                try:
                    data = json.load(f)
                except Exception:
                    pass
        data["api_key"] = api_key
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"[Server] Error saving config: {e}")
        return False

BLACKLISTED_MODELS = {
    "01-ai/yi-large",
    "adept/fuyu-8b",
    "ai21labs/jamba-1.5-large-instruct",
    "aisingapore/sea-lion-7b-instruct",
    "baai/bge-m3",
    "bigcode/starcoder2-15b",
    "databricks/dbrx-instruct",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "google/codegemma-1.1-7b",
    "google/codegemma-7b",
    "google/deplot",
    "google/gemma-2b",
    "google/gemma-3-12b-it",
    "google/gemma-3-4b-it",
    "google/recurrentgemma-2b",
    "ibm/granite-3.0-3b-a800m-instruct",
    "ibm/granite-3.0-8b-instruct",
    "ibm/granite-34b-code-instruct",
    "ibm/granite-8b-code-instruct",
    "meta/codellama-70b",
    "meta/llama2-70b",
    "microsoft/phi-3-vision-128k-instruct",
    "microsoft/phi-3.5-moe-instruct",
    "mistralai/codestral-22b-instruct-v0.1",
    "mistralai/mistral-7b-instruct-v0.3",
    "mistralai/mistral-large",
    "mistralai/mistral-large-2-instruct",
    "mistralai/mixtral-8x22b-v0.1",
    "nv-mistralai/mistral-nemo-12b-instruct",
    "nvidia/cosmos-reason2-8b",
    "nvidia/llama-3.1-nemotron-51b-instruct",
    "nvidia/ai-synthetic-video-detector",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/gliner-pii",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nvidia/llama3-chatqa-1.5-70b",
    "nvidia/mistral-nemo-minitron-8b-8k-instruct",
    "nvidia/nemotron-4-340b-instruct",
    "nvidia/nemotron-4-340b-reward",
    "nvidia/nemotron-nano-3-30b-a3b",
    "nvidia/nemoretriever-parse",
    "nvidia/riva-translate-4b-instruct",
    "nvidia/vila",
    "nvidia/nemotron-parse",
    "nvidia/riva-translate-4b-instruct-v1.1",
    "writer/palmyra-creative-122b",
    "writer/palmyra-fin-70b-32k",
    "writer/palmyra-med-70b",
    "writer/palmyra-med-70b-32k",
    "upstage/solar-10.7b-instruct",
    "zyphra/zamba2-7b-instruct"
}

def is_chat_model(model_id):
    if model_id in BLACKLISTED_MODELS:
        return False
    model_id_lower = model_id.lower()
    exclude_keywords = [
        'embed', 'rerank', 'clip', 'similarity', 'image', 'stable-diffusion', 'whisper', 'sdxl', 
        'vision-language', 'neva', 'kosmos', 'nemoguard', 'safety-guard', 'guard', 'safety', 'moderation'
    ]
    for kw in exclude_keywords:
        if kw in model_id_lower:
            return False
    return True

def fetch_models(api_key):
    if not api_key:
        return ["deepseek-ai/deepseek-r1", "meta/llama3-8b-instruct", "meta/llama3-70b-instruct", "nvidia/nemotron-4-340b-instruct", "mistralai/mixtral-8x22b-instruct-v0.1"]
    
    url = "https://integrate.api.nvidia.com/v1/models"
    req = urllib.request.Request(url, method='GET')
    req.add_header('Authorization', f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res_content = response.read()
            data = json.loads(res_content.decode('utf-8'))
            models = [m['id'] for m in data.get('data', []) if 'id' in m]
            return models
    except Exception as e:
        print(f"[Benchmark] Error fetching cloud models: {e}. Falling back to mock model list.")
        return ["deepseek-ai/deepseek-r1", "meta/llama3-8b-instruct", "meta/llama3-70b-instruct", "nvidia/nemotron-4-340b-instruct", "mistralai/mixtral-8x22b-instruct-v0.1"]

def run_mock_trial(model_id):
    import random
    is_reasoning = "r1" in model_id or "reasoning" in model_id
    
    # Simulate API latency
    time.sleep(random.uniform(0.1, 0.3) if not is_reasoning else random.uniform(0.4, 0.8))
    ttft = random.uniform(150, 300) if not is_reasoning else random.uniform(400, 900)
    
    # Simulate generating a 1000-word story (around 400-800 tokens)
    tokens = random.randint(300, 750)
    tps = random.uniform(40, 85) if not is_reasoning else random.uniform(15, 32)
    # 95% success rate for mock trials
    success = random.random() < 0.95
    if success:
        tpot = 1000.0 / tps if tps > 0 else 0.0
        return {
            "success": True,
            "ttft_ms": ttft,
            "tokens": tokens,
            "tps": tps,
            "tpot_ms": tpot
        }
    else:
        return {
            "success": False,
            "error": "Mock connection reset / timeout"
        }

def run_trial(model_id, api_key, prompt_or_messages, max_tokens):
    # If the API key starts with a mock prefix or is not set, run mock trial
    if not api_key or api_key.startswith("mock"):
        return run_mock_trial(model_id)
        
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    
    # Retry transient errors up to 3 times
    last_res = {"success": False, "error": "Unknown error"}
    for attempt in range(3):
        res = _run_trial_internal(model_id, api_key, prompt_or_messages, max_tokens, url)
        if res.get("success"):
            return res
            
        last_res = res
        err_msg = res.get("error", "")
        # If it's a structural failure (404, 401, 403) or a timeout/connection error, do not retry
        if any(h in err_msg.lower() for h in ["404", "401", "403", "timed out", "timeout", "10060", "connection", "host", "refused", "reset"]):
            return res
            
        # Wait 2 seconds before retrying transient issues (e.g. timeouts, 503s, empty responses, or 429s)
        if attempt < 2:
            time.sleep(2.0)
            
    return last_res
 
def _run_trial_internal(model_id, api_key, prompt_or_messages, max_tokens, url):
    if isinstance(prompt_or_messages, list):
        messages = list(prompt_or_messages)
    else:
        messages = [{"role": "user", "content": prompt_or_messages}]
        
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
        
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.7,
        "max_tokens": max_tokens,
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
    ttft = 0.0
    total_time = 0.0
    generated_text = ""
    actual_token_count = None
    
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            if response.status != 200:
                return {"success": False, "error": f"HTTP status {response.status}"}
                
            first_chunk = True
            first_token_time = None
            for line in response:
                # Absolute timeout check to prevent thread hangs on silent/slow startup streams
                if (time.time() - start_time) > 360.0:
                    print(f"[Benchmark] {model_id} exceeded absolute 360-second timeout limit. Stopping stream.")
                    try:
                        response.close()
                    except Exception:
                        pass
                    break
                    
                if first_chunk:
                    first_token_time = time.time()
                    ttft = (first_token_time - start_time) * 1000.0
                    first_chunk = False
                
                if not first_chunk and first_token_time is not None:
                    if (time.time() - first_token_time) > 180.0:
                        print(f"[Benchmark] {model_id} exceeded 3-minute limit after first token. Stopping stream early.")
                        try:
                            response.close()
                        except Exception:
                            pass
                        break
                    
                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str.startswith("data:"):
                    data_str = line_str[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        if "error" in data_json:
                            err_msg = data_json["error"].get("message", "Mid-stream API error")
                            return {"success": False, "error": f"API Error: {err_msg}"}
                        
                        # Extract exact token count from usage metadata if provided
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
            
            # Check if we generated any text. If not, treat as failure.
            if not generated_text.strip():
                return {"success": False, "error": "Empty response or stream error"}
                
            # If the API returned a valid actual token count, use it. Otherwise, fallback to estimation.
            if actual_token_count is not None and actual_token_count > 0:
                token_count = actual_token_count
            else:
                token_count = count_completion_tokens(generated_text)


                
            latency_delta_s = (total_time - ttft) / 1000.0
            if latency_delta_s > 0 and token_count > 1:
                tps = (token_count - 1) / latency_delta_s
                tpot = (total_time - ttft) / (token_count - 1)
            else:
                tps = 0.0
                tpot = 0.0
                
            return {
                "success": True,
                "ttft_ms": ttft,
                "tokens": token_count,
                "tps": tps,
                "tpot_ms": tpot,
                "total_time_ms": total_time,
                "text": generated_text
            }
    except urllib.error.HTTPError as e:
        error_msg = f"HTTP Error {e.code}"
        try:
            error_body = e.read().decode('utf-8')
            error_json = json.loads(error_body)
            if "detail" in error_json:
                error_msg += f": {error_json['detail']}"
            elif "error" in error_json:
                error_msg += f": {error_json['error'].get('message', '')}"
        except Exception:
            pass
        return {"success": False, "error": error_msg}
    except Exception as e:
        return {"success": False, "error": str(e)}

def save_incremental_model_result(model_summary):
    global active_run_timestamp
    try:
        history = []
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, "r") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
                    
        for run in reversed(history):
            if run.get("timestamp") == active_run_timestamp:
                results = run.get("results", [])
                results = [r for r in results if r.get("model") != model_summary["model"]]
                results.append(model_summary)
                run["results"] = results
                break
                
        with open(RESULTS_PATH, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[Benchmark] Error saving incremental result: {e}")

def mark_run_completed():
    global active_run_timestamp
    try:
        history = []
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, "r") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
                    
        for run in reversed(history):
            if run.get("timestamp") == active_run_timestamp:
                run["status"] = "completed"
                break
                
        with open(RESULTS_PATH, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[Benchmark] Error marking run completed: {e}")

SYSTEM_PROMPTS_POOL = [
    "You are a helpful assistant."
]

CONVERSATION_TURNS_POOL = [
    {
        "user": "I have always wondered: how fast can a unicorn run compared to a normal horse? Is there a theoretical limit?",
        "assistant": "Unicorns are magical creatures, so their speed isn't bound by normal biology. While a thoroughbred horse can reach speeds around 40-44 mph (64-70 km/h) in short bursts, legend has it that a unicorn can easily exceed that, reaching speeds of 80 to 100 mph (128-160 km/h) by leveraging their magical essence. Some legends even suggest that they can run so fast that they leave a trail of stardust behind them!"
    },
    {
        "user": "Wow, 100 mph! What role does their horn play in their speed? Does it act like a lightning rod or an aerodynamic stabilizer?",
        "assistant": "According to ancient magical lore, the horn is central to their velocity. It acts as a focal point for kinetic magic, channeling natural energy forces to reduce aerodynamic drag to absolute zero. Essentially, it creates a small localized vacuum in front of the unicorn, allowing them to sprint without facing any air resistance. This is why they don't experience wind friction even at extreme speeds!"
    },
    {
        "user": "That is fascinating! What about acceleration? Can they go from 0 to 100 mph instantly, or do they need a running start?",
        "assistant": "Unicorns have incredible acceleration! Because their movement is partially powered by spatial magic, they don't rely solely on muscle contraction. They can reach their top speed in a single leap. This instantaneous acceleration is often described in myth as a 'flash of white light' where the unicorn appears to teleport from a standstill to full sprint."
    },
    {
        "user": "Is there any myth about a unicorn running so fast that it traveled through time or reached another dimension?",
        "assistant": "Yes, indeed! There are several legends about special speed unicorns that could outrun time itself. In these stories, when a speed unicorn is running from a danger or seeking to save a life, its speed can break the temporal barrier. The friction of its hooves on the space-time fabric creates a temporal rift, allowing it to arrive at its destination before it even departed!"
    },
    {
        "user": "What do speed unicorns eat to maintain that level of energy? Standard grass doesn't seem like enough.",
        "assistant": "Standard grass is definitely not enough for their high-speed metabolism! Speed unicorns are said to feed on starlight, morning dew, and wild berries infused with pure mana. Starlight contains high-energy cosmic particles that their magic cores convert directly into kinetic energy. Some legends also say they drink from waterfalls of liquid light to recharge their speed reserves after a long sprint."
    },
    {
        "user": "Do they have races against other magical creatures, like pegasi or dragons? Who usually wins?",
        "assistant": "They frequently race! Pegasi have the advantage of flight and altitude, and dragons have raw, brute power. However, in a flat race, a speed unicorn almost always wins due to their zero-drag acceleration. Dragons are too heavy to accelerate quickly, and pegasi are slowed by air turbulence. The only creature that can challenge a speed unicorn is a phoenix in a dive, but on land, the unicorn is undefeated!"
    }
]

def generate_random_messages():
    system_prompt = random.choice(SYSTEM_PROMPTS_POOL)
    messages = [{"role": "system", "content": system_prompt}]
    
    # Choose between 3 and 6 turns
    num_turns = random.randint(3, 6)
    # Get a sequential selection of turns from the pool to keep it a normal conversation
    selected_turns = CONVERSATION_TURNS_POOL[:num_turns]
    
    for turn in selected_turns:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
        
    final_query = "Write a 1000-word story about a speed unicorn. MAKE IT SPEED UNICORN SPEEED"
    messages.append({"role": "user", "content": final_query})
    
    # Log details about the generated context
    total_chars = len(system_prompt) + sum(len(t["user"]) + len(t["assistant"]) for t in selected_turns) + len(final_query)
    approx_tokens = int(total_chars / 4)
    print(f"[Benchmark] Generated dynamic context: {num_turns} turns, approx {approx_tokens} prefill tokens.")
    
    return messages

def execute_trial_task(model_id, trial_idx, total_tasks, task_idx):
    global benchmark_status, completed_tasks_count, benchmark_running, active_run_results
    
    api_key = get_api_key()
    
    # Construct a long, multi-turn context dynamically
    messages = generate_random_messages()
    
    max_tokens = 2000
    
    trial_res = {"success": False, "error": "Unknown error"}
    
    try:
        trial_res = run_trial(model_id, api_key, messages, max_tokens)
    except Exception as e:
        trial_res = {"success": False, "error": str(e)}
        print(f"[Benchmark] Error running trial for {model_id}: {e}")
    finally:
        with benchmark_lock:
            if model_id not in active_run_results:
                active_run_results[model_id] = []
            active_run_results[model_id].append(trial_res)
            
            # Check if all 3 trials for this model are done
            if len(active_run_results[model_id]) == 3:
                trials = active_run_results[model_id]
                success_trials = [t for t in trials if t.get("success", False)]
                success_rate = len(success_trials) / 3.0
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
                            
                if not is_filtered_guardrail:
                    # Clean trials for JSON results storage to keep file size lightweight (excluding text)
                    json_trials = []
                    for t in trials:
                        cleaned = dict(t)
                        cleaned.pop("text", None)
                        json_trials.append(cleaned)
                        
                    model_summary = {
                        "model": model_id,
                        "avg_ttft_ms": avg_ttft,
                        "avg_tps": avg_tps,
                        "avg_tpot_ms": avg_tpot,
                        "avg_tokens": avg_tokens,
                        "success_rate": success_rate,
                        "trials": json_trials
                    }
                    save_incremental_model_result(model_summary)
                else:
                    print(f"[Benchmark] Skipping saving results for filtered guardrail model {model_id}")
                
            completed_tasks_count += 1
            print(f"[Benchmark] Trial {task_idx+1}/{total_tasks} complete ({model_id} trial {trial_idx+1}/3). Total completed: {completed_tasks_count}/{total_tasks}")
            
            if completed_tasks_count == total_tasks:
                benchmark_status = "Idle"
                benchmark_running = False
                print("[Benchmark] All benchmark trials completed.")
                mark_run_completed()

def run_benchmark_suite():
    global benchmark_running, benchmark_status, active_run_results, completed_tasks_count, active_run_timestamp
    with benchmark_lock:
        benchmark_running = True
    
    benchmark_status = "Starting benchmark suite..."
    print("[Benchmark] Starting benchmark suite...")
    
    try:
        api_key = get_api_key()
        
        benchmark_status = "Fetching active models..."
        all_models = fetch_models(api_key)
        
        models_to_test = [m for m in all_models if is_chat_model(m)]

        print(f"[Benchmark] Models to test ({len(models_to_test)}): {models_to_test}")
        
        # Reset tracking state
        active_run_results = {}
        completed_tasks_count = 0
        active_run_timestamp = datetime.now().isoformat()
        
        # Initialize the run placeholder in the results file
        history = []
        if os.path.exists(RESULTS_PATH):
            try:
                with open(RESULTS_PATH, "r") as f:
                    history = json.load(f)
                    if not isinstance(history, list):
                        history = []
            except Exception:
                pass
                
        new_run = {
            "timestamp": active_run_timestamp,
            "results": [],
            "status": "running"
        }
        history.append(new_run)
        if len(history) > 336:
            history = history[-336:]
            
        try:
            with open(RESULTS_PATH, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"[Benchmark] Error saving initial run placeholder: {e}")
            
        # Build tasks queue (interleave trials across different models)
        all_tasks = []
        for trial_idx in range(3):
            for model_id in models_to_test:
                all_tasks.append((model_id, trial_idx))
                
        total_tasks = len(all_tasks)
        print(f"[Benchmark] Starting queue with {total_tasks} total trials...")
        
        # Spawn thread for each task every 2.0 seconds (30 requests/min)
        for idx, (model_id, trial_idx) in enumerate(all_tasks):
            t = threading.Thread(
                target=execute_trial_task, 
                args=(model_id, trial_idx, total_tasks, idx),
                daemon=True
            )
            t.start()
            
            # Wait 2.0 seconds before starting the next trial to run at 30 RPM
            time.sleep(2.0)
            
        # Wait for all tasks to complete
        print("[Benchmark] All trials spawned. Waiting for active runs to complete...")
        while True:
            running_state = False
            with benchmark_lock:
                running_state = benchmark_running
            if not running_state:
                break
            time.sleep(1.0)
            
        # Print results at the end
        print("\n==================================================")
        print("ALL BENCHMARK TRIALS COMPLETED. PRINTING RESULTS:")
        print("==================================================")
        
        for model_id in sorted(models_to_test):
            print(f"\n##################################################")
            print(f"MODEL: {model_id}")
            print(f"##################################################")
            
            trials = active_run_results.get(model_id, [])
            for trial_idx, trial in enumerate(trials):
                print(f"\n--- TRIAL {trial_idx + 1} ---")
                if trial.get("success"):
                    print(f"Success: True")
                    print(f"TTFT: {trial['ttft_ms']:.2f} ms")
                    print(f"TPS: {trial['tps']:.2f} tokens/sec")
                    print(f"Tokens: {trial['tokens']}")
                    print(f"Total Time: {trial.get('total_time_ms', 0.0):.2f} ms")
                    if "text" in trial:
                        print(f"\n--- Response Text ---")
                        print(trial["text"])
                else:
                    print(f"Success: False")
                    print(f"Error: {trial.get('error')}")
            
    except Exception as e:
        print(f"[Benchmark] Error in benchmark suite: {e}")
        benchmark_status = f"Error: {str(e)}"
        with benchmark_lock:
            benchmark_running = False

def trigger_benchmark_async():
    global benchmark_running, benchmark_status
    with benchmark_lock:
        if benchmark_running:
            return False, "Benchmark is already running."
        benchmark_running = True
        
    def worker():
        try:
            run_benchmark_suite()
        except Exception as e:
            print(f"[Benchmark] Worker thread error: {e}")
            
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return True, "Benchmark triggered successfully."

def benchmark_scheduler_loop():
    # Wait 10 seconds after server startup
    time.sleep(10)
    last_run_time = time.time()
    
    while True:
        current_time = time.time()
        # Run hourly (3600 seconds)
        if current_time - last_run_time >= 3600:
            # Check if not already running before triggering
            is_running = False
            with benchmark_lock:
                is_running = benchmark_running
            if not is_running:
                try:
                    run_benchmark_suite()
                except Exception as e:
                    print(f"[Benchmark Scheduler] Error: {e}")
                last_run_time = time.time()
        time.sleep(10)

def start_benchmark_thread():
    t = threading.Thread(target=benchmark_scheduler_loop, daemon=True)
    t.start()
    print("[Benchmark] Background scheduler thread started.")

class NIMLocalServerHandler(BaseHTTPRequestHandler):

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        # 1. Serve index.html (benchmark dashboard) at root '/' or '/index.html' or '/benchmark' or '/benchmark.html'
        if self.path in ('/', '/index.html', '/benchmark', '/benchmark.html'):
            try:
                if not os.path.exists(INDEX_HTML_PATH):
                    raise FileNotFoundError(f"index.html not found at: {INDEX_HTML_PATH}")
                    
                with open(INDEX_HTML_PATH, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(f"Error serving index.html: {str(e)}".encode('utf-8'))
            return

        # 1.5 Serve playground.html at '/playground', '/playground.html', '/chat', or '/chat.html'
        if self.path in ('/playground', '/playground.html', '/chat', '/chat.html'):
            try:
                if not os.path.exists(PLAYGROUND_HTML_PATH):
                    raise FileNotFoundError(f"playground.html not found at: {PLAYGROUND_HTML_PATH}")
                    
                with open(PLAYGROUND_HTML_PATH, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(f"Error serving playground.html: {str(e)}".encode('utf-8'))
            return

        # 2. Mock endpoint GET /v1/models
        if self.path == '/v1/models':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            models_data = {
                "object": "list",
                "data": [
                    {"id": "deepseek-ai/deepseek-r1", "object": "model"},
                    {"id": "meta/llama3-8b-instruct", "object": "model"},
                    {"id": "meta/llama3-70b-instruct", "object": "model"},
                    {"id": "nvidia/nemotron-4-340b-instruct", "object": "model"},
                    {"id": "mistralai/mixtral-8x22b-instruct-v0.1", "object": "model"}
                ]
            }
            self.wfile.write(json.dumps(models_data).encode('utf-8'))
            return

        # 2.1 Get benchmark results
        if self.path == '/v1/benchmark/results':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            history = []
            if os.path.exists(RESULTS_PATH):
                try:
                    with open(RESULTS_PATH, "r") as f:
                        history = json.load(f)
                except Exception:
                    pass
            self.wfile.write(json.dumps(history).encode('utf-8'))
            return

        # 2.2 Get benchmark status
        if self.path == '/v1/benchmark/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            status_data = {
                "running": benchmark_running,
                "status": benchmark_status
            }
            self.wfile.write(json.dumps(status_data).encode('utf-8'))
            return

        # 3. Real NVIDIA Cloud Proxy GET /proxy/models -> https://integrate.api.nvidia.com/v1/models
        if self.path.startswith('/proxy/'):
            target_path = self.path[len('/proxy/'):]
            target_url = f"https://integrate.api.nvidia.com/v1/{target_path}"
            
            auth_header = self.headers.get('Authorization')
            auth_header = resolve_auth_header(auth_header)
                    
            req = urllib.request.Request(target_url, method='GET')
            if auth_header:
                req.add_header('Authorization', auth_header)
                
            try:
                with urllib.request.urlopen(req) as response:
                    res_content = response.read()
                    self.send_response(200)
                    self.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
                    self.end_headers()
                    self.wfile.write(res_content)
            except urllib.error.HTTPError as e:
                res_content = e.read()
                self.send_response(e.code)
                self.send_header('Content-Type', e.headers.get('Content-Type', 'application/json'))
                self.end_headers()
                self.wfile.write(res_content)
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        # 1. Mock endpoint POST /v1/chat/completions
        if self.path == '/v1/chat/completions':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode('utf-8'))
            model = req.get('model', 'unknown')
            thinking = req.get('thinking', {})
            chat_kwargs = req.get('chat_template_kwargs', {})
            thinking_enabled = thinking.get('type') == 'enabled' or chat_kwargs.get('enable_thinking', False)
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            
            chunks = []
            
            if model == "deepseek-ai/deepseek-r1":
                if thinking_enabled:
                    # Stream thoughts first
                    chunks.append("<think>")
                    chunks.append("Analysing ")
                    chunks.append("user ")
                    chunks.append("request... ")
                    chunks.append("Locating ")
                    chunks.append("workspace ")
                    chunks.append("files... ")
                    chunks.append("Preparing ")
                    chunks.append("response ")
                    chunks.append("structure... ")
                    chunks.append("Ready. ")
                    chunks.append("</think>\n")
                    chunks.append("Hello! ")
                    chunks.append("I ")
                    chunks.append("am ")
                    chunks.append("DeepSeek-R1 ")
                    chunks.append("running ")
                    chunks.append("in ")
                    chunks.append("mock ")
                    chunks.append("mode. ")
                    chunks.append("You ")
                    chunks.append("should ")
                    chunks.append("see ")
                    chunks.append("my ")
                    chunks.append("thought ")
                    chunks.append("process ")
                    chunks.append("captured ")
                    chunks.append("neatly ")
                    chunks.append("in ")
                    chunks.append("the ")
                    chunks.append("collapsible ")
                    chunks.append("accordion ")
                    chunks.append("above. ")
                else:
                    chunks.append("Hello! ")
                    chunks.append("I ")
                    chunks.append("am ")
                    chunks.append("DeepSeek-R1 ")
                    chunks.append("running ")
                    chunks.append("in ")
                    chunks.append("mock ")
                    chunks.append("mode. ")
                    chunks.append("Thinking ")
                    chunks.append("was ")
                    chunks.append("disabled, ")
                    chunks.append("so ")
                    chunks.append("I ")
                    chunks.append("am ")
                    chunks.append("answering ")
                    chunks.append("directly ")
                    chunks.append("without ")
                    chunks.append("any ")
                    chunks.append("thought ")
                    chunks.append("trace. ")
            else:
                response_text = "Hello! This is a mock response from the model `" + model + "`. I am streaming tokens to verify your telemetry metrics dashboard. Let's make sure code formatting looks great:\n\n```python\n# Telemetry Verification Script\ndef run_check():\n    ttft_ms = 400\n    tps = 20\n    print(f'Verifying TTFT: {ttft_ms}ms, Speed: {tps} tok/s')\n```\nAll calculations are verified!"
                chunks = [word + (" " if i < len(response_text.split(" ")) - 1 else "") for i, word in enumerate(response_text.split(" "))]
            
            time.sleep(0.4) # Wait 400ms to simulate TTFT
            for chunk in chunks:
                data = {
                    "choices": [{
                        "delta": {
                            "content": chunk
                        }
                    }]
                }
                self.wfile.write(f"data: {json.dumps(data)}\n\n".encode('utf-8'))
                self.wfile.flush()
                time.sleep(0.04) # delay to simulate throughput
                
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        # 2. Real NVIDIA Cloud Proxy POST /proxy/chat/completions -> https://integrate.api.nvidia.com/v1/chat/completions
        if self.path.startswith('/proxy/'):
            target_path = self.path[len('/proxy/'):]
            target_url = f"https://integrate.api.nvidia.com/v1/{target_path}"
            
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            auth_header = self.headers.get('Authorization')
            auth_header = resolve_auth_header(auth_header)
                    
            content_type = self.headers.get('Content-Type', 'application/json')
            
            req = urllib.request.Request(target_url, data=post_data, method='POST')
            req.add_header('Content-Type', content_type)
            if auth_header:
                req.add_header('Authorization', auth_header)
                
            try:
                with urllib.request.urlopen(req) as response:
                    self.send_response(200)
                    for header, val in response.headers.items():
                        if header.lower() in ('content-type', 'cache-control', 'connection'):
                            self.send_header(header, val)
                    self.end_headers()
                    
                    while True:
                        chunk = response.read(1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except urllib.error.HTTPError as e:
                res_content = e.read()
                self.send_response(e.code)
                self.send_header('Content-Type', e.headers.get('Content-Type', 'application/json'))
                self.end_headers()
                self.wfile.write(res_content)
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))
            return

        # 3. Save benchmark configuration
        if self.path == '/v1/benchmark/config':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                req_data = json.loads(post_data.decode('utf-8'))
                new_key = req_data.get('api_key', '')
                success = save_api_key(new_key)
                
                self.send_response(200 if success else 500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": success, "message": "API key updated." if success else "Failed to save key."}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "message": str(e)}).encode('utf-8'))
            return

        # 4. Trigger manual benchmark run
        if self.path == '/v1/benchmark/run':
            success, msg = trigger_benchmark_async()
            self.send_response(200 if success else 409)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"success": success, "message": msg}).encode('utf-8'))
            return

        self.send_response(404)
        self.end_headers()

def run(server_class=ThreadingHTTPServer, handler_class=NIMLocalServerHandler, port=8000, start_scheduler=False):
    if start_scheduler:
        start_benchmark_thread()
    else:
        print("[Benchmark] Background scheduled benchmarks are disabled by default when running locally.")
        print("[Benchmark] To enable hourly scheduled benchmarks, run with: python nim_server.py --scheduler")
        print("[Benchmark] Or set environment variable ENABLE_SCHEDULER=true")
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"============================================================")
    print(f"  NVIDIA NIM Playground and Proxy Local Server Running")
    print(f"  --> Visit: http://localhost:{port}/")
    print(f"  --> Benchmarks: http://localhost:{port}/benchmark")
    print(f"============================================================")
    httpd.serve_forever()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--run-once':
        run_benchmark_suite()
    else:
        # Check command line flags or environment variables
        start_scheduler = False
        port = 8000
        for arg in sys.argv[1:]:
            if arg == '--scheduler':
                start_scheduler = True
            elif arg.startswith('--port='):
                try:
                    port = int(arg.split('=')[1])
                except ValueError:
                    pass
        if os.environ.get("ENABLE_SCHEDULER", "").lower() in ("true", "1", "yes"):
            start_scheduler = True

        run(port=port, start_scheduler=start_scheduler)
