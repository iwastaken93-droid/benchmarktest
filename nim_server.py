import json
import time
import urllib.request
import urllib.error
import os
import threading
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML_PATH = os.path.join(SCRIPT_DIR, "index.html")
PLAYGROUND_HTML_PATH = os.path.join(SCRIPT_DIR, "playground.html")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "benchmark_results.json")

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

def is_chat_model(model_id):
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
    latency = ttft + (tokens / tps) * 1000.0
    
    # 95% success rate for mock trials
    success = random.random() < 0.95
    if success:
        tpot = (latency - ttft) / (tokens - 1) if tokens > 1 else 0.0
        return {
            "success": True,
            "ttft_ms": ttft,
            "latency_ms": latency,
            "tokens": tokens,
            "tps": tps,
            "tpot_ms": tpot
        }
    else:
        return {
            "success": False,
            "error": "Mock connection reset / timeout"
        }

def run_trial(model_id, api_key, prompt, max_tokens):
    # If the API key starts with a mock prefix or is not set, run mock trial
    if not api_key or api_key.startswith("mock"):
        return run_mock_trial(model_id)
        
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True
    }
    
    req_data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=req_data, method='POST')
    req.add_header('Content-Type', 'application/json')
    req.add_header('Authorization', f"Bearer {api_key}")
    
    start_time = time.time()
    ttft = 0.0
    total_time = 0.0
    generated_text = ""
    
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            if response.status != 200:
                return {"success": False, "error": f"HTTP status {response.status}"}
                
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
                            err_msg = data_json["error"].get("message", "Mid-stream API error")
                            return {"success": False, "error": f"API Error: {err_msg}"}
                        choices = data_json.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            reasoning = delta.get("reasoning", "") or delta.get("reasoning_content", "")
                            generated_text += content + reasoning
                    except Exception:
                        pass
            
            total_time = (time.time() - start_time) * 1000.0
            
            # Check if we generated any text. If not, treat as failure.
            if not generated_text.strip():
                return {"success": False, "error": "Empty response or stream error"}
                
            # Estimate token count based on generated text
            word_count = len(generated_text.split())
            char_count = len(generated_text)
            if word_count > 0:
                token_count = int(word_count * 1.33)
            else:
                token_count = int(char_count / 4)
            if token_count == 0:
                token_count = 1

            # Reject safety / guardrail models that generate very short responses
            if token_count < 100:
                return {"success": False, "error": f"Generated too few tokens ({token_count} < 100)"}
                
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
                "latency_ms": total_time,
                "tokens": token_count,
                "tps": tps,
                "tpot_ms": tpot
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

def execute_trial_task(model_id, trial_idx, total_tasks, task_idx):
    global benchmark_status, completed_tasks_count, benchmark_running, active_run_results
    
    api_key = get_api_key()
    prompt = "Write a 1000-word story about a speed unicorn. MAKE IT SPEED UNICORN SPEEED"
    max_tokens = 10000
    
    trial_res = {"success": False, "error": "Unknown error"}
    
    try:
        trial_res = run_trial(model_id, api_key, prompt, max_tokens)
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
                    avg_latency = sum(t["latency_ms"] for t in success_trials) / len(success_trials)
                    avg_tps = sum(t["tps"] for t in success_trials) / len(success_trials)
                    avg_tokens = sum(t["tokens"] for t in success_trials) / len(success_trials)
                    avg_tpot = sum(t.get("tpot_ms", 0.0) for t in success_trials) / len(success_trials)
                else:
                    avg_ttft = 0.0
                    avg_latency = 0.0
                    avg_tps = 0.0
                    avg_tokens = 0.0
                    avg_tpot = 0.0
                    
                if success_rate > 0.0 and avg_tokens >= 100:
                    model_summary = {
                        "model": model_id,
                        "avg_ttft_ms": avg_ttft,
                        "avg_latency_ms": avg_latency,
                        "avg_tps": avg_tps,
                        "avg_tpot_ms": avg_tpot,
                        "avg_tokens": avg_tokens,
                        "success_rate": success_rate,
                        "trials": trials
                    }
                    save_incremental_model_result(model_summary)
                else:
                    print(f"[Benchmark] Skipping saving results for {model_id} (success_rate={success_rate}, avg_tokens={avg_tokens:.1f})")
                
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
            
        # Build tasks queue
        all_tasks = []
        for model_id in models_to_test:
            for trial_idx in range(3):
                all_tasks.append((model_id, trial_idx))
                
        total_tasks = len(all_tasks)
        print(f"[Benchmark] Starting queue with {total_tasks} total trials...")
        
        # Spawn thread for each task every 6.0 seconds (10 requests/min, well under 20 requests/min limit)
        for idx, (model_id, trial_idx) in enumerate(all_tasks):
            t = threading.Thread(
                target=execute_trial_task, 
                args=(model_id, trial_idx, total_tasks, idx),
                daemon=True
            )
            t.start()
            
            # Wait 6 seconds before starting the next trial to respect rate limit and run in parallel
            time.sleep(6.0)
            
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
    last_run_time = 0.0
    
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
            # Inject API key from config.json if not provided or empty in client headers
            if not auth_header or auth_header.strip() in ("Bearer", "Bearer null", "Bearer undefined"):
                key = get_api_key()
                if key:
                    auth_header = f"Bearer {key}"
                    
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
            # Inject API key from config.json if not provided or empty in client headers
            if not auth_header or auth_header.strip() in ("Bearer", "Bearer null", "Bearer undefined"):
                key = get_api_key()
                if key:
                    auth_header = f"Bearer {key}"
                    
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

def run(server_class=ThreadingHTTPServer, handler_class=NIMLocalServerHandler, port=8000):
    start_benchmark_thread()
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
        run()
