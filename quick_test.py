import sys
import os
import json
import time
import urllib.request
import urllib.error

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
            {"role": "user", "content": "Write a 2-sentence story about a speed unicorn."}
        ],
        "temperature": 0.7,
        "max_tokens": 150,
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
        
    if error_occurred:
        print(f"[{model_id}] Error: {error_occurred}\n")
        return
        
    # Calculate stats
    token_count = nim_server.count_completion_tokens(generated_text)
    latency_delta_s = (total_time - ttft) / 1000.0 if total_time and ttft else 0
    tps = (token_count - 1) / latency_delta_s if latency_delta_s > 0 and token_count > 1 else 0.0
    
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
    for model in models:
        print(f"Testing {model}...")
        test_model(model, api_key)
        time.sleep(2.0)

if __name__ == '__main__':
    main()
