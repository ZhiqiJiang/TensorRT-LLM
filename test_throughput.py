from transformers import AutoTokenizer
import requests
import time
import threading
import numpy as np

model_path = "/root/TensorRT-LLM/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
max_tokens = 20

# Define the URL and headers
url = "http://localhost:8000/v1/chat/completions"
url = "http://localhost:8000/v2/models/tensorrt_llm_bls/generate"
headers = {
    "Content-Type": "application/json"
}
data = {
    "model": model_path,
    "messages": [
        {"role": "user", "content": "如何复现deepseek r1中的知识蒸馏" * 200} # 2200 tokens
    ],
    "max_tokens": max_tokens
}
data = {
    "text_input": "如何复现deepseek r1中的知识蒸馏" * 200,
    "max_tokens": max_tokens
}

# Function to send a request and return the token counts
def send_request(results, lock):
    t1 = time.time()
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    prompt = data["messages"][0]["content"]
    out_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
    input_tokens = len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt)))
    output_tokens = len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(out_text)))
    t2 = time.time()
    # print(f"request time: {(t2 - t1) * 1000:.2f} ms, input_tokens_num: {input_tokens}, "
    #       f"output_tokens_num: {output_tokens}, TPOT: {(t2 - t1) * 1000 / output_tokens:.2f} ms")
    with lock:
        results.append((input_tokens, output_tokens, (t2 - t1) * 1000))

def send_request_triton(results, lock):
    t1 = time.time()
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    prompt = data["text_input"]
    out_text = response_json["text_output"]
    input_tokens = len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt)))
    output_tokens = len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(out_text)))
    t2 = time.time()
    # print(f"request time: {(t2 - t1) * 1000:.2f} ms, input_tokens_num: {input_tokens}, "
    #       f"output_tokens_num: {output_tokens}, TPOT: {(t2 - t1) * 1000 / output_tokens:.2f} ms")
    with lock:
        results.append((input_tokens, output_tokens, (t2 - t1) * 1000))

def request():
    t1 = time.time()
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    prompt = data["text_input"]
    out_text = response_json["text_output"]
    print(prompt)
    print(out_text)
    t2 = time.time()

# Function to test different concurrency levels
def test_concurrency(concurrency_level, duration=5):
    threads = []
    results = []
    lock = threading.Lock()
    start_time = time.time()

    def worker():
        while time.time() - start_time < duration:
            send_request_triton(results, lock)

    for _ in range(concurrency_level):
        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    input_tokens = [result[0] for result in results]
    output_tokens = [result[1] for result in results]
    rt = np.mean([result[2] for result in results])
    
    return input_tokens, output_tokens, rt

def test_diff_concurrency():
    concurrency_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    duration = 5
    for level in concurrency_levels:
        t1 = time.time()
        input_tokens, output_tokens, rt = test_concurrency(level, duration)
        # print(f"Concurrency Level: {level}, Total Input Tokens: {sum(input_tokens)}, Total Output Tokens: {sum(output_tokens)}, "
        #       f"QPS: {sum(output_tokens) / max_tokens / duration:.2f}")
        t2 = time.time()
        print(f"concurrency: {level}, cost time: {t2 - t1:.2f} s, "
              f"RT: {rt:.2f} ms, "
              f"QPS: {sum(output_tokens) / max_tokens / (t2 - t1):.2f}")

def test_request():
    results = []
    lock = threading.Lock()
    send_request(results, lock)

if __name__ == "__main__":
    test_diff_concurrency()
    # request()
