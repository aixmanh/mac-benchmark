"""Chạy mô hình ngôn ngữ cục bộ bằng MLX và đo tốc độ token/giây.

Cách dùng: llm.py <model>
"""
import sys
import time

from common import PROMPT, ensure_model, fail, offline_if_cached, progress, result, stream

MODELS = {
    # repo, tên hiển thị, mức lượng tử
    "llama3b": ("mlx-community/Llama-3.2-3B-Instruct-4bit", "Llama 3.2 3B", "4-bit"),
    "qwen8b": ("mlx-community/Qwen3-8B-4bit", "Qwen3 8B", "4-bit"),
    "qwen27b": ("mlx-community/Qwen3.8-27B-8bit", "Qwen3.8 27B", "8-bit"),
}

CONTEXT = (
    "Apple silicon combines CPU cores, GPU cores, a neural engine and media engines on a single chip that shares one "
    "pool of unified memory. Because the processor and graphics share memory, large machine learning models can be "
    "loaded once and used by every engine without copying. "
)
QUESTION = "Based on the notes above, write a detailed explanation of why unified memory helps run local AI models."
MAX_TOKENS = 512


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "llama3b"
    if key not in MODELS:
        fail(f"Model không hợp lệ: {key}")
    repo, name, quant = MODELS[key]
    offline_if_cached(repo)
    try:
        from mlx_lm import load, stream_generate
    except ImportError:
        fail("Chưa cài mlx-lm. Bấm 'Cài đặt AI' hoặc chạy ./setup_ai.sh")

    path = ensure_model(repo, 0.0, 0.5)
    progress(0.52, f"Nạp {name} vào bộ nhớ")
    t_load = time.perf_counter()
    model, tok = load(path)
    load_s = time.perf_counter() - t_load

    content = PROMPT or CONTEXT * 24 + "\n\n" + QUESTION
    messages = [{"role": "user", "content": content}]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    progress(0.6, f"{name}: xử lý prompt")
    last = None
    text, pending = "", ""
    t0 = time.perf_counter()
    for i, resp in enumerate(stream_generate(model, tok, prompt, max_tokens=MAX_TOKENS)):
        last = resp
        text += resp.text
        pending += resp.text
        if i % 8 == 0:
            stream(pending)
            pending = ""
        if i % 16 == 0:
            progress(0.6 + 0.4 * i / MAX_TOKENS, f"{name}: {i} token, {resp.generation_tps:.1f} tok/s")
    if pending:
        stream(pending)
    dt = time.perf_counter() - t0
    if last is None:
        fail("Model không sinh ra token nào")
    result(
        dt, f"{last.generation_tps:.1f}", "token/giây", last.generation_tps,
        f"{name} ({quant}) trên MLX: sinh {last.generation_tokens} token",
        [f"Xử lý prompt {last.prompt_tokens} token: {last.prompt_tps:.0f} token/giây",
         f"Bộ nhớ đỉnh {last.peak_memory:.1f} GB, nạp model {load_s:.1f} s"],
        text=text, prompt=PROMPT or "(prompt mẫu ~1.300 token về bộ nhớ hợp nhất)",
    )


if __name__ == "__main__":
    main()
