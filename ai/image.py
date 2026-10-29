"""Tạo ảnh bằng FLUX.2 klein 4B trên MLX (mflux).

Cách dùng: image.py <độ phân giải> <lượng tử: bf16|q8|q4>
Prompt lấy từ biến môi trường BENCH_PROMPT (để trống = 4 prompt mẫu).
"""
import os
import shutil
import sys
import time

from common import OUT, PROMPT, ROOT, ensure_model, fail, offline_if_cached, progress, result

REPO = "black-forest-labs/FLUX.2-klein-4B"
PATTERNS = ["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*"]
PROMPTS = [
    "Photorealistic close-up of a hummingbird hovering near red flowers, frozen wings, detailed feathers, soft green bokeh",
    "A futuristic Mac Studio on a wooden desk at night, neon city skyline through the window, cinematic lighting",
    "A red fox resting in fresh snow under soft winter light, detailed fur, gentle bokeh, natural color grading",
    "Aerial view of a winding river through autumn forest, golden hour, ultra detailed landscape photography",
]
COUNT = len(PROMPTS)
STEPS = 4
LAST_IMAGE = os.path.join(ROOT, "build", "last_image.png")  # bài VIDEO "từ ảnh cuối" dùng ảnh này


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    quant = sys.argv[2] if len(sys.argv) > 2 else "bf16"
    offline_if_cached(REPO)
    bits = {"bf16": None, "q8": 8, "q4": 4}.get(quant, "bad")
    if bits == "bad":
        fail(f"Lượng tử không hợp lệ: {quant}")
    try:
        from mflux.models.common.config import ModelConfig
        from mflux.models.flux2.variants import Flux2Klein
    except ImportError:
        fail("Chưa cài mflux. Bấm 'Cài đặt AI' hoặc chạy ./setup_ai.sh")

    path = ensure_model(REPO, 0.0, 0.35, allow_patterns=PATTERNS)
    progress(0.36, f"Nạp FLUX.2 klein 4B ({quant})")
    t_load = time.perf_counter()
    model = Flux2Klein(quantize=bits, model_path=path, model_config=ModelConfig.flux2_klein_4b())
    load_s = time.perf_counter() - t_load

    prompts = [PROMPT] * COUNT if PROMPT else PROMPTS
    progress(0.4, "Khởi động (ảnh 512×512)")
    model.generate_image(seed=1, prompt=prompts[0], num_inference_steps=STEPS, width=512, height=512)

    times = []
    images = []
    for i, prompt in enumerate(prompts):
        progress(0.45 + 0.55 * i / COUNT, f"Tạo ảnh {i + 1}/{COUNT} ({size}×{size}, {STEPS} bước)")
        t0 = time.perf_counter()
        img = model.generate_image(seed=100 + i, prompt=prompt, num_inference_steps=STEPS, width=size, height=size)
        times.append(time.perf_counter() - t0)
        name = f"image_{i + 1}.png"
        img.image.save(os.path.join(OUT, name))
        images.append(name)
    shutil.copyfile(os.path.join(OUT, images[-1]), LAST_IMAGE)
    total = sum(times)
    per = total / COUNT
    result(
        total, f"{per:.2f}", "giây/ảnh", 60.0 / per,
        f"FLUX.2 klein 4B ({quant}) trên MLX: {COUNT} ảnh {size}×{size}, {STEPS} bước",
        [f"{60.0 / per:.1f} ảnh/phút, nhanh nhất {min(times):.2f} s, chậm nhất {max(times):.2f} s",
         f"{STEPS / per:.2f} bước/giây, nạp model {load_s:.1f} s"],
        images=images, prompt=prompts[0],
    )


if __name__ == "__main__":
    main()
