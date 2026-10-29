"""Tạo video AI bằng LTX-2.5 (distilled, DiT int8) trên MLX.

Cách dùng: video.py <độ phân giải: 512|720> <giây: 2|5> <nguồn: t2v|i2v>
i2v = lấy ảnh cuối của bài IMAGE làm khung hình đầu. Prompt lấy từ BENCH_PROMPT.
Chạy bằng Python trong third_party/ltx-2-mlx/.venv (runtime xocialize/ltx-2-mlx, nhánh ltx-2.5).
"""
import os
import re
import subprocess
import sys
import time

from common import OUT, PROMPT, ROOT, ensure_model, fail, offline_if_cached, progress, result

TREE_REPO = "mlx-community/ltx-2.5-mlx-q8"          # text encoder int8 + VAE + upscaler (không có DiT)
DIT_REPO = "mlx-community/ltx-2.5-mlx-ditq8"        # DiT distilled int8
SIZES = {"512": (512, 768), "720": (704, 1280)}
FPS = 24
PROMPT_DEFAULT = ("A red fox walks through deep fresh snow in a quiet pine forest at golden hour, "
          "soft snowflakes falling, warm sunlight through the trees, cinematic wildlife shot, shallow depth of field")


def build_model_dir(tree, dit):
    """Ghép cây model q8 với DiT int8 thành một thư mục (bằng symlink, không tốn thêm dung lượng)."""
    out = os.path.join(ROOT, "build", "models", "ltx-2.5-distilled-q8")
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(tree):
        if name.startswith("transformer"):
            continue
        dst = os.path.join(out, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(os.path.join(tree, name)), dst)
    dst = os.path.join(out, "transformer-distilled.safetensors")
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(os.path.join(dit, "transformer-distilled.safetensors")), dst)
    return out


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else "512"
    seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    if res not in SIZES:
        fail(f"Độ phân giải không hợp lệ: {res}")
    h, w = SIZES[res]
    offline_if_cached(DIT_REPO, TREE_REPO)
    source = sys.argv[3] if len(sys.argv) > 3 else "t2v"
    frames = seconds * FPS + 1  # LTX cần 8k+1 khung hình
    prompt = PROMPT or PROMPT_DEFAULT
    last_image = os.path.join(ROOT, "build", "last_image.png")
    if source == "i2v" and not os.path.exists(last_image):
        fail("Chưa có ảnh nào. Chạy bài IMAGE trước rồi chọn 'Từ ảnh cuối'.")
    cli = os.path.join(os.path.dirname(sys.executable), "ltx-2-mlx")
    if not os.path.exists(cli):
        fail("Chưa cài runtime LTX-2.5. Bấm 'Cài đặt AI' hoặc chạy ./setup_ai.sh")

    dit = ensure_model(DIT_REPO, 0.0, 0.2, allow_patterns=["transformer-distilled.safetensors", "*.json"])
    tree = ensure_model(TREE_REPO, 0.2, 0.4)
    model_dir = build_model_dir(tree, dit)

    out_mp4 = os.path.join(OUT, "video.mp4")
    if os.path.exists(out_mp4):
        os.remove(out_mp4)
    env = dict(os.environ, PATH=os.path.join(ROOT, "build", "bin") + ":" + os.environ.get("PATH", ""))
    cmd = [cli, "generate", "--distilled", "--model", model_dir, "--prompt", prompt,
           "-H", str(h), "-W", str(w), "-f", str(frames), "--frame-rate", str(FPS),
           "--seed", "42", "-o", out_mp4]
    if source == "i2v":
        cmd += ["--image", last_image]

    progress(0.42, f"LTX-2.5: nạp model và mã hoá prompt ({w}×{h}, {frames} khung hình)")
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
    tail = []
    buf = ""
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        if ch not in "\r\n":
            buf += ch
            continue
        line, buf = buf.strip(), ""
        if not line:
            continue
        tail = (tail + [line])[-15:]
        low = line.lower()
        m = re.search(r"(\d+)/(\d+)", line)
        if m and int(m.group(2)) > 0:
            done, total = int(m.group(1)), int(m.group(2))
            # Distilled: giai đoạn 1 có 8 bước (nửa độ phân giải), giai đoạn 2 có 3 bước (tinh chỉnh)
            stage = 2 if total <= 3 else 1
            base, span = (0.45, 0.35) if stage == 1 else (0.8, 0.15)
            progress(base + span * done / total, f"LTX-2.5 giai đoạn {stage}: bước {done}/{total}")
        elif "decod" in low:
            progress(0.95, "Giải mã video + âm thanh")
    rc = proc.wait()
    dt = time.perf_counter() - t0
    if rc != 0 or not os.path.exists(out_mp4):
        fail("LTX-2.5 lỗi: " + " | ".join(tail[-3:]))

    preview = os.path.join(OUT, "video.png")
    subprocess.run([os.path.join(ROOT, "build", "bin", "ffmpeg"), "-y", "-loglevel", "error", "-ss", str(seconds / 2),
                    "-i", out_mp4, "-frames:v", "1", preview], check=False)
    size_mb = os.path.getsize(out_mp4) / 1e6
    result(
        dt, f"{dt:.1f}", "giây", seconds / dt * 60,
        f"LTX-2.5 distilled (DiT int8) trên MLX: {seconds} s video {w}×{h} @ {FPS} fps, có âm thanh"
        + (", từ ảnh cuối" if source == "i2v" else ""),
        [f"{dt / seconds:.1f} giây tính toán cho mỗi giây video ({frames} khung hình)",
         f"Tính cả nạp model và mã hoá prompt; file mp4 {size_mb:.1f} MB"],
        poster="video.png", video="video.mp4", prompt=prompt,
    )


if __name__ == "__main__":
    main()
