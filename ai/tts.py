"""Chuyển văn bản thành giọng nói (TTS) trên MLX bằng mlx-audio.

Cách dùng: tts.py <model: kokoro|qwen3|vieneu>
Văn bản lấy từ BENCH_PROMPT (để trống = đoạn văn mẫu ~35 giây).
VieNeu-TTS chạy bằng Python trong venv của repo VieNeu-TTS (server chọn đúng venv).
"""
import os
import sys
import time
import wave

from common import OUT, PROMPT, ensure_model, fail, offline_if_cached, progress, result

MODELS = {
    # repo, tên hiển thị, giọng mặc định
    "kokoro": ("mlx-community/Kokoro-82M-bf16", "Kokoro 82M", "af_heart"),
    "qwen3": ("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", "Qwen3-TTS 1.7B (8-bit)", "ryan"),
    "vieneu": (None, "VieNeu-TTS v3 Turbo", "Mai Anh"),
}
SAMPLE = (
    "Welcome to the Mac Studio M5 Ultra test. Everything you are hearing right now was generated on this machine, "
    "with no internet connection and no cloud service. The model reads the text, predicts the sound of each word, "
    "and turns it into a natural voice in a fraction of a second. "
    "Unified memory lets the processor and the graphics cores share the same data, so even large speech models "
    "load quickly and run smoothly. This makes it possible to narrate videos, read documents aloud, "
    "or build a private voice assistant that never sends your words anywhere."
)


SAMPLE_VI = (
    "Chào mừng các bạn đến với bài kiểm tra Mac Studio M5 Ultra. Giọng nói bạn đang nghe được tạo ra ngay trên chiếc máy này, "
    "không cần kết nối internet và không dùng dịch vụ đám mây nào. Mô hình đọc văn bản, đoán cách phát âm từng từ, "
    "rồi chuyển thành giọng nói tự nhiên chỉ trong tích tắc. Nhờ vậy, bạn có thể đọc sách nói, lồng tiếng cho video, "
    "hay xây dựng một trợ lý giọng nói riêng tư, không gửi lời nói của mình đi đâu cả."
)


def save_wav(path, chunks, sr):
    import numpy as np

    audio = np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks])
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return len(pcm) / sr


def run_vieneu(name, voice_name):
    """VieNeu-TTS v3 Turbo: ONNX Runtime trên CPU (không dùng GPU/MLX)."""
    # VieNeu tự tải model của nó: chạy offline, nếu chưa có model thì chạy lại một lần có mạng để tải.
    online_retry = os.environ.get("BENCH_VIENEU_ONLINE") == "1"
    if not online_retry:
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from vieneu import Vieneu
    except ImportError:
        fail("Không tìm thấy VieNeu-TTS. Đặt biến VIENEU_DIR trỏ tới repo VieNeu-TTS (có sẵn .venv).")
    text = PROMPT or SAMPLE_VI
    progress(0.1, f"Nạp {name}")
    t_load = time.perf_counter()
    try:
        engine = Vieneu()
    except Exception:
        if online_retry:
            raise
        progress(0.1, f"{name}: chưa có model trong máy, tải từ Hugging Face (lần đầu)")
        env = dict(os.environ, HF_HUB_OFFLINE="0", BENCH_VIENEU_ONLINE="1")
        os.execve(sys.executable, [sys.executable, "-u"] + sys.argv, env)
    load_s = time.perf_counter() - t_load
    voices = engine.list_preset_voices()
    voice_id = next((vid for label, vid in voices if voice_name in label), voices[0][1])

    progress(0.5, "Khởi động (câu ngắn)")
    t_warm = time.perf_counter()
    engine.infer("Khởi động.", voice=voice_id)
    warm_s = time.perf_counter() - t_warm
    progress(0.6, f"{name}: đang đọc {len(text)} ký tự")
    t0 = time.perf_counter()
    audio = engine.infer(text, voice=voice_id)
    dt = time.perf_counter() - t0
    sr = getattr(engine, "sample_rate", None) or 48000
    audio_s = save_wav(os.path.join(OUT, "speech.wav"), [audio], sr)
    rtf = audio_s / dt
    result(
        dt, f"{rtf:.1f}", "× thời gian thực", rtf,
        f"{name} (ONNX trên CPU): {audio_s:.1f} giây giọng đọc tiếng Việt trong {dt:.2f} giây (giọng {voice_name})",
        [f"{len(text) / dt:.0f} ký tự/giây, {sr // 1000} kHz",
         f"Chỉ tính thời gian đọc; trước đó nạp model {load_s:.1f} s + khởi động {warm_s:.1f} s. VieNeu chạy bằng CPU"],
        audio="speech.wav", prompt=text if PROMPT else "(đoạn văn mẫu tiếng Việt)",
    )


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "kokoro"
    if key not in MODELS:
        fail(f"Model không hợp lệ: {key}")
    repo, name, voice = MODELS[key]
    if repo:
        offline_if_cached(repo)
    if key == "vieneu":
        return run_vieneu(name, voice)
    try:
        import mlx.core as mx
        from mlx_audio.tts.utils import load
    except ImportError:
        fail("Chưa cài mlx-audio. Bấm 'Cài đặt AI' hoặc chạy ./setup_ai.sh")

    text = PROMPT or SAMPLE
    ensure_model(repo, 0.0, 0.4)
    progress(0.42, f"Nạp {name}")
    t_load = time.perf_counter()
    model = load(repo)  # mlx-audio nhận loại model từ tên repo, không từ đường dẫn snapshot (đã có trong cache)
    load_s = time.perf_counter() - t_load

    progress(0.5, "Khởi động (câu ngắn)")
    t_warm = time.perf_counter()
    for _ in model.generate("Hello, this is a warm up.", voice=voice):
        pass
    warm_s = time.perf_counter() - t_warm

    progress(0.6, f"{name}: đang đọc {len(text)} ký tự")
    mx.reset_peak_memory()
    chunks, sr = [], 24000
    t0 = time.perf_counter()
    for r in model.generate(text, voice=voice):
        chunks.append(r.audio)
        sr = r.sample_rate
    dt = time.perf_counter() - t0
    if not chunks:
        fail("Model không tạo ra âm thanh")
    audio_s = save_wav(os.path.join(OUT, "speech.wav"), chunks, sr)
    rtf = audio_s / dt
    result(
        dt, f"{rtf:.1f}", "× thời gian thực", rtf,
        f"{name} trên MLX: {audio_s:.1f} giây giọng đọc trong {dt:.2f} giây (giọng {voice})",
        [f"{len(text) / dt:.0f} ký tự/giây, {sr // 1000} kHz",
         f"Chỉ tính thời gian đọc; trước đó nạp model {load_s:.1f} s + khởi động {warm_s:.1f} s. Bộ nhớ đỉnh {mx.get_peak_memory() / 1e9:.1f} GB"],
        audio="speech.wav", prompt=text if PROMPT else "(đoạn văn mẫu tiếng Anh)",
    )


if __name__ == "__main__":
    main()
