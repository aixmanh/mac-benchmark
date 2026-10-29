"""Chuyển giọng nói thành văn bản bằng Whisper trên MLX.

Cách dùng: transcribe.py <model> <phút>
Âm thanh mẫu được tạo bằng lệnh `say` của macOS (chỉ lần đầu) và lưu trong build/audio.
"""
import os
import subprocess
import sys
import time
import wave

from common import ROOT, ensure_model, fail, offline_if_cached, progress, result

MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}

PARAGRAPHS = [
    "Welcome back to the channel. Today we are putting a brand new workstation through a series of real world tests, "
    "from video encoding to speech recognition, to find out how much faster it really is.",
    "The first thing you notice is how quiet the machine stays under load. Even when every core is busy compiling code, "
    "the fans barely spin up, and the case stays cool to the touch.",
    "Unified memory changes the way we think about large models. Instead of copying data between the processor and the "
    "graphics card, both can read the same memory directly, which saves time and energy.",
    "For video editors, the media engines are the real story. Hardware encoders for high efficiency video and ProRes "
    "let you export long timelines in a fraction of the time it used to take.",
    "Machine learning workloads are becoming part of everyday software. Transcription, image generation and local "
    "language models all benefit from fast memory bandwidth and a capable neural engine.",
    "Of course, benchmarks only tell part of the story. What matters is how the computer feels when you are working on "
    "a deadline, switching between applications and juggling dozens of browser tabs.",
    "Storage speed matters too. A fast solid state drive means projects open instantly, caches rebuild quickly and large "
    "files copy in seconds rather than minutes.",
    "Let us talk about power efficiency. A machine that finishes the same job while drawing less power runs cooler, "
    "lasts longer and costs less to operate over its lifetime.",
    "Developers will appreciate the number of performance cores. Large projects build in parallel, test suites finish "
    "sooner, and container workloads run without slowing down the rest of the system.",
    "Finally, remember that software keeps improving. Frameworks are being optimized for this architecture every month, "
    "so the numbers you see today will likely get even better over time.",
]


def pick_voice():
    """Chọn một giọng đọc tiếng Anh có sẵn (giọng mặc định của máy có thể là tiếng Việt)."""
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    voices = {line.split()[0]: line for line in out.splitlines() if line.strip()}
    for name in ("Samantha", "Daniel", "Karen", "Moira", "Alex", "Tessa", "Fred"):
        if name in voices:
            return name
    for line in out.splitlines():
        if " en_" in line:
            return line.split("  ")[0].strip()
    return None


def make_audio(minutes):
    audio_dir = os.path.join(ROOT, "build", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    wav = os.path.join(audio_dir, f"speech_en_{minutes:g}min.wav")
    if os.path.exists(wav):
        return wav
    progress(0.02, f"Tạo {minutes:g} phút âm thanh mẫu bằng giọng đọc macOS (chỉ lần đầu)")
    words_needed = int(minutes * 165 * 1.15)
    text, count, i = [], 0, 0
    while count < words_needed:
        p = PARAGRAPHS[i % len(PARAGRAPHS)]
        text.append(p)
        count += len(p.split())
        i += 1
    txt = os.path.join(audio_dir, "speech.txt")
    aiff = os.path.join(audio_dir, "speech.aiff")
    with open(txt, "w") as f:
        f.write("\n\n".join(text))
    voice = pick_voice()
    subprocess.run(["say"] + (["-v", voice] if voice else []) + ["-r", "165", "-o", aiff, "-f", txt], check=True)
    full = os.path.join(audio_dir, "speech_full.wav")
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, full], check=True)
    os.remove(aiff)
    # Cắt đúng độ dài yêu cầu
    with wave.open(full) as src:
        params = src.getparams()
        data = src.readframes(min(src.getnframes(), int(minutes * 60 * src.getframerate())))
    with wave.open(wav, "wb") as dst:
        dst.setparams(params)
        dst.writeframes(data)
    os.remove(full)
    return wav


def load_wav(path):
    import numpy as np

    with wave.open(path) as w:
        frames = w.readframes(w.getnframes())
        rate = w.getframerate()
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, rate


def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "small"
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 13.8
    repo = MODELS.get(model_key)
    if not repo:
        fail(f"Model không hợp lệ: {model_key}")
    offline_if_cached(repo)

    try:
        import mlx_whisper
    except ImportError:
        fail("Chưa cài mlx-whisper. Bấm 'Cài đặt AI' hoặc chạy ./setup_ai.sh")

    wav = make_audio(minutes)
    audio, rate = load_wav(wav)
    audio_sec = len(audio) / rate
    path = ensure_model(repo, 0.05, 0.3)

    progress(0.32, f"Nạp model Whisper {model_key}")
    opts = dict(path_or_hf_repo=path, language="en", verbose=None, condition_on_previous_text=False)
    mlx_whisper.transcribe(audio[: rate * 5], **opts)

    progress(0.4, f"Đang chuyển {audio_sec / 60:.1f} phút âm thanh thành văn bản")
    t0 = time.perf_counter()
    out = mlx_whisper.transcribe(audio, **opts)
    dt = time.perf_counter() - t0
    text = out.get("text", "").strip()
    words = len(text.split())
    result(
        dt, f"{dt:.1f}", "giây", audio_sec / dt,
        f"{audio_sec / 60:.1f} phút âm thanh, Whisper {model_key} trên MLX",
        [f"{words} từ đã nhận dạng", f"Nhanh gấp {audio_sec / dt:.0f}× thời gian thực",
         f"“{text[:140]}…”"],
    )


if __name__ == "__main__":
    main()
