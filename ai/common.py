"""Tiện ích chung cho các bài test AI: in tiến độ/kết quả theo giao thức của server và tải model."""
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Server truyền thư mục kết quả riêng cho mỗi lần chạy và prompt người dùng nhập (nếu có)
OUT = os.environ.get("BENCH_OUT") or os.path.join(ROOT, "build")
PROMPT = os.environ.get("BENCH_PROMPT", "").strip()
os.makedirs(OUT, exist_ok=True)


def progress(p, msg):
    print(f"PROGRESS {max(0.0, min(1.0, p)):.3f} {msg}", flush=True)


def stream(text):
    """Gửi một đoạn chữ (LLM đang sinh) để giao diện hiện dần."""
    print("STREAM " + json.dumps(text, ensure_ascii=False), flush=True)


def fail(msg):
    print(f"ERROR {msg}", flush=True)
    sys.exit(1)


def result(seconds, big, unit, metric, summary, details, **extra):
    data = dict(seconds=seconds, big=big, unit=unit, metric=metric, summary=summary, details=details, **extra)
    print("RESULT " + json.dumps(data, ensure_ascii=False), flush=True)


def hf_cached(repo):
    """Model đã nằm trong bộ nhớ đệm Hugging Face của máy chưa (không cần mạng để kiểm tra)."""
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    snaps = os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots")
    return os.path.isdir(snaps) and any(os.scandir(snaps))


def offline_if_cached(*repos):
    """Model đã có trong máy -> chặn mọi kết nối tới Hugging Face.

    Không đăng nhập thì Hugging Face giới hạn lượt hỏi; các thư viện vẫn hỏi lại model mỗi lần nạp nên có lúc phải
    chờ/thử lại rất lâu (đo được VieNeu: 0.8 s offline so với 36–197 s online). Phải gọi trước khi import
    huggingface_hub hay thư viện model.
    """
    if repos and all(hf_cached(r) for r in repos):
        os.environ["HF_HUB_OFFLINE"] = "1"
        return True
    return False


def _dir_size(path):
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    return total


def ensure_model(repo, p_from=0.0, p_to=0.3, allow_patterns=None):
    """Tải model từ Hugging Face (chỉ lần đầu), báo tiến độ theo dung lượng đã tải."""
    import fnmatch
    from huggingface_hub import HfApi, snapshot_download

    try:
        return snapshot_download(repo, local_files_only=True, allow_patterns=allow_patterns)
    except Exception:
        pass

    try:
        info = HfApi().model_info(repo, files_metadata=True)
        total = sum((s.size or 0) for s in info.siblings
                    if not allow_patterns or any(fnmatch.fnmatch(s.rfilename, pat) for pat in allow_patterns))
    except Exception:
        total = 0
    from huggingface_hub.constants import HF_HUB_CACHE

    cache = os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"))
    done = threading.Event()

    def watch():
        while not done.wait(0.7):
            got = _dir_size(cache) if os.path.isdir(cache) else 0
            if total and got:
                progress(p_from + (p_to - p_from) * got / total,
                         f"Tải model {repo} (lần đầu): {got / 1e9:.2f}/{total / 1e9:.2f} GB")
            else:
                progress(p_from, f"Tải model {repo} (lần đầu, {total / 1e9:.1f} GB)" if total else f"Tải model {repo} (lần đầu)")

    threading.Thread(target=watch, daemon=True).start()
    try:
        path = snapshot_download(repo, allow_patterns=allow_patterns)
    except Exception as e:
        fail(f"Không tải được model {repo}: {e}")
    finally:
        done.set()
    return path
