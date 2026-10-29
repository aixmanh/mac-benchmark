#!/usr/bin/env python3
"""Máy chủ web cho bộ test hiệu năng M5 Ultra. Chạy: python3 server.py rồi mở http://localhost:9800"""
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "build")
WEB = os.path.join(ROOT, "web")
RESULTS = os.path.join(ROOT, "results", "history.json")
OUTPUTS = os.path.join(BUILD, "outputs")  # ảnh/video/khung hình của từng lần chạy
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
IMAGE_PY = os.path.join(ROOT, ".venv-image", "bin", "python")
TTS_PY = os.path.join(ROOT, ".venv-tts", "bin", "python")
# VieNeu-TTS dùng repo + venv riêng của nó (không cài lại trong dự án này)
VIENEU_DIR = os.environ.get("VIENEU_DIR", os.path.expanduser("~/Du_an/TTS/VieNeu-TTS"))
VIENEU_PY = os.path.join(VIENEU_DIR, ".venv", "bin", "python")
VIDEO_PY = os.path.join(ROOT, "third_party", "ltx-2-mlx", ".venv", "bin", "python")
PORT = int(os.environ.get("PORT", "9800"))
CONFIG = os.path.join(ROOT, "config.json")


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def sysctl(name):
    return sh(["sysctl", "-n", name])


# ---------- Biên dịch phần native ----------

def build_native():
    os.makedirs(BUILD, exist_ok=True)
    targets = [
        ("stats", "native/stats.m",
         ["clang", "-O2", "-fobjc-arc", "-framework", "Foundation", "-framework", "IOKit", "-lIOReport"]),
        ("bench", "native/bench.swift", ["swiftc", "-O", "-suppress-warnings"]),
    ]
    for name, src, cmd in targets:
        out, src_path = os.path.join(BUILD, name), os.path.join(ROOT, src)
        if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src_path):
            continue
        print(f"Biên dịch {src} ...", flush=True)
        r = subprocess.run(cmd + [src_path, "-o", out], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            sys.exit(f"Không biên dịch được {src}")


# ---------- Thông số trực tiếp ----------

class Stats:
    def __init__(self):
        self.latest = {}
        self.lock = threading.Lock()
        self.listeners = []

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            proc = subprocess.Popen([os.path.join(BUILD, "stats")], stdout=subprocess.PIPE, text=True)
            for line in proc.stdout:
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                with self.lock:
                    self.latest = data
                    for fn in list(self.listeners):
                        fn(data)
            time.sleep(1)

    def get(self):
        with self.lock:
            return dict(self.latest)


STATS = Stats()


# ---------- Thông tin máy ----------

def machine_info():
    gpu_cores = ""
    prof = sh(["system_profiler", "SPDisplaysDataType"])
    for line in prof.splitlines():
        if "Total Number of Cores" in line:
            gpu_cores = line.split(":")[-1].strip()
    levels = []
    for i in range(4):
        n = sysctl(f"hw.perflevel{i}.logicalcpu")
        if not n:
            break
        levels.append({"name": sysctl(f"hw.perflevel{i}.name") or f"Cấp {i}", "cores": int(n)})
    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "model": sysctl("hw.model"),
        "cpu_cores": int(sysctl("hw.ncpu") or 0),
        "core_levels": levels,
        "gpu_cores": gpu_cores,
        "ram_gb": round(int(sysctl("hw.memsize") or 0) / 2**30),
        "macos": platform.mac_ver()[0],
        "host": platform.node().split(".")[0],
    }


INFO = {}


def ai_status():
    """Môi trường nào đã cài: transcribe/llm (.venv), tts (.venv-tts), image (.venv-image), video (runtime LTX-2.5)."""
    def ok(py, code):
        return os.path.exists(py) and subprocess.run([py, "-c", code], capture_output=True).returncode == 0
    base = ok(VENV_PY, "import mlx_whisper, mlx_lm")
    return {
        "transcribe": base, "llm": base,
        "tts": ok(TTS_PY, "import mlx_audio, misaki, en_core_web_sm") or ok(VIENEU_PY, "import vieneu"),
        "image": ok(IMAGE_PY, "import mflux"),
        "video": ok(VIDEO_PY, "import ltx_pipelines_mlx") and os.path.exists(os.path.join(BUILD, "bin", "ffmpeg")),
    }


# ---------- Chạy test ----------

# Kiểm tra môi trường AI tốn ~3 giây (nạp thử thư viện của 5 venv) nên chỉ làm lúc khởi động và sau khi cài đặt.
AI_STATUS = {}


def refresh_ai_status():
    AI_STATUS.update(ai_status())
    return AI_STATUS


def command_for(test, opts, outdir):
    bench = os.path.join(BUILD, "bench")
    simple = {"cpu", "gpu", "draw", "hevc", "prores", "memory", "ssd"}
    if test in simple:
        return [bench, test, opts.get("mode", ""), outdir]
    if test == "transcribe":
        return [VENV_PY, "-u", os.path.join(ROOT, "ai", "transcribe.py"), opts.get("model", "small"), opts.get("minutes", "13.8")]
    if test == "llm":
        return [VENV_PY, "-u", os.path.join(ROOT, "ai", "llm.py"), opts.get("model", "llama3b")]
    if test == "tts":
        model = opts.get("model", "kokoro")
        py = VIENEU_PY if model == "vieneu" else TTS_PY
        return [py, "-u", os.path.join(ROOT, "ai", "tts.py"), model]
    if test == "image":
        return [IMAGE_PY, "-u", os.path.join(ROOT, "ai", "image.py"), opts.get("size", "1024"), opts.get("quant", "bf16")]
    if test == "video":
        return [VIDEO_PY, "-u", os.path.join(ROOT, "ai", "video.py"), opts.get("res", "512"), opts.get("seconds", "5"),
                opts.get("source", "t2v")]
    if test == "setup":
        return ["/bin/bash", os.path.join(ROOT, "setup_ai.sh")]
    return None


class Job:
    def __init__(self, test, opts, label, prompt="", free=True):
        self.id = str(int(time.time() * 1000))
        self.test, self.opts, self.label, self.prompt = test, opts, label, prompt
        self.outdir = os.path.join(OUTPUTS, self.id)
        self.stream = ""
        self.free = free      # tắt mô hình nền + Docker trước khi chạy
        self.freed = []       # những gì đã tắt
        self.status = "running"
        self.progress, self.message = 0.0, "Đang khởi động"
        self.result, self.error = None, None
        self.started = time.time()
        self.ended = None
        self.proc = None
        self.peak = {}
        self.sums, self.samples = {"cpu": 0.0, "gpu": 0.0}, 0
        self.log = []

    def on_stats(self, s):
        for k in ("cpu_w", "gpu_w", "ane_w", "soc_w", "sys_w", "cpu_temp", "gpu_temp", "cpu", "gpu"):
            if k in s and s[k] is not None and s[k] >= 0:
                self.peak[k] = max(self.peak.get(k, 0), s[k])
        self.sums["cpu"] += max(s.get("cpu", 0), 0)
        self.sums["gpu"] += max(s.get("gpu", 0), 0)
        self.samples += 1

    def to_dict(self):
        n = max(self.samples, 1)
        return {
            "id": self.id, "test": self.test, "opts": self.opts, "label": self.label,
            "status": self.status, "progress": self.progress, "message": self.message,
            "result": self.result, "error": self.error, "log": self.log[-30:],
            "prompt": self.prompt, "stream": self.stream, "freed": self.freed,
            "elapsed": (self.ended or time.time()) - self.started,
            "peak": self.peak, "avg": {"cpu": self.sums["cpu"] / n, "gpu": self.sums["gpu"] / n},
        }


JOB = None
JOB_LOCK = threading.Lock()


def load_history():
    try:
        with open(RESULTS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def save_history(entry):
    hist = load_history()
    hist.append(entry)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(hist[-500:], f, ensure_ascii=False, indent=1)


def load_config():
    try:
        with open(CONFIG) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def free_resources(job):
    """Trước mỗi bài test: tắt các mô hình chạy nền (launchd), Docker và server mô hình chạy tay.

    Danh sách nằm trong config.json (free_before_test). Dịch vụ launchd bị bootout nên không tự bật lại
    cho tới lần đăng nhập/khởi động máy sau. Trả về tên những thứ đã tắt.
    """
    cfg = load_config().get("free_before_test", {})
    uid = os.getuid()
    stopped = []
    for label in cfg.get("launchd", []):
        target = f"gui/{uid}/{label}"
        if subprocess.run(["launchctl", "print", target], capture_output=True).returncode == 0:
            job.message = f"Tắt mô hình nền: {label}"
            subprocess.run(["launchctl", "bootout", target], capture_output=True, timeout=60)
            stopped.append(label.rsplit(".", 1)[-1])
    for pattern in cfg.get("kill_patterns", []):
        if subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0:
            job.message = f"Tắt tiến trình: {pattern}"
            subprocess.run(["pkill", "-f", pattern], capture_output=True)
            stopped.append(pattern)
    if cfg.get("docker") and subprocess.run(["pgrep", "-f", "com.docker.backend"], capture_output=True).returncode == 0:
        job.message = "Tắt Docker Desktop"
        docker = shutil.which("docker") or os.path.expanduser("~/.docker/bin/docker")
        ok = False
        if os.path.exists(docker):
            try:
                ok = subprocess.run([docker, "desktop", "stop"], capture_output=True, timeout=120).returncode == 0
            except subprocess.TimeoutExpired:
                ok = False
        if not ok:
            subprocess.run(["osascript", "-e", 'quit app "Docker"'], capture_output=True, timeout=60)
        stopped.append("docker")
    if stopped:
        time.sleep(2)  # chờ hệ điều hành thu hồi bộ nhớ
    return stopped


def run_job(job, cmd):
    STATS.listeners.append(job.on_stats)
    try:
        if job.free and job.test != "setup":
            job.freed = free_resources(job)
            if job.status == "cancelled":
                return
        os.makedirs(job.outdir, exist_ok=True)
        job.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    cwd=ROOT, start_new_session=True,
                                    env=dict(os.environ, PYTHONUNBUFFERED="1",
                                             BENCH_OUT=job.outdir, BENCH_PROMPT=job.prompt,
                                             PATH=os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", "")))
        for raw in job.proc.stdout:
            line = raw.rstrip("\n")
            if line.startswith("PROGRESS "):
                parts = line.split(" ", 2)
                try:
                    job.progress = float(parts[1])
                except ValueError:
                    pass
                job.message = parts[2] if len(parts) > 2 else ""
            elif line.startswith("STREAM "):
                try:
                    job.stream += json.loads(line[7:])
                except ValueError:
                    pass
            elif line.startswith("RESULT "):
                job.result = json.loads(line[7:])
                job.result["base"] = f"/out/{job.id}/"
            elif line.startswith("ERROR "):
                job.error = line[6:]
            elif line.strip():
                job.log.append(line[-300:])
                if job.test == "setup":
                    job.message = line[-160:]
        rc = job.proc.wait()
        if job.status == "cancelled":
            pass
        elif job.result is not None or (job.test == "setup" and rc == 0):
            job.status, job.progress = "done", 1.0
        else:
            job.status = "error"
            job.error = job.error or (job.log[-1] if job.log else f"Thoát với mã {rc}")
    except Exception as e:
        job.status, job.error = "error", str(e)
    finally:
        job.ended = time.time()
        if job.on_stats in STATS.listeners:
            STATS.listeners.remove(job.on_stats)
    if job.test == "setup" and job.status == "done":
        refresh_ai_status()
    if job.status == "done" and job.result:
        d = job.to_dict()
        save_history({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "test": job.test, "opts": job.opts, "label": job.label,
            "prompt": job.prompt,
            "result": job.result, "peak": d["peak"], "avg": d["avg"],
        })


def start_job(test, opts, label, prompt="", free=True):
    global JOB
    with JOB_LOCK:
        if JOB and JOB.status == "running":
            return None, "Đang có bài test khác chạy"
        if test in ("transcribe", "llm", "tts", "image", "video") and not (AI_STATUS or refresh_ai_status())[test]:
            return None, "Chưa cài môi trường AI cho bài này. Bấm nút CÀI ĐẶT AI trước."
        job = Job(test, opts, label, prompt, free)
        cmd = command_for(test, opts, job.outdir)
        if not cmd:
            return None, f"Không có bài test {test}"
        JOB = job
        threading.Thread(target=run_job, args=(JOB, cmd), daemon=True).start()
        return JOB, None


def cancel_job():
    if JOB and JOB.status == "running":
        JOB.status = "cancelled"  # nếu đang ở bước tắt mô hình nền, run_job dừng trước khi chạy test
        if JOB.proc:
            try:
                os.killpg(JOB.proc.pid, signal.SIGTERM)
            except OSError:
                pass


# ---------- HTTP ----------

class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404)
            return
        # Hỗ trợ Range để Safari phát được video
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            start_s, _, end_s = rng[6:].partition("-")
            start = int(start_s or 0)
            end = min(int(end_s) if end_s else len(body) - 1, len(body) - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
            body = body[start:end + 1]
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self.send_file(os.path.join(WEB, "index.html"), "text/html; charset=utf-8")
        if path == "/api/info":
            return self.send_json(dict(INFO, ai=AI_STATUS or refresh_ai_status()))
        if path == "/api/stats":
            return self.send_json(STATS.get())
        if path == "/api/job":
            return self.send_json(JOB.to_dict() if JOB else None)
        if path == "/api/history":
            return self.send_json(load_history())
        if path.startswith("/out/"):
            parts = path.split("/")
            if len(parts) == 4 and parts[2].isdigit() and parts[3] and ".." not in parts[3]:
                name = parts[3]
                ctype = {"mp4": "video/mp4", "png": "image/png", "wav": "audio/wav"}.get(name.rsplit(".", 1)[-1], "application/octet-stream")
                return self.send_file(os.path.join(OUTPUTS, parts[2], name), ctype)
            return self.send_error(404)
        if path.startswith("/preview/"):
            name = os.path.basename(path)
            ctype = "video/mp4" if name.endswith(".mp4") else "image/png"
            return self.send_file(os.path.join(BUILD, name), ctype)
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            body = {}
        if path == "/api/run":
            job, err = start_job(body.get("test", ""), body.get("opts", {}), body.get("label", ""),
                                 str(body.get("prompt", ""))[:2000], bool(body.get("free", True)))
            if err:
                return self.send_json({"error": err}, 409)
            return self.send_json(job.to_dict())
        if path == "/api/cancel":
            cancel_job()
            return self.send_json({"ok": True})
        if path == "/api/history/clear":
            try:
                os.remove(RESULTS)
            except OSError:
                pass
            return self.send_json({"ok": True})
        self.send_error(404)


def main():
    if platform.system() != "Darwin":
        sys.exit("Ứng dụng này chỉ chạy trên macOS (Apple Silicon).")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        sys.exit(f"Cổng {PORT} đang được dùng — có thể ứng dụng đã chạy rồi: mở http://localhost:{PORT}")
    build_native()
    INFO.update(machine_info())
    threading.Thread(target=refresh_ai_status, daemon=True).start()
    STATS.start()
    print(f"{INFO['chip']} — mở http://localhost:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        cancel_job()


if __name__ == "__main__":
    main()
