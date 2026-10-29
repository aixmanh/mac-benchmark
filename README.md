# Test máy M5 Ultra

Ứng dụng web chạy cục bộ để đo hiệu năng chiếc Mac này (giao diện theo kiểu video so sánh, nhưng chỉ test một máy).

## Chạy

```bash
./run.sh            # biên dịch (lần đầu), chạy server và mở http://localhost:9800
```

Hoặc `python3 server.py` rồi tự mở trình duyệt. Bấm Ctrl+C trong Terminal để tắt.

Bài test AI (LLM, Transcribe, TTS, Image, Video) cần cài môi trường một lần: bấm **CÀI ĐẶT AI** trong ứng dụng hoặc chạy `./setup_ai.sh`.

## Các bài test

| Tab | Đo gì | Cách đo |
|---|---|---|
| LLM | AI ngôn ngữ | Llama 3.2 3B, Qwen3 8B, Qwen3.8 27B (8-bit) trên MLX, token/giây |
| Transcribe | AI giọng nói | Whisper (tiny → large-v3) trên MLX với 1/5/13.8 phút giọng đọc tạo bằng `say` |
| TTS | AI giọng nói | VieNeu-TTS v3 Turbo (tiếng Việt, ONNX trên CPU), Kokoro 82M / Qwen3-TTS 1.7B (tiếng Anh, mlx-audio); đo × thời gian thực |
| Image | AI tạo ảnh | FLUX.2 klein 4B (mflux), 4 ảnh 512–2048 px, BF16/8-bit/4-bit, giây/ảnh (tải ~16 GB lần đầu) |
| Video | AI tạo video | LTX-2.5 distilled, DiT int8 (runtime `xocialize/ltx-2-mlx`), 2–5 s video có âm thanh (tải ~45 GB lần đầu) |
| CPU | Tốc độ đơn nhân / đa nhân | Tính tập Mandelbrot (Double), Giter/s |
| GPU | Sức tính GPU | Nhân ma trận 8192×8192 FP32/FP16 bằng Metal Performance Shaders, TFLOPS |
| Draw | Đồ hoạ | Render 150k hạt phát sáng bằng Metal trong 6 s, FPS ở 1080p/4K/8K |
| HEVC | Media engine | Mã hoá HEVC 4K60/8K60 (hoặc 4 luồng song song) bằng VideoToolbox |
| ProRes | Bộ tăng tốc ProRes | Mã hoá ProRes 422 HQ / 4444, 4K/8K |
| Memory | Băng thông RAM | Sao chép 4 GB từ CPU (memcpy đa luồng) hoặc GPU (compute shader) |
| SSD | Ổ cứng | Ghi rồi đọc tuần tự 8/16/32 GB, bỏ qua bộ nhớ đệm, xoá file sau khi xong |

Thông số trực tiếp (không cần sudo): % CPU/GPU, RAM, công suất GPU/Neural Engine (IOReport), công suất SoC và toàn máy, nhiệt độ CPU/GPU (SMC).
Trên M5 + macOS 27, công suất riêng của CPU không đọc được nếu không có quyền root, nên ứng dụng hiển thị công suất SoC thay thế.

Kết quả được lưu ở `results/history.json`.

Mỗi lần bấm START (khi ô **Tắt mô hình nền & Docker khi START** đang tích), ứng dụng tắt trước những thứ khai báo trong `config.json` (chép từ `config.example.json`: `cp config.example.json config.json`; không có file thì không tắt gì) → `free_before_test`: các dịch vụ launchd (`launchd`, bị bootout nên chỉ bật lại khi khởi động/đăng nhập lại máy), tiến trình khớp `kill_patterns`, và Docker Desktop (`docker`). Thêm/bớt tên trong file đó cho phù hợp.

VieNeu-TTS dùng repo và `.venv` có sẵn ở `~/Du_an/TTS/VieNeu-TTS`; đặt biến `VIENEU_DIR` nếu repo nằm chỗ khác.

## Cấu trúc

- `server.py` — server web (Python có sẵn trên macOS, không cần thư viện ngoài)
- `config.example.json` — mẫu cấu hình tắt mô hình nền/Docker trước khi test
- `web/index.html` — giao diện
- `native/bench.swift` — các bài test CPU/GPU/Draw/HEVC/ProRes/Memory/SSD
- `native/stats.m` — đọc thông số trực tiếp
- `ai/` — bài test AI: Whisper và LLM (`.venv`), TTS (`.venv-tts`), FLUX.2 (`.venv-image`), LTX-2.5 (`third_party/ltx-2-mlx/.venv`)
- `third_party/ltx-2-mlx` — runtime LTX-2.5 cho MLX; `setup_ai.sh` vá đường dẫn README trong `packages/*/pyproject.toml` để build được với hatchling mới
- `build/bin/ffmpeg` — ffmpeg tĩnh (từ `imageio-ffmpeg`) dùng để ghi mp4
- `tools/obs_webcam.py` — giữ webcam Osmo Pocket trong OBS (cảnh "Test M5 Ultra") thành khung dọc 9:16 thẳng đứng quanh mặt; tự xoay khi đổi hướng Pocket. Chạy: `.venv-obs/bin/python tools/obs_webcam.py` (cần bật WebSocket trong OBS); `tools/face.swift` tìm mặt bằng Vision
