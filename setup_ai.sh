#!/bin/bash
# Cài các môi trường Python riêng cho bài test AI:
#   .venv                     — Whisper + LLM (mlx-whisper, mlx-lm)
#   .venv-image               — tạo ảnh FLUX.2 klein (mflux)
#   .venv-tts                 — giọng nói Kokoro / Qwen3-TTS (mlx-audio)
#   third_party/ltx-2-mlx     — tạo video LTX-2.5 (xocialize/ltx-2-mlx, nhánh ltx-2.5) + ffmpeg tĩnh
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "Cài uv (trình quản lý Python)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "== Whisper + LLM"
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx-whisper mlx-lm

echo "== FLUX.2 klein (mflux)"
[ -x .venv-image/bin/python ] || uv venv --python 3.12 .venv-image
uv pip install --python .venv-image/bin/python mflux

echo "== TTS (mlx-audio: Kokoro, Qwen3-TTS)"
[ -x .venv-tts/bin/python ] || uv venv --python 3.12 .venv-tts
uv pip install --python .venv-tts/bin/python mlx-audio "misaki[en]"
# Kokoro dùng spaCy để tách từ tiếng Anh: cài model en_core_web_sm đúng phiên bản spaCy
SPACY_V=$(.venv-tts/bin/python -c "import spacy; print('.'.join(spacy.__version__.split('.')[:2]))")
uv pip install --python .venv-tts/bin/python \
    "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-$SPACY_V.0/en_core_web_sm-$SPACY_V.0-py3-none-any.whl"

echo "== LTX-2.5 (ltx-2-mlx)"
mkdir -p third_party
if [ ! -d third_party/ltx-2-mlx ]; then
    git clone --depth 1 -b ltx-2.5 https://github.com/xocialize/ltx-2-mlx third_party/ltx-2-mlx
fi
(
    cd third_party/ltx-2-mlx
    # hatchling mới không cho README nằm ngoài thư mục gói: chép README vào từng gói con
    for d in packages/*/; do
        if grep -q '"\.\./\.\./README.md"' "$d/pyproject.toml"; then
            cp README.md "$d/README.md"
            sed -i '' 's|"\.\./\.\./README.md"|"README.md"|' "$d/pyproject.toml"
        fi
    done
    uv sync --python 3.12
    uv pip install --python .venv/bin/python imageio-ffmpeg
    # Text encoder Gemma-4 (gemma4_unified) của LTX-2.5 cần mlx-lm bản git main
    uv pip install --python .venv/bin/python "mlx-lm @ git+https://github.com/ml-explore/mlx-lm"
)
mkdir -p build/bin
ln -sf "$(third_party/ltx-2-mlx/.venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" build/bin/ffmpeg

echo "Đã cài xong môi trường AI."
