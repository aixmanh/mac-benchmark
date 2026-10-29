#!/bin/bash
# Khởi động ứng dụng test máy và mở trình duyệt.
cd "$(dirname "$0")"
PORT="${PORT:-9800}"
(sleep 2 && open "http://localhost:$PORT") &
exec python3 server.py
