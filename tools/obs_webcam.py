"""Tự giữ webcam Osmo Pocket thành khung dọc 9:16, thẳng đứng, ở góc dưới phải cảnh OBS.

Pocket ở chế độ webcam không tự xoay hình theo hướng máy: để dọc thì nó gửi một dải hình nằm nghiêng giữa hai
viền đen. Công cụ này cứ vài giây lấy một khung hình qua obs-websocket, tìm mặt bằng Vision (build/face) để biết
hình đang nghiêng chiều nào, rồi đặt lại xoay + cắt + vị trí cho nguồn webcam trong cảnh.

Cách dùng:
  .venv-obs/bin/python tools/obs_webcam.py           # chạy liên tục (mặc định, kiểm tra mỗi 3 giây)
  .venv-obs/bin/python tools/obs_webcam.py --once    # chỉnh một lần rồi thoát
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import obsws_python as obs
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE = os.path.join(ROOT, "build", "face")
WS_CONFIG = os.path.expanduser("~/Library/Application Support/obs-studio/plugin_config/obs-websocket/config.json")

CANVAS_W, CANVAS_H = 1920, 1080
BOX_H, MARGIN = 420, 24          # khung dọc 236×420, cách mép 24 px
BOX_W = BOX_H * 9 / 16
ROTATION = {"up": 0.0, "right": 90.0, "left": 270.0}
RECENTER = 0.15                  # chỉ dời khung khi mặt lệch hơn 15% bề rộng khung (tránh rung)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def ensure_face_tool():
    src = os.path.join(ROOT, "tools", "face.swift")
    if not os.path.exists(FACE) or os.path.getmtime(FACE) < os.path.getmtime(src):
        subprocess.run(["swiftc", "-O", "-suppress-warnings", src, "-o", FACE], check=True)


def connect():
    cfg = json.load(open(WS_CONFIG))
    return obs.ReqClient(host="127.0.0.1", port=cfg.get("server_port", 4455),
                         password=cfg.get("server_password", ""), timeout=15)


def find_camera(cl, scene):
    for it in cl.get_scene_item_list(scene).scene_items:
        if it.get("inputKind") in ("av_capture_input_v2", "macos-avcapture", "macos-avcapture-fast"):
            return it["sourceName"], it["sceneItemId"]
    raise SystemExit(f"Không có nguồn webcam trong cảnh {scene}")


def grab(cl, source, w, h):
    data = cl.get_source_screenshot(source, "png", w, h, -1).image_data
    return Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1]))).convert("RGB")


def detect_face(img):
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        img.save(f.name)
        out = subprocess.run([FACE, f.name], capture_output=True, text=True).stdout
    try:
        return json.loads(out)
    except ValueError:
        return {"orientation": None}


def plan(frame, face):
    """Tính xoay/cắt (đơn vị: tỉ lệ 0..1 của khung gốc) để ra khung dọc 9:16 quanh mặt."""
    arr = np.asarray(frame).astype(int)
    cols = np.where(arr.max(axis=(0, 2)) > 24)[0]          # bỏ viền đen hai bên
    fw, fh = frame.size
    x0, x1 = (cols.min(), cols.max() + 1) if len(cols) else (0, fw)
    cw, ch = (x1 - x0) / fw, 1.0                            # vùng có hình (tỉ lệ)
    orient = face["orientation"]
    # mặt tìm trên dải hình đã cắt viền: đổi toạ độ ra dải đó
    fx = face["x"] + face["w"] / 2
    rot = ROTATION[orient]
    crop = {"cropLeft": x0 / fw, "cropRight": 1 - x1 / fw, "cropTop": 0.0, "cropBottom": 0.0}
    aspect = 9 / 16
    if rot == 0:
        # hình thẳng: rộng cw·fw, cao fh -> cắt ngang
        up_w, up_h = cw * fw, ch * fh
        keep = min(up_w, up_h * aspect)
        left = min(max(fx * up_w - keep / 2, 0), up_w - keep)
        crop["cropLeft"] += left / fw
        crop["cropRight"] += (up_w - left - keep) / fw
        out_h = up_h
        center = (left + keep / 2) / up_w
    else:
        # hình nghiêng: sau khi xoay, rộng = chiều cao gốc, cao = bề rộng dải -> cắt theo chiều cao gốc
        up_w, up_h = ch * fh, cw * fw
        keep = min(up_w, up_h * aspect)
        u0 = min(max(fx * up_w - keep / 2, 0), up_w - keep)
        if rot == 90:   # xoay theo chiều kim đồng hồ: cột u của hình thẳng = hàng (cao - u) của hình gốc
            crop["cropTop"], crop["cropBottom"] = (up_w - u0 - keep) / fh, u0 / fh
        else:           # 270°: cột u = hàng u
            crop["cropTop"], crop["cropBottom"] = u0 / fh, (up_w - u0 - keep) / fh
        out_h = up_h
        center = (u0 + keep / 2) / up_w
    return orient, rot, crop, out_h / fh, center, fx


def apply(cl, scene, item_id, src_w, src_h, rot, crop, out_h_ratio):
    scale = BOX_H / (out_h_ratio * src_h)
    cl.set_scene_item_transform(scene, item_id, {
        "boundsType": "OBS_BOUNDS_NONE", "alignment": 0, "rotation": rot,
        "cropLeft": round(crop["cropLeft"] * src_w), "cropRight": round(crop["cropRight"] * src_w),
        "cropTop": round(crop["cropTop"] * src_h), "cropBottom": round(crop["cropBottom"] * src_h),
        "scaleX": scale, "scaleY": scale,
        "positionX": CANVAS_W - MARGIN - BOX_W / 2, "positionY": CANVAS_H - MARGIN - BOX_H / 2,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="chỉnh một lần rồi thoát")
    ap.add_argument("--scene", default="Test M5 Ultra")
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()

    ensure_face_tool()
    cl = connect()
    cam, item_id = find_camera(cl, args.scene)
    log(f"Theo dõi '{cam}' trong cảnh '{args.scene}'")
    state = None  # (hướng, vị trí mặt lúc đặt khung)
    while True:
        try:
            t = cl.get_scene_item_transform(args.scene, item_id).scene_item_transform
            src_w, src_h = t["sourceWidth"], t["sourceHeight"]
            frame = grab(cl, cam, 960, 540)
            arr = np.asarray(frame).astype(int)
            cols = np.where(arr.max(axis=(0, 2)) > 24)[0]
            strip = frame.crop((cols.min(), 0, cols.max() + 1, frame.size[1])) if len(cols) else frame
            face = detect_face(strip)
            if face.get("orientation"):
                orient, rot, crop, out_h, center, fx = plan(frame, face)
                moved = state is None or state[0] != orient or abs(fx - state[1]) > RECENTER
                if moved:
                    apply(cl, args.scene, item_id, src_w, src_h, rot, crop, out_h)
                    log(f"Hình {'đã thẳng' if rot == 0 else 'đang nghiêng'} -> xoay {rot:.0f}°, khung dọc quanh mặt (x={fx:.2f})")
                    state = (orient, fx)
            elif state is None:
                log("Chưa thấy mặt, chờ…")
        except Exception as e:  # OBS tắt / mất kết nối: thử lại
            log(f"Lỗi: {e}")
            if args.once:
                sys.exit(1)
            time.sleep(5)
            try:
                cl = connect()
            except Exception:
                pass
        if args.once and state:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
