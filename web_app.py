import cv2
import threading
import os
import time
import uuid
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import config
from utils import draw_boxes, FPS
import sys
import traceback
import platform

sys.excepthook = lambda exc_type, exc_value, exc_tb: print(
    "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
)


UPLOAD_FOLDER = "uploads"
DOWNLOAD_FOLDER = "downloads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs("data", exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# SHARED STATE
# ============================================================
model = None
streaming = False
conf_threshold = config.CONFIDENCE_THRESHOLD
current_model_name = config.MODEL_NAME
source = 0
source_version = 0
error_message = None

_worker = None
_worker_running = False
_frame_buffer = None
_frame_count = 0
_buffer_lock = threading.Lock()
_stats_lock = threading.Lock()
_last_stats = {"fps": 0.0, "detections": 0}


# ============================================================
# MODEL
# ============================================================
def load_model(model_name):
    global model
    try:
        model = YOLO(model_name)
        model.to(config.DEVICE)
        model.fuse()
        print(f"Model '{model_name}' loaded on {config.DEVICE}")
        return {"status": "success", "message": f"Model '{model_name}' loaded"}
    except Exception as e:
        print(f"Error loading model: {e}")
        return {"status": "error", "message": str(e)}


load_model(current_model_name)

# Warm up model: run dummy inference to compile CUDA kernels
try:
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False, device=config.DEVICE)
    print("Model warmed up")
except Exception as e:
    print(f"Model warmup skipped: {e}")




# ============================================================
# SOURCE NORMALISATION
# ============================================================
def normalise_source(src):
    if isinstance(src, int):
        return src
    src = str(src).strip()
    if (src.startswith('"') and src.endswith('"')) or (
        src.startswith("'") and src.endswith("'")
    ):
        src = src[1:-1]
    src = src.replace("\\", "/")
    return src


# ============================================================
# SINGLE WORKER THREAD
# ============================================================
def _worker_loop():
    global _frame_buffer, _frame_count, _worker_running, _last_stats, _stats_lock
    global error_message, source_version

    cap = None
    fps_ctr = FPS()
    last_ver = -1
    w, h = config.FRAME_WIDTH, config.FRAME_HEIGHT
    synth_ctr = 0
    output_writer = None

    while _worker_running:
        # --- Source management ---
        if source_version != last_ver:
            if cap is not None:
                cap.release()
                cap = None
            src = normalise_source(source)
            if isinstance(src, int) and platform.system() == "Windows":
                cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(src)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, config.WEBCAM_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.FRAME_WIDTH
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.FRAME_HEIGHT
                error_message = None
                print(f"Opened source: {src} ({w}x{h})")
            else:
                error_message = f"Cannot open: {src}"
                if cap is not None:
                    cap.release()
                    cap = None
            last_ver = source_version
            output_writer = None

        # --- Capture ---
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                src = normalise_source(source)
                cap.release()
                if isinstance(src, int) and platform.system() == "Windows":
                    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(src)
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                continue
        else:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            synth_ctr += 1
            cx = w // 2 + int(150 * np.sin(synth_ctr * 0.02))
            cy = h // 2 + int(100 * np.cos(synth_ctr * 0.03))
            cv2.circle(frame, (cx, cy), 40, (0, 229, 255), -1)
            cv2.putText(
                frame, "Waiting for source", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )
            if error_message:
                cv2.putText(
                    frame, error_message, (50, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )
            fps = fps_ctr.update()
            with _stats_lock:
                _last_stats["fps"] = fps
                _last_stats["detections"] = 0
            _write_buffer(frame)
            time.sleep(0.03)
            continue

        # --- Detection ---
        fps = fps_ctr.update()
        try:
            results = model.track(
                frame,
                persist=True,
                conf=conf_threshold,
                iou=config.NMS_IOU_THRESHOLD,
                max_det=config.MAX_DETECTIONS,
                device=config.DEVICE,
                tracker=config.TRACKER_CONFIG,
                verbose=False,
            )
        except Exception as e:
            print(f"Detection error: {e}")
            continue

        # --- Render (boxes ALWAYS drawn, even without track IDs) ---
        det_count = 0
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            det_count = len(boxes)
            track_ids = (
                results[0].boxes.id.cpu().numpy().astype(int)
                if results[0].boxes.id is not None
                else None
            )
            frame = draw_boxes(
                frame, boxes, class_ids, track_ids,
                model.names, confs, fps, det_count
            )

        with _stats_lock:
            _last_stats["fps"] = fps
            _last_stats["detections"] = det_count

        # --- Output file ---
        if output_writer is None and config.OUTPUT_VIDEO_PATH and w > 0 and h > 0:
            output_writer = cv2.VideoWriter(
                config.OUTPUT_VIDEO_PATH,
                cv2.VideoWriter_fourcc(*"XVID"),
                20.0,
                (w, h),
            )
        if output_writer is not None:
            output_writer.write(frame)

        # --- Buffer ---
        _write_buffer(frame)

    if cap is not None:
        cap.release()
    if output_writer is not None:
        output_writer.release()
    print("Worker exited")


def _write_buffer(frame):
    global _frame_buffer, _frame_count
    ret, jpeg = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
    )
    if ret:
        with _buffer_lock:
            _frame_buffer = jpeg.tobytes()
            _frame_count += 1


def _start_worker():
    global _worker, _worker_running
    if _worker is not None and _worker.is_alive():
        return
    _worker_running = True
    _worker = threading.Thread(target=_worker_loop, daemon=True)
    _worker.start()


def _stop_worker():
    global _worker, _worker_running
    _worker_running = False
    if _worker is not None:
        _worker.join(timeout=3)
        _worker = None
    print("Worker stopped")


# ============================================================
# HTTP STREAMING GENERATOR
# ============================================================
def generate_frames():
    last_sent = -1
    deadline = time.time() + 5.0
    with _buffer_lock:
        ready = _frame_buffer is not None
    while not ready and time.time() < deadline:
        time.sleep(0.05)
        with _buffer_lock:
            ready = _frame_buffer is not None

    while streaming:
        with _buffer_lock:
            buf = _frame_buffer
            cnt = _frame_count
        if buf is not None and cnt != last_sent:
            last_sent = cnt
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(buf)).encode() + b"\r\n\r\n"
                + buf + b"\r\n"
            )
        else:
            time.sleep(0.01)


# ============================================================
# FLASK ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/start", methods=["POST"])
def start_stream():
    global streaming
    if not streaming:
        streaming = True
        _start_worker()
        return jsonify({"status": "success", "message": "Stream started"})
    return jsonify({"status": "info", "message": "Already streaming"})


@app.route("/stop", methods=["POST"])
def stop_stream():
    global streaming
    if streaming:
        streaming = False
        _stop_worker()
        return jsonify({"status": "success", "message": "Stream stopped"})
    return jsonify({"status": "info", "message": "Already stopped"})


@app.route("/set_source", methods=["POST"])
def set_source():
    global source, source_version
    data = request.get_json()
    if not data or "source" not in data:
        return jsonify({"status": "error", "message": "Missing 'source'"}), 400
    new_source = data["source"]
    new_source = normalise_source(new_source)
    if isinstance(new_source, str) and new_source.isdigit():
        new_source = int(new_source)
    source = new_source
    source_version += 1
    print(f"Source set to: {source} (v{source_version})")
    return jsonify({"status": "success", "message": f"Source set to {source}"})


@app.route("/set_model", methods=["POST"])
def set_model():
    global current_model_name
    data = request.get_json()
    if not data or "model" not in data:
        return jsonify({"status": "error", "message": "Missing 'model'"}), 400
    model_name = data["model"]
    result = load_model(model_name)
    if result["status"] == "success":
        current_model_name = model_name
    return jsonify(result)


@app.route("/set_conf", methods=["POST"])
def set_conf():
    global conf_threshold
    data = request.get_json()
    if not data or "conf" not in data:
        return jsonify({"status": "error", "message": "Missing 'conf'"}), 400
    new_conf = float(data["conf"])
    if 0.1 <= new_conf <= 0.9:
        conf_threshold = new_conf
        return jsonify({"status": "success", "message": f"Confidence set to {new_conf}"})
    return jsonify({"status": "error", "message": "Confidence must be 0.1-0.9"}), 400


@app.route("/status")
def status():
    return jsonify({
        "streaming": streaming,
        "model": current_model_name,
        "confidence": conf_threshold,
        "source": str(source),
    })


@app.route("/upload_video", methods=["POST"])
def upload_video():
    global source, source_version
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No file"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid file type"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    abs_path = os.path.abspath(filepath).replace("\\", "/")
    if abs_path != source:
        source = abs_path
        source_version += 1
        print(f"Upload set source to: {source} (v{source_version})")
    return jsonify({"status": "success", "message": f"Uploaded {filename}", "path": abs_path})


if __name__ == "__main__":
    print("VisionTrack AI on http://127.0.0.1:5000")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
