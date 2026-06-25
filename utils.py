import cv2
import numpy as np

GOLDEN_RATIO = 0.618033988749895
_COLOR_CACHE = {}

def _get_color(tid):
    if tid not in _COLOR_CACHE:
        h = (tid * GOLDEN_RATIO) % 1.0
        hsv = np.array([[[h * 180, 204, 229]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        _COLOR_CACHE[tid] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return _COLOR_CACHE[tid]

class FPS:
    def __init__(self):
        self.prev = cv2.getTickCount()
        self.fps = 0.0
        self._smooth = 0.0

    def update(self):
        now = cv2.getTickCount()
        dt = (now - self.prev) / cv2.getTickFrequency()
        if dt > 0:
            self._smooth = 0.9 * self._smooth + 0.1 * (1.0 / dt) if self._smooth > 0 else 1.0 / dt
            self.fps = self._smooth
        self.prev = now
        return self.fps

    def get(self):
        return self.fps

def draw_boxes(frame, boxes, class_ids, track_ids, class_names, confidences=None, fps=None, det_count=None):
    if boxes is None or len(boxes) == 0:
        if fps is not None:
            _draw_fps(frame, fps, 0)
        return frame

    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(1.0, 0.6 * w / 800.0))
    thickness = max(1, int(round(2 * w / 800.0)))

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        cid = class_ids[i]
        conf = confidences[i] if confidences is not None and i < len(confidences) else None
        tid = track_ids[i] if track_ids is not None else i + 1
        color = _get_color(tid)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        label = class_names[cid] if cid < len(class_names) else str(cid)
        if track_ids is not None:
            label += f" ID:{tid}"
        if conf is not None:
            label += f" {conf:.0%}"

        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        pad = int(4 * font_scale + 2)
        lx, ly = x1, y1 - pad - th if y1 - pad - th > 0 else y2 + pad + th

        cv2.rectangle(frame, (lx - pad, ly - pad), (lx + tw + pad, ly + th + pad), (0, 0, 0), -1)
        cv2.putText(frame, label, (lx, ly + th), font, font_scale, color, thickness, cv2.LINE_AA)

    if fps is not None:
        dets = det_count if det_count is not None else len(boxes)
        _draw_fps(frame, fps, dets)

    return frame

def _draw_fps(frame, fps, dets):
    txt = f"FPS:{fps:.1f} DET:{dets}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(txt, font, 0.7, 2)
    x, y = 10, 10
    bw, bh = tw + 16, th + 14
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 0), -1)
    cv2.putText(frame, txt, (x + 8, y + bh - 6), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
