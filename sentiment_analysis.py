import cv2
import urllib.request
import os
import sys

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# OpenCV DNN face detector (SSD ResNet, ~10MB, much better than Haar cascade)
FACE_PROTO  = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_WEIGHTS = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
FACE_PROTO_URL   = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_WEIGHTS_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

MOOD_COLORS = {
    "Happiness": (0, 220, 80),
    "Neutral":   (180, 180, 180),
    "Sadness":   (200, 80, 40),
    "Anger":     (0, 0, 230),
    "Fear":      (160, 0, 230),
    "Surprise":  (0, 210, 255),
    "Disgust":   (0, 160, 0),
    "Contempt":  (100, 100, 200),
}


def download(url, dest, label):
    if os.path.exists(dest):
        return
    print(f"[INFO] Downloading {label}...")
    def _prog(count, block, total):
        print(f"\r  {min(int(count * block * 100 / total), 100)}%", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_prog)
    print()


def setup_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    download(FACE_PROTO_URL,   FACE_PROTO,   "face detector config (~30 KB)")
    download(FACE_WEIGHTS_URL, FACE_WEIGHTS, "face detector weights (~10 MB)")


def detect_faces_dnn(net, frame, conf_thresh=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                  (104.0, 177.0, 123.0), swapRB=False)
    net.setInput(blob)
    dets = net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        x1 = max(0, int(dets[0, 0, i, 3] * w))
        y1 = max(0, int(dets[0, 0, i, 4] * h))
        x2 = min(w, int(dets[0, 0, i, 5] * w))
        y2 = min(h, int(dets[0, 0, i, 6] * h))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


def draw_rounded_rect(img, pt1, pt2, color, thickness=2, r=12):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
    cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
    cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
    cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
    cv2.ellipse(img, (x1+r, y1+r), (r, r), 180,  0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y1+r), (r, r), 270,  0, 90, color, thickness)
    cv2.ellipse(img, (x1+r, y2-r), (r, r),  90,  0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y2-r), (r, r),   0,  0, 90, color, thickness)


def draw_label(frame, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.65, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 6
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - th - pad*2), (x + tw + pad*2, y), color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x + pad, y - pad), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_bars(frame, emotions_dict, x, y, width=120):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bar_h, gap = 9, 14
    sorted_items = sorted(emotions_dict.items(), key=lambda e: e[1], reverse=True)[:5]
    for i, (label, score) in enumerate(sorted_items):
        color = MOOD_COLORS.get(label, (160, 160, 160))
        by = y + i * (bar_h + gap)
        filled = int(score * width)
        cv2.rectangle(frame, (x, by), (x + width, by + bar_h), (50, 50, 50), -1)
        if filled > 0:
            cv2.rectangle(frame, (x, by), (x + filled, by + bar_h), color, -1)
        cv2.putText(frame, f"{label[:6]} {int(score*100)}%",
                    (x + width + 4, by + bar_h),
                    font, 0.36, (210, 210, 210), 1, cv2.LINE_AA)


def run(source=0):
    setup_models()

    # Load face detector
    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_WEIGHTS)

    # Load emotion recognizer (downloads ~16 MB model on first run)
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    print("[INFO] Loading emotion model (downloads ~16 MB on first run)...")
    emotion_model = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf")
    print("[INFO] Ready — press Q to quit")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return

    cv2.namedWindow("Sentiment Analysis", cv2.WINDOW_NORMAL)

    frame_n = 0
    cache = []  # [(x,y,w,h, emotion_str, emotions_dict)]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        h, w = frame.shape[:2]

        if frame_n % 4 == 0:
            boxes = detect_faces_dnn(face_net, frame, conf_thresh=0.5)
            cache = []
            for (fx, fy, fw, fh) in boxes:
                # Add padding around the face for better emotion context
                pad = int(min(fw, fh) * 0.1)
                x1 = max(0, fx - pad)
                y1 = max(0, fy - pad)
                x2 = min(w, fx + fw + pad)
                y2 = min(h, fy + fh + pad)
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                emotion_str, scores = emotion_model.predict_emotions(face_img, logits=False)
                labels = emotion_model.idx_to_class
                emotions_dict = {labels[i]: float(scores[i]) for i in range(len(scores))}
                cache.append((fx, fy, fw, fh, emotion_str, emotions_dict))

        for (fx, fy, fw, fh, emotion_str, emotions_dict) in cache:
            color = MOOD_COLORS.get(emotion_str, (180, 180, 180))
            conf = int(emotions_dict.get(emotion_str, 0) * 100)

            draw_rounded_rect(frame, (fx, fy), (fx+fw, fy+fh), color, 2)
            cx, cy = fx + fw//2, fy + fh//2
            cv2.ellipse(frame, (cx, cy), (int(fw*0.6), int(fh*0.65)), 0, 0, 360, color, 2)

            draw_label(frame, f"{emotion_str}  {conf}%", max(fx, 0), max(fy - 8, 22), color)

            bar_x = fx + fw + 10
            if bar_x + 200 > w:
                bar_x = max(fx - 200, 0)
            draw_bars(frame, emotions_dict, bar_x, fy)

        cv2.putText(frame, "Real-Time Sentiment Analysis",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Q = quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1, cv2.LINE_AA)

        cv2.imshow("Sentiment Analysis", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    run(src)
