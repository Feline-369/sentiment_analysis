# Real-Time Sentiment Analysis

A real-time facial emotion recognition app that uses your webcam (or a video file) to detect faces, circle them, and predict their mood live.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green) ![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange)

---

## Features

- **Live webcam or video file** input
- **Face detection** using OpenCV's DNN-based SSD ResNet detector (more accurate than Haar cascades)
- **Emotion recognition** using EfficientNet trained on AffectNet (450k real-world faces)
- **8 emotions detected:** Happiness, Sadness, Anger, Fear, Surprise, Disgust, Contempt, Neutral
- **Visual overlay:** ellipse + rounded bounding box around each face, color-coded by mood
- **Live confidence bar chart** showing scores for all 8 emotions
- **Lightweight:** ~90 MB total — no TensorFlow or PyTorch required

---

## Demo

Each detected face gets:
- A colored ellipse and bounding box (color changes per mood)
- A label showing the dominant emotion and confidence (e.g. `Happiness 87%`)
- A side panel with live bars for all 8 emotion scores

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sentiment_analysis.git
cd sentiment_analysis
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Webcam (default)

```bash
python3 sentiment_analysis.py
```

### Video file

```bash
python3 sentiment_analysis.py path/to/video.mp4
```

### Specific camera index

```bash
python3 sentiment_analysis.py 1
```

Press **Q** or **Esc** to quit.

---

## First Run

On the first run, the app automatically downloads two small model files into the `models/` folder:

| File | Size | Purpose |
|------|------|---------|
| `deploy.prototxt` | ~30 KB | Face detector config |
| `res10_300x300_ssd_iter_140000.caffemodel` | ~10 MB | Face detector weights |
| `enet_b0_8_best_vgaf.onnx` | ~16 MB | Emotion recognition model |

All subsequent runs start instantly with no download.

---

## Requirements

```
opencv-python>=4.5
hsemotion-onnx
numpy
```

Install with:

```bash
pip install -r requirements.txt
```

---

## How It Works

1. **Face Detection** — Each frame is passed through a ResNet-10 SSD model (via OpenCV DNN) to locate faces with bounding boxes and confidence scores.
2. **Preprocessing** — Each detected face region (with slight padding) is cropped from the frame.
3. **Emotion Inference** — The cropped face is fed into an EfficientNet-B0 ONNX model trained on [AffectNet](http://mohammadmahoor.com/affectnet/), which outputs probability scores for 8 emotions.
4. **Rendering** — Results are overlaid on the frame with color-coded ellipses, labels, and bar charts.

Inference runs every 4 frames for smooth real-time performance.

---

## Project Structure

```
sentiment_analysis/
├── sentiment_analysis.py   # Main app
├── requirements.txt        # Python dependencies
├── models/                 # Downloaded model files (auto-created)
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   └── enet_b0_8_best_vgaf.onnx
└── README.md
```

---

## Troubleshooting

**Camera not opening**
- Make sure no other app is using the webcam.
- Try a different camera index: `python3 sentiment_analysis.py 1`

**Slow performance**
- Inference runs every 4 frames by default. For a faster machine you can lower the `frame_n % 4` check in the code.

**Inaccurate emotions**
- Ensure good, even lighting on your face.
- Face the camera directly — the model works best on near-frontal faces.

---

## Credits

- Face detector: [OpenCV DNN face detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)
- Emotion model: [HSEmotionONNX](https://github.com/HSE-asavchenko/face-emotion-recognition) by Andrey Savchenko, trained on [AffectNet](http://mohammadmahoor.com/affectnet/)
