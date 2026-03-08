# 🪖 Helmet Detection using YOLOv8

Real-time helmet detection system using YOLOv8 and Python. Detects whether a person is wearing a helmet or not from a live camera feed, images, or video files.

## Results

| Metric | Helmet | No-Helmet | Overall |
|--------|--------|-----------|---------|
| Precision | 94.3% | 89.2% | 91.8% |
| Recall | 88.8% | 88.8% | 88.8% |
| mAP50 | 95.6% | 93.1% | **94.3%** |

Inference speed: **2.6ms per image** (~380 FPS)

## Tech Stack

- **Python** — Main language
- **YOLOv8** (Ultralytics) — Object detection model
- **PyTorch** — Deep learning backend
- **OpenCV** — Image/video processing
- **Label Studio** — Custom dataset annotation
- **CUDA** — GPU acceleration

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/helmet-detection-yolov8.git
cd helmet-detection-yolov8

# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Webcam (live detection)
```bash
python detect.py --source 0
```

### Image
```bash
python detect.py --source path/to/image.jpg --save
```

### Video
```bash
python detect.py --source path/to/video.mp4 --save
```

## Dataset

Trained on ~5,800 images merged from:
- Roboflow Hard Hat Workers dataset (~5,000 images)
- Kaggle Bike Helmet dataset (764 images)
- Custom images annotated using Label Studio

Classes: `helmet`, `no-helmet`

## Team

- Gowri Manoj
- Jebin Boby Mathew

Built for competition submission.
