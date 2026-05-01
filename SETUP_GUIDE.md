# 🚀 S.H.A.D.Y - Setup & Run Guide (Both Options)

## Quick Start (Option B - Recommended, No Compilation)

```powershell
cd Detection_With_Yolo
pip install opencv-python numpy flask yt-dlp
python setup_project.py          # Downloads YOLO weights & configs
python test_setup.py             # Verify everything works
python Fall_Detection.py --source 0   # Run with webcam!
```

---

## Option A vs Option B - Comparison

| Feature | Option A (Darknet) | Option B (OpenCV DNN) |
|---------|-------------------|----------------------|
| **Setup Time** | 2-3 hours | 5 minutes |
| **Compilation** | Required (CMake + VS) | Not needed |
| **GPU Support** | CUDA (fast) | CUDA via OpenCV (if built with CUDA) |
| **CPU Support** | Slow | Works well |
| **Compatibility** | May-June 2020 darknet only | Any OpenCV 4.x |
| **Switching** | Change 1 line in config.py | Change 1 line in config.py |
| **Accuracy** | Same (same weights) | Same (same weights) |

### How to Switch Between Options

In `config.py`, change ONE line:

```python
# Option B (OpenCV DNN - no compilation needed)
BACKEND = "opencv"

# Option A (Compiled Darknet - faster with GPU)
# BACKEND = "darknet"
```

---

## OPTION B: OpenCV DNN (Easy - No Compilation)

### Step 1: Install Python Packages

```powershell
pip install opencv-python numpy flask yt-dlp
```

### Step 2: Download YOLO Files

```powershell
cd "e:\S.H.A.D.Y-main\S.H.A.D.Y-main\S.H.A.D.Y-main\Detection_With_Yolo"
python setup_project.py
```

This will download:
- `cfg/yolov4-tiny.cfg` (YOLO architecture)
- `yolov4-tiny.weights` (~23 MB, pre-trained model)
- `cfg/coco.data` (metadata)
- `data/coco.names` (80 class names)

### Step 3: Verify Setup

```powershell
python test_setup.py
```

### Step 4: Run Detection!

```powershell
# Webcam
python Object_Detection.py --source 0
python Fall_Detection.py --source 0
python Social_Distance.py --source 0
python Vehicle_Crash.py --source 0

# Local video file
python Fall_Detection.py --source "path/to/video.mp4"

# YouTube video
python Fall_Detection.py --source "youtube:https://www.youtube.com/watch?v=xxxxx"
```

### Step 5: (Optional) Get GPU Acceleration

If you want CUDA-accelerated OpenCV DNN:

```powershell
# Uninstall normal opencv
pip uninstall opencv-python opencv-python-headless

# Install CUDA-enabled OpenCV (pre-built)
pip install opencv-contrib-python
```

> Note: Pre-built pip OpenCV usually doesn't have CUDA. For full CUDA support,
> you need to compile OpenCV from source with CUDA flags. The CPU version still
> works fine for testing/development.

---

## OPTION A: Compiled Darknet (Original - Full GPU Speed)

### Prerequisites

| Software | Download Link |
|----------|-------------|
| Visual Studio 2019 Community | https://visualstudio.microsoft.com/vs/older-downloads/ |
| CMake | https://cmake.org/download/ |
| CUDA Toolkit 12.4 | Already installed on your system |
| cuDNN | Already installed on your system |

### Step 1: Install CMake

1. Download from https://cmake.org/download/
2. Run installer, select "Add CMake to system PATH"
3. Verify: `cmake --version`

### Step 2: Install Visual Studio 2019/2022

1. Download VS Community from Microsoft
2. During installation, select:
   - "Desktop development with C++"
   - Windows 10/11 SDK
3. Restart PC

### Step 3: Fix CUDA PATH

Your CUDA is installed but not in PATH. Add to System Environment Variables:

```
Path += C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin
Path += C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp
```

Verify: Open new terminal → `nvcc --version`

### Step 4: Download Darknet (May-June 2020 version)

```powershell
cd E:\
git clone https://github.com/AlexeyAB/darknet.git
cd darknet

# IMPORTANT: Checkout a specific commit from that era
# The code was written for this timeframe
git log --oneline --after="2020-05-01" --before="2020-07-01" | Select-Object -Last 1
```

### Step 5: Build Darknet with CMake

```powershell
cd E:\darknet
mkdir build_release
cd build_release

cmake .. -DENABLE_CUDA=ON -DENABLE_CUDNN=ON -DENABLE_OPENCV=ON -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

> If using VS 2022, replace "Visual Studio 16 2019" with "Visual Studio 17 2022"

### Step 6: Verify Darknet Build

```powershell
cd E:\darknet\build\darknet\x64
.\darknet.exe
# Should print usage info without errors
```

### Step 7: Copy Project Files

```powershell
# Copy detection scripts to darknet x64 folder
Copy-Item "E:\S.H.A.D.Y-main\...\Detection_With_Yolo\*.py" "E:\darknet\build\darknet\x64\"

# Download weights
cd E:\darknet\build\darknet\x64
Invoke-WebRequest -Uri "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights" -OutFile "yolov4-tiny.weights"
```

### Step 8: Switch to Darknet Backend

Edit `config.py`:
```python
BACKEND = "darknet"   # Now uses compiled darknet.dll
```

### Step 9: Run

```powershell
cd E:\darknet\build\darknet\x64
python Object_Detection.py --source 0
```

---

## Troubleshooting

### Common Issues

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'cv2'` | `pip install opencv-python` |
| `Weights file not found` | Run `python setup_project.py` |
| `Could not open video source` | Check webcam connection or video path |
| `CUDA not available` | Works without CUDA (just slower on CPU) |
| `import darknet` fails (Option A) | Darknet not compiled, use Option B |
| YouTube video fails | `pip install yt-dlp` |

### Performance Tips

1. **Use YOLOv4-tiny** (not full YOLOv4) for real-time detection
2. **Reduce resolution** - Set `RESIZE_FACTOR = 2` or `3` in config.py
3. **Lower threshold** - Set `CONFIDENCE_THRESHOLD = 0.3` for fewer detections
4. **GPU matters** - Even Option B is faster with CUDA-enabled OpenCV

---

## File Structure (New Scripts)

```
Detection_With_Yolo/
│
├── config.py               ← MAIN CONFIG (switch backend here!)
├── darknet_opencv.py       ← Option B: OpenCV DNN wrapper
├── detector_engine.py      ← Shared detection engine
│
├── Object_Detection.py     ← Run: detects all 80 classes
├── Fall_Detection.py       ← Run: detects human falls
├── Social_Distance.py      ← Run: monitors social distancing
├── Vehicle_Crash.py        ← Run: detects vehicle crashes
│
├── setup_project.py        ← Run FIRST: downloads weights/configs
├── test_setup.py           ← Run SECOND: verifies everything works
│
├── cfg/                    ← Created by setup_project.py
│   ├── yolov4-tiny.cfg
│   ├── yolov4.cfg
│   └── coco.data
│
├── data/                   ← Created by setup_project.py
│   └── coco.names
│
└── yolov4-tiny.weights     ← Downloaded by setup_project.py (~23MB)
```

---

## Switching Between Backends (Future Reference)

Kabhi bhi switch karna ho:

```python
# config.py mein sirf ek line change karo:

BACKEND = "opencv"    # Option B - No compilation, works everywhere
# ya
BACKEND = "darknet"   # Option A - Compiled darknet (faster GPU)
```

Baaki sab code SAME rahega. Detection scripts automatically correct backend load kar lenge.
