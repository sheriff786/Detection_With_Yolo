# ==============================================================
# S.H.A.D.Y Project - Configuration File
# ==============================================================
# Switch between Option A (Darknet) and Option B (OpenCV DNN)
# by changing the BACKEND variable below.
# ==============================================================

# ┌─────────────────────────────────────────────────────────────┐
# │  BACKEND SELECTION                                          │
# │                                                             │
# │  "darknet"      → Option A: Original compiled darknet      │
# │  "opencv"       → Option B: OpenCV DNN (no compilation)    │
# │                                                             │
# └─────────────────────────────────────────────────────────────┘

BACKEND = "opencv"    # Change to "darknet" if you have compiled darknet


# ==============================================================
# Model Configuration
# ==============================================================

# YOLOv4-tiny (Fast - for Fall/Social/Crash detection)
YOLO_TINY_CONFIG = "./cfg/yolov4-tiny.cfg"
YOLO_TINY_WEIGHTS = "./yolov4-tiny.weights"

# YOLOv4 Full (Accurate - for Object detection)
YOLO_FULL_CONFIG = "./cfg/yolov4.cfg"
YOLO_FULL_WEIGHTS = "./yolov4.weights"

# Metadata
META_PATH = "./cfg/coco.data"

# ==============================================================
# Detection Thresholds
# ==============================================================

CONFIDENCE_THRESHOLD = 0.25        # Minimum confidence to consider a detection
NMS_THRESHOLD = 0.45               # Non-Maximum Suppression threshold
SOCIAL_DISTANCE_THRESHOLD = 75.0   # Pixels - social distancing distance
FALL_FRAME_THRESHOLD = 20          # Consecutive frames to trigger fall alert
CRASH_FRAME_THRESHOLD = 8          # Consecutive frames to trigger crash alert

# ==============================================================
# Video Input Configuration
# ==============================================================
# Uncomment ONE of the following video sources:

# VIDEO_SOURCE = 0                                          # Webcam
# VIDEO_SOURCE = "test_video.mp4"                           # Local file
# VIDEO_SOURCE = "youtube:https://www.youtube.com/watch?v=xxx"  # YouTube
# VIDEO_SOURCE = "http://192.168.0.106:4747/mjpegfeed"      # DroidCam

VIDEO_SOURCE = 0  # Default: Webcam

# ==============================================================
# Email Alert Configuration
# ==============================================================

EMAIL_ALERTS_ENABLED = False        # Set to True to enable email alerts
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"  # Use App Password, not account password
RECEIVER_EMAIL = "receiver@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# ==============================================================
# Display Configuration
# ==============================================================

DISPLAY_FPS = True                  # Show FPS in console
RESIZE_FACTOR = 2                   # Divide frame by this (2 = half size)
WINDOW_NAME = "S.H.A.D.Y Detection"
