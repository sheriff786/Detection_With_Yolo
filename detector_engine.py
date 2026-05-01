"""
detector_engine.py - Unified Detection Engine
==============================================
This module handles loading the correct backend (darknet or opencv)
based on config.py settings. All detection scripts use this.

Usage:
    from detector_engine import load_detector, detect_frame, get_video_capture
"""

import os
import sys
import cv2
import numpy as np
import time

# Import configuration
from config import (
    BACKEND, YOLO_TINY_CONFIG, YOLO_TINY_WEIGHTS, 
    YOLO_FULL_CONFIG, YOLO_FULL_WEIGHTS, META_PATH,
    CONFIDENCE_THRESHOLD, VIDEO_SOURCE, RESIZE_FACTOR
)


def _load_darknet_module():
    """Load the appropriate darknet module based on BACKEND config"""
    if BACKEND == "opencv":
        import darknet_opencv as darknet
        print("[Engine] Using OpenCV DNN Backend (Option B)")
    elif BACKEND == "darknet":
        import darknet
        print("[Engine] Using Compiled Darknet Backend (Option A)")
    else:
        raise ValueError(f"Unknown BACKEND: {BACKEND}. Use 'opencv' or 'darknet'")
    return darknet


# Load the darknet module once
darknet = _load_darknet_module()


def load_detector(model="tiny"):
    """
    Load YOLO model and return (network, meta, class_names)
    
    Args:
        model: "tiny" for YOLOv4-tiny (fast), "full" for YOLOv4 (accurate)
    
    Returns:
        (netMain, metaMain, altNames)
    """
    if model == "tiny":
        configPath = YOLO_TINY_CONFIG
        weightPath = YOLO_TINY_WEIGHTS
    elif model == "full":
        configPath = YOLO_FULL_CONFIG
        weightPath = YOLO_FULL_WEIGHTS
    else:
        raise ValueError(f"Unknown model: {model}. Use 'tiny' or 'full'")
    
    metaPath = META_PATH
    
    # Validate paths
    if not os.path.exists(configPath):
        raise FileNotFoundError(
            f"Config file not found: {os.path.abspath(configPath)}\n"
            f"Download it or check path in config.py"
        )
    if not os.path.exists(weightPath):
        raise FileNotFoundError(
            f"Weights file not found: {os.path.abspath(weightPath)}\n"
            f"Download from: https://github.com/AlexeyAB/darknet/releases\n"
            f"  - yolov4-tiny.weights: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights\n"
            f"  - yolov4.weights: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
        )
    if not os.path.exists(metaPath):
        raise FileNotFoundError(
            f"Meta file not found: {os.path.abspath(metaPath)}\n"
            f"Create cfg/coco.data with 'names = data/coco.names'"
        )
    
    # Load network
    netMain = darknet.load_net_custom(
        configPath.encode("ascii"), 
        weightPath.encode("ascii"), 0, 1
    )
    
    # Load metadata
    metaMain = darknet.load_meta(metaPath.encode("ascii"))
    
    # Load class names
    altNames = None
    try:
        with open(metaPath) as metaFH:
            metaContents = metaFH.read()
            import re
            match = re.search("names *= *(.*)$", metaContents,
                              re.IGNORECASE | re.MULTILINE)
            if match:
                result = match.group(1)
                if os.path.exists(result):
                    with open(result) as namesFH:
                        namesList = namesFH.read().strip().split("\n")
                        altNames = [x.strip() for x in namesList]
    except Exception:
        pass
    
    print(f"[Engine] Model loaded: {model} ({configPath})")
    return netMain, metaMain, altNames


def get_video_capture(source=None):
    """
    Get video capture object from configured source.
    
    Args:
        source: Override VIDEO_SOURCE from config. Can be:
                - int (webcam index)
                - str (file path)
                - str starting with "youtube:" (YouTube URL)
                - str starting with "http" (DroidCam/IP camera)
    
    Returns:
        cv2.VideoCapture object
    """
    if source is None:
        source = VIDEO_SOURCE
    
    if isinstance(source, int):
        # Webcam
        print(f"[Engine] Opening webcam {source}...")
        cap = cv2.VideoCapture(source)
        
    elif isinstance(source, str) and source.startswith("youtube:"):
        # YouTube video
        url = source.replace("youtube:", "")
        print(f"[Engine] Opening YouTube video: {url}")
        try:
            import yt_dlp
            ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_url = info.get('url')
                if not video_url:
                    formats = info.get('formats', [])
                    if formats:
                        video_url = formats[-1].get('url')
            cap = cv2.VideoCapture(video_url)
        except ImportError:
            try:
                import pafy
                video = pafy.new(url)
                best = video.getbest(preftype="mp4")
                cap = cv2.VideoCapture(best.url)
            except ImportError:
                raise ImportError(
                    "Install yt-dlp (recommended) or pafy for YouTube support:\n"
                    "  pip install yt-dlp"
                )
    
    elif isinstance(source, str) and source.startswith("http"):
        # IP Camera / DroidCam
        print(f"[Engine] Opening IP camera: {source}")
        cap = cv2.VideoCapture(source)
        
    else:
        # Local file
        print(f"[Engine] Opening local video: {source}")
        if not os.path.exists(source):
            raise FileNotFoundError(f"Video file not found: {source}")
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    
    print(f"[Engine] Video source opened successfully!")
    return cap


def detect_frame(netMain, metaMain, frame, img_w, img_h):
    """
    Run YOLO detection on a single frame.
    
    Args:
        netMain: loaded network
        metaMain: loaded metadata
        frame: RGB frame (numpy array)
        img_w: width for darknet image
        img_h: height for darknet image
    
    Returns:
        list of detections: [(class_name_bytes, confidence, (cx, cy, w, h)), ...]
    """
    darknet_image = darknet.make_image(img_w, img_h, 3)
    
    frame_resized = cv2.resize(frame, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
    
    detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=CONFIDENCE_THRESHOLD)
    
    return detections


def run_detection_loop(netMain, metaMain, cap, draw_function, window_name="S.H.A.D.Y"):
    """
    Main detection loop - reads frames, runs YOLO, applies custom drawing function.
    
    Args:
        netMain: loaded network
        metaMain: loaded metadata
        cap: cv2.VideoCapture object
        draw_function: function(detections, img) -> img with drawings
        window_name: display window title
    """
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_width = frame_width // RESIZE_FACTOR
    new_height = frame_height // RESIZE_FACTOR
    
    print(f"[Engine] Original resolution: {frame_width}x{frame_height}")
    print(f"[Engine] Processing resolution: {new_width}x{new_height}")
    print(f"[Engine] Starting detection loop... Press 'q' to quit.")
    print()
    
    darknet_image = darknet.make_image(new_width, new_height, 3)
    
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        
        if not ret:
            break
        
        # Convert BGR -> RGB and resize
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height), 
                                   interpolation=cv2.INTER_LINEAR)
        
        # Run YOLO detection
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(netMain, metaMain, darknet_image, 
                                          thresh=CONFIDENCE_THRESHOLD)
        
        # Apply custom drawing function
        image = draw_function(detections, frame_resized)
        
        # Convert back to BGR for display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Show FPS
        fps = 1.0 / (time.time() - prev_time)
        cv2.putText(image, f"FPS: {fps:.1f}", (image.shape[1] - 150, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, image)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("[Engine] Detection loop ended.")
