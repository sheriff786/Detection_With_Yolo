"""
Object_Detection.py - General Object Detection (All 80 COCO classes)
=====================================================================
Uses YOLOv4 (full) for maximum accuracy.
Works with both Option A (darknet) and Option B (OpenCV DNN).

Usage:
    python Object_Detection.py
    python Object_Detection.py --source 0                    # Webcam
    python Object_Detection.py --source video.mp4            # Local file
    python Object_Detection.py --source "youtube:URL"        # YouTube
"""

import sys
import os
import cv2
import numpy as np
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BACKEND, CONFIDENCE_THRESHOLD, VIDEO_SOURCE, RESIZE_FACTOR

# Load correct backend
if BACKEND == "opencv":
    import darknet_opencv as darknet
else:
    import darknet

from detector_engine import load_detector, get_video_capture


# ================================================================
# Colored labels for 80 COCO classes
# ================================================================
color_dict = {
    'person': [0, 255, 255], 'bicycle': [238, 123, 158], 'car': [24, 245, 217],
    'motorbike': [224, 119, 227], 'aeroplane': [154, 52, 104], 'bus': [179, 50, 247],
    'train': [180, 164, 5], 'truck': [82, 42, 106], 'boat': [201, 25, 52],
    'traffic light': [62, 17, 209], 'fire hydrant': [60, 68, 169],
    'stop sign': [199, 113, 167], 'parking meter': [19, 71, 68],
    'bench': [161, 83, 182], 'bird': [75, 6, 145], 'cat': [100, 64, 151],
    'dog': [156, 116, 171], 'horse': [88, 9, 123], 'sheep': [181, 86, 222],
    'cow': [116, 238, 87], 'elephant': [74, 90, 143], 'bear': [249, 157, 47],
    'zebra': [26, 101, 131], 'giraffe': [195, 130, 181], 'backpack': [242, 52, 233],
    'umbrella': [131, 11, 189], 'handbag': [221, 229, 176], 'tie': [193, 56, 44],
    'suitcase': [139, 53, 137], 'frisbee': [102, 208, 40], 'skis': [61, 50, 7],
    'snowboard': [65, 82, 186], 'sports ball': [65, 82, 186],
    'kite': [153, 254, 81], 'baseball bat': [233, 80, 195],
    'baseball glove': [165, 179, 213], 'skateboard': [57, 65, 211],
    'surfboard': [98, 255, 164], 'tennis racket': [205, 219, 146],
    'bottle': [140, 138, 172], 'wine glass': [23, 53, 119],
    'cup': [102, 215, 88], 'fork': [198, 204, 245], 'knife': [183, 132, 233],
    'spoon': [14, 87, 125], 'bowl': [221, 43, 104], 'banana': [181, 215, 6],
    'apple': [16, 139, 183], 'sandwich': [150, 136, 166], 'orange': [219, 144, 1],
    'broccoli': [123, 226, 195], 'carrot': [230, 45, 209], 'hot dog': [252, 215, 56],
    'pizza': [234, 170, 131], 'donut': [36, 208, 234], 'cake': [19, 24, 2],
    'chair': [115, 184, 234], 'sofa': [125, 238, 12], 'pottedplant': [57, 226, 76],
    'bed': [77, 31, 134], 'diningtable': [208, 202, 204], 'toilet': [208, 202, 204],
    'tvmonitor': [208, 202, 204], 'laptop': [159, 149, 163],
    'mouse': [148, 148, 87], 'remote': [171, 107, 183],
    'keyboard': [33, 154, 135], 'cell phone': [206, 209, 108],
    'microwave': [206, 209, 108], 'oven': [97, 246, 15],
    'toaster': [147, 140, 184], 'sink': [157, 58, 24],
    'refrigerator': [117, 145, 137], 'book': [155, 129, 244],
    'clock': [53, 61, 6], 'vase': [145, 75, 152], 'scissors': [8, 140, 38],
    'teddy bear': [37, 61, 220], 'hair drier': [129, 12, 229],
    'toothbrush': [11, 126, 158]
}


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    """Draw colored bounding boxes for all detected objects"""
    for detection in detections:
        x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]
        name_tag = str(detection[0].decode())
        
        if name_tag in color_dict:
            color = color_dict[name_tag]
        else:
            color = [255, 255, 255]
        
        xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, color, 1)
        cv2.putText(img,
                    name_tag + " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)
    return img


def main():
    # Parse command line source override
    source = VIDEO_SOURCE
    if len(sys.argv) > 2 and sys.argv[1] == "--source":
        source = sys.argv[2]
        try:
            source = int(source)
        except ValueError:
            pass
    
    # Load model (full YOLOv4 for object detection)
    print("=" * 50)
    print("  S.H.A.D.Y - Object Detection")
    print(f"  Backend: {BACKEND.upper()}")
    print("=" * 50)
    
    # Use "tiny" if full weights not available
    try:
        netMain, metaMain, altNames = load_detector("full")
    except FileNotFoundError:
        print("[!] YOLOv4 full weights not found, falling back to YOLOv4-tiny...")
        netMain, metaMain, altNames = load_detector("tiny")
    
    # Open video source
    cap = get_video_capture(source)
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_width = frame_width // RESIZE_FACTOR
    new_height = frame_height // RESIZE_FACTOR
    
    print(f"Resolution: {new_width}x{new_height}")
    print("Press 'q' to quit")
    print()
    
    darknet_image = darknet.make_image(new_width, new_height, 3)
    
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)
        
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=CONFIDENCE_THRESHOLD)
        
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        fps = 1.0 / (time.time() - prev_time)
        cv2.putText(image, f"FPS: {fps:.1f}", (10, image.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('S.H.A.D.Y - Object Detection', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Detection completed.")


if __name__ == "__main__":
    main()
