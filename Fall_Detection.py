"""
Fall_Detection.py - Human Fall Detection
==========================================
Detects falls by comparing bounding box height vs width.
If width > height → Person has fallen.

Uses YOLOv4-tiny for real-time speed.
Works with both Option A (darknet) and Option B (OpenCV DNN).

Usage:
    python Fall_Detection.py
    python Fall_Detection.py --source 0
    python Fall_Detection.py --source fall_video.mp4
"""

import sys
import os
import cv2
import numpy as np
import time
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (BACKEND, CONFIDENCE_THRESHOLD, VIDEO_SOURCE, 
                    RESIZE_FACTOR, FALL_FRAME_THRESHOLD, EMAIL_ALERTS_ENABLED)

if BACKEND == "opencv":
    import darknet_opencv as darknet
else:
    import darknet

from detector_engine import load_detector, get_video_capture

# Alert counter
alert_var = 0


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    """
    Fall Detection Algorithm:
    - Filter only 'person' class
    - For each person: compare width (dx) vs height (dy)
    - If dy < dx (height < width) → FALL DETECTED
    """
    global alert_var
    
    if len(detections) > 0:
        centroid_dict = dict()
        objectId = 0
        
        for detection in detections:
            name_tag = str(detection[0].decode())
            if name_tag == 'person':
                x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
                centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax)
                objectId += 1
        
        # Determine falls
        fall_alert_list = []
        for id, p in centroid_dict.items():
            dx = p[4] - p[2]   # width of bounding box
            dy = p[5] - p[3]   # height of bounding box
            difference = dy - dx
            if difference < 0:  # width > height → FALL
                fall_alert_list.append(id)
        
        # Draw bounding boxes
        for idx, box in centroid_dict.items():
            if idx in fall_alert_list:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2)  # RED
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)  # GREEN
        
        # Display result
        if len(fall_alert_list) != 0:
            text = "FALL DETECTED!"
            cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Email alert logic
            if EMAIL_ALERTS_ENABLED:
                alert_var += 1
                if alert_var == FALL_FRAME_THRESHOLD:
                    img_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite('fall_alert.jpg', img_save)
                    try:
                        import image_email_fall
                        image_email_fall.SendMail('fall_alert.jpg')
                        print("[ALERT] Fall email sent!")
                    except Exception as e:
                        print(f"[ALERT] Email failed: {e}")
        else:
            text = "Fall Not Detected"
            cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            alert_var = 0
    
    return img


def main():
    source = VIDEO_SOURCE
    if len(sys.argv) > 2 and sys.argv[1] == "--source":
        source = sys.argv[2]
        try:
            source = int(source)
        except ValueError:
            pass
    
    print("=" * 50)
    print("  S.H.A.D.Y - Fall Detection")
    print(f"  Backend: {BACKEND.upper()}")
    print(f"  Alert Threshold: {FALL_FRAME_THRESHOLD} frames")
    print("=" * 50)
    
    netMain, metaMain, altNames = load_detector("tiny")
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
        
        cv2.imshow('S.H.A.D.Y - Fall Detection', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Detection completed.")


if __name__ == "__main__":
    main()
