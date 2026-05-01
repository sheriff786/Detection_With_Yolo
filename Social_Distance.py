"""
Social_Distance.py - Social Distancing Monitor
================================================
Detects persons and calculates Euclidean distance between all pairs.
If distance < threshold → Violation detected (Red box + line).

Uses YOLOv4-tiny for real-time speed.
Works with both Option A (darknet) and Option B (OpenCV DNN).

Usage:
    python Social_Distance.py
    python Social_Distance.py --source 0
    python Social_Distance.py --source crowd_video.mp4
"""

import sys
import os
import cv2
import numpy as np
import time
import math
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (BACKEND, CONFIDENCE_THRESHOLD, VIDEO_SOURCE,
                    RESIZE_FACTOR, SOCIAL_DISTANCE_THRESHOLD)

if BACKEND == "opencv":
    import darknet_opencv as darknet
else:
    import darknet

from detector_engine import load_detector, get_video_capture


def is_close(p1, p2):
    """Calculate Euclidean distance between two points"""
    dst = math.sqrt(p1**2 + p2**2)
    return dst


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    """
    Social Distance Algorithm:
    - Filter only 'person' class
    - Calculate Euclidean distance between ALL pairs
    - If distance < threshold → Both persons are "at risk" (Red box)
    - Draw red line between violating pairs
    """
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
        
        # Check all pairs for social distancing violations
        red_zone_list = []
        red_line_list = []
        
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]
            distance = is_close(dx, dy)
            
            if distance < SOCIAL_DISTANCE_THRESHOLD:
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)
                    red_line_list.append(p1[0:2])
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)
                    red_line_list.append(p2[0:2])
        
        # Draw bounding boxes
        for idx, box in centroid_dict.items():
            if idx in red_zone_list:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2)  # RED
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)  # GREEN
        
        # Display risk count
        text = "People at Risk: %s" % str(len(red_zone_list))
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (246, 86, 86), 2, cv2.LINE_AA)
        
        # Draw red lines between close people
        for check in range(0, len(red_line_list) - 1):
            start_point = red_line_list[check]
            end_point = red_line_list[check + 1]
            check_line_x = abs(end_point[0] - start_point[0])
            check_line_y = abs(end_point[1] - start_point[1])
            if (check_line_x < SOCIAL_DISTANCE_THRESHOLD) and (check_line_y < 25):
                cv2.line(img, start_point, end_point, (255, 0, 0), 2)
    
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
    print("  S.H.A.D.Y - Social Distance Detection")
    print(f"  Backend: {BACKEND.upper()}")
    print(f"  Distance Threshold: {SOCIAL_DISTANCE_THRESHOLD} pixels")
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
        
        cv2.imshow('S.H.A.D.Y - Social Distance', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Detection completed.")


if __name__ == "__main__":
    main()
