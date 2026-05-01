"""
app.py - S.H.A.D.Y Flask Web Application
==========================================
Web interface for all detection modules.
Uses OpenCV DNN backend (no compiled darknet needed).

Usage:
    conda activate yolo_shady
    python app.py
    → Open http://127.0.0.1:5000 in browser
"""

import sys
import os
import math
import cv2
import numpy as np
import time
import logging
from itertools import combinations
from flask import Flask, render_template, Response, request

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup file logging so we can always see what's happening
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_debug.log")
file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler])
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from config import BACKEND, CONFIDENCE_THRESHOLD

# Load correct backend
if BACKEND == "opencv":
    import darknet_opencv as darknet
else:
    import darknet

# ================================================================
# Global variables
# ================================================================
netMain = None
metaMain = None
altNames = None
video_link = None
case = None

# ================================================================
# Utility Functions
# ================================================================

def is_close(p1, p2):
    """Calculate Euclidean Distance between two points"""
    dst = math.sqrt(p1**2 + p2**2)
    return dst


def convertBack(x, y, w, h):
    """Converts center coordinates to rectangle coordinates"""
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


# ================================================================
# Detection Drawing Functions
# ================================================================

def cvDrawBoxes_fall(detections, img):
    """Fall Detection: width > height = FALL"""
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

        fall_alert_list = []
        for id, p in centroid_dict.items():
            dx, dy = p[4] - p[2], p[5] - p[3]
            difference = dy - dx
            if difference < 0:
                fall_alert_list.append(id)

        for idx, box in centroid_dict.items():
            if idx in fall_alert_list:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2)
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)

        if len(fall_alert_list) != 0:
            text = "FALL DETECTED!"
            cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            text = "Fall Not Detected"
            cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return img


def cvDrawBoxes_social(detections, img):
    """Social Distance: Euclidean distance between persons"""
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

        red_zone_list = []
        red_line_list = []
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]
            distance = is_close(dx, dy)
            if distance < 75.0:
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)
                    red_line_list.append(p1[0:2])
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)
                    red_line_list.append(p2[0:2])

        for idx, box in centroid_dict.items():
            if idx in red_zone_list:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2)
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)

        text = "People at Risk: %s" % str(len(red_zone_list))
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (246, 86, 86), 2, cv2.LINE_AA)

        for check in range(0, len(red_line_list) - 1):
            start_point = red_line_list[check]
            end_point = red_line_list[check + 1]
            check_line_x = abs(end_point[0] - start_point[0])
            check_line_y = abs(end_point[1] - start_point[1])
            if (check_line_x < 75) and (check_line_y < 25):
                cv2.line(img, start_point, end_point, (255, 0, 0), 2)
    return img


def cvDrawBoxes_vehicle(detections, img):
    """Vehicle Crash: Bounding box overlap = crash"""
    if len(detections) > 0:
        centroid_dict = dict()
        objectId = 0
        for detection in detections:
            name_tag = str(detection[0].decode())
            if name_tag == 'car':
                x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
                centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax)
                objectId += 1

        vehicle_red_zone_list = []
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
            if not ((p1[2] >= p2[4]) or (p1[4] <= p2[2]) or (p1[5] <= p2[3]) or (p1[3] >= p2[5])):
                if id1 not in vehicle_red_zone_list:
                    vehicle_red_zone_list.append(id1)
                if id2 not in vehicle_red_zone_list:
                    vehicle_red_zone_list.append(id2)

        for idx, box in centroid_dict.items():
            if idx in vehicle_red_zone_list:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2)
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)

        if len(vehicle_red_zone_list) != 0:
            text = "CRASH DETECTED!"
            cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            text = "Crash Not Detected"
            cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return img


def cvDrawBoxes_object(detections, img):
    """Object Detection: All 80 COCO classes with colored boxes"""
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

    for detection in detections:
        x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]
        name_tag = str(detection[0].decode())
        color = color_dict.get(name_tag, [255, 255, 255])
        xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, color, 2)
        cv2.putText(img,
                    name_tag + " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)
    return img


# ================================================================
# Video Frame Generator
# ================================================================

def load_model():
    """Load YOLO model once"""
    global netMain, metaMain, altNames
    
    configPath = "./cfg/yolov4-tiny.cfg"
    weightPath = "./yolov4-tiny.weights"
    metaPath = "./cfg/coco.data"

    if not os.path.exists(configPath):
        raise ValueError("Config not found: " + os.path.abspath(configPath))
    if not os.path.exists(weightPath):
        raise ValueError("Weights not found: " + os.path.abspath(weightPath) + 
                         "\nRun: python setup_project.py")
    if not os.path.exists(metaPath):
        raise ValueError("Meta not found: " + os.path.abspath(metaPath))

    if netMain is None:
        netMain = darknet.load_net_custom(
            configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
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


def get_video_url(link):
    """Extract video URL from YouTube link using yt-dlp"""
    logger.info(f"Attempting to extract URL from: {link}")
    
    if not link or not link.strip():
        logger.error("No video link provided!")
        return None
    
    link = link.strip()
    
    # If it's a direct video file URL (mp4, avi, etc.), return as-is
    if link.endswith(('.mp4', '.avi', '.mkv', '.webm', '.mov')):
        logger.info(f"Direct video URL detected: {link}")
        return link
    
    # Clean URL (remove anything after &)
    if '&' in link:
        link = link[:link.index('&')]
    
    try:
        import yt_dlp
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'quiet': False,
            'no_warnings': False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(link, download=False)
            url = info.get('url')
            if not url:
                # Try formats list
                formats = info.get('formats', [])
                if formats:
                    url = formats[-1].get('url')
            logger.info(f"Successfully extracted video URL (length={len(url) if url else 0})")
            return url
    except Exception as e:
        logger.error(f"yt-dlp failed: {type(e).__name__}: {e}", exc_info=True)
        return None


def gen_frames():
    """Generator: yields JPEG frames with detection drawn"""
    global case, netMain, metaMain, video_link

    logger.info(f"gen_frames() called - case={case}, video_link={video_link}")

    try:
        load_model()
    except Exception as e:
        logger.error(f"Model loading failed: {e}", exc_info=True)
        return

    video_url = get_video_url(video_link)
    if video_url is None:
        logger.error("Could not get video URL - aborting gen_frames")
        return

    logger.info(f"Opening video with OpenCV...")
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        logger.error(f"Could not open video stream. URL length: {len(video_url)}")
        logger.error(f"First 200 chars of URL: {video_url[:200]}")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_width = frame_width // 2
    new_height = frame_height // 2

    logger.info(f"Starting YOLO loop... Case: {case}, Resolution: {new_width}x{new_height}")

    darknet_image = darknet.make_image(new_width, new_height, 3)

    while True:
        ret, frame_read = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

        if case == 'object':
            image = cvDrawBoxes_object(detections, frame_resized)
        elif case == 'social':
            image = cvDrawBoxes_social(detections, frame_resized)
        elif case == 'fall':
            image = cvDrawBoxes_fall(detections, frame_resized)
        elif case == 'vehicle':
            image = cvDrawBoxes_vehicle(detections, frame_resized)
        else:
            image = cvDrawBoxes_object(detections, frame_resized)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    logger.info("Video stream ended.")


# ================================================================
# Flask App
# ================================================================

app = Flask(__name__,
            template_folder='Deployed App/templates',
            static_folder='Deployed App/static')


@app.route('/')
def Shady():
    return render_template('Shady.html')


@app.route('/FallDetection', methods=['GET', 'POST'])
def FallDetection():
    global case
    case = 'fall'
    return render_template('FallDetection.html')


@app.route('/ObjectDetection', methods=['GET', 'POST'])
def ObjectDetection():
    global case
    case = 'object'
    return render_template('ObjectDetection.html')


@app.route('/SocialDistancingDetection', methods=['GET', 'POST'])
def SocialDistancingDetection():
    global case
    case = 'social'
    return render_template('SocialDistancingDetection.html')


@app.route('/VehicleCrashDetection', methods=['GET', 'POST'])
def VehicleCrashDetection():
    global case
    case = 'vehicle'
    return render_template('VehicleCrashDetection.html')


@app.route('/ContactUs')
def ContactUs():
    return render_template('ContactUs.html')


@app.route('/Video', methods=['GET', 'POST'])
def Video():
    global video_link
    video_link = request.form.get('videolink')
    logger.info(f"/Video route - video_link: {video_link}, case: {case}")
    return render_template('Video.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    print("=" * 50)
    print("  S.H.A.D.Y - Flask Web Application")
    print(f"  Backend: {BACKEND.upper()}")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 50)
    
    # Pre-load model at startup
    load_model()
    
    app.run(debug=False, host='127.0.0.1', port=5000)
