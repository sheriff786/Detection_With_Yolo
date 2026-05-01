"""
setup_project.py - Download all required files for S.H.A.D.Y
==============================================================
Run this script to download YOLO weights, config, and data files.

Usage:
    python setup_project.py
"""

import os
import sys
import urllib.request
import hashlib

# Base directory (where this script lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_file(url, dest_path, description=""):
    """Download a file with progress indicator"""
    if os.path.exists(dest_path):
        print(f"  [SKIP] Already exists: {os.path.basename(dest_path)}")
        return True
    
    print(f"  [DOWNLOADING] {description}")
    print(f"    URL:  {url}")
    print(f"    Dest: {dest_path}")
    
    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 // total_size)
                mb_done = (count * block_size) / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r    Progress: {percent}% ({mb_done:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print(f"\n  [DONE] Downloaded: {os.path.basename(dest_path)}")
        return True
    except Exception as e:
        print(f"\n  [ERROR] Failed to download: {e}")
        return False


def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def create_coco_names(path):
    """Create the coco.names file with 80 COCO class names"""
    names = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(names))
    print(f"  [CREATED] {path} ({len(names)} classes)")


def create_coco_data(path, names_path):
    """Create the coco.data metadata file"""
    content = f"""classes = 80
names = {names_path}
"""
    with open(path, 'w') as f:
        f.write(content)
    print(f"  [CREATED] {path}")


def main():
    print("=" * 60)
    print("  S.H.A.D.Y Project Setup")
    print("  Downloading YOLO weights, config, and data files")
    print("=" * 60)
    print()
    
    # Create directories
    print("[1/5] Creating directories...")
    cfg_dir = os.path.join(BASE_DIR, "cfg")
    data_dir = os.path.join(BASE_DIR, "data")
    create_directory(cfg_dir)
    create_directory(data_dir)
    print()
    
    # Create coco.names
    print("[2/5] Creating data files...")
    names_path = os.path.join(data_dir, "coco.names")
    if not os.path.exists(names_path):
        create_coco_names(names_path)
    else:
        print(f"  [SKIP] Already exists: coco.names")
    
    # Create coco.data
    coco_data_path = os.path.join(cfg_dir, "coco.data")
    if not os.path.exists(coco_data_path):
        create_coco_data(coco_data_path, "data/coco.names")
    else:
        print(f"  [SKIP] Already exists: coco.data")
    print()
    
    # Download YOLOv4-tiny config
    print("[3/5] Downloading YOLOv4-tiny config...")
    download_file(
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        os.path.join(cfg_dir, "yolov4-tiny.cfg"),
        "YOLOv4-tiny configuration"
    )
    print()
    
    # Download YOLOv4-tiny weights (~23 MB)
    print("[4/5] Downloading YOLOv4-tiny weights (~23 MB)...")
    download_file(
        "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
        os.path.join(BASE_DIR, "yolov4-tiny.weights"),
        "YOLOv4-tiny weights (23 MB)"
    )
    print()
    
    # Download YOLOv4 full config (optional)
    print("[5/5] Downloading YOLOv4 full config (optional)...")
    download_file(
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
        os.path.join(cfg_dir, "yolov4.cfg"),
        "YOLOv4 full configuration"
    )
    print()
    
    # Summary
    print("=" * 60)
    print("  SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("  Files created/downloaded:")
    print(f"    cfg/yolov4-tiny.cfg  : {'OK' if os.path.exists(os.path.join(cfg_dir, 'yolov4-tiny.cfg')) else 'MISSING'}")
    print(f"    cfg/yolov4.cfg       : {'OK' if os.path.exists(os.path.join(cfg_dir, 'yolov4.cfg')) else 'MISSING'}")
    print(f"    cfg/coco.data        : {'OK' if os.path.exists(coco_data_path) else 'MISSING'}")
    print(f"    data/coco.names      : {'OK' if os.path.exists(names_path) else 'MISSING'}")
    print(f"    yolov4-tiny.weights  : {'OK' if os.path.exists(os.path.join(BASE_DIR, 'yolov4-tiny.weights')) else 'MISSING'}")
    print()
    print("  [OPTIONAL] For full YOLOv4 object detection, also download:")
    print("    yolov4.weights (~245 MB):")
    print("    https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights")
    print()
    print("  Next steps:")
    print("    1. Make sure config.py has BACKEND = 'opencv'")
    print("    2. Run: python Object_Detection.py --source 0")
    print("    3. Or:  python Fall_Detection.py --source 0")
    print()


if __name__ == "__main__":
    main()
