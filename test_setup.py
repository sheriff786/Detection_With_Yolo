"""
test_setup.py - Quick test to verify everything is working
============================================================
Run this after setup_project.py to verify installation.

Usage:
    python test_setup.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required packages are importable"""
    print("[TEST 1] Checking Python packages...")
    results = {}
    
    packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy', 
        'flask': 'flask',
    }
    
    for module, pip_name in packages.items():
        try:
            __import__(module)
            results[module] = True
            print(f"  [OK] {module}")
        except ImportError:
            results[module] = False
            print(f"  [MISSING] {module} → pip install {pip_name}")
    
    # Optional packages
    optional = {
        'yt_dlp': 'yt-dlp (for YouTube videos)',
    }
    for module, desc in optional.items():
        try:
            __import__(module)
            print(f"  [OK] {module} (optional)")
        except ImportError:
            print(f"  [INFO] {module} not installed - {desc}")
    
    return all(results.values())


def test_opencv_dnn():
    """Test OpenCV DNN capabilities"""
    print("\n[TEST 2] Checking OpenCV DNN backend...")
    import cv2
    print(f"  OpenCV version: {cv2.__version__}")
    
    # Check CUDA support
    try:
        backends = cv2.dnn.getAvailableBackends()
        print(f"  Available backends: {backends}")
        
        # Check if CUDA backend is available
        cuda_available = any('CUDA' in str(b) for b in backends)
        if cuda_available:
            print(f"  [OK] CUDA backend available (GPU acceleration)")
        else:
            print(f"  [INFO] CUDA not available in OpenCV - will use CPU")
            print(f"         (Still works fine, just slower)")
    except Exception as e:
        print(f"  [WARN] Could not check backends: {e}")
    
    return True


def test_files():
    """Test that all required files exist"""
    print("\n[TEST 3] Checking required files...")
    
    base = os.path.dirname(os.path.abspath(__file__))
    required_files = {
        'cfg/yolov4-tiny.cfg': 'YOLO tiny config',
        'cfg/coco.data': 'COCO metadata',
        'data/coco.names': 'Class names (80 classes)',
        'yolov4-tiny.weights': 'YOLOv4-tiny weights (23MB)',
    }
    
    all_ok = True
    for filepath, desc in required_files.items():
        full_path = os.path.join(base, filepath)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  [OK] {filepath} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {filepath} - {desc}")
            all_ok = False
    
    # Optional
    optional_files = {
        'cfg/yolov4.cfg': 'YOLOv4 full config',
        'yolov4.weights': 'YOLOv4 full weights (245MB)',
    }
    for filepath, desc in optional_files.items():
        full_path = os.path.join(base, filepath)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  [OK] {filepath} ({size_mb:.1f} MB) (optional)")
        else:
            print(f"  [INFO] {filepath} not found (optional - {desc})")
    
    return all_ok


def test_darknet_opencv():
    """Test the OpenCV DNN wrapper module"""
    print("\n[TEST 4] Testing darknet_opencv module...")
    
    try:
        import darknet_opencv
        print(f"  [OK] darknet_opencv module imported successfully")
        
        # Test make_image
        img = darknet_opencv.make_image(416, 416, 3)
        print(f"  [OK] make_image(416, 416, 3) works")
        
        # Test with actual model if weights exist
        base = os.path.dirname(os.path.abspath(__file__))
        cfg = os.path.join(base, "cfg", "yolov4-tiny.cfg")
        weights = os.path.join(base, "yolov4-tiny.weights")
        meta = os.path.join(base, "cfg", "coco.data")
        
        if os.path.exists(cfg) and os.path.exists(weights) and os.path.exists(meta):
            print(f"  [LOADING] Loading YOLOv4-tiny network...")
            net = darknet_opencv.load_net_custom(cfg.encode(), weights.encode(), 0, 1)
            print(f"  [OK] Network loaded!")
            
            meta_obj = darknet_opencv.load_meta(meta.encode())
            print(f"  [OK] Metadata loaded ({meta_obj.classes} classes)")
            
            # Quick detection test with dummy image
            import numpy as np
            dummy_frame = np.zeros((416, 416, 3), dtype=np.uint8)
            darknet_image = darknet_opencv.make_image(416, 416, 3)
            darknet_opencv.copy_image_from_bytes(darknet_image, dummy_frame.tobytes())
            detections = darknet_opencv.detect_image(net, meta_obj, darknet_image, thresh=0.25)
            print(f"  [OK] Detection test passed! ({len(detections)} detections on blank image)")
            
            return True
        else:
            print(f"  [INFO] Weights not yet downloaded - run setup_project.py first")
            return True  # Module works, just no weights
            
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def test_webcam():
    """Quick test to see if webcam is accessible"""
    print("\n[TEST 5] Checking webcam access...")
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"  [OK] Webcam accessible ({w}x{h})")
        else:
            print(f"  [WARN] Webcam opened but could not read frame")
        cap.release()
    else:
        print(f"  [INFO] No webcam detected (use local video or YouTube instead)")
    return True


def main():
    print("=" * 60)
    print("  S.H.A.D.Y - Setup Verification")
    print("=" * 60)
    print()
    
    results = []
    results.append(("Python packages", test_imports()))
    results.append(("OpenCV DNN", test_opencv_dnn()))
    results.append(("Required files", test_files()))
    results.append(("darknet_opencv module", test_darknet_opencv()))
    results.append(("Webcam", test_webcam()))
    
    print()
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        icon = "[OK]" if passed else "[!!]"
        print(f"  {icon} {name}: {status}")
    
    print()
    if all(r[1] for r in results):
        print("  ALL TESTS PASSED! You can run the detection scripts.")
        print()
        print("  Quick start:")
        print("    python Object_Detection.py --source 0")
        print("    python Fall_Detection.py --source 0")
        print("    python Social_Distance.py --source 0")
        print("    python Vehicle_Crash.py --source 0")
    else:
        print("  Some tests failed. Check the output above for details.")
        print("  Run 'python setup_project.py' to download missing files.")


if __name__ == "__main__":
    main()
