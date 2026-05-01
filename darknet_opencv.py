"""
darknet_opencv.py - OpenCV DNN Backend (Option B)
=================================================
This module provides the SAME API as the original darknet.py module
but uses OpenCV's DNN module internally. No need to compile darknet!

Usage:
    Instead of: import darknet
    Use:        import darknet_opencv as darknet

All function signatures match the original darknet.py so you can
switch between Option A (compiled darknet) and Option B (OpenCV DNN)
by simply changing the import statement.
"""

import cv2
import numpy as np
import os


# ============================================================
# Internal state - mimics darknet's C structures
# ============================================================
class _NetworkMeta:
    """Mimics darknet's metadata structure"""
    def __init__(self, names_file):
        with open(names_file, 'r') as f:
            self.names = [line.strip() for line in f.readlines() if line.strip()]
        self.classes = len(self.names)


class _Network:
    """Mimics darknet's network structure using OpenCV DNN"""
    def __init__(self, config_path, weights_path):
        print(f"[OpenCV DNN] Loading network...")
        print(f"  Config:  {config_path}")
        print(f"  Weights: {weights_path}")
        
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        # Try to use CUDA if available, otherwise fallback to CPU
        cuda_works = False
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            # Test with a dummy forward pass to confirm CUDA actually works
            dummy = cv2.dnn.blobFromImage(np.zeros((416, 416, 3), dtype=np.uint8), 1/255.0, (416, 416))
            self.net.setInput(dummy)
            self.net.forward(self.net.getUnconnectedOutLayersNames())
            cuda_works = True
            print("  Backend: CUDA (GPU)")
        except Exception:
            cuda_works = False
        
        if not cuda_works:
            self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("  Backend: OpenCV CPU (CUDA not available - still works fine)")
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        out_layers = self.net.getUnconnectedOutLayers()
        # Handle both old and new OpenCV versions
        if isinstance(out_layers[0], (list, np.ndarray)):
            self.output_layers = [layer_names[i[0] - 1] for i in out_layers]
        else:
            self.output_layers = [layer_names[i - 1] for i in out_layers]
        
        # Read input size from config
        self.width = 416
        self.height = 416
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    if line.startswith('width'):
                        self.width = int(line.split('=')[1].strip())
                    elif line.startswith('height'):
                        self.height = int(line.split('=')[1].strip())
        except Exception:
            pass
        
        print(f"  Input Size: {self.width}x{self.height}")
        print(f"  [OpenCV DNN] Network loaded successfully!")


class _DarknetImage:
    """Mimics darknet's image structure"""
    def __init__(self, w, h, c):
        self.w = w
        self.h = h
        self.c = c
        self.data = None  # Will store the frame bytes


# ============================================================
# Public API - MATCHES original darknet.py signatures
# ============================================================

def load_net_custom(config_path, weights_path, batch_size_unused, flag_unused):
    """
    Load neural network from config and weights files.
    Matches: darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)
    
    Parameters match original darknet API (bytes encoded paths).
    """
    # Handle both bytes and string input
    if isinstance(config_path, bytes):
        config_path = config_path.decode("ascii")
    if isinstance(weights_path, bytes):
        weights_path = weights_path.decode("ascii")
    
    return _Network(config_path, weights_path)


def load_meta(meta_path):
    """
    Load metadata (class names) from .data file.
    Matches: darknet.load_meta(metaPath.encode("ascii"))
    """
    if isinstance(meta_path, bytes):
        meta_path = meta_path.decode("ascii")
    
    # Parse the .data file to find the names file
    names_file = None
    with open(meta_path, 'r') as f:
        for line in f:
            if line.strip().startswith('names'):
                names_file = line.split('=')[1].strip()
                break
    
    if names_file is None:
        raise ValueError(f"Could not find 'names' entry in {meta_path}")
    
    if not os.path.exists(names_file):
        # Try relative to meta_path directory
        meta_dir = os.path.dirname(meta_path)
        alt_path = os.path.join(meta_dir, names_file)
        if os.path.exists(alt_path):
            names_file = alt_path
        else:
            # Try common locations
            for try_path in [f"./data/{os.path.basename(names_file)}", 
                           f"../data/{os.path.basename(names_file)}",
                           names_file]:
                if os.path.exists(try_path):
                    names_file = try_path
                    break
    
    if not os.path.exists(names_file):
        raise ValueError(f"Names file not found: {names_file}")
    
    print(f"[OpenCV DNN] Loaded {names_file}")
    return _NetworkMeta(names_file)


def make_image(w, h, c):
    """
    Create a darknet-compatible image buffer.
    Matches: darknet.make_image(frame_width, frame_height, 3)
    """
    return _DarknetImage(w, h, c)


def copy_image_from_bytes(darknet_image, frame_bytes):
    """
    Copy frame bytes into darknet image structure.
    Matches: darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
    """
    # Reconstruct numpy array from bytes
    frame = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = frame.reshape((darknet_image.h, darknet_image.w, darknet_image.c))
    darknet_image.data = frame.copy()


def detect_image(network, meta, darknet_image, thresh=0.25, nms=0.45):
    """
    Run YOLO detection on an image.
    Matches: darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
    
    Returns list of tuples: [(class_name_bytes, confidence, (center_x, center_y, width, height)), ...]
    NOTE: class_name is returned as BYTES (b'person') to match original darknet API
    """
    if darknet_image.data is None:
        return []
    
    frame = darknet_image.data
    
    # Create blob from image (OpenCV DNN expects BGR, but our frame is RGB)
    # Convert RGB to BGR for OpenCV DNN
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    blob = cv2.dnn.blobFromImage(
        frame_bgr, 
        1/255.0,                          # Scale factor
        (network.width, network.height),   # Input size for network
        swapRB=True,                       # BGR -> RGB internally
        crop=False
    )
    
    network.net.setInput(blob)
    outputs = network.net.forward(network.output_layers)
    
    # Parse detections
    img_h, img_w = darknet_image.h, darknet_image.w
    
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = float(scores[class_id])
            
            if confidence > thresh:
                # YOLO returns center_x, center_y, width, height (normalized 0-1)
                center_x = detection[0] * img_w
                center_y = detection[1] * img_h
                w = detection[2] * img_w
                h = detection[3] * img_h
                
                boxes.append([center_x, center_y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression
    detections = []
    if len(boxes) > 0:
        # Convert to format needed by NMSBoxes (x, y, w, h where x,y is top-left)
        nms_boxes = []
        for box in boxes:
            cx, cy, w, h = box
            x = int(cx - w/2)
            y = int(cy - h/2)
            nms_boxes.append([x, y, int(w), int(h)])
        
        indices = cv2.dnn.NMSBoxes(nms_boxes, confidences, thresh, nms)
        
        # Handle both old and new OpenCV versions
        if len(indices) > 0:
            if isinstance(indices[0], (list, np.ndarray)):
                indices = [i[0] for i in indices]
            
            for i in indices:
                class_name = meta.names[class_ids[i]]
                confidence = confidences[i]
                cx, cy, w, h = boxes[i]
                
                # Return format matches original darknet:
                # (class_name_as_bytes, confidence_float, (center_x, center_y, width, height))
                detections.append((
                    class_name.encode('ascii'),  # bytes like b'person'
                    confidence,
                    (cx, cy, w, h)
                ))
    
    return detections


# ============================================================
# Utility function to check if GPU/CUDA is available
# ============================================================
def check_gpu_support():
    """Check and print GPU/CUDA support status for OpenCV DNN"""
    build_info = cv2.getBuildInformation()
    cuda_available = 'CUDA' in build_info and 'YES' in build_info.split('CUDA')[1][:50]
    print(f"OpenCV version: {cv2.__version__}")
    print(f"CUDA support in OpenCV: {'Yes' if cuda_available else 'No'}")
    print(f"Available backends: {cv2.dnn.getAvailableBackends()}")
    return cuda_available


if __name__ == "__main__":
    print("=" * 60)
    print("  darknet_opencv.py - OpenCV DNN Backend for YOLO")
    print("=" * 60)
    print()
    check_gpu_support()
    print()
    print("This module replaces the compiled darknet library.")
    print("Import it as: import darknet_opencv as darknet")
    print()
    print("Required files:")
    print("  - cfg/yolov4-tiny.cfg  (or yolov4.cfg)")
    print("  - yolov4-tiny.weights  (or yolov4.weights)")
    print("  - cfg/coco.data")
    print("  - data/coco.names")
