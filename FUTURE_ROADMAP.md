# S.H.A.D.Y — Future Roadmap & Product Plan

**Owner:** Mohammad Sheriff Mehmood  
**Email:** mdsheriff2702@gmail.com  
**Last Updated:** May 2026

---

## Current Modules (v1.0)

| Module | Status | Model |
|--------|--------|-------|
| Object Detection | ✅ Working | YOLOv4-tiny |
| Fall Detection | ✅ Working | YOLOv4-tiny |
| Social Distance Detection | ✅ Working | YOLOv4-tiny |
| Vehicle Crash Detection | ✅ Working | YOLOv4-tiny |

---

## Phase 1: New Detection Modules

### Priority 1 — Easy to Add (Use existing COCO model + new logic)

| Detection | Logic | Use Case | Effort |
|-----------|-------|----------|--------|
| **Crowd Counting** | Count `person` class per frame, threshold alert | Events, malls, stations | Low |
| **Intrusion Detection** | Person enters a user-defined ROI zone | Restricted areas, night security | Low |
| **Abandoned Object** | Object appears and stays static for X seconds | Airports, train stations | Medium |

### Priority 2 — Need Custom Trained Model

| Detection | How | Use Case | Effort |
|-----------|-----|----------|--------|
| **Fire & Smoke Detection** | Custom YOLOv8 model (datasets available on Roboflow) | Building safety, factories, kitchens | Medium |
| **Weapon Detection** | Custom YOLO model (gun, knife datasets) | Schools, public places, ATMs, banks | Medium |
| **PPE Detection** | Helmet, vest, gloves detection | Construction sites, factories | Medium |
| **Face Mask Detection** | Custom YOLO (small dataset) | Hospitals, offices | Medium |

### Priority 3 — Advanced (Requires additional techniques)

| Detection | How | Use Case | Effort |
|-----------|-----|----------|--------|
| **Fight/Violence Detection** | Pose estimation + action recognition (MediaPipe/OpenPose) | Schools, prisons, public areas | High |
| **Drowning Detection** | Person tracking in pool + motion analysis | Swimming pools, beaches | High |
| **Wrong-Way Driver** | Vehicle direction tracking with optical flow | Highways, one-way roads | High |
| **Speed Estimation** | Vehicle tracking + camera calibration | Traffic enforcement | High |
| **License Plate Recognition (ANPR)** | OCR on detected car region (EasyOCR/Tesseract) | Parking, toll, traffic | High |

---

## Phase 2: Technical Upgrades

| Upgrade | Why | How |
|---------|-----|-----|
| **YOLOv4-tiny → YOLOv8/v11** | 2-3x better accuracy, same speed | `pip install ultralytics`, swap model loading |
| **GPU Inference (TensorRT)** | 10x faster on RTX 4060 | Export ONNX → TensorRT engine |
| **RTSP Live Camera** | Real surveillance cameras | OpenCV `cv2.VideoCapture("rtsp://...")` |
| **Multi-Camera** | Process 4-8 feeds simultaneously | Threading/multiprocessing per stream |
| **Edge Deployment** | Run on Jetson Nano/Orin at customer site | TensorRT + Docker container |

---

## Phase 3: Product Features

| Feature | Description | Tech Stack |
|---------|-------------|------------|
| **Real-time Alerts** | WhatsApp/SMS/Email when incident detected | Twilio API / SMTP |
| **Incident Recording** | Save 10s video clip of detected event | OpenCV VideoWriter + MinIO/S3 |
| **Dashboard** | Live camera status, incident count, heatmaps | React + FastAPI |
| **User Authentication** | Login system, roles (admin/viewer) | JWT + PostgreSQL |
| **Database** | Store all incidents with timestamp, camera ID, type | PostgreSQL + SQLAlchemy |
| **REST API** | Let other systems integrate with S.H.A.D.Y | FastAPI endpoints |
| **Mobile App** | Push notifications on incidents | React Native / Flutter |
| **Cloud Deployment** | AWS/Azure with GPU for customers without hardware | Docker + Kubernetes |
| **Analytics & Reports** | "Falls increased 30% this week in Ward B" | Grafana / custom charts |

---

## Phase 4: Market Verticals

| Market | Modules Used | Revenue Model |
|--------|-------------|---------------|
| **Elder Care / Hospitals** | Fall Detection + Intrusion | ₹2000-5000/camera/month |
| **Traffic Management** | Vehicle Crash + Speed + ANPR | Government contracts |
| **Workplace Safety** | PPE + Social Distance + Fire | Enterprise license |
| **Retail / Warehouse** | Object + Crowd + Intrusion | Per-store pricing |
| **Smart City** | All modules combined | Annual municipal contracts |
| **Schools / Campuses** | Weapon + Violence + Intrusion | Institutional license |
| **Banks / ATMs** | Weapon + Intrusion + Face | Per-branch SaaS |

---

## Recommended First Steps

1. **Pick ONE vertical** → Fall Detection for hospitals/old age homes (high urgency, less competition in India)
2. **Add RTSP camera support** → Replace YouTube with live camera feed
3. **Add WhatsApp alert** → Twilio API sends message when fall detected
4. **Get 1 pilot customer** → Free trial at local hospital/old age home
5. **Collect real data** → Improve model accuracy with actual footage
6. **Monetize** → ₹2000-5000/camera/month subscription

---

## Production Tech Stack (Target)

```
Frontend:   React / Next.js (Dashboard)
Backend:    FastAPI (replace Flask)
Inference:  TensorRT / ONNX Runtime (GPU)
Database:   PostgreSQL + Redis (caching)
Queue:      Celery + RabbitMQ (async processing)
Storage:    MinIO / AWS S3 (video clips)
Alerts:     Twilio (WhatsApp/SMS) + SMTP (Email)
Deploy:     Docker + Kubernetes
Monitoring: Grafana + Prometheus
```

---

## Resources & Datasets

| Detection | Dataset Source |
|-----------|---------------|
| Fire & Smoke | [Roboflow Fire Detection](https://universe.roboflow.com/search?q=fire) |
| Weapons | [Roboflow Weapon Detection](https://universe.roboflow.com/search?q=weapon) |
| PPE | [Roboflow PPE](https://universe.roboflow.com/search?q=ppe) |
| Face Mask | [Kaggle Face Mask](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) |
| Violence | [UCF Crime Dataset](https://www.crcv.ucf.edu/projects/real-world/) |
| License Plates | [OpenALPR datasets](https://github.com/openalpr/openalpr) |

---

## Notes

- Current model (YOLOv4-tiny) runs at ~15-20 FPS on CPU, ~100+ FPS on GPU
- For custom models, train using Ultralytics YOLOv8: `yolo train data=custom.yaml model=yolov8n.pt epochs=100`
- Keep the dual-backend architecture (config.py BACKEND switch) for flexibility
- All new modules should follow the same pattern: `New_Detection.py` + template page + Flask route
