# S.H.A.D.Y — System Architecture

**Multi-Platform AI Surveillance System**  
**Owner:** Mohammad Sheriff Mehmood  
**Last Updated:** May 2026

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER (Platforms)                          │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ Android  │   iOS    │  Web App │  Drone   │  Desktop │  IP Camera/NVR  │
│  App     │   App    │ (React)  │  SDK     │  (Win/   │  (RTSP Direct)  │
│          │          │          │          │  Linux)  │                 │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┴────────┬────────┘
     │          │          │          │          │               │
     └──────────┴──────────┴──────────┴──────────┴───────────────┘
                                   │
                          ┌────────▼────────┐
                          │   API GATEWAY   │
                          │  (REST + WS)    │
                          │  Auth / Rate    │
                          │  Limit / Route  │
                          └────────┬────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
     ┌────────▼────────┐  ┌───────▼───────┐  ┌────────▼────────┐
     │  VIDEO INGESTION │  │  STREAM MGR   │  │   USER SERVICE  │
     │  SERVICE         │  │               │  │                 │
     │  - RTSP pull     │  │  - Session    │  │  - Auth (JWT)   │
     │  - Upload (MP4)  │  │  - Routing    │  │  - Profiles     │
     │  - YouTube       │  │  - Load       │  │  - Permissions  │
     │  - WebRTC        │  │    Balance    │  │  - Billing      │
     └────────┬────────┘  └───────┬───────┘  └─────────────────┘
              │                    │
              └────────┬───────────┘
                       │
              ┌────────▼────────────────────────────────────┐
              │           MESSAGE QUEUE (Redis/RabbitMQ)      │
              │   Frames distributed to available workers     │
              └────────┬────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
│ AI WORKER 1  │ │ WORKER 2 │ │ WORKER N    │
│              │ │          │ │             │
│ ┌──────────┐ │ │  (same)  │ │   (same)    │
│ │ YOLO     │ │ │          │ │             │
│ │ Engine   │ │ │          │ │             │
│ │(TensorRT)│ │ │          │ │             │
│ └──────────┘ │ │          │ │             │
│              │ │          │ │             │
│ Detections:  │ │          │ │             │
│ - Object     │ │          │ │             │
│ - Fall       │ │          │ │             │
│ - Crash      │ │          │ │             │
│ - Social Dist│ │          │ │             │
│ - Fire       │ │          │ │             │
│ - Weapon     │ │          │ │             │
│ - Intrusion  │ │          │ │             │
└───────┬──────┘ └────┬─────┘ └──────┬──────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
              ┌────────▼────────┐
              │  EVENT SERVICE  │
              │                 │
              │ - Incident DB   │
              │ - Alert Engine  │
              │ - Clip Recorder │
              └───┬─────────┬───┘
                  │         │
         ┌────────▼──┐  ┌──▼──────────┐
         │  ALERT    │  │  STORAGE    │
         │  SERVICE  │  │  SERVICE    │
         │           │  │             │
         │ - Email   │  │ - Video     │
         │ - SMS     │  │   Clips     │
         │ - WhatsApp│  │ - Snapshots │
         │ - Push    │  │ - Logs      │
         │   Notif   │  │ (S3/MinIO)  │
         └───────────┘  └─────────────┘
```

---

## How Each Platform Connects

### 1. Web App (React/Next.js)
```
Browser → API Gateway → Stream Manager → WebSocket → Live feed + annotations
                      → User Service → Auth
                      → Event Service → Dashboard data
```
- User opens dashboard, sees live camera feeds with detection overlay
- Receives real-time alerts via WebSocket

### 2. Android / iOS App
```
Mobile App → API Gateway (REST + WebSocket)
           → Push Notification Service (Firebase FCM / APNs)
```
- View live feeds, receive push alerts
- Upload video from phone camera for analysis
- **Key:** App is a thin client — all AI runs on server

### 3. Drone
```
Drone Camera → RTSP/WebRTC → Video Ingestion Service → AI Workers
                            ← Annotated stream back to ground station
```
- Drone streams live video to server
- Server processes and sends back annotated feed + alerts
- For offline: **Edge AI** on drone (Jetson Nano mounted on drone)

### 4. IP Cameras / NVR
```
Camera (RTSP) → Video Ingestion Service (pulls stream) → AI Workers
             ← No return needed (camera is output-only)
             → Alerts go to dashboard/phone
```
- Most common deployment — existing CCTV systems
- S.H.A.D.Y sits between cameras and monitoring station

### 5. Desktop App
```
Desktop App → Local AI Engine (GPU) → Direct camera access
            → Syncs incidents to cloud
```
- For customers who want on-premise (no internet needed)
- Runs the full AI engine locally with GPU

---

## Core Components (What to Build)

### 1. AI Engine (The Brain) — `shady-engine`
```
shady-engine/
├── models/
│   ├── yolov8n.pt          # Nano (fastest)
│   ├── yolov8s.pt          # Small (balanced)
│   ├── fire_smoke.pt       # Custom fire model
│   └── weapon.pt           # Custom weapon model
├── detectors/
│   ├── base_detector.py    # Abstract class
│   ├── object_detector.py
│   ├── fall_detector.py
│   ├── crash_detector.py
│   ├── fire_detector.py
│   ├── intrusion_detector.py
│   └── weapon_detector.py
├── engine.py               # Main pipeline: frame → detect → annotate → event
├── tracker.py              # Object tracking (DeepSORT/ByteTrack)
└── config.yaml             # Model configs, thresholds, zones
```

**Key Design:** Each detector is a plugin. Enable/disable per camera.

### 2. API Server — `shady-api`
```
shady-api/
├── routes/
│   ├── auth.py             # Login, register, JWT
│   ├── cameras.py          # CRUD cameras, start/stop
│   ├── detections.py       # Configure which detections per camera
│   ├── events.py           # Get incidents, clips, reports
│   ├── alerts.py           # Configure alert rules
│   └── stream.py           # WebSocket live feed
├── models/                 # Database models
├── services/               # Business logic
└── main.py                 # FastAPI app
```

### 3. Video Ingestion — `shady-ingest`
```
Handles:
- RTSP pull (IP cameras)
- WebRTC receive (drones, mobile)
- File upload (MP4, recorded videos)
- YouTube/URL extraction (demo/testing)
```

### 4. Alert Service — `shady-alerts`
```
Handles:
- Email (SMTP)
- SMS (Twilio)
- WhatsApp (Twilio/Meta Business API)
- Push Notification (Firebase FCM)
- Webhook (for 3rd party integration)
```

### 5. Storage — `shady-storage`
```
Handles:
- Incident video clips (10s before + 10s after)
- Snapshot images
- Audit logs
- Model files
```

---

## Deployment Options

### Option A: Cloud (SaaS)
```
┌─────────────────────────────────────────┐
│           AWS / Azure / GCP             │
│                                         │
│  ┌─────────┐  ┌──────────┐  ┌───────┐  │
│  │ API     │  │ AI Worker│  │  DB   │  │
│  │ (ECS)   │  │ (GPU EC2)│  │(RDS)  │  │
│  └─────────┘  └──────────┘  └───────┘  │
│  ┌─────────┐  ┌──────────┐  ┌───────┐  │
│  │ Redis   │  │   S3     │  │ CDN   │  │
│  │ (Queue) │  │ (Clips)  │  │(Stream)│  │
│  └─────────┘  └──────────┘  └───────┘  │
└─────────────────────────────────────────┘
```
- Best for: Multiple customers, scale on demand
- Cost: ₹50K-2L/month depending on cameras

### Option B: On-Premise (Edge)
```
┌──────────────────────────────┐
│  Customer Site               │
│  ┌────────────────────────┐  │
│  │  Edge Box              │  │
│  │  (Jetson Orin / PC+GPU)│  │
│  │                        │  │
│  │  All services run      │  │
│  │  locally in Docker     │  │
│  │                        │  │
│  │  Cameras ──→ Engine    │  │
│  │  Engine ──→ Alerts     │  │
│  └────────────────────────┘  │
│           │                  │
│           │ (Optional sync)  │
│           ▼                  │
│    Cloud Dashboard           │
└──────────────────────────────┘
```
- Best for: Privacy-sensitive customers (hospitals, military)
- Hardware: Jetson Orin (₹40K) or PC with GPU

### Option C: Hybrid
- AI runs on-premise (low latency)
- Dashboard + alerts run on cloud (accessible anywhere)
- Best of both worlds

---

## Communication Protocols

| Platform | Protocol | Why |
|----------|----------|-----|
| Web Dashboard | WebSocket | Real-time annotated frames |
| Mobile App | REST + Push (FCM) | Battery efficient, alerts when app closed |
| IP Camera → Server | RTSP | Industry standard for cameras |
| Drone → Server | WebRTC | Low latency, works over 4G/5G |
| Desktop App | Local (no network) | Direct GPU access, fastest |
| Server → Server | gRPC | Fast internal communication |

---

## Database Schema (Core)

```sql
-- Users & Auth
users (id, name, email, password_hash, role, org_id)
organizations (id, name, plan, camera_limit)

-- Cameras
cameras (id, org_id, name, rtsp_url, location, status, enabled_detections[])

-- Events/Incidents
events (id, camera_id, type, confidence, timestamp, clip_url, snapshot_url, acknowledged)

-- Alert Rules
alert_rules (id, org_id, event_type, channel, recipients[], cooldown_seconds)

-- Zones (for intrusion detection)
zones (id, camera_id, name, polygon_points[], zone_type)
```

---

## API Endpoints (Core)

```
POST   /auth/login              → JWT token
POST   /auth/register           → Create account

GET    /cameras                 → List all cameras
POST   /cameras                 → Add camera (RTSP URL, name)
PUT    /cameras/:id/detections  → Enable/disable detection types
DELETE /cameras/:id             → Remove camera

GET    /events                  → List incidents (filter by type, date, camera)
GET    /events/:id/clip         → Download video clip
POST   /events/:id/acknowledge  → Mark as reviewed

WS     /stream/:camera_id      → Live annotated video feed (WebSocket)

GET    /analytics/summary       → Dashboard stats
GET    /analytics/heatmap       → Detection heatmap data
```

---

## Implementation Priority

| Step | What | Effort | Impact |
|------|------|--------|--------|
| 1 | Refactor AI Engine into plugin-based detector system | 1 week | Foundation for everything |
| 2 | Build FastAPI server with camera CRUD + WebSocket stream | 1 week | Replace Flask |
| 3 | Add RTSP camera support | 2 days | Real camera deployment |
| 4 | Build React dashboard (live view + events list) | 2 weeks | Visual product |
| 5 | Add alert service (Email + WhatsApp) | 3 days | Core value prop |
| 6 | Docker-compose for easy deployment | 2 days | Deployable anywhere |
| 7 | Mobile app (React Native) | 2-3 weeks | Android + iOS together |
| 8 | Edge deployment (Jetson) | 1 week | On-premise option |

---

## Tech Stack (Final)

| Layer | Technology | Why |
|-------|-----------|-----|
| AI Engine | YOLOv8 + TensorRT | Best speed/accuracy |
| Tracking | ByteTrack | Fast multi-object tracking |
| API | FastAPI (Python) | Fast, async, auto-docs |
| Database | PostgreSQL | Reliable, scalable |
| Cache/Queue | Redis | Fast pub/sub + caching |
| Storage | MinIO (self-hosted S3) | Free, S3-compatible |
| Web Frontend | React + TailwindCSS | Modern, fast |
| Mobile | React Native | One codebase, both platforms |
| Deployment | Docker + Docker Compose | Easy setup anywhere |
| Monitoring | Grafana + Prometheus | System health |
| CI/CD | GitHub Actions | Auto deploy |

---

## Single Codebase Strategy

The key insight: **ONE AI engine, MANY interfaces**

```
                    ┌──────────────────┐
                    │   shady-engine   │
                    │   (Python pkg)   │
                    │                  │
                    │  - All models    │
                    │  - All detectors │
                    │  - All logic     │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───┐  ┌──────▼─────┐  ┌────▼────────┐
     │ shady-api  │  │ shady-edge │  │ shady-cli   │
     │ (Cloud)    │  │ (Jetson)   │  │ (Desktop)   │
     └────────────┘  └────────────┘  └─────────────┘
```

- `shady-engine` is a Python package used by ALL deployment targets
- Cloud API imports it, Edge box imports it, Desktop app imports it
- Mobile/Web are just UIs that talk to whichever backend is running

---

## Notes

- Start with monolith (all in one server), split into microservices only when needed
- Use Docker from day 1 — makes deployment consistent everywhere
- Every new detection module = just a new file in `detectors/` folder
- Keep config-driven: which detections run on which camera = database config, not code change
