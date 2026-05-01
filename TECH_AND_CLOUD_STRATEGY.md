# S.H.A.D.Y — Technology & Cloud Strategy

**Multi-Platform UI + Free Cloud Services Guide**  
**Owner:** Mohammad Sheriff Mehmood  
**Last Updated:** May 2026

---

## 🖥️ PART 1: UI/Frontend Technology per Platform

---

### Decision Matrix

| Platform | Technology | Why This? | Alternative |
|----------|-----------|-----------|-------------|
| **Web Dashboard** | React + TailwindCSS | Industry standard, huge ecosystem, fast | Next.js (React + SSR) |
| **Android App** | React Native | Same codebase for iOS, JavaScript | Kotlin (native, harder) |
| **iOS App** | React Native | One codebase = both platforms done | Swift (native, harder) |
| **Desktop (Win/Linux/Mac)** | Electron + React | Reuse web code as desktop app | PyQt (Python native) |
| **Drone Ground Station** | React (Web) + WebRTC | Drone streams to browser, no app needed | Custom C++ (complex) |
| **Camera Config Panel** | React (Web) | Configure cameras from any browser | — |
| **Admin Panel** | React + Ant Design | Pre-built admin components, tables, charts | — |

---

### Why React Native for Mobile (Not Flutter/Native)?

```
┌──────────────────────────────────────────────────────────────┐
│                    ONE CODEBASE STRATEGY                       │
│                                                               │
│   React (Web)  ──── Shared Logic ────  React Native (Mobile) │
│                         │                                     │
│                    Same language (JS/TS)                       │
│                    Same state management                       │
│                    Same API calls                              │
│                    Same developers                             │
│                                                               │
│   Result: 1 team builds Web + Android + iOS + Desktop         │
└──────────────────────────────────────────────────────────────┘
```

| Approach | Languages Needed | Teams Needed | Time |
|----------|-----------------|--------------|------|
| Native (Kotlin + Swift + React) | 3 | 3 | 3x |
| Flutter + React | 2 (Dart + JS) | 2 | 2x |
| **React + React Native** | **1 (JavaScript)** | **1** | **1x** |

---

### Platform-Specific Details

#### Web Dashboard (React)
```
Tech Stack:
├── React 18+ (UI framework)
├── TailwindCSS (styling - fast, utility-first)
├── React Query / TanStack Query (API data fetching)
├── Zustand (lightweight state management)
├── Recharts (charts for analytics)
├── React Player (video streaming display)
└── Socket.io-client (real-time WebSocket)

Key Pages:
├── /login
├── /dashboard (overview, stats, alerts)
├── /cameras (list, add, configure)
├── /cameras/:id/live (live feed with detection overlay)
├── /events (incident history with filters)
├── /events/:id (clip playback, details)
├── /analytics (heatmaps, trends, reports)
└── /settings (alerts, zones, users)
```

#### Mobile App (React Native)
```
Tech Stack:
├── React Native 0.73+
├── Expo (easier builds, OTA updates)
├── React Navigation (screen navigation)
├── React Native Video (live stream playback)
├── Firebase Cloud Messaging (push notifications)
├── AsyncStorage (local settings cache)
└── Socket.io-client (real-time alerts)

Key Screens:
├── Login
├── Dashboard (camera grid, alert count)
├── Camera Live View (tap to see detection feed)
├── Alerts (push notification list)
├── Event Detail (play clip, acknowledge)
└── Settings (notification preferences)
```

#### Drone Interface
```
┌─────────────────────────────────────────────────┐
│  Drone doesn't need a custom app!               │
│                                                  │
│  Drone Camera → RTSP/WebRTC → Our Server        │
│  Ground Station = Same Web Dashboard            │
│                                                  │
│  For offline/edge:                              │
│  Drone + Jetson Nano → onboard detection        │
│  Results sent when back in range                │
└─────────────────────────────────────────────────┘

Drone SDKs (if needed later):
├── DJI Mobile SDK (React Native plugin exists)
├── MAVLink protocol (for custom drones)
└── WebRTC for live streaming over 4G/5G
```

#### Camera/NVR Interface
```
No UI needed on camera side!

Camera → RTSP stream → Our server pulls it
Configuration = done from Web Dashboard

Supported protocols:
├── RTSP (most IP cameras)
├── ONVIF (camera discovery + control)
├── RTMP (some Chinese cameras)
└── HTTP MJPEG (basic webcams)
```

---

## ☁️ PART 2: Free Cloud Services

---

### AWS Free Tier

| Service | Free Amount | Use For |
|---------|------------|---------|
| **EC2** | 750 hrs/month (t2.micro) for 12 months | API server, small workloads |
| **S3** | 5 GB storage + 20K GET, 2K PUT | Video clips, snapshots |
| **RDS** | 750 hrs/month (db.t2.micro) + 20 GB | PostgreSQL database |
| **Lambda** | 1M requests + 400K GB-seconds/month | Alert triggers, webhooks |
| **SNS** | 1M publishes, 1K emails | Push notifications |
| **SES** | 62K emails/month (from EC2) | Email alerts |
| **CloudWatch** | 10 custom metrics, 5 GB logs | Monitoring |
| **ECR** | 500 MB storage | Docker image registry |
| **SageMaker** | 250 hrs (ml.t2.medium) for 2 months | Model training |
| **Rekognition** | 5K images/month for 12 months | Compare with our custom model |

**⚠️ Limitation:** No free GPU instances. Training needs paid or alternatives.

---

### GCP Free Tier (BEST for AI/ML)

| Service | Free Amount | Use For |
|---------|------------|---------|
| **Compute Engine** | 1 e2-micro VM (always free!) | API server |
| **Cloud Storage** | 5 GB (always free) | Video clips |
| **Cloud SQL** | — (not free, use VM instead) | — |
| **Cloud Functions** | 2M invocations/month | Alert triggers |
| **Pub/Sub** | 10 GB/month | Message queue |
| **Firebase** | Spark plan (free): Auth, Firestore (1GB), FCM (unlimited push) | Mobile auth + push |
| **Colab** | FREE GPU (T4/A100) with limits | **MODEL TRAINING** ⭐ |
| **Vertex AI** | $300 credit (new accounts) | Training + deployment |
| **Cloud Build** | 120 min/day | CI/CD pipeline |
| **Artifact Registry** | 500 MB | Docker images |

**⭐ BEST FOR US:** Google Colab = Free GPU for training custom models!

---

### Azure Free Tier

| Service | Free Amount | Use For |
|---------|------------|---------|
| **Virtual Machines** | 750 hrs B1S (12 months) | API server |
| **Blob Storage** | 5 GB LRS (12 months) | Video clips |
| **Azure SQL** | 250 GB (always free, serverless) | Database |
| **Functions** | 1M executions/month (always free) | Alert triggers |
| **Notification Hubs** | 1M push/month (free tier) | Mobile push |
| **Cognitive Services** | 5K transactions/month | Vision API (compare) |
| **DevOps** | 5 users, unlimited repos | Code hosting + CI/CD |
| **$200 credit** | First 30 days | Try GPU VMs for training |

---

### Other Free Services (Platform Agnostic)

| Service | Free Tier | Use For |
|---------|-----------|---------|
| **Roboflow** | 10K images, 3 model versions | Dataset management + annotation |
| **Kaggle** | 30 hrs/week GPU (P100) | **MODEL TRAINING** ⭐ |
| **Google Colab** | ~12 hrs/session GPU (T4) | **MODEL TRAINING** ⭐ |
| **Lightning.ai** | 22 hrs/month GPU | Model training |
| **Hugging Face** | Free model hosting (CPU inference) | Model serving (small scale) |
| **Render** | Free web service (750 hrs/month) | API hosting (no GPU) |
| **Railway** | $5 credit/month | Small deployments |
| **Vercel** | Unlimited (frontend) | React web dashboard hosting |
| **Netlify** | Unlimited (frontend) | Alternative frontend hosting |
| **Supabase** | Free: 500MB DB, 1GB storage, 50K auth | Database + Auth |
| **Firebase** | Free: Auth, FCM, 1GB Firestore | Mobile backend |
| **Twilio** | $15 trial credit | SMS/WhatsApp alerts (testing) |
| **GitHub Actions** | 2000 min/month (public repos) | CI/CD |
| **Docker Hub** | Unlimited public images | Container registry |

---

## 🏋️ PART 3: Model Training Strategy (FREE)

---

### Where to Train (Free GPU)

```
┌─────────────────────────────────────────────────────────────┐
│                FREE GPU TRAINING OPTIONS                      │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Platform    │  GPU         │  Time Limit  │  Storage       │
├──────────────┼──────────────┼──────────────┼────────────────┤
│  Colab Free  │  T4 (16GB)  │  ~12 hrs     │  15 GB (temp)  │
│  Colab Pro   │  A100/V100  │  24 hrs      │  (₹750/month)  │
│  Kaggle      │  P100/T4x2  │  30 hrs/week │  20 GB         │
│  Lightning   │  T4          │  22 hrs/mon  │  —             │
│  Local RTX   │  4060 (8GB) │  Unlimited   │  Unlimited     │
│  4060                                                        │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

### Training Workflow

```
Step 1: Collect Dataset
    └── Roboflow (annotate + augment + export)

Step 2: Train Model
    └── Google Colab / Kaggle (free GPU)
    └── Command: yolo train data=custom.yaml model=yolov8n.pt epochs=100

Step 3: Export Model
    └── Export to ONNX / TensorRT for fast inference

Step 4: Deploy
    └── Download trained .pt file → put in shady-engine/models/

Step 5: Test
    └── Run locally on RTX 4060 (your machine)
```

### Datasets for Future Detections

| Detection | Free Dataset | Size | Source |
|-----------|-------------|------|--------|
| Fire & Smoke | D-Fire Dataset | 21K images | Roboflow |
| Weapons | Pistol Detection | 3K images | Roboflow |
| PPE (Helmet/Vest) | Construction Safety | 10K images | Roboflow |
| Face Mask | Face Mask Detection | 7K images | Kaggle |
| License Plate | ANPR Dataset | 5K images | Roboflow |
| Violence | UCF Crime Dataset | 1900 videos | UCF |

---

## 💰 PART 4: Cost Comparison (When Scaling)

---

### Hosting Cost Estimate (10 cameras, 1 customer)

| Option | Monthly Cost | Notes |
|--------|-------------|-------|
| **Free Tier Only** | ₹0 | Limited to 1 VM, no GPU inference |
| **Budget Cloud** | ₹3,000-5,000 | Small GPU VM (spot instances) |
| **Production Cloud** | ₹15,000-30,000 | Dedicated GPU + DB + storage |
| **On-Premise (One-time)** | ₹40,000-80,000 | Jetson Orin / PC with GPU |

### Recommended Start (₹0 Budget)

```
Phase 1 (NOW - Free):
├── API Server      → GCP e2-micro (always free)
├── Database        → Supabase free (500MB PostgreSQL)
├── Frontend        → Vercel (free React hosting)
├── Push Notif      → Firebase FCM (free unlimited)
├── Email Alerts    → Gmail SMTP (500/day free)
├── Model Training  → Google Colab + Kaggle (free GPU)
├── Code Hosting    → GitHub (free)
├── CI/CD           → GitHub Actions (free)
├── Docker Registry → Docker Hub (free public)
└── AI Inference    → Your RTX 4060 locally (free!)

Phase 2 (First Customer - ₹3-5K/month):
├── Upgrade VM      → GCP/AWS small GPU instance
├── Add storage     → S3/GCS for video clips
└── Add database    → Cloud SQL (managed PostgreSQL)

Phase 3 (10+ Customers - ₹15-30K/month):
├── Scale VMs       → Multiple GPU workers
├── Add CDN         → CloudFront for video streaming
├── Add monitoring  → Grafana Cloud (free tier)
└── Add load balancer
```

---

## 📱 PART 5: Platform Summary Table

| Platform | UI Tech | Backend Connection | Install Size | Offline? |
|----------|---------|-------------------|-------------|----------|
| Web | React + Tailwind | REST + WebSocket | 0 (browser) | No |
| Android | React Native | REST + FCM Push | ~30 MB | Partial |
| iOS | React Native | REST + APNs Push | ~30 MB | Partial |
| Desktop | Electron + React | Local engine + sync | ~200 MB | Yes |
| Drone | Web (browser) | WebRTC stream | 0 | No* |
| Camera | None (headless) | RTSP pull by server | 0 | — |

*Drone with Jetson = offline capable

---

## 🎯 FINAL RECOMMENDATION: Start Here

```
Week 1-2: ┌─────────────────────────────────────┐
           │ 1. Refactor AI engine (plugin-based) │
           │ 2. Build FastAPI backend              │
           │ 3. Add RTSP camera support            │
           └─────────────────────────────────────┘

Week 3-4: ┌─────────────────────────────────────┐
           │ 4. React dashboard (live view)        │
           │ 5. Deploy API on GCP free tier        │
           │ 6. Deploy frontend on Vercel          │
           └─────────────────────────────────────┘

Week 5-6: ┌─────────────────────────────────────┐
           │ 7. Firebase push + alert service      │
           │ 8. React Native app (basic)           │
           │ 9. Train fire detection on Colab      │
           └─────────────────────────────────────┘

Week 7-8: ┌─────────────────────────────────────┐
           │ 10. Add 1 real camera (pilot test)    │
           │ 11. Polish UI, fix bugs               │
           │ 12. Demo to potential customer         │
           └─────────────────────────────────────┘
```

---

## Notes

- **React everywhere** = 1 language (JavaScript/TypeScript), 1 team, all platforms
- **Free GPU training** via Colab/Kaggle is enough for custom models
- **GCP always-free tier** is the best for hosting (e2-micro never expires)
- **Your RTX 4060** is your development + inference machine (no cloud GPU needed for testing)
- **Firebase FCM** = unlimited free push notifications (Android + iOS)
- Start free, scale only when you have paying customers
