# Video Intelligence & Temporal Forensics Workstation

An enterprise surveillance analytics and multimodal reasoning workstation. The system normalizes raw video footage to **1.0 FPS (Frame Per Second)**, indexes temporal frames, uploads the synchronized stream to Google GenAI API (`gemma-4-26b-a4b-it`), and exposes a corporate forensic console with real-time token streaming and frame-accurate timestamp synchronization.

---

## 1. System Architecture

```
                 RAW SURVEILLANCE VIDEO (MP4 / MOV / AVI)
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │         OPENCV 1.0 FPS TEMPORAL NORMALIZER             │
       │  • Samples strictly 1 frame per second (1.000 Hz)      │
       │  • Builds 1 FPS synchronized video (video_1fps.mp4)    │
       │  • Caches individual frame thumbnails (frame_XXXX.jpg) │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │             GOOGLE GENAI FILES & VLM INFERENCE         │
       │  • Uploads 1 FPS video via Google GenAI Files API      │
       │  • Model: gemma-4-26b-a4b-it                           │
       │  • Thinking level: HIGH | Web Search grounding         │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │         ENTERPRISE SURVEILLANCE WORKSTATION UI         │
       │  • Surveillance HUD (SMPTE timecode, REC, FPS, Clock)  │
       │  • 1.0 FPS Telemetry & Frame Inspector                 │
       │  • Transport controls (-1s, +1s, 0.5x-4x, scrubber)   │
       │  • Operator Console: Live chunk-by-chunk token stream  │
       │  • Clickable timestamps seeking original video & frame │
       │  • Simultaneous Dark & Light enterprise themes         │
       └────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
Karunya-05/
├── .env                         # Environment variables (API keys, ports)
├── .env.example                 # Environment template
├── .gitignore                   # Excludes caches, venv, secrets, uploads
├── README.md                    # System documentation and developer guide
├── yolov8n.pt                   # Local YOLOv8 weights (for local detection tasks)
└── base/
    ├── requirements.txt         # Python dependencies
    ├── run_demo.bat             # One-click Windows launch script
    ├── .env                     # Local environment file
    ├── backend/
    │   ├── app.py               # FastAPI server & GenAI streaming engine
    │   ├── uploads/             # Stores uploaded raw videos (.gitkeep)
    │   └── rendered/
    │       ├── frames/          # Extracted 1.0 FPS frame thumbnails (.gitkeep)
    │       └── video_1fps.mp4   # 1 FPS normalized video passed to API
    └── frontend/
        ├── index.html           # Surveillance workstation interface
        ├── css/
        │   └── style.css        # Enterprise Dark & Light theme stylesheets
        └── js/
            └── app.js           # Real-time SSE stream reader, player & telemetry sync
```

---

## 3. Setup & Installation

### Step 1: Clone or Navigate to Project
```bash
cd /d "c:\Users\KAVIN\Documents\Karunya-05\base"
```

### Step 2: Install Dependencies
Ensure Python 3.10+ is installed. Run the following command:
```bash
pip install -r requirements.txt
```
Or install individually:
```bash
pip install google-genai opencv-python fastapi uvicorn python-multipart pydantic
```

### Step 3: Configure Environment Variables
Copy `.env.example` to `.env` (or update existing `.env`):
```ini
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_NAME=gemma-4-26b-a4b-it
PORT=8000
HOST=127.0.0.1
```

---

## 4. Running the Application

### Option A: Using Windows Batch Script
Double-click or run:
```cmd
base\run_demo.bat
```

### Option B: Manual CLI
```bash
cd base
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 5. API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Ingests video file, extracts 1 FPS frames, uploads to Google GenAI API, and returns video duration and total frames. |
| `POST` | `/api/chat` | Accepts `{ "message": "query" }`, streams back token-by-token forensic response via Server-Sent Events (SSE). |
| `GET` | `/api/video` | Serves the original uploaded video with HTTP range headers for smooth timeline scrubbing. |
| `GET` | `/api/frames/{sec}` | Serves the extracted JPEG frame thumbnail for any given second index. |
| `GET` | `/api/status` | Returns system telemetry (active file ID, duration, frames count, model name). |

---

## 6. Future Developer Integration Roadmap

This codebase provides the foundational ingestion and reasoning baseline. The following integration hooks are prepared for expansion:

1. **Local Object & Vehicle Detector (CUDA YOLOv8)**:
   - Hook into `process_video_to_1fps` in [`base/backend/app.py`](file:///c:/Users/KAVIN/Documents/Karunya-05/base/backend/app.py) to run `yolov8n.pt` bounding box inference per second.
   - Extract vehicle license plates and route them to local OCR (`PaddleOCR` or `Tesseract`).

2. **Person Identification & Outfit Re-ID Bank**:
   - Crop person bounding boxes and compute normalized 512-d feature vectors for facial embeddings.
   - On occlusion or turned heads, calculate color histogram and outfit vectors to link tracking across consecutive 1 FPS frames.

3. **Multi-Stream Camera Ingestion**:
   - Expand `session_state` to support multiple camera IDs (`cam_01`, `cam_02`, etc.) and multi-video uploads.
   - Synchronize cross-camera SMPTE timecodes to track movement trajectories from gate to lobby.

4. **Clarify-Once Persistent Knowledge Graph**:
   - Store spatial zones (e.g., "Main Gate" = Cam 1 Bounding Box `[x1, y1, x2, y2]`) in an SQLite/JSON knowledge store.
   - When the user clarifies a referent once, persist it so the model references it in all future queries.
