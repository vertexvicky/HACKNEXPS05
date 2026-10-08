# DETAILED ARCHITECTURAL PLANNING: REAL MULTI-STREAM GENAI VIDEO INTELLIGENCE

## 1. Executive Summary & Vision

This system replaces mock/synthetic heuristics with a **True GenAI Multimodal Intelligence Engine** running on local **CUDA (NVIDIA GeForce RTX 3050)** for local vision segmentation/tracking, integrated with the **Google GenAI API** (`google-genai` SDK) for multimodal event understanding and conversational reasoning.

---

## 2. Core Architecture Pipeline

```
                    ┌──────────────────────────────────────────────┐
                    │          RAW CCTV VIDEO STREAMS              │
                    │         (cam1.mp4, cam2.mp4, etc.)           │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │    2 FPS OPENCV FRAME SAMPLER WITH HUD       │
                    │  - 1 frame every 0.5 seconds                 │
                    │  - Burns high-contrast timestamp & camera HUD │
                    └──────────────────────┬───────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
      ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
      │   YOLOv8 DETECTOR ON CUDA       │   │   VEHICLE & OBJECT TRACKING     │
      │   - Person & Vehicle bboxes     │   │   - License plates, cars, vans  │
      │   - Backpacks, bags, suitcases  │   │   - Spatial containment: CARRIES│
      └─────────────────┬───────────────┘   └─────────────────────────────────┘
                        │
                        ▼
      ┌──────────────────────────────────────────────────────────────┐
      │     PERSON IDENTIFICATION, FACE BANK & OUTFIT RE-ID ENGINE   │
      │                                                              │
      │  1. Face Detection & Crop                                    │
      │     └─► Feature extraction (512-d normalized embeddings)     │
      │                                                              │
      │  2. Multi-Embedding Identity Bank                            │
      │     └─► Compare against DB: [v_1, v_2, ..., v_n] per person   │
      │     └─► Match found -> Link ID & append updated embedding    │
      │     └─► No match -> Register new entity 'unknown_XXX'        │
      │                                                              │
      │  3. Outfit & Body Re-ID Fallback                             │
      │     └─► When face is occluded/turned away:                   │
      │         Compute dense torso/outfit color & feature vector    │
      │                                                              │
      │  4. Occlusion Waitlist Buffer                                │
      │     └─► Buffer unconfirmed tracks across consecutive frames  │
      │     └─► When face reappears: link coords + outfit -> update  │
      └──────────────────────────────┬───────────────────────────────┘
                                     │
                                     ▼
      ┌──────────────────────────────────────────────────────────────┐
      │      LOCAL STORAGE: SQLITE (events.db) + VECTOR STORE        │
      │  - Events table: timestamp, camera_id, bbox, entity_id, crop │
      │  - Relations table: (subject, CARRIES/ENTERS_ZONE, object)   │
      │  - Multi-layer vectors: Face, Outfit, Action, Captions       │
      │  - Persistent Memory: knowledge.json (Clarify-Once)          │
      └──────────────────────────────┬───────────────────────────────┘
                                     │
                                     ▼
      ┌──────────────────────────────────────────────────────────────┐
      │          GENAI MULTIMODAL REASONER & CHAT ENGINE             │
      │                (Google GenAI API / gemma-4-31b-it)           │
      │                                                              │
      │  - Formats retrieved 2 FPS clips with rendered time HUD      │
      │  - Ingests structured multi-camera event timeline records    │
      │  - Executes deep multimodal reasoning & chain-of-thought     │
      │  - Generates conversational answers with grounded bboxes     │
      └──────────────────────────────────────────────────────────────┘
```

---

## 3. Person & Facial Recognition Deep Dive

### A. Multi-Embedding Identity Bank
Human facial features vary across camera angles, head pose, lighting, and expressions. A single static embedding leads to false negatives.
* **Architecture:** Every registered entity ($P_i$) maintains an embedding bank:
  $$\mathcal{B}(P_i) = \{\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_k\}$$
* **Matching Metric:**
  $$\text{Sim}(f, P_i) = \max_{j=1..k} \frac{f \cdot \mathbf{e}_j}{\|f\| \|\mathbf{e}_j\|}$$
* If $\text{Sim}(f, P_i) \ge \tau_{\text{face}}$ ($0.65$), the frame is attributed to $P_i$, and $f$ is added to $\mathcal{B}(P_i)$ (capped at recent $M=20$ exemplars).
* If no entity matches above threshold, a new identity cluster `unknown_001`, `unknown_002` is initialized.

### B. Outfit / Body Re-ID with Occlusion Waitlist Buffer
When a subject turns their back or walks under bad lighting:
1. **Outfit Feature Extraction:** Deep torso/clothing embedding vector $\mathbf{o}_t$ is computed from the lower 75% body crop on CUDA.
2. **Spatial-Temporal Tracking:** The spatial trajectory $[x_1, y_1, x_2, y_2]_t$ is propagated via velocity estimation.
3. **Waitlist Buffer:** If face is lost, the track enters a *Waitlist State* with active outfit signature $\mathbf{o}_t$ for up to $N$ frames ($8$ seconds at 2 FPS).
4. **Retroactive Resolution:** As soon as the subject turns towards a camera and their face is identified, the entire waitlisted trajectory is retroactively tagged with the confirmed identity.

---

## 4. Vehicle, Object & Spatial Relational Tracking

1. **Object Containment (`CARRIES`):**
   - If a detected object (backpack, handbag, box) bounding box is spatially contained within a person's outer bounding box for $\ge 3$ consecutive sampled frames:
     $$\text{Emit: } (P_i, \text{CARRIES}, \text{object\_class}, t, \text{camera\_id})$$
2. **Zone Crossings (`ENTERS_ZONE`):**
   - Centroid intersection with defined camera regions (e.g., Gate 1, Parking Bay 3, Corridor Sector).

---

## 5. GenAI Multimodal Prompting & Reasoning Engine

### System Prompt Design for GenAI Model:
```
You are an expert Multi-Stream Video Surveillance Reasoning Intelligence.
You receive:
1. Structured spatial-temporal video logs extracted from multi-camera footage at 2 FPS.
2. Grounded visual frames with burned timestamp HUDs and bounding boxes.
3. Persistent entity and spatial knowledge mappings.

Rules:
1. Every claim must reference the exact Camera ID and timestamp range (e.g., Cam 1 at t=14.0s).
2. Distinguish confirmed identities (from facial recognition) from predicted identities (from outfit/trajectory tracking).
3. If an unmapped location or alias is queried, explicitly request clarification.
4. Output structured step-by-step reasoning followed by the direct conversational answer.
```

---

## 6. Implementation Checklist & File Mapping

| File | Role |
|---|---|
| `base/backend/models.py` | PyTorch CUDA vision engine (YOLOv8 + Deep Face & Outfit embeddings) |
| `base/backend/face_engine.py` | Multi-embedding Face Bank, Unknown cluster manager, and Outfit Waitlist Buffer |
| `base/backend/indexer.py` | 2 FPS OpenCV decoder with burned time HUD, YOLO detection & relational extraction |
| `base/backend/query_engine.py` | Google GenAI SDK integration with the requested model & conversational reasoner |
| `base/backend/database.py` | SQLite schema with multi-embedding identity tables, events, and relations |
| `base/backend/memory.py` | Persistent `knowledge.json` managing clarify-once mappings |
| `base/backend/app.py` | FastAPI application serving real upload endpoints, video streaming, and chat API |
| `base/frontend/index.html` | Clean UI with simultaneous Dark and Light themes, video uploader, and AI chat |
| `base/frontend/css/style.css` | Complete Dark/Light styling tokens, chat bubbles, and evidence cards |
| `base/frontend/js/app.js` | Chat client, theme switcher, real-time upload progress, and clarify prompt handler |
