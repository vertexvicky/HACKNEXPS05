# PDR_base — HNX26EPS05 MVP
## Product Design Reference · Base / Hackathon MVP

> **Core Architecture Principle:**
> Video is NEVER fed into the LLM at query time.
> Video is processed ONCE offline → embeddings + events extracted and stored.
> The LLM reasons over the retrieved embeddings and structured events — not raw video.
> This is what makes it scalable, fast, and capable of querying ACTIONS — not just appearances.

---

## 0. The Fundamental Problem With Naive Approaches

| Naive Approach | Why It Fails |
|----------------|--------------|
| Feed video directly to Gemini at query time | Hours of video = millions of tokens. Context window overflows. Re-processes everything on every query. |
| Just use CLIP frame embeddings | CLIP is per-frame, static appearance only. Cannot capture "carrying", "entering", "driving". No action understanding. |
| Keyword search on detection labels | "person carrying bag" ≠ "bag" AND "person". Requires relational understanding. |

**Our approach:** Build a rich **multi-layer index** ONCE offline. At query time — search the index, retrieve evidence, let LLM reason over the retrieved text + metadata (not the video).

---

## 1. What Gets Embedded Offline (Index Time)

We extract **4 types of embeddings** per video segment/event:

```
For each detected entity track:
  
  1. VISUAL EMBEDDING (appearance)
     → OpenCLIP ViT-B/32 on detection crop
     → Captures: color, shape, clothing, vehicle type
     → Answers: "red car", "person in blue jacket"

  2. ACTION/TEMPORAL EMBEDDING (what it's doing)
     → TC-CLIP or VideoMAE on 16-frame sliding window around entity
     → Captures: walking, running, carrying, sitting, entering
     → Answers: "person carrying a bag", "car reversing", "person running"

  3. RELATIONAL RECORD (who + what + with what)
     → Rule-based extraction from tracker outputs
     → Stored as typed tuples: (person_A, CARRIES, bag_B, cam1, t=123.4s)
     → Answers: "the bag that Santhya was carrying"

  4. NATURAL LANGUAGE DESCRIPTION (auto-captioned)
     → Small VLM (MiniCPM-V / LLaVA-1.5-7B) captions each event window
     → "A person in a blue jacket is walking through a gate carrying a black backpack"
     → This is what the LLM reads at query time — NOT the video
```

---

## 2. Architecture — Two Strict Phases

### Phase 1: OFFLINE INDEXING (runs once on the recorded videos)

```
cam1.mp4, cam2.mp4
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  FRAME EXTRACTOR                                            │
│  → 2 FPS baseline + shot-change keyframes                   │
│  → ~7,200 frames per hour per camera (16-frame window = 8s) │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼────────────────┐
         │             │                │
         ▼             ▼                ▼
   YOLO-World      RetinaFace      ByteTrack
   (open-vocab     (face detect)   (stable IDs
    detection)                      per camera)
         │             │                │
         └─────────────┼────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  PER-ENTITY PROCESSING      │
         │                             │
         │  ┌──────────────────────┐   │
         │  │ OpenCLIP Visual Emb  │   │  ← appearance
         │  └──────────────────────┘   │
         │  ┌──────────────────────┐   │
         │  │ TC-CLIP Action Emb   │   │  ← what it's DOING
         │  │ (16-frame window)    │   │
         │  └──────────────────────┘   │
         │  ┌──────────────────────┐   │
         │  │ ArcFace Face Emb     │   │  ← who it is
         │  └──────────────────────┘   │
         │  ┌──────────────────────┐   │
         │  │ PaddleOCR Plate Text │   │  ← vehicle ID
         │  └──────────────────────┘   │
         │  ┌──────────────────────┐   │
         │  │ LLaVA-1.5 Caption    │   │  ← text description
         │  │ (event description)  │   │    of the event window
         │  └──────────────────────┘   │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  ASSOCIATION ENGINE         │
         │  (spatial + temporal rules) │
         │                             │
         │  person ←→ object tracking: │
         │  If object moves with       │
         │  person for >15 frames      │
         │  → emit CARRIES tuple       │
         │                             │
         │  Zone crossings:            │
         │  bbox centroid crosses      │
         │  polygon → ENTERS_ZONE      │
         └─────────────┬───────────────┘
                       │
         ┌─────────────┼──────────────────────────────────┐
         │             │                                   │
         ▼             ▼                                   ▼
  ┌────────────┐ ┌────────────────┐              ┌──────────────────┐
  │  SQLite    │ │  ChromaDB      │              │  knowledge.json  │
  │  events.db │ │  Vector Store  │              │  Persistent KB   │
  │            │ │                │              │  (clarify-once)  │
  │  Structured│ │  4 collections:│              │                  │
  │  event log │ │  - visual_emb  │              │  Built/updated   │
  │  + relation│ │  - action_emb  │              │  at query time   │
  │  tuples    │ │  - face_emb    │              │  via user input  │
  │            │ │  - captions_emb│              │                  │
  └────────────┘ └────────────────┘              └──────────────────┘
```

### Phase 2: ONLINE QUERY (live at demo — no video touched)

```
User types: "Did a red car pass the main gate in the last hour?"
                       │
                       ▼
         ┌─────────────────────────────┐
         │  MEMORY RESOLVER            │
         │  knowledge.json lookup      │
         │  "main gate" → cam1         │
         │  If MISS → ask once → save  │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  LLM QUERY PLANNER          │
         │  (Gemini Flash)             │
         │  → slot-fill the query:     │
         │    entity=car               │
         │    color=red                │
         │    action=passing/moving    │
         │    location=cam1            │
         │    time=last_hour           │
         │  → decides WHICH indexes    │
         │    to search                │
         └─────────────┬───────────────┘
                       │
          ┌────────────┼──────────────────┐
          │            │                  │
          ▼            ▼                  ▼
   SQL Query     Vector Search      Vector Search
   events.db     visual_emb         action_emb
   WHERE         "red car"          "car driving
   class=car,    cosine sim         through gate"
   cam=cam1,
   time>T-3600
          │            │                  │
          └────────────┴──────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  EVIDENCE MERGER + RANKER   │
         │  Merge SQL + vector hits    │
         │  Deduplicate by event_id    │
         │  Rank by confidence + time  │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  LLM ANSWER SYNTHESIZER     │
         │  Input: top-3 event records │
         │  = {camera, timestamp,      │
         │     description text,       │
         │     relation tuples}        │
         │                             │
         │  LLM reads TEXT, not video  │
         │  → generates grounded answer│
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  CLIP RENDERER              │
         │  Extract ±5s clip from      │
         │  original video at event ts │
         │  Draw bbox overlay          │
         └─────────────────────────────┘
                       │
                       ▼
         Chat UI: Text answer + video clip + camera + timestamp
```

---

## 3. The Action Embedding — Why TC-CLIP

This is the key that allows querying **actions**, not just appearances.

OpenCLIP (image model) embeds: **"what things look like"**  
TC-CLIP / VideoMAE (temporal model) embeds: **"what things are doing"**

```python
# TC-CLIP: temporal contextualization across 16 frames
# Applied as a sliding window over each tracked entity's crop sequence

import torch
from tc_clip import TCCLIPModel  # or VideoMAE

model = TCCLIPModel.from_pretrained("naver-ai/tc-clip")

def embed_action_window(frame_crops: list):
    # frame_crops: list of 16 consecutive crops of one entity
    # Returns: 512-dim action embedding vector
    frames_tensor = preprocess_frames(frame_crops)  # [16, C, H, W]
    with torch.no_grad():
        action_emb = model.encode_video(frames_tensor)
    return action_emb.numpy()
```

**What this enables:**

| Query | Without Action Emb | With TC-CLIP Action Emb |
|-------|--------------------|--------------------------|
| "person carrying something" | Only visual match on "bag near person" | Matches the MOTION of carrying |
| "car reversing" | Cannot distinguish direction | Matches the backward motion pattern |
| "person entering a door" | Just detects person + door separately | Matches the entry motion sequence |
| "running" vs "walking" | Both look like "person" in CLIP | Distinguished by motion vector |

---

## 4. The Caption Layer — What the LLM Actually Reads

Instead of the LLM seeing raw video frames, it sees **auto-generated text descriptions** of events.

```python
# LLaVA-1.5-7B (or MiniCPM-V for lightweight) runs offline during indexing
# Input: keyframe + context (camera, time, track info)
# Output: rich natural language description

prompt = """
Camera: cam1 (Main Gate), Time: 09:14:23
Tracked entity: person (track_id=4, face_id=unknown_002)
Nearby tracked objects: bag (track_id=12), car (track_id=7)

Describe precisely what this person is doing in this moment.
Include: movement direction, posture, objects they're interacting with.
"""

# Output stored in SQLite + embedded into captions collection in ChromaDB:
# "A person is walking through a gate from outside. They are carrying a black 
#  backpack on their left shoulder and appear to be looking at their phone. 
#  A white SUV is parked nearby."
```

**At query time, the LLM receives ONLY:**
```
Event #42:
  Camera: cam1 | Timestamp: 09:14:23 | Duration: 8s
  Description: "A person is walking through a gate from outside carrying a 
                black backpack, looking at their phone. A white SUV is nearby."
  Relations: [person_4 CARRIES bag_12, person_4 ENTERS_ZONE main_gate]
  Clip: clips/cam1_091423.mp4
```

The LLM reads this text → reasons → generates the answer. **Never sees the raw video.**

---

## 5. SQLite Schema — The Structured Index

```sql
CREATE TABLE events (
    event_id     TEXT PRIMARY KEY,
    camera_id    TEXT NOT NULL,        -- "cam1", "cam2"
    timestamp    REAL NOT NULL,        -- seconds from video start
    track_id     INTEGER,
    entity_class TEXT,                 -- "person", "car", "bag"
    color        TEXT,                 -- dominant color bucket
    plate_text   TEXT,                 -- OCR if vehicle
    face_id      TEXT,                 -- "unknown_002" or "santhya"
    bbox         TEXT,                 -- JSON [x1,y1,x2,y2]
    description  TEXT,                 -- LLaVA auto-caption
    clip_path    TEXT,                 -- path to pre-extracted clip
    visual_emb_id TEXT,               -- FK to ChromaDB visual collection
    action_emb_id TEXT,               -- FK to ChromaDB action collection
    caption_emb_id TEXT               -- FK to ChromaDB caption collection
);

CREATE TABLE relations (
    relation_id  TEXT PRIMARY KEY,
    source_id    INTEGER,              -- track_id of subject
    target_id    INTEGER,              -- track_id of object
    relation_type TEXT,               -- CARRIES, ENTERS_ZONE, NEAR, DRIVES
    camera_id    TEXT,
    start_ts     REAL,
    end_ts       REAL,
    confidence   REAL
);
```

---

## 6. ChromaDB Collections (Vector Store)

```python
import chromadb
client = chromadb.PersistentClient(path="./db/chroma")

# 3 separate collections — each with different embedding types
visual_col  = client.get_or_create_collection("visual_embeddings")   # OpenCLIP crops
action_col  = client.get_or_create_collection("action_embeddings")   # TC-CLIP windows
caption_col = client.get_or_create_collection("caption_embeddings")  # CLIP-text on descriptions

# At index time: store with metadata for filtered search
visual_col.add(
    embeddings=[visual_emb.tolist()],
    metadatas=[{"camera": "cam1", "timestamp": 553.2, 
                "class": "person", "face_id": "unknown_002"}],
    ids=[event_id]
)

# At query time: semantic search + metadata filter
results = action_col.query(
    query_embeddings=[query_action_emb.tolist()],
    where={"camera": "cam1", "timestamp": {"$gt": T-3600}},
    n_results=5
)
```

---

## 7. Persistent Memory — knowledge.json

```json
{
  "locations": {
    "main gate": {"camera": "cam1", "zone": null},
    "parking": {"camera": "cam1", "zone": [[200,300],[800,300],[800,600],[200,600]]}
  },
  "persons": {
    "santhya": {"face_id": "face_cluster_023"}
  },
  "vehicles": {
    "delivery van": {"plate": "TN09AB1234"}
  }
}
```

**Clarify-once protocol:**
1. Every query → load `knowledge.json` first
2. Named entity/location not found → LLM asks user ONCE → save to disk immediately
3. App restart → load same file → zero re-asks

---

## 8. Model Stack

| Component | Model | Role | When It Runs |
|-----------|-------|------|--------------|
| Open-vocab detector | YOLO-World-S | Detect anything by text prompt | **Offline** |
| Multi-object tracker | ByteTrack | Stable track IDs per camera | **Offline** |
| Face detector | RetinaFace (InsightFace) | Small face detection | **Offline** |
| Face embedder | ArcFace R50 | 512-dim face vectors | **Offline** |
| Body Re-ID | OSNet-x0.25 | Body appearance fallback | **Offline** |
| Plate OCR | PaddleOCR | License plate text | **Offline** |
| Visual embedding | OpenCLIP ViT-B/32 | Appearance search | **Offline** |
| **Action embedding** | **TC-CLIP** | **What entities are DOING** | **Offline** |
| Event captioner | LLaVA-1.5-7B or MiniCPM-V | Auto-describe each event | **Offline** |
| Caption embedding | OpenCLIP text encoder | Embed descriptions | **Offline** |
| LLM query planner | Gemini Flash | Parse query, plan retrieval | **Online** |
| LLM answer synth | Gemini Flash | Reason over retrieved events | **Online** |

> **The LLM (Gemini) only runs at query time and only reads TEXT — never video.**

---

## 9. What "Querying Actions" Now Means

```
Query: "Show me all the times someone entered carrying something"

1. LLM decomposes:
   → action = "person entering" → search action_col with TC-CLIP embedding of "person entering door"
   → relation = CARRIES → SQL: SELECT * FROM relations WHERE relation_type='CARRIES'
   → JOIN both → find events where SAME track has both CARRIES relation AND entry action embedding match

2. Retrieve top events from both indexes

3. LLM receives 3 event records:
   Event #23: cam1, 09:14:23, "person walking through gate carrying backpack"
   Event #67: cam2, 10:32:11, "person entering lobby holding large bag"
   Event #91: cam1, 11:45:05, "person entering with box in arms"

4. LLM synthesizes: "Found 3 instances of people entering while carrying objects:
   - 09:14 at Camera 1 (Main Gate): person with backpack [clip attached]
   - 10:32 at Camera 2 (Lobby): person with large bag [clip attached]
   - 11:45 at Camera 1 (Main Gate): person with box [clip attached]"
```

---

## 10. File Structure

```
Karunya-05/
├── data/
│   ├── cam1.mp4
│   ├── cam2.mp4
│   └── clips/                   ← pre-extracted 10s clips per event
├── db/
│   ├── events.db                ← SQLite: events + relations tables
│   ├── chroma/
│   │   ├── visual_embeddings/   ← OpenCLIP appearance vectors
│   │   ├── action_embeddings/   ← TC-CLIP action vectors
│   │   └── caption_embeddings/  ← CLIP-text on event descriptions
│   └── knowledge.json           ← clarify-once persistent memory
├── src/
│   ├── indexer.py               ← OFFLINE: full extraction pipeline
│   ├── embedder.py              ← visual + action + caption embedding
│   ├── associator.py            ← relation extraction (CARRIES, ENTERS etc.)
│   ├── captioner.py             ← LLaVA auto-captioning per event
│   ├── query_engine.py          ← ONLINE: LLM parse + multi-index retrieval
│   ├── memory.py                ← knowledge.json load/save/resolve
│   ├── clip_renderer.py         ← extract + annotate clips at query time
│   └── app.py                   ← Gradio chat UI
├── PDR_base.md
├── PDR_full.md
└── requirements.txt
```

---

## 11. Requirements

```
# Vision
ultralytics>=8.1.0          # YOLO-World
insightface                  # RetinaFace + ArcFace
torchreid                    # OSNet
paddleocr
paddlepaddle

# Embeddings
open_clip_torch              # OpenCLIP visual + text
tc_clip                      # TC-CLIP action embeddings (or use VideoMAE via transformers)
transformers                 # For LLaVA / MiniCPM-V captioner

# Storage
chromadb
opencv-python
ffmpeg-python

# Query layer
gradio>=4.0
google-generativeai

# Utils
supervision
numpy
Pillow
torch
```

---

## 12. 24-Hour Build Timeline

| Hours | Task | Output |
|-------|------|--------|
| 0–2 | Setup env, install all deps, get test videos | Working Python env |
| 2–4 | `indexer.py`: YOLO-World + ByteTrack on both videos | Detection + tracking on video |
| 4–6 | `embedder.py`: OpenCLIP visual + TC-CLIP action embeddings | Two embedding types per event |
| 6–8 | `captioner.py`: LLaVA captions per event → stored in SQLite | Text description per event |
| 8–10 | `associator.py`: CARRIES / ENTERS_ZONE relation extraction | Relations table populated |
| 10–11 | InsightFace face embedding + unknown clustering | Face IDs in events table |
| 11–12 | PaddleOCR plates → stored in events table | Plate text queryable |
| 12–13 | ChromaDB: ingest all 3 embedding collections | Vector search working |
| 13–15 | `memory.py`: knowledge.json clarify-once protocol | Persistent memory working |
| 15–18 | `query_engine.py`: LLM planner + multi-index retrieval | End-to-end query flow |
| 18–19 | `clip_renderer.py`: clip extraction + bbox overlay | Visual evidence works |
| 19–21 | `app.py`: Gradio chat UI wired up | Full demo functional |
| 21–23 | Run baseline comparison + ablation table | Numbers ready |
| 23–24 | Polish demo, write short research write-up | Judge-ready |

---

## 13. Demo Script

**Step 1:** "The system pre-processed 2 camera recordings offline. No video is processed during queries."

**Step 2:** Query — *"Did anyone enter the main gate carrying something in the last hour?"*
> System hits action_emb + SQL JOIN on relations. Returns 2 events with clips.

**Step 3:** Query — *"Show me where Santhya went"*
> System: "I don't know who Santhya is — can you identify them?" → User points to face cluster.
> System saves. Returns Santhya's full timeline across both cameras.

**Step 4:** Restart app. Ask "Where did Santhya go?" → **Answers immediately. No re-ask.**

**Step 5:** Show ablation — action embedding retrieval vs. flat CLIP on "carrying", "entering", "running" queries.

---

## 14. Baseline vs. MVP — Ablation Table

| Query | Flat CLIP Baseline | Our MVP (3-layer index) |
|-------|--------------------|--------------------------|
| "Red car" | ✅ Appearance works | ✅ SQL + visual emb |
| "Person carrying bag" | ❌ Only finds person + bag separately | ✅ Action emb + CARRIES relation |
| "Person entering" | ❌ No motion understanding | ✅ TC-CLIP action emb |
| "Last 30 minutes" | ❌ No temporal filter | ✅ SQL timestamp filter |
| "Where is Santhya" | ❌ No named entity concept | ✅ Face ID → all events |
| "Plate TN09AB1234" | ❌ CLIP can't read plates | ✅ PaddleOCR → SQL |
| Clarify-once (restart) | ❌ Not possible | ✅ knowledge.json persists |
