# PDR_full — HNX26EPS05 Full Product
## Product Design Reference · Complete Vision / Final Ideation

> **Core Principle (same as base, scaled up):**
> Video is processed ONCE offline. Everything gets embedded into a searchable, reasoned index.
> At query time: embed the query → retrieve from index → LLM reasons over text + metadata.
> The system understands ACTIONS, RELATIONS, and CAUSAL SEQUENCES — not just appearances.
> **Video is never re-processed at query time. Ever.**

---

## 0. Vision Statement

> A multi-camera video intelligence system where every frame ever recorded is distilled into a
> **living, queryable knowledge graph of actions, identities, and relationships.**
>
> Ask it anything — *"What did Santhya do after the delivery van arrived?"* — and it
> reasons across cameras, time, and embedded understanding to give you a grounded, traceable answer.
>
> It is not a search engine. It is a **memory that understands.**

---

## 1. Why Embedding Offline Is Non-Negotiable

```
Hours of CCTV video:
  1 camera × 8 hours = ~57,600 frames at 2 FPS
  4 cameras × 8 hours = ~230,400 frames

If fed to Gemini at query time:
  ~115,200 frames × ~500 tokens/frame = 57,600,000 tokens per query
  → Impossible. Context window maxes at ~1M tokens.
  → Even if it worked: $$$, minutes of latency, can't do it live.

Our architecture:
  Index offline ONCE → store compressed representations
  At query time: retrieve top-3 to top-10 relevant events (~2,000 tokens total)
  → Gemini reads 2,000 tokens, not 57 million.
  → Sub-second reasoning. Scales to any video length.
```

---

## 2. The Four Embedding Layers (Built Offline)

Every detected entity track produces **4 independent indexes**:

### Layer 1 — Visual (Appearance)
```
Model: OpenCLIP ViT-L/14 (full product) / ViT-B/32 (MVP)
Input: Cropped detection image
Output: 512-dim vector
Answers: "red car", "person in blue jacket", "black backpack"
```

### Layer 2 — Action (Temporal / Motion)
```
Model: TC-CLIP (naver-ai/tc-clip) or VideoMAE-v2
Input: 16-frame sliding window of entity crop sequence
Output: 512-dim action embedding vector
Answers: "carrying a bag", "running", "entering a door",
         "handing something over", "sitting", "driving"

This is what makes action-based queries possible.
Flat CLIP has zero capability here.
```

### Layer 3 — Caption (Natural Language Description)
```
Model: MiniCPM-V 2.6 (lightweight) / LLaVA-1.5-7B
Input: Keyframe + tracker context + relation tuples
Output: Rich text description of the event

Example output:
"At 09:14 at Camera 1, a young person wearing a blue jacket enters through
 the main gate from outside. They are carrying a black backpack on their left
 shoulder and briefly look back before continuing. A white SUV is visible
 parked to the right."

Stored in: SQLite events.description + ChromaDB caption_embeddings
At query time: LLM reads THIS TEXT, not the video.
```

### Layer 4 — Relational (Typed Graph Tuples)
```
Extraction: Rule-based from tracker + spatial analysis
Output: (subject, relation, object, camera, start_ts, end_ts, confidence)

Examples:
  (person_4, CARRIES, bag_12, cam1, 553.1, 612.4, 0.91)
  (person_4, ENTERS_ZONE, "main_gate", cam1, 551.2, 553.5, 1.0)
  (person_4, SAME_AS, person_9, cam2, 980.3, 980.3, 0.83) ← Re-ID
  (vehicle_7, HAS_PLATE, "TN09AB1234", cam1, 480.0, 480.0, 0.97)
  (person_4, ACCOMPANIED_BY, person_5, cam2, 985.0, 1024.0, 0.88)

Stored in: SQLite relations table
Enables: "the bag carried by Santhya", "who was with the delivery driver"
```

---

## 3. Full System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     LIVE STREAM INGESTION                            │
│         RTSP / HLS / USB cameras (N streams via multiprocessing)     │
│         OR: Recorded video files (Phase 1 / hackathon mode)          │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────┐
│                    PERCEPTION ENGINE (Per Camera)                    │
│                                                                      │
│  ┌─────────────┐  ┌───────────────┐  ┌──────────────────────────┐   │
│  │ YOLO-World-M│  │  RetinaFace   │  │       ByteTrack           │   │
│  │ Open-vocab  │  │  Face detect  │  │  Stable per-camera IDs    │   │
│  │ detection   │  │  (small faces)│  │                           │   │
│  └──────┬──────┘  └──────┬────────┘  └────────────┬─────────────┘   │
│         └────────────────┴──────────────────────────┘               │
│                          │                                           │
│  ┌───────────────────────▼──────────────────────────────────────┐   │
│  │              PER-ENTITY EMBEDDING PIPELINE                   │   │
│  │                                                              │   │
│  │  OpenCLIP ViT-L/14  → visual_emb (appearance)               │   │
│  │  TC-CLIP (16 frames) → action_emb (motion/action)           │   │
│  │  ArcFace R100        → face_emb (identity)                  │   │
│  │  OSNet-x1.0          → body_emb (Re-ID fallback)            │   │
│  │  PaddleOCR           → plate_text                           │   │
│  │  MiniCPM-V 2.6       → event_description (NL caption)       │   │
│  └───────────────────────┬──────────────────────────────────────┘   │
└─────────────────────────┬─────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────────┐
│                   RELATIONAL REASONING ENGINE                        │
│                                                                      │
│  Association Engine (spatial + temporal rules, per-camera):          │
│  ─ CARRIES: bbox overlap + co-travel for >15 frames                  │
│  ─ DRIVES: person enters vehicle bbox, vehicle moves                 │
│  ─ ACCOMPANIES: two persons within 150px for >60 frames              │
│  ─ ENTERS_ZONE / EXITS_ZONE: centroid crosses polygon boundary       │
│  ─ HANDED_TO: object changes custody between tracks                  │
│                                                                      │
│  Cross-Camera Re-ID Registry:                                        │
│  Stage 1: ArcFace face cosine match (threshold 0.60)                 │
│  Stage 2: OSNet body cosine match (threshold 0.70, fallback)         │
│  → Emits: SAME_AS relation linking global_id across cameras          │
│                                                                      │
│  Temporal Causal Chainer:                                            │
│  → Groups all events for a global_id into an ordered timeline        │
│  → Enables: "What did person X do across all cameras?"               │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────────┐
│                    KNOWLEDGE & STORAGE LAYER                         │
│                                                                      │
│  ┌──────────────────┐  ┌───────────────────────┐                     │
│  │   PostgreSQL     │  │        Qdrant          │                     │
│  │  (prod) /        │  │   Vector Store         │                     │
│  │   SQLite (MVP)   │  │                        │                     │
│  │                  │  │  Collections:           │                     │
│  │  events table    │  │  - visual_embeddings   │                     │
│  │  relations table │  │  - action_embeddings   │                     │
│  │  identities table│  │  - face_embeddings     │                     │
│  │  captions table  │  │  - body_embeddings     │                     │
│  │                  │  │  - caption_embeddings  │                     │
│  └──────────────────┘  └───────────────────────┘                     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  PERSISTENT KNOWLEDGE BASE                   │   │
│  │  knowledge.json (MVP) / PostgreSQL KB table (Full)           │   │
│  │                                                              │   │
│  │  - Location → camera + zone polygon mappings                │   │
│  │  - Person name → face_embedding_id + aliases                │   │
│  │  - Entity → attributes (plate, color, type)                 │   │
│  │  - Standing query subscriptions                             │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────────┐
│                  CONVERSATIONAL REASONING ENGINE                     │
│                                                                      │
│  Query flow (ONLINE — no video touched):                             │
│  1. Load persistent KB → resolve all named entities/locations        │
│  2. If unknown referent → ask user once → save to KB                 │
│  3. LLM decomposes query into sub-retrieval plans:                   │
│     → SQL sub-query (structured: class, color, plate, time)          │
│     → Visual vector search (appearance match)                        │
│     → Action vector search (motion/behavior match)                   │
│     → Caption vector search (semantic NL match)                      │
│     → Relation graph traversal (who+what+with whom)                  │
│  4. Merge results → rank → top-K events                              │
│  5. LLM receives: event records as TEXT → generates answer           │
│  6. Evidence: pre-extracted clips + bbox overlays                    │
│                                                                      │
│  Standing Query Monitor (background process):                        │
│  → Checks every new event against registered standing queries        │
│  → Triggers alert on match                                           │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────────┐
│                        INTERFACE LAYER                               │
│                                                                      │
│  Chat UI (web, mobile)          → conversational queries             │
│  Live camera grid               → annotated real-time view           │
│  Entity timeline explorer       → full path reconstruction UI        │
│  Alert dashboard                → standing query notifications       │
│  Zone editor                    → draw polygons on camera frames     │
│  Privacy toggle                 → face blur on/off per camera        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. The Query Reasoning Flow — Full Depth

```
User: "What did the person who arrived with the delivery van do after they entered?"

Step 1: MEMORY RESOLUTION
  "delivery van" → check KB → found: plate=TN09AB1234, entity=vehicle_7

Step 2: LLM QUERY DECOMPOSITION
  Sub-query A: Find vehicle_7 arrival event
    → SQL: SELECT * FROM events WHERE entity_class='car' AND plate='TN09AB1234'
    → Result: cam1, ts=480.0, "white van arrives at main gate"

  Sub-query B: Find persons who arrived WITH vehicle_7
    → SQL: SELECT * FROM relations WHERE relation_type='ACCOMPANIED_BY'
            AND (source_id=7 OR target_id=7) AND start_ts BETWEEN 475 AND 495
    → Result: person_4 was near vehicle_7 at arrival

  Sub-query C: What did person_4 do after ts=480?
    → SQL: SELECT * FROM events WHERE track_id=4 AND timestamp > 480 ORDER BY timestamp
    → SQL: SELECT * FROM relations WHERE source_id=4 AND start_ts > 480
    → Cross-camera: SAME_AS links person_4 → person_9 (cam2, via face Re-ID)
    → Timeline reconstruction across cam1 + cam2

  Sub-query D: Action embedding search for "entering" after ts=480
    → Vector search on action_embeddings, filter timestamp>480
    → Match: event_67 (cam1, ts=551), event_83 (cam2, ts=980)

Step 3: EVIDENCE AGGREGATION
  Merge SQL results + vector hits
  Deduplicate by event_id
  Build ordered timeline for person_4 / person_9

Step 4: LLM ANSWER SYNTHESIS
  Input (TEXT, not video):
    "Event 1: vehicle_7 (delivery van, TN09AB1234) arrives at cam1 main gate, ts=480
     Event 2: person_4 exits van at cam1, ts=491, 'person in grey hoodie steps out of white van'
     Relation: person_4 CARRIES bag_15 from ts=495
     Event 3: person_4 ENTERS_ZONE main_gate at cam1, ts=553
     Event 4: person_4 SAME_AS person_9 at cam2, ts=980 (face match, confidence=0.87)
     Event 5: person_9 (same individual) ENTERS_ZONE lobby at cam2, ts=982"

  LLM output:
    "The person who arrived with the delivery van (grey hoodie) exited the
     vehicle at 08:01 at Camera 1, picked up a bag, entered through the main
     gate at 09:13, and was then identified at Camera 2 (Lobby) at 16:20.
     [3 video clips attached]"
```

---

## 5. Action Embedding in Detail — TC-CLIP

```python
# TC-CLIP produces action-aware embeddings from video clips
# Unlike CLIP (image-based), TC-CLIP understands temporal dynamics

from transformers import AutoProcessor, AutoModel
import torch

# Load TC-CLIP (temporal contextualization CLIP)
processor = AutoProcessor.from_pretrained("naver-ai/tc-clip")
model = AutoModel.from_pretrained("naver-ai/tc-clip")

def embed_action(frame_crops: list[np.ndarray], stride: int = 1) -> np.ndarray:
    """
    frame_crops: 16 consecutive BGR frames of one tracked entity (cropped)
    Returns: 512-dim action embedding
    """
    # Preprocess to [1, T, C, H, W]
    inputs = processor(videos=[frame_crops[::stride]], return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_video_features(**inputs)
    return outputs.cpu().numpy()[0]

def embed_action_query(text: str) -> np.ndarray:
    """
    Embed a text description of an action into the same space.
    "person carrying a bag" → 512-dim vector that matches action clips.
    """
    inputs = processor(text=[text], return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    return outputs.cpu().numpy()[0]

# This allows: action_embeddings.query("person handing something to another person")
# → retrieves clips of HANDED_TO events even without explicit label
```

---

## 6. Caption Generation — MiniCPM-V 2.6 (Offline)

```python
from transformers import AutoTokenizer, AutoModel
from PIL import Image

model = AutoModel.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)

def caption_event(keyframe: np.ndarray, context: dict) -> str:
    """
    keyframe: the representative frame of the event
    context: {camera, timestamp, track_id, class, relations, nearby_entities}
    """
    system_prompt = "You are a precise CCTV event analyst. Describe only what you observe."
    user_prompt = f"""
Camera: {context['camera']} | Timestamp: {context['timestamp']:.1f}s
Tracked: {context['class']} (ID={context['track_id']})
Relations detected: {context['relations']}
Nearby: {context['nearby']}

Describe: (1) what this entity is doing, (2) direction of movement,
(3) objects being interacted with, (4) any other notable details.
Be specific and literal. No guessing.
"""
    image = Image.fromarray(cv2.cvtColor(keyframe, cv2.COLOR_BGR2RGB))
    response = model.chat(tokenizer, [image], user_prompt, system_prompt=system_prompt)
    return response

# Output stored in events.description (SQLite)
# Also embedded via OpenCLIP text encoder → caption_embeddings (Qdrant)
```

---

## 7. Cross-Camera Re-ID — Two-Stage Cascade

```python
class GlobalIdentityRegistry:
    def __init__(self, face_threshold=0.60, body_threshold=0.70):
        self.identities = {}   # global_id → {face_embs, body_embs, name, cameras_seen}
        self.face_threshold = face_threshold
        self.body_threshold = body_threshold

    def match_or_create(self, face_emb, body_emb, camera_id, timestamp):
        best_match = None
        best_score = 0

        for gid, identity in self.identities.items():
            # Stage 1: Face Re-ID
            if face_emb is not None and identity['face_embs']:
                scores = [cosine_sim(face_emb, fe) for fe in identity['face_embs']]
                score = max(scores)
                if score > self.face_threshold and score > best_score:
                    best_score = score
                    best_match = (gid, "face", score)

            # Stage 2: Body Re-ID (fallback when face not matched)
            if best_match is None and body_emb is not None:
                scores = [cosine_sim(body_emb, be) for be in identity['body_embs']]
                score = max(scores) if scores else 0
                if score > self.body_threshold and score > best_score:
                    best_score = score
                    best_match = (gid, "body", score)

        if best_match:
            gid, method, confidence = best_match
            self.identities[gid]['cameras_seen'].add(camera_id)
            # Emit SAME_AS relation
            emit_relation(gid, "SAME_AS", camera_id, timestamp, confidence, method)
            return gid
        else:
            # New identity
            gid = f"global_{len(self.identities):04d}"
            self.identities[gid] = {
                'face_embs': [face_emb] if face_emb is not None else [],
                'body_embs': [body_emb] if body_emb is not None else [],
                'name': None,  # filled via clarify-once
                'cameras_seen': {camera_id}
            }
            return gid
```

---

## 8. Full Standing Query System

```python
# Runs as a background process, continuously monitoring new events
import threading

class StandingQueryMonitor:
    def __init__(self, kb_path: str, db_path: str):
        self.kb = load_knowledge(kb_path)
        self.standing_queries = self.kb.get("standing_queries", [])
        self.last_checked_ts = {}

    def evaluate_new_event(self, event: dict, relations: list):
        """Called every time a new event is written to events.db"""
        for sq in self.standing_queries:
            if not sq['active']:
                continue
            if self.matches(event, relations, sq['conditions']):
                self.fire_alert(sq, event)

    def matches(self, event, relations, conditions) -> bool:
        # Zone condition
        if 'zone' in conditions:
            zone = self.kb['locations'].get(conditions['zone'])
            if zone and event.get('camera_id') != zone['camera']:
                return False
            zone_entered = any(r['relation_type'] == 'ENTERS_ZONE'
                               and r['target'] == conditions['zone']
                               for r in relations)
            if not zone_entered:
                return False
        # Time condition
        if 'time_after' in conditions:
            h, m = map(int, conditions['time_after'].split(':'))
            event_hour = int(event['timestamp'] // 3600) % 24
            if event_hour < h:
                return False
        return True

    def fire_alert(self, sq, event):
        alert = {
            "standing_query": sq['query'],
            "triggered_by": event,
            "clip": event.get('clip_path'),
            "timestamp": event['timestamp'],
            "camera": event['camera_id']
        }
        # Send to webhook / push notification / in-app
        send_alert(alert)
```

---

## 9. Full Model Stack

| Component | MVP | Full Product | When |
|-----------|-----|-------------|------|
| Detection | YOLO-World-S | YOLO-World-M | Offline |
| Tracking | ByteTrack | BoT-SORT (appearance-guided) | Offline |
| Face detect | RetinaFace | RetinaFace | Offline |
| Face embed | ArcFace R50 | ArcFace R100 | Offline |
| Body Re-ID | OSNet-x0.25 | OSNet-x1.0 | Offline |
| Plate OCR | PaddleOCR | PaddleOCR | Offline |
| Visual embed | OpenCLIP ViT-B/32 | OpenCLIP ViT-L/14 | Offline |
| **Action embed** | **TC-CLIP** | **TC-CLIP** | **Offline** |
| **Captioner** | **LLaVA-1.5-7B** | **MiniCPM-V 2.6** | **Offline** |
| Caption embed | CLIP text encoder | CLIP text encoder | Offline |
| Vector DB | ChromaDB | Qdrant | Offline write / Online read |
| SQL DB | SQLite | PostgreSQL | Offline write / Online read |
| Graph DB | SQLite relations table | Neo4j / Kuzu | Offline write / Online read |
| LLM (cloud) | Gemini Flash | Gemini 2.0 Flash / GPT-4o | Online only |
| LLM (on-prem) | — | Ollama + Llama 3.1 8B | Online only |
| UI | Gradio | Next.js + Socket.IO | Online |

---

## 10. Full Capability Matrix

| Capability | Base MVP | Full Product |
|------------|----------|--------------|
| Input | 2 recorded videos | N live RTSP streams |
| Open-vocab detection | ✅ YOLO-World-S | ✅ YOLO-World-M |
| **Action embedding (TC-CLIP)** | ✅ | ✅ |
| **Auto-captioning (LLaVA/MiniCPM)** | ✅ | ✅ |
| Face recognition (ArcFace) | ✅ | ✅ R100 |
| Body Re-ID (OSNet) | ✅ basic | ✅ full cascade |
| Plate OCR | ✅ | ✅ |
| Object-person association (CARRIES etc.) | ✅ rule-based | ✅ + HANDED_TO |
| Cross-camera Re-ID | ✅ face only | ✅ face + body cascade |
| Temporal causal chaining | ❌ | ✅ |
| Standing queries / alerts | ❌ | ✅ background monitor |
| Clarify-once memory | ✅ JSON | ✅ rich KB with aliases |
| Zone drawing UI | ❌ | ✅ polygon editor |
| Face blur (privacy) | ❌ | ✅ GDPR mode |
| On-prem LLM | ❌ | ✅ Ollama |
| Live camera grid UI | ❌ | ✅ Next.js + Socket.IO |
| Entity timeline explorer | ❌ | ✅ |
| Ablation study | ✅ table | ✅ full paper |

---

## 11. Research Contribution — The Five Novel Claims

1. **Multi-layer Video Indexing for Relational Retrieval**
   Four embedding types (visual + action + caption + relational tuples) indexed offline.
   First open-source system to combine temporal action embeddings + typed relation graph + LLM reasoning for CCTV queries.

2. **Action-Queryable Video Archive**
   TC-CLIP action embeddings allow querying *what entities are doing*, not just what they look like.
   Enables: "Show me all carrying events", "Find everyone who entered running" — impossible with flat CLIP.

3. **Auto-Captioned Event Records as LLM Context**
   VLM-generated event captions serve as the bridge between pixel-space and reasoning-space.
   LLM never sees raw video — it reads structured event narratives.

4. **Clarify-Once Knowledge Base with Relational Memory**
   Beyond simple camera labeling: the KB learns entity aliases, associations between people and vehicles,
   and standing query conditions — persisted across restarts.

5. **Cross-Modal Re-ID Fallback Chain**
   Face → Body → Outfit cascade with per-stage confidence logging.
   Enables Re-ID even when face is obscured (mask, angle, distance, darkness).

---

## 12. Ablation Plan

| Ablation | Removes | What It Shows |
|----------|---------|---------------|
| A1 | TC-CLIP action embeddings (use only visual CLIP) | Value of action understanding |
| A2 | Auto-captions (use class labels only) | Value of rich event descriptions |
| A3 | Relational tuples (use visual search only) | Value of CARRIES/ENTERS relations |
| A4 | Cross-camera Re-ID | Value of multi-camera continuity |
| A5 | Clarify-once memory | Value of persistent KB |
| A6 | Body Re-ID fallback | Value of two-stage cascade |
| A7 | Temporal SQL filter | Value of structured time indexing |

---

## 13. Roadmap

```
Phase 1 — Hackathon (24h) → PDR_base
  Recorded videos, 3-layer index, clarify-once, Gradio UI, ablation table

Phase 2 — Post-hackathon (1–2 weeks)
  PostgreSQL + Qdrant migration
  Zone polygon editor UI
  Entity timeline visualization
  Full TC-CLIP integration

Phase 3 — Full Product (1–3 months)
  Live RTSP stream ingestion (multiprocessing)
  Standing queries + alert system
  Neo4j graph DB
  Next.js frontend (camera grid, timeline, alert dashboard)
  Privacy mode (face blur + Ollama LLM)

Phase 4 — Production (3–6 months)
  TensorRT model optimization
  Multi-server distributed deployment
  Enterprise auth + audit logging
  VMS integration API
```
