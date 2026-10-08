import os
import time
import json
import shutil
from pathlib import Path
from typing import List, Optional
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
RENDERED_DIR = BASE_DIR / "rendered"
FRAMES_DIR = RENDERED_DIR / "frames"

def load_environment():
    targets = [BASE_DIR / ".env", ROOT_DIR / ".env"]
    for env_path in targets:
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k not in os.environ:
                            os.environ[k] = v

load_environment()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is missing. Please set it in .env file.")

MODEL_NAME = os.environ.get("MODEL_NAME", "gemma-4-26b-a4b-it")

client = genai.Client(api_key=API_KEY)

app = FastAPI(title="Surveillance Video Analytics Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_state = {
    "uploaded_file_id": None,
    "video_path": None,
    "video_1fps_path": None,
    "duration_sec": 0,
    "total_frames": 0,
    "fps": 1.0,
    "chat_history": []
}

class ChatRequest(BaseModel):
    message: str

def process_video_to_1fps(input_path: Path) -> dict:
    for f in FRAMES_DIR.glob("*.jpg"):
        try:
            f.unlink()
        except Exception:
            pass

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = int(total_frames / native_fps) if native_fps > 0 else 0

    output_1fps_path = RENDERED_DIR / "video_1fps.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_1fps_path), fourcc, 1.0, (width, height))

    sec = 0
    extracted_frames = 0
    while cap.isOpened():
        frame_idx = int(sec * native_fps)
        if frame_idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = FRAMES_DIR / f"frame_{sec:04d}.jpg"
        cv2.imwrite(str(frame_filename), frame)
        out.write(frame)
        sec += 1
        extracted_frames += 1

    cap.release()
    out.release()

    return {
        "duration_sec": sec,
        "total_frames_extracted": extracted_frames,
        "width": width,
        "height": height,
        "path_1fps": output_1fps_path
    }

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    target_video_path = UPLOAD_DIR / "original_video.mp4"
    with open(target_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    info = process_video_to_1fps(target_video_path)

    uploaded_gemini_file = client.files.upload(file=str(info["path_1fps"]))
    while uploaded_gemini_file.state.name == "PROCESSING":
        time.sleep(1)
        uploaded_gemini_file = client.files.get(name=uploaded_gemini_file.name)

    session_state["uploaded_file_id"] = uploaded_gemini_file.name
    session_state["video_path"] = target_video_path
    session_state["video_1fps_path"] = info["path_1fps"]
    session_state["duration_sec"] = info["duration_sec"]
    session_state["total_frames"] = info["total_frames_extracted"]
    session_state["chat_history"] = []

    return {
        "status": "success",
        "duration": info["duration_sec"],
        "frames_extracted": info["total_frames_extracted"],
        "file_name": uploaded_gemini_file.name,
        "resolution": f"{info['width']}x{info['height']}"
    }

@app.get("/api/video")
def get_original_video():
    path = session_state.get("video_path")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="No video uploaded")
    return FileResponse(str(path), media_type="video/mp4")

@app.get("/api/frames/{sec}")
def get_frame(sec: int):
    frame_path = FRAMES_DIR / f"frame_{sec:04d}.jpg"
    if not frame_path.exists():
        fallback = sorted(list(FRAMES_DIR.glob("*.jpg")))
        if fallback:
            return FileResponse(str(fallback[0]), media_type="image/jpeg")
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(str(frame_path), media_type="image/jpeg")

@app.post("/api/chat")
async def chat_with_video(req: ChatRequest):
    file_id = session_state.get("uploaded_file_id")
    if not file_id:
        raise HTTPException(status_code=400, detail="Please upload a video first")

    uploaded_file = client.files.get(name=file_id)

    system_instruction = (
        "You are an enterprise surveillance video intelligence system analyzing footage sampled strictly at 1.0 FPS. "
        "Provide factual, objective, evidence-grounded reports. "
        "Analyze vehicles, pedestrians, recognitions, interactions, luggage, directions, and temporal patterns. "
        "Every single reference to an event, object, or detection MUST include a precise timestamp or time range "
        "enclosed in brackets, e.g. [00:04] or [00:10 - 00:15]. "
        "Organize longer responses with clear sections: Summary, Timestamped Evidence Log, and Key Observations."
    )

    contents = []
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(file_uri=uploaded_file.uri, mime_type="video/mp4"),
                types.Part.from_text(text="Reference stream initialized at 1.0 FPS.")
            ]
        )
    )

    for item in session_state["chat_history"]:
        contents.append(
            types.Content(
                role=item["role"],
                parts=[types.Part.from_text(text=item["text"])]
            )
        )

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=req.message)]
        )
    )

    tools = [
        types.Tool(googleSearch=types.GoogleSearch()),
    ]

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="HIGH",
        ),
        audio_transcription_config=types.AudioTranscriptionConfig(),
        tools=tools,
        system_instruction=system_instruction
    )

    async def event_generator():
        full_text = ""
        response_stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=generate_content_config,
        )
        for chunk in response_stream:
            if chunk.text:
                full_text += chunk.text
                data = json.dumps({"token": chunk.text})
                yield f"data: {data}\n\n"
        session_state["chat_history"].append({"role": "user", "text": req.message})
        session_state["chat_history"].append({"role": "model", "text": full_text})
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@app.get("/api/status")
def get_status():
    return {
        "has_video": bool(session_state.get("uploaded_file_id")),
        "duration": session_state.get("duration_sec", 0),
        "total_frames": session_state.get("total_frames", 0),
        "model": MODEL_NAME,
        "history_count": len(session_state.get("chat_history", []))
    }

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
