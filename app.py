import os
import re
import io
import time
import logging
import tempfile
import asyncio
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import edge_tts
from pydub import AudioSegment

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bangla-tts")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_TEXT_LENGTH = 25000
CHUNK_SIZE = 4000
FALLBACK_VOICE = "bn-IN-TanishaaNeural"

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bangla TTS Backend",
    description="Production-ready Bangla Text-to-Speech API using Microsoft Edge TTS",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Voice selection
# ---------------------------------------------------------------------------
async def select_bangla_voice() -> str:
    """
    Automatically detect the best available Bangla Neural voice.
    Preference order: bn-BD Neural → any bn-* Neural → fallback.
    """
    try:
        voices = await edge_tts.list_voices()
        bangla_neural = [
            v for v in voices
            if str(v.get("Locale", "")).startswith("bn")
            and "Neural" in str(v.get("ShortName", ""))
        ]
        if not bangla_neural:
            logger.warning("No Bangla Neural voices discovered – using fallback")
            return FALLBACK_VOICE

        # Prefer Bangladesh locale when present
        for v in bangla_neural:
            if "bn-BD" in v["ShortName"]:
                logger.info("Selected voice: %s", v["ShortName"])
                return v["ShortName"]

        selected = bangla_neural[0]["ShortName"]
        logger.info("Selected voice: %s", selected)
        return selected
    except Exception as exc:
        logger.error("Voice discovery failed: %s – using fallback", exc)
        return FALLBACK_VOICE

# ---------------------------------------------------------------------------
# Text extraction & chunking
# ---------------------------------------------------------------------------
def extract_curly_blocks(raw: str) -> Optional[str]:
    """
    Extract every {...} block and join them with two newlines.
    Returns None when no valid curly-bracket blocks exist.
    """
    matches = re.findall(r"\{([^{}]*)\}", raw, flags=re.DOTALL)
    cleaned = [m.strip() for m in matches if m.strip()]
    if not cleaned:
        return None
    return "\n\n".join(cleaned)

def split_into_chunks(text: str, max_chars: int = CHUNK_SIZE) -> List[str]:
    """
    Split text into chunks of approximately max_chars without cutting
    Bangla words. Prefer whitespace, newline or Bangla punctuation.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        # Walk backwards looking for a safe split point
        split_at = max_chars
        look_back = min(300, max_chars)
        for i in range(max_chars, max_chars - look_back, -1):
            ch = remaining[i]
            if ch in " \t\n\r।.!?၊,;:":
                split_at = i + 1
                break

        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].strip()

    return chunks

# ---------------------------------------------------------------------------
# Audio generation helpers
# ---------------------------------------------------------------------------
async def _synthesize_chunk(text: str, voice: str, directory: str) -> str:
    """Synthesize a single chunk to a temporary mp3 file."""
    fd, path = tempfile.mkstemp(suffix=".mp3", dir=directory)
    os.close(fd)
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(path)
        return path
    except Exception:
        if os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
        raise

async def generate_speech(text: str, voice: str) -> bytes:
    """
    Generate a complete mp3 for the given text.
    Handles chunking, synthesis, merging and full temporary-file cleanup.
    """
    temp_dir = tempfile.mkdtemp(prefix="bangla_tts_")
    chunk_files: List[str] = []

    try:
        chunks = split_into_chunks(text)
        logger.info("Text split into %d chunk(s)", len(chunks))

        for idx, chunk in enumerate(chunks):
            logger.debug("Synthesizing chunk %d/%d (%d chars)", idx + 1, len(chunks), len(chunk))
            path = await _synthesize_chunk(chunk, voice, temp_dir)
            chunk_files.append(path)

        if not chunk_files:
            raise RuntimeError("No audio chunks were produced")

        # Merge with pydub
        combined = AudioSegment.from_mp3(chunk_files[0])
        for path in chunk_files[1:]:
            combined += AudioSegment.from_mp3(path)

        buffer = io.BytesIO()
        combined.export(buffer, format="mp3")
        buffer.seek(0)
        return buffer.read()

    finally:
        # Guaranteed cleanup of every temporary artefact
        for path in chunk_files:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError as exc:
                logger.warning("Failed to delete temp file %s: %s", path, exc)
        try:
            os.rmdir(temp_dir)
        except OSError as exc:
            logger.warning("Failed to remove temp directory %s: %s", temp_dir, exc)

# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "online"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# ---------------------------------------------------------------------------
# Main TTS endpoint
# ---------------------------------------------------------------------------
@app.post("/tts")
async def tts(request: Request):
    """
    Accept JSON {"text": "{বাংলা লেখা}"}, extract curly blocks,
    synthesise Bangla speech and return a single audio/mpeg response.
    """
    started = time.monotonic()
    logger.info("Incoming TTS request")

    # ---- Parse JSON -------------------------------------------------------
    try:
        body = await request.json()
    except Exception:
        logger.error("Invalid JSON body")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Invalid JSON",
                "message": "Request body must be valid JSON.",
            },
        )

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Invalid JSON",
                "message": "Request body must be a JSON object.",
            },
        )

    if "text" not in body:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Missing field",
                "message": "Missing required field 'text'.",
            },
        )

    raw_text = body["text"]
    if not isinstance(raw_text, str):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Invalid type",
                "message": "Field 'text' must be a string.",
            },
        )

    raw_text = raw_text.strip()
    if not raw_text:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Empty text",
                "message": "Text cannot be empty.",
            },
        )

    if len(raw_text) > MAX_TEXT_LENGTH:
        logger.warning("Rejected oversized request (%d chars)", len(raw_text))
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Text too long",
                "message": f"Text exceeds maximum allowed length of {MAX_TEXT_LENGTH} characters.",
            },
        )

    # ---- Curly-bracket validation -----------------------------------------
    extracted = extract_curly_blocks(raw_text)
    if extracted is None:
        logger.warning("Curly Bracket Error – no valid blocks found")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Curly Bracket Error",
                "message": "Text must be enclosed within { }.",
            },
        )

    if not extracted.strip():
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Empty text",
                "message": "No usable text found inside curly brackets.",
            },
        )

    # ---- Synthesis --------------------------------------------------------
    try:
        voice = await select_bangla_voice()
        audio_bytes = await generate_speech(extracted, voice)

        elapsed = time.monotonic() - started
        logger.info(
            "TTS completed successfully in %.2fs | voice=%s | bytes=%d",
            elapsed,
            voice,
            len(audio_bytes),
        )

        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": 'attachment; filename="bangla_tts.mp3"',
                "Cache-Control": "no-store",
            },
        )

    except Exception as exc:
        logger.exception("TTS generation failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "TTS Error",
                "message": "Failed to generate speech. Please try again later.",
            },
        )

# ---------------------------------------------------------------------------
# Entrypoint (Render / local)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting Bangla TTS server on 0.0.0.0:%d", port)
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
    )
