"""
Bangla Telegram TTS Bot
========================
Production-ready Telegram bot that converts long Bangla text into natural
MP3 voice using Microsoft Edge-TTS, overcoming Telegram's message length
limits via a session-based text collection system.

Environment: Python 3.11+
Compatible with: Google Colab, Termux, Linux VPS, Render.com

Dependencies:
    python-telegram-bot==21.6
    edge-tts
    pydub
    ffmpeg-python
    nest_asyncio
"""

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import edge_tts
import nest_asyncio
from pydub import AudioSegment
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

nest_asyncio.apply()

# ============================================================
# Configuration
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")  # <-- placeholder, replace with your bot token

PREFERRED_VOICE = "bn-IN-TanishaaNeural"
CHUNK_SIZE = 4000  # approx characters per TTS chunk
MAX_RETRIES = 3
COLLECTION_TIMEOUT_SECONDS = 30 * 60  # 30 minutes
WORDS_PER_MINUTE_ESTIMATE = 150  # rough speaking rate for duration estimate
TIMEOUT_CHECK_INTERVAL = 60  # seconds, how often the timeout job runs

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("bangla_tts_bot")

# Globally cached Bangla voice name, resolved once at startup.
CACHED_VOICE: Optional[str] = None


# ============================================================
# Session Management
# ============================================================

@dataclass
class UserSession:
    """Holds the state of a single user's text collection session."""

    user_id: int
    chat_id: int
    collecting: bool = False
    messages: list = field(default_factory=list)
    characters: int = 0
    start_time: float = 0.0
    last_activity: float = 0.0
    status_message_id: Optional[int] = None

    def reset(self) -> None:
        """Reset the session back to an idle state."""
        self.collecting = False
        self.messages = []
        self.characters = 0
        self.start_time = 0.0
        self.last_activity = 0.0
        self.status_message_id = None


# In-memory session store. Key: user_id
SESSIONS: dict[int, UserSession] = {}


def get_session(user_id: int, chat_id: int) -> UserSession:
    """Get or create the session for a given user."""
    if user_id not in SESSIONS:
        SESSIONS[user_id] = UserSession(user_id=user_id, chat_id=chat_id)
    return SESSIONS[user_id]


# ============================================================
# Helper Functions
# ============================================================

def estimate_duration(character_count: int) -> str:
    """Estimate spoken duration from a character count.

    Uses a rough heuristic based on average speaking rate, converting
    an approximate word count into minutes and seconds.
    """
    approx_words = max(character_count / 5.5, 0)
    minutes_float = approx_words / WORDS_PER_MINUTE_ESTIMATE
    total_seconds = int(minutes_float * 60)
    minutes, seconds = divmod(total_seconds, 60)
    return f"~{minutes}m {seconds:02d}s"


def split_text_safely(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split text into chunks without breaking words.

    Splits primarily on whitespace boundaries so that no word (Bangla or
    otherwise) is ever cut in half across chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > chunk_size:
        split_at = chunk_size
        window = remaining[:chunk_size]
        last_space = max(
            window.rfind(" "),
            window.rfind("\n"),
            window.rfind("\t"),
        )
        if last_space > 0:
            split_at = last_space

        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def format_status_text(session: UserSession) -> str:
    """Format the live collection status message."""
    duration = estimate_duration(session.characters)
    return (
        "📝 Collecting Text\n\n"
        f"Messages : {len(session.messages)}\n"
        f"Characters : {session.characters}\n"
        f"Estimated Voice : {duration}\n\n"
        "Waiting for }"
    )


def format_final_status_text(session: UserSession) -> str:
    """Format status text used for /status command replies."""
    if session.collecting:
        duration = estimate_duration(session.characters)
        return (
            "Collection : Running\n"
            f"Messages : {len(session.messages)}\n"
            f"Characters : {session.characters}\n"
            f"Estimated Voice : {duration}"
        )
    return "Collection : Idle\n\nSend { to start collecting text."


async def resolve_voice() -> str:
    """Resolve and cache the Bangla neural voice to use.

    Prefers PREFERRED_VOICE if available, otherwise falls back to the
    first available Bangla neural voice. This lookup happens only once
    at startup and the result is cached globally.
    """
    global CACHED_VOICE
    if CACHED_VOICE:
        return CACHED_VOICE

    try:
        voices = await edge_tts.list_voices()
        bangla_voices = [v for v in voices if v["Locale"].startswith("bn-")]

        for v in bangla_voices:
            if v["ShortName"] == PREFERRED_VOICE:
                CACHED_VOICE = PREFERRED_VOICE
                logger.info("Using preferred Bangla voice: %s", CACHED_VOICE)
                return CACHED_VOICE

        if bangla_voices:
            CACHED_VOICE = bangla_voices[0]["ShortName"]
            logger.info("Preferred voice unavailable. Falling back to: %s", CACHED_VOICE)
            return CACHED_VOICE

        CACHED_VOICE = PREFERRED_VOICE
        logger.warning("No Bangla voices found via list_voices(). Defaulting to: %s", CACHED_VOICE)
        return CACHED_VOICE

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to resolve voice list: %s", exc)
        CACHED_VOICE = PREFERRED_VOICE
        return CACHED_VOICE


async def generate_tts_chunk(text: str, voice: str, index: int, work_dir: str) -> str:
    """Generate a single TTS audio chunk with retry logic.

    Returns the path to the generated MP3 file. Raises RuntimeError if
    generation fails after MAX_RETRIES attempts.
    """
    output_path = os.path.join(work_dir, f"chunk_{index:04d}.mp3")
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
            raise RuntimeError("Generated file is empty")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Chunk %d generation failed (attempt %d/%d): %s",
                index, attempt, MAX_RETRIES, exc,
            )
            await asyncio.sleep(1.5 * attempt)

    raise RuntimeError(f"Chunk {index} failed after {MAX_RETRIES} attempts: {last_error}")


def merge_audio_files(paths: list[str], output_path: str) -> str:
    """Merge a list of MP3 files into a single MP3 using pydub."""
    combined = AudioSegment.empty()
    for path in paths:
        segment = AudioSegment.from_file(path, format="mp3")
        combined += segment
    combined.export(output_path, format="mp3")
    return output_path


def cleanup_files(paths: list[str]) -> None:
    """Delete a list of temporary files, ignoring missing files."""
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            logger.warning("Failed to delete temp file %s: %s", path, exc)

async def close_status_message(
    context: ContextTypes.DEFAULT_TYPE,
    session: UserSession,
    text: str,
    delay: int = 3,
) -> None:
    """Edit the status message, wait a few seconds, then delete it."""

    if not session.status_message_id:
        return

    try:
        await context.bot.edit_message_text(
            chat_id=session.chat_id,
            message_id=session.status_message_id,
            text=text,
        )

        await asyncio.sleep(delay)

        await context.bot.delete_message(
            chat_id=session.chat_id,
            message_id=session.status_message_id,
        )

    except Exception:
        pass

# ============================================================
# Command Handlers
# ============================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    text = (
        "👋 Welcome to Bangla TTS Bot!\n\n"
        "This bot converts long Bangla text into natural MP3 voice, "
        "with no limit on how much text you send.\n\n"
        "How to use:\n"
        "1. Send { to start collecting text\n"
        "2. Send as many messages as you want\n"
        "3. Send } to finish and generate voice\n\n"
        "Commands:\n"
        "/start - Show this welcome message\n"
        "/help - Show detailed usage instructions\n"
        "/status - Show current collection status\n"
        "/cancel - Cancel the current collection"
    )
    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    text = (
        "📖 Help\n\n"
        "This bot collects your Bangla text across multiple messages and "
        "converts the whole thing into one MP3 voice file.\n\n"
        "Starting a collection:\n"
        "Send a message containing only { to begin.\n\n"
        "Adding text:\n"
        "Send any number of messages. Each one is appended to your "
        "collection exactly as sent, including line breaks and any "
        "curly braces inside the text.\n\n"
        "Finishing a collection:\n"
        "Send a message containing only } to stop collecting and start "
        "voice generation.\n\n"
        "Commands:\n"
        "/status - Check messages collected, characters, and estimated "
        "voice duration\n"
        "/cancel - Cancel the current collection and delete all "
        "collected text\n\n"
        "Notes:\n"
        "- A collection automatically times out after 30 minutes of "
        "inactivity.\n"
        "- Long text is automatically split into safe chunks and merged "
        "back into one final MP3."
    )
    await update.message.reply_text(text)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    session = get_session(user_id, chat_id)
    await update.message.reply_text(format_final_status_text(session))


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cancel command."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    session = get_session(user_id, chat_id)

    if not session.collecting:
        await update.message.reply_text("No collection is running.")
        return

    await close_status_message(
    context,
    session,
    "❌ Collection Cancelled."
    )

    session.reset()

    await update.message.reply_text("✅ Collection Cancelled.")

# ============================================================
# Message Handler
# ============================================================

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all incoming text messages: collection start/end/collect."""
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    raw_text = update.message.text
    stripped = raw_text.strip()

    session = get_session(user_id, chat_id)

    # ---- Collection start trigger: exact "{" ----
    if stripped == "{":
        if session.collecting:
            await update.message.reply_text("Collection already running.")
            return

        session.reset()
        session.collecting = True
        session.start_time = time.time()
        session.last_activity = time.time()

        status_message = await update.message.reply_text(format_status_text(session))
        session.status_message_id = status_message.message_id
        return

    # ---- Collection end trigger: exact "}" ----
    if stripped == "}":
        if not session.collecting:
            await update.message.reply_text(
                "No collection is running.\n\nSend { to start collecting text."
            )
            return

        if not session.messages:
            await update.message.reply_text("No text collected.")
            session.reset()
            return

        session.collecting = False
        await run_voice_generation(update, context, session)
        return

    # ---- Normal text while collecting: append to session ----
    if session.collecting:
        session.messages.append(raw_text + "\n")
        session.characters += len(raw_text)
        session.last_activity = time.time()

        if session.status_message_id:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=session.status_message_id,
                    text=format_status_text(session),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to edit status message: %s", exc)
        return

    # ---- Normal text with no collection running ----
    await update.message.reply_text(
        "No collection is running.\n\nSend { to start collecting text."
    )


# ============================================================
# Voice Generation Pipeline
# ============================================================

async def run_voice_generation(
    update: Update, context: ContextTypes.DEFAULT_TYPE, session: UserSession
) -> None:
    """Run the full text-to-speech generation pipeline for a session."""
    chat_id = session.chat_id
    full_text = "".join(session.messages)
    message_count = len(session.messages)
    character_count = session.characters
    duration_estimate = estimate_duration(character_count)

    progress_message = await context.bot.send_message(chat_id=chat_id, text="🎙️ Preparing...")

    work_dir = tempfile.mkdtemp(prefix="bangla_tts_")
    chunk_paths: list[str] = []
    merged_path: Optional[str] = None

    try:
        voice = await resolve_voice()

        chunks = split_text_safely(full_text, CHUNK_SIZE)
        total_chunks = len(chunks)

        await progress_message.edit_text(f"Generating Voice...\n\nChunk 0 / {total_chunks}")

        async def generate_with_progress(index: int, chunk_text: str) -> str:
            return await generate_tts_chunk(chunk_text, voice, index, work_dir)

        tasks = [
            generate_with_progress(i, chunk_text)
            for i, chunk_text in enumerate(chunks, start=1)
        ]

        completed = 0
        for coro in asyncio.as_completed(tasks):
            path = await coro
            chunk_paths.append(path)
            completed += 1
            try:
                await progress_message.edit_text(
                    f"Generating Voice...\n\nChunk {completed} / {total_chunks}"
                )
            except Exception:  # noqa: BLE001
                pass

        # Chunk filenames are zero-padded by index, so sorting by path
        # restores the original text order regardless of completion order.
        chunk_paths.sort(key=lambda p: p)

        await progress_message.edit_text("🔗 Merging Audio...")
        merged_path = os.path.join(work_dir, "final_output.mp3")
        await asyncio.to_thread(merge_audio_files, chunk_paths, merged_path)

        await progress_message.edit_text("📤 Uploading...")

        caption = (
            "🎧 Bangla TTS\n\n"
            f"Characters : {character_count}\n"
            f"Messages : {message_count}\n"
            f"Chunks : {total_chunks}\n"
            f"Duration : {duration_estimate}\n"
            f"Voice : {voice}"
        )

        with open(merged_path, "rb") as audio_file:
            await context.bot.send_audio(
                chat_id=chat_id,
                audio=audio_file,
                caption=caption,
                title="Bangla TTS",
            )

        await progress_message.edit_text("✅ Completed.")
        logger.info(
            "Voice generation completed for user %s: %d chars, %d chunks",
            session.user_id, character_count, total_chunks,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("Voice generation failed for user %s: %s", session.user_id, exc)
        try:
            await progress_message.edit_text("⚠️ Voice Generate করা যায়নি।\n\nআবার চেষ্টা করুন।")
        except Exception:  # noqa: BLE001
            await context.bot.send_message(
                chat_id=chat_id,
                text="⚠️ Voice Generate করা যায়নি।\n\nআবার চেষ্টা করুন।",
            )

    finally:
        cleanup_files(chunk_paths)
        cleanup_files([merged_path] if merged_path else [])
        try:
            os.rmdir(work_dir)
        except OSError:
            pass
        session.reset()


# ============================================================
# Timeout Job
# ============================================================

async def check_timeouts(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Periodic job: cancel any session inactive for over 30 minutes."""
    now = time.time()
    for user_id, session in list(SESSIONS.items()):
        if not session.collecting:
            continue
        if now - session.last_activity >= COLLECTION_TIMEOUT_SECONDS:
            chat_id = session.chat_id
            session.reset()
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="⌛ Collection Timeout.\n\nSession Cancelled.",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to notify user %s of timeout: %s", user_id, exc)
            logger.info("Session for user %s timed out and was cancelled", user_id)


# ============================================================
# Error Handler
# ============================================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler: log every exception, never crash the bot."""
    logger.error(
        "Unhandled exception while processing update: %s",
        context.error,
        exc_info=context.error,
    )

    if isinstance(update, Update) and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⚠️ An unexpected error occurred. Please try again.",
            )
        except Exception:  # noqa: BLE001
            pass


# ============================================================
# Application Setup
# ============================================================

async def on_startup(application: Application) -> None:
    """Resolve and cache the TTS voice once when the bot starts."""
    voice = await resolve_voice()
    logger.info("Bot starting up. Cached voice: %s", voice)


def main() -> None:
    """Build and run the Telegram bot application."""
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(on_startup)
        .build()
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("cancel", cancel_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    application.add_error_handler(error_handler)

    if application.job_queue is not None:
        application.job_queue.run_repeating(
            check_timeouts, interval=TIMEOUT_CHECK_INTERVAL, first=TIMEOUT_CHECK_INTERVAL
        )
    else:
        logger.warning(
            "JobQueue unavailable. Install python-telegram-bot[job-queue] for auto-timeout support."
        )

    logger.info("Bot polling started.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
