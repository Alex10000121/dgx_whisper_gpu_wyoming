import asyncio
import subprocess
import logging
import uuid
import os

from wyoming.server import AsyncTcpServer, AsyncEventHandler
from wyoming.audio import AudioChunk, AudioStop, AudioStart
from wyoming.asr import Transcript
from wyoming.info import AsrModel, AsrProgram, Attribution, Info, Describe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WHISPER_BIN = "/app/whisper-core/build/bin/whisper-cli"
MODEL_NAME  = os.environ.get("WHISPER_MODEL", "large-v3")
MODEL_PATH  = f"/data/ggml-{MODEL_NAME}.bin"
LANGUAGE    = os.environ.get("WHISPER_LANG", "de")
BEAM_SIZE   = os.environ.get("WHISPER_BEAM_SIZE", "5")

HF_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"


def ensure_model():
    if os.path.exists(MODEL_PATH):
        logger.info(f"Modell gefunden: {MODEL_PATH}")
        return
    logger.info(f"Modell nicht gefunden – lade {MODEL_NAME} herunter...")
    os.makedirs("/data", exist_ok=True)
    url = f"{HF_BASE}/ggml-{MODEL_NAME}.bin"
    result = subprocess.run(["wget", "-q", "--show-progress", "-O", MODEL_PATH, url])
    if result.returncode != 0:
        raise RuntimeError(f"Download fehlgeschlagen: {url}")
    logger.info(f"Modell gespeichert: {MODEL_PATH}")


class WhisperHandler(AsyncEventHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio = b''
        self._session_id = uuid.uuid4().hex
        self._pcm = f"/tmp/audio_{self._session_id}.pcm"
        self._wav = f"/tmp/audio_{self._session_id}.wav"

    async def handle_event(self, event):
        if Describe.is_type(event.type):
            info = Info(asr=[AsrProgram(
                name="whisper-gpu",
                description="Whisper.cpp GPU (GB10 Grace Blackwell)",
                attribution=Attribution(name="ggerganov", url="https://github.com/ggerganov/whisper.cpp"),
                installed=True,
                models=[AsrModel(
                    name=MODEL_NAME,
                    description=f"Whisper {MODEL_NAME}",
                    attribution=Attribution(name="OpenAI", url="https://openai.com"),
                    installed=True,
                    languages=[LANGUAGE]
                )]
            )])
            await self.write_event(info.event())
            return True

        elif AudioStart.is_type(event.type):
            self.audio = b''
            return True

        elif AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.audio += chunk.audio
            return True

        elif AudioStop.is_type(event.type):
            logger.info(f"[{self._session_id[:8]}] Audio empfangen ({len(self.audio)} bytes) – starte Transkription")
            try:
                with open(self._pcm, "wb") as f:
                    f.write(self.audio)

                subprocess.run([
                    "ffmpeg", "-y",
                    "-f", "s16le", "-ar", "16000", "-ac", "1",
                    "-i", self._pcm, self._wav
                ], stderr=subprocess.DEVNULL, check=True)

                cmd = [
                    WHISPER_BIN,
                    "-m", MODEL_PATH,
                    "-f", self._wav,
                    "-l", LANGUAGE,
                    "--beam-size", BEAM_SIZE,
                    "-nt", "-np"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

                if res.returncode != 0:
                    logger.error(f"whisper-cli Fehler: {res.stderr}")
                    text = ""
                else:
                    text = res.stdout.strip()

                logger.info(f"[{self._session_id[:8]}] Erkannt: {text!r}")
                await self.write_event(Transcript(text=text).event())

            except Exception as e:
                logger.exception(f"Fehler bei Transkription: {e}")
                await self.write_event(Transcript(text="").event())
            finally:
                for f in [self._pcm, self._wav]:
                    try:
                        os.remove(f)
                    except FileNotFoundError:
                        pass
            return True

        return True


async def main():
    ensure_model()
    logger.info(f"Starte Wyoming Whisper GPU Server auf 0.0.0.0:10300")
    logger.info(f"Modell: {MODEL_PATH} | Sprache: {LANGUAGE} | Beam-Size: {BEAM_SIZE}")
    server = AsyncTcpServer("0.0.0.0", 10300)
    await server.run(WhisperHandler)


if __name__ == "__main__":
    asyncio.run(main())
