"""Push-to-talk voice loop: mic -> local Whisper -> intent -> spoken reply.

Press Enter, speak for a few seconds, and the transcript is matched against the
same intents as server.py. Arm actions are dry-run only (logged), nothing moves.

Run with the isolated env so the model cache lives in whisper-local/hf-cache:
    ./run-listen.sh
"""
from __future__ import annotations

import logging
import subprocess
import sys

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from intents import dispatch, parse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

RATE = 16000
SECONDS = 4


def record(seconds: int = SECONDS) -> np.ndarray:
    audio = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def say(text: str) -> None:
    print(f"robot: {text}")
    subprocess.run(["say", text], check=False)


def main() -> None:
    print("loading model (first run downloads ~75 MB)...")
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    say("Arm ready. Press enter, then tell me what to do.")
    while True:
        try:
            input("\n[enter] to talk, ctrl-c to quit ")
        except (KeyboardInterrupt, EOFError):
            break
        print(f"listening for {SECONDS}s...")
        segments, _ = model.transcribe(record(), language="en", vad_filter=True)
        heard = " ".join(s.text.strip() for s in segments).strip()
        print(f"heard: {heard!r}")
        if not heard:
            say("I did not hear anything.")
            continue
        intent, color, reply = parse(heard)
        if intent in ("stop", "pick_up", "open", "get_water"):
            dispatch(intent, color)
        say(reply)
        if intent == "goodbye":
            break


if __name__ == "__main__":
    sys.exit(main())
