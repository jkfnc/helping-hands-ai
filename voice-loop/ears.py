"""Local Whisper as a helper process: the demo writes "listen" on stdin, this records one
utterance from the mic, transcribes it on this machine, and prints "HEARD: <text>".

Recording starts at once and ends after a short silence once speech has started (or at
MAX_SECONDS), so a quick "stop" comes back fast. Run with the isolated env (run-ears.sh), so the
model cache stays in whisper-local/hf-cache.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

RATE = 16000
BLOCK = 1600            # 100 ms
MAX_SECONDS = 6.0
END_SILENCE_S = 0.8     # this much quiet after speech ends the utterance
START_TIMEOUT_S = 4.0   # give up if nobody speaks
# words Whisper should expect, so "banana" and "bowl" come through clean
VOCAB = "robot, apple, orange, banana, bowl, plate, stop, reset, clean up the table, get me water"


def out(line: str) -> None:
    print(line, flush=True)


def record() -> np.ndarray:
    """One utterance: wait for speech, then stop after END_SILENCE_S of quiet."""
    blocks, started, quiet, t0 = [], False, 0.0, time.monotonic()
    noise = None
    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=BLOCK) as stream:
        while time.monotonic() - t0 < MAX_SECONDS:
            block, _ = stream.read(BLOCK)
            block = block[:, 0]
            blocks.append(block)
            level = float(np.sqrt(np.mean(block ** 2)))
            if noise is None:
                noise = level
            loud = level > max(0.01, 3.0 * noise)
            if loud:
                started, quiet = True, 0.0
            elif started:
                quiet += BLOCK / RATE
                if quiet >= END_SILENCE_S:
                    break
            else:
                noise = 0.9 * noise + 0.1 * level
                if time.monotonic() - t0 > START_TIMEOUT_S:
                    break
    return np.concatenate(blocks)


def main() -> None:
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    out("READY")
    for line in sys.stdin:
        line = line.strip()
        if line.startswith("file "):
            # a recorded utterance (the demo recorder's): same model, same settings, no mic
            audio = line[5:]
        elif line == "listen":
            out("LISTENING")
            audio = record()
        else:
            continue
        segments, _ = model.transcribe(audio, language="en", vad_filter=True, initial_prompt=VOCAB)
        out("HEARD: " + " ".join(s.text.strip() for s in segments).strip())


if __name__ == "__main__":
    main()
