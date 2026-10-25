"""Vonage Voice webhook server: call the number, speak a command, the arm hears it.

Vonage does the speech-to-text (NCCO `input` with speech) and the text-to-speech
(NCCO `talk`). This server only maps the transcript to an intent and replies.

Arm actions are dry-run (logged only) unless ARM_LIVE=1, and live actions are not
wired yet, so a recognised command is announced and logged but nothing moves.
"""
from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, request

from intents import dispatch, log, parse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

app = Flask(__name__)
BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")


def listen(prompt: str):
    """NCCO that says `prompt`, then listens for one spoken command."""
    return [
        {"action": "talk", "text": prompt, "bargeIn": False},
        {
            "action": "input",
            "type": ["speech"],
            "eventUrl": [f"{BASE_URL}/speech"],
            "eventMethod": "POST",
            "speech": {"language": "en-US", "endOnSilence": 1.5, "startTimeout": 8, "maxDuration": 10},
        },
    ]


@app.route("/answer", methods=["GET", "POST"])
def answer():
    return jsonify(listen("Arm ready. Tell me what to do."))


@app.route("/speech", methods=["POST"])
def speech():
    body = request.get_json(silent=True) or {}
    results = (body.get("speech") or {}).get("results") or []
    heard = results[0]["text"] if results else ""
    log.info("heard: %r", heard)
    if not heard:
        return jsonify(listen("I did not hear anything. Try again."))
    intent, color, reply = parse(heard)
    if intent in ("stop", "pick_up", "open", "get_water"):
        dispatch(intent, color)
    if intent == "goodbye":
        return jsonify([{"action": "talk", "text": reply}])
    return jsonify([{"action": "talk", "text": reply}] + listen("What next?"))


@app.route("/event", methods=["POST"])
def event():
    log.info("event: %s", (request.get_json(silent=True) or {}).get("status"))
    return "", 204


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5055)))
