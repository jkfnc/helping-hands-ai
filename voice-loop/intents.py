"""Transcript -> intent mapping and (dry-run) arm dispatch, shared by server.py and listen.py.

Arm actions are logged only unless ARM_LIVE=1, and live actions are not wired yet,
so a recognised command is announced and logged but nothing moves.
"""
from __future__ import annotations

import logging
import os
import re

log = logging.getLogger("voice-loop")
ARM_LIVE = os.environ.get("ARM_LIVE") == "1"

INTENTS = [
    # (name, pattern, spoken reply)
    ("stop", r"\b(stop|halt|freeze|emergency)\b", "Stopping."),
    ("goodbye", r"\b(goodbye|bye|hang up|that's all)\b", "Goodbye."),
    ("open", r"\b(open|release|let go)\b", "Yes, I will open the gripper."),
    ("pick_up", r"\b(pick up|grab|hand me|give me|get)\b.*\b(cube|block)\b", "Yes, I will."),
    ("get_water", r"\b(get|bring|fetch|give|hand)\b.*\b(water|cup)\b", "Yes, I will get your water."),
]
COLOR = re.compile(r"\b(green|red|blue|yellow)\b")


def parse(text: str):
    text = text.lower().strip()
    for name, pattern, reply in INTENTS:
        if re.search(pattern, text):
            color = COLOR.search(text)
            return name, color.group(1) if color else None, reply
    return None, None, "Sorry, I did not catch that. Try: pick up the green cube."


def dispatch(intent: str, color: str | None):
    # Imported here so the voice loop starts without numpy/viam-sdk until needed.
    import water_fetch

    if intent == "stop":
        log.info("ARM stop requested (live=%s)", ARM_LIVE)
        water_fetch.ABORT.set()
    elif intent == "open":
        log.info("ARM release requested (live=%s)", ARM_LIVE)
        water_fetch.RELEASE.set()
    elif intent == "pick_up":
        log.info("ARM pick up %s cube (live=%s) - not wired yet", color or "any", ARM_LIVE)
    elif intent == "get_water":
        started = water_fetch.start_in_background(dry_run=not ARM_LIVE)
        log.info("ARM water fetch %s (live=%s)", "started" if started else "already running", ARM_LIVE)
