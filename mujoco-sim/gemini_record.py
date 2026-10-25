"""Record the demo video with Gemini Live as the robot's voice and an older woman's voice as the person.

    .venv/bin/python gemini_record.py                  # runs the robot, writes demo_gemini.mp4
    .venv/bin/python gemini_record.py --compose-only   # re-makes it from the saved recording (demo_rec_gemini/)

Margaret's lines (voices/*.wav, Gemini TTS; voices/make_voices.py) go into gemini_demo.py's Live
session as if from a microphone; Gemini answers aloud and calls the robot tools, which run
Dream-1 and the SO-101 through Viam. The Viam camera frames, both voices and the captions are put
together by demo_record.compose.
"""
from __future__ import annotations

import argparse
import asyncio
import builtins
import os
import shutil
import sys
import tempfile
import time
import wave

from viam.robot.client import RobotClient

import demo_record as dr
import gemini_demo as gd
import yam_viam as yv

HERE = os.path.dirname(os.path.abspath(__file__))
VOICES = os.path.join(HERE, "voices")
LINES = ["1_apple", "2_orange", "3_banana", "stop_4", "5_water"]
STOP_AFTER_S = 6.0      # the stop line comes this long after the banana task starts

dr.OUT = os.path.join(HERE, "demo_gemini.mp4")
dr.REC_DIR = os.path.join(HERE, "demo_rec_gemini")
dr.TITLE = "Talk to a helper robot   |   voice: Gemini Live  >  brain: Dream-1  >  body: Viam"
dr.SUBTITLE = "Just talk · Gemini Live understands and answers · Dream-1 plans the moves · Viam runs the robot"
dr.HEARD_LABEL = "GEMINI HEARD"
dr.FAST = 2.4           # the silent robot stretches (labelled on screen), so the Gemini take fits 2 minutes


class Retake(Exception):
    pass


class Take:
    """One recording: frames from Viam, Margaret's lines, Gemini's audio, captions."""

    def __init__(self, robot, tmp):
        self.rec = dr.Recorder(robot, tmp)
        self.tmp = tmp
        self.out_chunks = []          # (t, pcm) as the speaker plays Gemini's audio
        self.line_t = 0.0
        self.task_started = None
        self.results = []             # "[put apple into the bowl: done ..." lines
        self.report_t = None          # when the robot last reported a result (Gemini then speaks about it)

    # ---- what the demo prints becomes captions --------------------------------------
    def on_print(self, line):
        r = self.rec
        if line.startswith("you: "):
            r.events.append((self.line_t, "you", line[5:]))
            r.events.append((self.line_t + 0.01, "whisper", line[5:]))
        elif line.startswith("robot: "):
            # printed when Gemini's turn completes: the caption goes where that utterance began
            r.events.append((self.utterance_start(r.now()), "robot", line[7:]))
            r.events.sort(key=lambda e: e[0])
        elif line.startswith("  [tool] put_fruit"):
            self.task_started = r.now()
            r.events.append((r.now(), "status", "Dream-1 is moving the arms, live through Viam"))
        elif line.startswith("  [tool] get_water"):
            self.task_started = r.now()
            r.camera = "mj-cam-front"
            r.events.append((r.now(), "status", "SO-101 + Viam motion planner: fetch and hand over"))
        elif line.startswith("  [tool] reset_table"):
            r.events.append((r.now(), "status", "Viam: resetting the table"))
        elif line.startswith("  [put ") or line.startswith("  [robot]") or line.startswith("  [tool] stop"):
            if line.startswith("  [put "):
                self.results.append(line)
            self.report_t = r.now()
            r.events.append((r.now(), "status", ""))

    def utterance_start(self, t):
        """Start of the newest stretch of Gemini audio received before t (0.3 s gaps split stretches)."""
        times = [ct for ct, _ in self.out_chunks if ct <= t]
        if not times:
            return t
        start = times[-1]
        for prev in reversed(times[:-1]):
            if start - prev > 0.3:
                break
            start = prev
        return start

    # ---- Gemini's voice, timed as it is played ---------------------------------------
    def speaker_class(self):
        take = self

        class RecSpeaker(gd.Speaker):
            def _cb(self, outdata, frames, t, status):
                n = frames * 2
                chunk = bytes(self.buf[:n])
                del self.buf[:n]
                if chunk:
                    self.last_audio = time.monotonic()
                    take.out_chunks.append((take.rec.now(), chunk))
                outdata[:] = chunk + b"\0" * (n - len(chunk))
        return RecSpeaker

    def robot_clips(self):
        """Gemini's audio as one WAV per utterance (a 0.3 s gap starts a new one)."""
        clips, cur, start, last = [], bytearray(), None, None
        for t, pcm in self.out_chunks:
            if last is not None and t - last > 0.3 and cur:
                clips.append((start, bytes(cur)))
                cur, start = bytearray(), None
            if start is None:
                start = t
            cur.extend(pcm)
            last = t + len(pcm) / 2 / gd.OUT_RATE
        if cur:
            clips.append((start, bytes(cur)))
        for n, (t, pcm) in enumerate(clips):
            path = os.path.join(self.tmp, f"gemini_{n:03d}.wav")
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(gd.OUT_RATE)
                w.writeframes(pcm)
            self.rec.audio.append((t, path))


async def wait_reply(take: Take, speaker, after, timeout=15.0):
    """Let Gemini finish what it says about a result: its audio starts after `after`, then goes quiet."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not any(ct > after for ct, _ in take.out_chunks):
        await asyncio.sleep(0.1)
    while speaker.talking():
        await asyncio.sleep(0.1)
    await asyncio.sleep(0.8)


async def play_lines(session, robot, speaker, take: Take):
    """gemini_demo.wav_loop, timed for the video and the stop line."""
    async def turn(timeout):
        try:
            await asyncio.wait_for(robot.turn_done.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        robot.turn_done.clear()

    await turn(20)
    for name in LINES:
        path = os.path.join(VOICES, f"{name}.wav")
        if name.startswith("stop"):
            while not robot.is_busy():
                await asyncio.sleep(0.1)
            while take.rec.now() - (take.task_started or take.rec.now()) < STOP_AFTER_S or speaker.talking():
                await asyncio.sleep(0.1)
        else:
            # every fruit asked for so far must have reported (Gemini may reset the table first, so the
            # task can start a few seconds after the line)
            expected = {"2_orange": 1, "3_banana": 2}.get(name, 0)
            deadline = time.monotonic() + 120
            while len(take.results) < expected and time.monotonic() < deadline:
                await asyncio.sleep(0.2)
            while speaker.talking() or robot.is_busy():
                await asyncio.sleep(0.2)
            if take.report_t is not None:                # never talk over Gemini's report of the last task
                await wait_reply(take, speaker, take.report_t)
            await asyncio.sleep(1.0)
            if name == "2_orange" and not any("apple into the bowl: done" in r for r in take.results):
                raise Retake("the apple missed")
            if name == "3_banana" and not any("orange into the bowl: done" in r for r in take.results):
                raise Retake("the orange missed")
        robot.turn_done.clear()
        take.line_t = take.rec.now()
        take.rec.audio.append((take.line_t, path))
        await gd.send_line(session, path, manual=True)
        await turn(20)
    # the last task, then Gemini telling the person how it went: wait for that reply to be spoken
    while robot.is_busy():
        await asyncio.sleep(0.3)
    done_at = take.rec.now()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not any(ct > done_at for ct, _ in take.out_chunks):
        await asyncio.sleep(0.2)
    await turn(15)
    while speaker.talking():
        await asyncio.sleep(0.3)
    await asyncio.sleep(2.0)


async def one_take(take_no):
    yv.load_env()
    tmp = tempfile.mkdtemp(prefix="gemini_rec_")
    cam_robot = await RobotClient.at_address(yv.ADDRESS, RobotClient.Options.with_api_key(
        api_key=os.environ["SO101_API_KEY"], api_key_id=os.environ["SO101_API_KEY_ID"]))
    take = Take(cam_robot, tmp)
    real_print = builtins.print

    def tee(*a, **k):
        real_print(*a, **k)
        if a:
            take.on_print(" ".join(str(x) for x in a))
    gd.print = tee                                  # gemini_demo's prints become captions
    gd.Speaker = take.speaker_class()
    gd.wav_loop = lambda session, paths, robot, speaker, manual=True: play_lines(session, robot, speaker, take)
    args = argparse.Namespace(headphones=False, wav=["(lines)"], model=gd.MODEL, voice="Puck", manual_turns=True)
    take.rec.t0 = time.monotonic()
    take.rec.events.append((0.0, "title", ""))
    cap = asyncio.create_task(take.rec.capture_loop())
    try:
        await asyncio.sleep(3.0)                    # the title card
        await gd.main(args)
        ok = True
    except Retake as why:
        print(f"take {take_no}: {why}")
        ok = False
    finally:
        take.rec.running = False
        await cap
        await cam_robot.close()
    if not ok:
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    take.robot_clips()
    print(f"captured {len(take.rec.frames)} frames over {take.rec.now():.0f}s; composing...", flush=True)
    shutil.rmtree(dr.REC_DIR, ignore_errors=True)
    dr.save(take.rec)
    dr.compose(take.rec, tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {dr.OUT}")
    return True


async def main():
    if "--compose-only" in sys.argv:
        tmp = tempfile.mkdtemp(prefix="gemini_rec_")
        dr.compose(dr.load(), tmp)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"wrote {dr.OUT}")
        return
    for n in (1, 2, 3):
        if await one_take(n):
            return
    print("no clean take; nothing written")


if __name__ == "__main__":
    asyncio.run(main())
