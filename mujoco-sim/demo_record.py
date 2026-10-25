"""Record the 2-minute demo video, with the Mac's voices playing both sides.

    .venv/bin/python demo_record.py                  # runs the robot, writes demo.mp4 next to this file
    .venv/bin/python demo_record.py --compose-only   # re-makes demo.mp4 from the saved recording (demo_rec/)

A person's lines are spoken by one macOS voice, written to audio, and transcribed by the demo's
own local Whisper helper (ears.py); what Whisper heard goes to the demo's own command handler
(demo.py), which runs Dream-1 and the SO-101 through Viam exactly as live. The robot answers in
a second voice. Meanwhile the Viam camera the audience would watch is captured, and at the end
the frames, the captions and both voices are put together with ffmpeg.
"""
from __future__ import annotations

import asyncio
import io
import os
import shutil
import subprocess
import tempfile
import time
import wave

from PIL import Image, ImageDraw, ImageFont
from viam.components.camera import Camera
from viam.robot.client import RobotClient

import yam_viam as yv
from demo import Demo
from robot_tasks import RobotTasks

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "demo.mp4")
PERSON_VOICE, ROBOT_VOICE = "Samantha", "Daniel"
FPS = 10
W, H = 1280, 720
FONT = "/System/Library/Fonts/Helvetica.ttc"
TITLE = "Talk to a helper robot   |   voice: local Whisper  >  brain: Dream-1  >  body: Viam"
SUBTITLE = "Voice stays on the laptop · Viam runs the robot · Dream-1 plans the moves"
HEARD_LABEL = "WHISPER HEARD"


def synth(text, voice, path):
    subprocess.run(["say", "-v", voice, "-r", "185", "-o", path, "--data-format=LEI16@16000", text], check=True)
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


class Recorder:
    def __init__(self, robot, tmp):
        self.robot, self.tmp = robot, tmp
        self.t0 = time.monotonic()
        self.frames = []          # (t, jpeg bytes)
        self.events = []          # (t, who, text)
        self.audio = []           # (t, wav path)
        self.camera = "mj-cam-top"
        self.n = 0
        self.running = True

    def now(self):
        return time.monotonic() - self.t0

    async def capture_loop(self):
        while self.running:
            t = time.monotonic()
            try:
                images, _ = await Camera.from_robot(self.robot, self.camera).get_images()
                self.frames.append((self.now(), images[0].data))
            except Exception:  # noqa: BLE001  a frame lost while a scene loads
                pass
            await asyncio.sleep(max(0.0, 1 / FPS - (time.monotonic() - t)))

    async def speak(self, who, text, voice):
        self.n += 1
        path = os.path.join(self.tmp, f"{self.n:03d}_{who}.wav")
        dur = synth(text, voice, path)
        self.events.append((self.now(), who, text))
        self.audio.append((self.now(), path))
        await asyncio.sleep(dur + 0.3)       # keep the timeline in step with the speech
        return path


class RecordedDemo(Demo):
    """demo.py's Demo with the robot's voice recorded instead of played."""

    def __init__(self, rt, rec):
        super().__init__(rt)
        self.rec = rec
        self.last = None          # the Result of the newest fruit task

    async def say(self, text):
        print(f"robot: {text}", flush=True)
        await self.rec.speak("robot", text, ROBOT_VOICE)

    def status(self, text):
        self.rec.events.append((self.rec.now(), "status", text))

    async def one_task(self, fruit, target, announce=True):
        if announce:
            hint = " I'll try, the banana is tricky." if fruit == "banana" else ""
            await self.say(f"Putting the {fruit} {'in the bowl' if target == 'bowl' else 'on the plate'}.{hint}")
        self.status("Dream-1 is moving the arms, live through Viam")
        try:
            self.last = await super().one_task(fruit, target, announce=False)
            return self.last
        finally:
            self.status("")

    async def water(self):
        self.status("SO-101 + Viam motion planner: fetch and hand over")
        try:
            await super().water()
        finally:
            self.status("")

    async def hear(self, wav):
        self.ears.stdin.write(f"file {wav}\n".encode())
        await self.ears.stdin.drain()
        while True:
            line = (await self.ears.stdout.readline()).decode().strip()
            if line.startswith("HEARD:"):
                return line[6:].strip()


async def person(demo, rec, text):
    print(f"you: {text}", flush=True)
    wav = await rec.speak("you", text, PERSON_VOICE)
    heard = await demo.hear(wav)
    rec.events.append((rec.now(), "whisper", heard))
    print(f"  whisper heard: {heard!r}", flush=True)
    await demo.handle(heard)


class Retake(Exception):
    """A task missed on camera: record the whole take again."""


async def wait_task(demo):
    if demo.busy:
        await demo.busy


async def script(demo, rec):
    rec.events.append((rec.now(), "title", ""))
    await asyncio.sleep(3.0)
    await person(demo, rec, "Robot, put the apple in the bowl.")
    await wait_task(demo)
    if not (demo.last and demo.last.success):
        raise Retake("the apple missed")
    # the orange only lands from a fresh table (with the apple already in the bowl it missed 3 of 3)
    await person(demo, rec, "Reset.")
    await wait_task(demo)
    await person(demo, rec, "Now put the orange in the bowl.")
    await wait_task(demo)
    if not (demo.last and demo.last.success):
        raise Retake("the orange missed")
    await person(demo, rec, "Put the banana on the plate.")
    await asyncio.sleep(8.0)                # let it get going, then...
    await person(demo, rec, "Stop!")
    await wait_task(demo)
    await asyncio.sleep(1.0)
    rec.camera = "mj-cam-front"             # act 2 is watched from the cup scene's camera
    await person(demo, rec, "Can you get me some water?")
    await wait_task(demo)
    await asyncio.sleep(2.0)


# ---- the video -----------------------------------------------------------------------
def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def caption_at(events, t):
    """The newest line of each kind before t; a new thing the person says clears the old reply."""
    shown = {}
    for et, who, text in events:
        if et > t:
            break
        if who == "you":
            shown = {}
        if who in ("you", "whisper", "robot", "status"):
            shown[who] = text
    return shown


FAST = 1.5          # playback speed while nobody is speaking (the robot working), to fit the 2-minute slot


def timeline(rec, end):
    """Recording time -> video time: 1x while a voice is speaking (and around the title), FAST otherwise.

    @return (src, out): matching times every 10 ms, for np.interp either way.
    """
    import numpy as np
    speech = [(0.0, 3.5)]
    for t, path in rec.audio:
        with wave.open(path) as w:
            speech.append((t - 0.3, t + w.getnframes() / w.getframerate() + 0.5))
    src = np.arange(0.0, end + 0.01, 0.01)
    slow = np.zeros_like(src, dtype=bool)
    for a, b in speech:
        slow |= (src >= a) & (src <= b)
    rate = np.where(slow, 1.0, 1.0 / FAST)
    out = np.concatenate([[0.0], np.cumsum(rate[:-1] * 0.01)])
    return src, out, slow


def compose(rec, tmp):
    import numpy as np
    f_title, f_big, f_small = font(34), font(30), font(22)
    frames_dir = os.path.join(tmp, "frames")
    os.makedirs(frames_dir)
    end = max(t for t, _ in rec.frames) + 1.0
    src, out_t, slow = timeline(rec, end)
    i, k, last = 0, 0, None
    while i / FPS < out_t[-1]:
        t = float(np.interp(i / FPS, out_t, src))       # the recording time this video frame shows
        fast = not slow[min(int(t / 0.01), len(slow) - 1)]
        while k + 1 < len(rec.frames) and rec.frames[k + 1][0] <= t:
            k += 1
        img = Image.open(io.BytesIO(rec.frames[k][1])).convert("RGB")
        iw, ih = img.size
        target_h = int(iw * 9 / 16)
        if ih > target_h:
            img = img.crop((0, (ih - target_h) // 2, iw, (ih - target_h) // 2 + target_h))
        img = img.resize((W, H), Image.BILINEAR)
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle((0, 0, W, 56), fill=(10, 20, 40, 200))
        d.text((24, 12), TITLE, font=f_title, fill="white")
        if t < 3.0:
            d.rectangle((0, H // 2 - 70, W, H // 2 + 70), fill=(10, 20, 40, 210))
            d.text((W // 2, H // 2 - 20), "Say it. The robot does it.", font=font(52), fill="white", anchor="mm")
            d.text((W // 2, H // 2 + 36), SUBTITLE, font=f_small, fill=(200, 215, 235), anchor="mm")
        shown = caption_at(rec.events, t)
        y = H - 20
        if shown.get("status"):
            d.rectangle((W - 24 - d.textlength(shown["status"], font=f_small) - 16, 68, W - 24, 104), fill=(0, 90, 60, 200))
            d.text((W - 32, 74), shown["status"], font=f_small, fill="white", anchor="ra")
        for who, label, color in (("robot", "ROBOT", (255, 214, 102)), ("whisper", HEARD_LABEL, (150, 220, 255)),
                                  ("you", "YOU", (255, 255, 255))):
            if who in shown:
                # wrap to the frame: the label on the first line, the rest indented under it
                words, lines, cur = f"{label}:  {shown[who]}".split(" "), [], ""
                for w_ in words:
                    trial = (cur + " " + w_).strip()
                    if cur and d.textlength(trial, font=f_big) > W - 64:
                        lines.append(cur)
                        cur = "      " + w_
                    else:
                        cur = trial if not cur.startswith("      ") else cur + " " + w_
                lines.append(cur)
                for line in reversed(lines):
                    tw = d.textlength(line, font=f_big)
                    y -= 44
                    d.rectangle((16, y - 4, 32 + tw, y + 38), fill=(0, 0, 0, 170))
                    d.text((24, y), line, font=f_big, fill=color)
                y -= 4
        if fast:
            d.rectangle((W - 104, H - 60, W - 24, H - 20), fill=(0, 0, 0, 170))
            d.text((W - 64, H - 40), f"{FAST:g}x", font=f_big, fill="white", anchor="mm")
        img.save(os.path.join(frames_dir, f"{i:05d}.jpg"), quality=88)
        i += 1
    video = os.path.join(tmp, "video.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i",
                    os.path.join(frames_dir, "%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", video], check=True)
    # both voices on one track, each clip at the moment it was spoken
    inputs, filters = [], []
    for n, (t, path) in enumerate(rec.audio):
        inputs += ["-i", path]
        ms = int(float(np.interp(t, src, out_t)) * 1000)
        filters.append(f"[{n + 1}:a]aresample=24000,adelay={ms}|{ms}[a{n}]")
    mix = "".join(f"[a{n}]" for n in range(len(rec.audio)))
    filters.append(f"{mix}amix=inputs={len(rec.audio)}:normalize=0[aout]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, *inputs, "-filter_complex", ";".join(filters),
                    "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-shortest", OUT], check=True)


REC_DIR = os.path.join(HERE, "demo_rec")   # the raw recording, kept so the video can be re-composed


def save(rec):
    import json
    os.makedirs(os.path.join(REC_DIR, "frames"), exist_ok=True)
    meta = {"frames": [], "events": rec.events, "audio": []}
    for n, (t, data) in enumerate(rec.frames):
        name = f"frames/{n:05d}.jpg"
        open(os.path.join(REC_DIR, name), "wb").write(data)
        meta["frames"].append([t, name])
    for t, path in rec.audio:
        shutil.copy(path, os.path.join(REC_DIR, os.path.basename(path)))
        meta["audio"].append([t, os.path.basename(path)])
    json.dump(meta, open(os.path.join(REC_DIR, "recording.json"), "w"))


def load():
    import json
    meta = json.load(open(os.path.join(REC_DIR, "recording.json")))
    rec = Recorder(None, REC_DIR)
    rec.frames = [(t, open(os.path.join(REC_DIR, name), "rb").read()) for t, name in meta["frames"]]
    rec.events = [tuple(e) for e in meta["events"]]
    rec.audio = [(t, os.path.join(REC_DIR, name)) for t, name in meta["audio"]]
    return rec


async def main():
    import sys
    if "--compose-only" in sys.argv:
        tmp = tempfile.mkdtemp(prefix="demo_rec_")
        compose(load(), tmp)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"wrote {OUT}")
        return
    yv.load_env()
    tmp = tempfile.mkdtemp(prefix="demo_rec_")
    robot = await RobotClient.at_address(yv.ADDRESS, RobotClient.Options.with_api_key(
        api_key=os.environ["SO101_API_KEY"], api_key_id=os.environ["SO101_API_KEY_ID"]))
    rt = RobotTasks(robot)
    takes = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--takes=")), 2))
    try:
        for take in range(1, takes + 1):
            rec = Recorder(robot, tmp)
            demo = RecordedDemo(rt, rec)
            try:
                if not await demo.start_ears():
                    raise SystemExit("Whisper helper did not start")
                await rt.reset_yam()
                rec.t0 = time.monotonic()
                cap = asyncio.create_task(rec.capture_loop())
                try:
                    await script(demo, rec)
                    break
                except Retake as why:
                    print(f"take {take}: {why}" + ("; recording again" if take < takes else "; out of takes"), flush=True)
                    if take == takes:
                        raise SystemExit("no clean take; nothing written")
                finally:
                    rec.running = False
                    await cap
            finally:
                if demo.ears:
                    demo.ears.kill()
    finally:
        rt.close()
        await robot.close()
    print(f"captured {len(rec.frames)} frames over {rec.now():.0f}s; composing...", flush=True)
    shutil.rmtree(REC_DIR, ignore_errors=True)
    save(rec)
    compose(rec, tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
