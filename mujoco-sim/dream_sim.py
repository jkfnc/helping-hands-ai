"""Dream-1, zero-shot, driving the MuJoCo SO-101 through Viam.

    .venv/bin/python dream_sim.py --check                    # no model calls, no credits: build and validate only
    .venv/bin/python dream_sim.py "pick up the white cup and hand it to the person" --seconds 60

Each get_action() call costs Dream credits; the run stops after --seconds or --max-calls.

Dream-1's base models were trained on two YAM arms with three cameras. On another arm the
model acts on the cameras and the prompt and answers where the gripper should go; the
dream library's IK (dream/so101_arm.json: Dream's own simulation SO-101 URDF and tool
point, option (c), confirmed by the user) turns that into our five joints. Our arm runs
as the right arm; the left is parked with a black wrist camera, as Dream's docs say.

Scenes (--scene): "dream" (default) is Dream's own YAM table, bowl, plate and calibrated top
camera with our SO-101 where Dream's zero-shot SO-101 stands (scene_dream.py); "cup" is ours.
Cameras: mj-cam-top (dream) or mj-cam-front (cup) -> video.overhead, mj-cam-wrist -> video.wrist_r,
black -> video.wrist_l.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import math
import os
import threading
import time

import numpy as np
from PIL import Image

import dream
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.generic import Generic
from viam.proto.component.arm import JointPositions
from viam.robot.client import RobotClient

HERE = os.path.dirname(os.path.abspath(__file__))
ADDRESS = os.environ.get("SO101_ADDRESS", "so101-main.xjrc4czns9.viam.cloud")
ARM_FILE = os.path.join(HERE, "dream", "so101_arm.json")
CHECKPOINT = "dreamlabs/dream-1-beta"
JOINTS = 5
CAMERAS = ("overhead", "wrist_l", "wrist_r")
FRAME_W, FRAME_H = 320, 176   # what the model reads
ROW_HZ = 20.0                 # rows a second we play (the docs' and dream's own loop's rate)
ROW_S = 1.0 / ROW_HZ
FRAME_EVERY = 2               # grab fresh camera frames every N rows (10 Hz)
REST = [-0.106, -0.838, 0.74, 1.145, 0.0]          # Dream's rest for the SO-101's right side (the arm file's)
LEFT_PARKED = [0.104, -0.838, 0.74, 1.145, 0.0]   # Dream's rest for the SO-101's left side (never commanded)
RAMP_RAD_S = 0.5              # first move onto the model's targets: no joint faster than this

STATE = {}
for side in ("right", "left"):
    for part, width in (("arm", JOINTS), ("grip", 1), ("eef_xyz", 3), ("eef_rot6", 6)):
        start = sum(end - begin for begin, end in STATE.values())
        STATE[f"{side}_{part}"] = [start, start + width]
CONFIG = {
    "checkpoint": CHECKPOINT,
    "embodiment": {"state": STATE, "action": STATE,
                   "video": {role: f"observation.images.{role}" for role in CAMERAS}},
    "ik": {"arm": ARM_FILE, "head": "eef"},
    "action_hz": ROW_HZ,
}


def load_env():
    for line in open(os.path.join(HERE, "..", ".env")):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def to_frame(jpeg: bytes) -> str:
    """A camera image as the model reads it: center-cropped to 320:176, resized, JPEG q85, base64."""
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    w, h = img.size
    ch = int(w * FRAME_H / FRAME_W)
    if ch <= h:
        img = img.crop((0, (h - ch) // 2, w, (h - ch) // 2 + ch))
    img = img.resize((FRAME_W, FRAME_H), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def black_frame() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (FRAME_W, FRAME_H)).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


class Planner:
    """get_action() on its own thread, one call at a time (the docs' pattern)."""

    def __init__(self, client):
        self.client, self.thread, self.result, self.lock = client, None, None, threading.Lock()
        self.calls = 0

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def request(self):
        t_obs = time.monotonic()
        self.calls += 1

        def call():
            try:
                action = self.client.get_action()
                result = (t_obs, np.asarray(action["q_r_chunk"]))   # [48, 5 joints + grip]
            except dream.DreamError as error:
                result = error
            with self.lock:
                self.result = result

        self.thread = threading.Thread(target=call, daemon=True)
        self.thread.start()

    def take(self):
        with self.lock:
            result, self.result = self.result, None
        return result

    def wait(self):
        if self.thread is not None:
            self.thread.join()


class SimRobot:
    def __init__(self, robot, scene):
        self.arm = Arm.from_robot(robot, "mj-arm")
        # Dream's scene: its station's calibrated top camera, the view the model was trained on
        self.front = Camera.from_robot(robot, "mj-cam-top" if scene == "dream" else "mj-cam-front")
        self.wrist = Camera.from_robot(robot, "mj-cam-wrist")
        self.world = Generic.from_robot(robot, "mj-world")
        self.black = black_frame()
        self.frames = None

    async def state(self):
        s = await self.arm.do_command({"get_state": True})
        return [math.radians(v) for v in s["joints_deg"]], float(s["grip"])

    async def grab_frames(self):
        (f,), _ = await self.front.get_images()
        (w,), _ = await self.wrist.get_images()
        self.frames = {"video.overhead": to_frame(f.data), "video.wrist_l": self.black, "video.wrist_r": to_frame(w.data)}

    async def command(self, q_rad, grip):
        await self.arm.do_command({"set_target": {"joints_deg": [math.degrees(v) for v in q_rad], "grip": grip}})


def observation(q, grip, prompt):
    return {
        "state.right_arm": np.asarray(q, np.float32),
        "state.right_grip": np.asarray([grip], np.float32),
        "state.left_arm": np.asarray(LEFT_PARKED, np.float32),
        "state.left_grip": np.asarray([1.0], np.float32),
        "prompt": prompt,
    }


async def run(prompt, seconds, max_calls, check, reset, scene):
    load_env()
    robot = await RobotClient.at_address(ADDRESS, RobotClient.Options.with_api_key(
        api_key=os.environ["SO101_API_KEY"], api_key_id=os.environ["SO101_API_KEY_ID"]))
    sim = SimRobot(robot, scene)
    try:
        if (await sim.world.do_command({"get_scene": True}))["scene"] != scene:
            print(f"loading the {scene} scene")
            await sim.world.do_command({"load_scene": scene})
        if reset:
            # cup back on the table, then the arm to Dream's rest pose for the SO-101 (its IK leans toward it)
            await sim.world.do_command({"reset": True})
            await sim.arm.move_to_joint_positions(JointPositions(values=[math.degrees(v) for v in REST]))
        with dream.DreamClient() as client:
            client.policy(CONFIG)            # sends nothing: checks the config and builds the IK model of the arm
            await sim.grab_frames()
            q, grip = await sim.state()
            client.set_observation(sim.frames, observation(q, grip, prompt))   # checked against the embodiment here
            print(f"policy ok, observation ok: joints {np.round(np.degrees(q), 1)} grip {grip:.2f}, "
                  f"frames {', '.join(sim.frames)}")
            if check:
                print("--check: stopping before any model call (no credits used)")
                return
            await play(client, sim, prompt, seconds, max_calls)
    finally:
        await robot.close()


async def play(client, sim, prompt, seconds, max_calls):
    planner = Planner(client)
    plan = None          # (t_obs, rows, row it took over at)
    last_cmd = None      # the joints (rad) last commanded
    ramp_until = 0.0
    retry_at = 0.0
    t_end = time.monotonic() + seconds
    tick = 0
    next_tick = time.monotonic()
    try:
        while time.monotonic() < t_end:
            if tick % FRAME_EVERY == 0:
                await sim.grab_frames()
            q, grip = await sim.state()
            client.set_observation(sim.frames, observation(q, grip, prompt))
            now = time.monotonic()

            result = planner.take()
            if isinstance(result, tuple):
                t_obs, rows = result
                first = plan is None
                plan = (t_obs, rows, int((now - t_obs) / ROW_S))
                if first:   # ramp in from where the arm is
                    dist = np.max(np.abs(rows[min(plan[2], len(rows) - 1), :JOINTS] - np.asarray(q)))
                    ramp_until = now + max(2.0, dist / RAMP_RAD_S)
                t = client.last_timing or {}
                print(f"chunk {planner.calls}: rtt {t.get('rtt_ms', 0):.0f} ms, mode {client.mode}")
            elif isinstance(result, dream.DreamSessionExpired):
                plan = None
            elif isinstance(result, (dream.DreamServerError, dream.DreamTimeout, dream.DreamProtocolError)) and getattr(
                    result, "retryable", True):
                print(f"  retrying after {type(result).__name__}: {result}")
                retry_at = now + (getattr(result, "retry_after_s", None) or 1.0)
            elif result is not None:
                raise result

            row, executed, remaining = None, 0, 0
            if plan is not None:
                t_obs, rows, start = plan
                k = int((now - t_obs) / ROW_S)
                if k < len(rows):
                    row, executed, remaining = rows[k], k - start, len(rows) - k
            if (now >= retry_at and not planner.busy() and planner.calls < max_calls
                    and client.refill_due(executed, remaining)):
                planner.request()

            if row is not None:
                target = np.asarray(row[:JOINTS], float)
                if now < ramp_until and last_cmd is not None:
                    step = RAMP_RAD_S * ROW_S
                    target = last_cmd + np.clip(target - last_cmd, -step, step)
                elif last_cmd is None:
                    step = RAMP_RAD_S * ROW_S
                    target = np.asarray(q) + np.clip(target - np.asarray(q), -step, step)
                last_cmd = target
                await sim.command(target, float(row[JOINTS]))

            if planner.calls >= max_calls and not planner.busy() and plan is not None and remaining == 0:
                break
            tick += 1
            next_tick += ROW_S
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        planner.wait()
        cup = (await sim.world.do_command({"get_cup": True}))["cup"]
        print(f"done: {planner.calls} model calls; cup at ({cup['x']:.0f}, {cup['y']:.0f}, {cup['z']:.0f}) mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="put the cup into the bowl")
    ap.add_argument("--scene", choices=("dream", "cup"), default="dream",
                    help="dream: Dream's YAM table and top camera (default); cup: our cup-and-hand scene")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--check", action="store_true", help="validate everything, make no model call")
    ap.add_argument("--no-reset", action="store_true", help="keep the scene and arm as they are")
    args = ap.parse_args()
    asyncio.run(run(args.prompt, args.seconds, args.max_calls, args.check, not args.no_reset, args.scene))


if __name__ == "__main__":
    main()
