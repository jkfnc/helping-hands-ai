"""Dream's YAM example, driven through Viam: Dream-1 runs the two YAM arms of Dream's own MuJoCo
scene, which the mujoco-so101 module serves as Viam components on machine so101.

    .venv/bin/python yam_viam.py "put apple into the bowl" --seconds 60
    .venv/bin/python yam_viam.py --check      # loads the scene, reads everything through Viam, no model call

The request is the one `dreamcli example yam-mujoco` makes (its dream_control helpers): model
dreamlabs/dream-1-beta, the YAM station keys, nine frames per camera 0.2 s apart at 320x176 from
the 640x360 crop, and 48 rows per reply played at 20 a second. Everything the loop reads or
writes goes through Viam: cameras mj-cam-top, yam-cam-left, yam-cam-right; both arms and grippers
in one mj-world command per row. The physics holds while a call is pending, as in Dream's example.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import math
import os
import time
from collections import deque

from PIL import Image

import dream
import dream_examples.yam_mujoco as _yam_pkg
import sys

sys.path.insert(0, list(_yam_pkg.__path__)[0])   # dream_control imports its siblings by bare name
from dream_examples.yam_mujoco import dream_control as dc  # noqa: E402
from viam.components.camera import Camera
from viam.components.generic import Generic
from viam.robot.client import RobotClient

HERE = os.path.dirname(os.path.abspath(__file__))
ADDRESS = os.environ.get("SO101_ADDRESS", "so101-main.xjrc4czns9.viam.cloud")
CAMERAS = {"top": "mj-cam-top", "left": "yam-cam-left", "right": "yam-cam-right"}
ROW_S = 0.05


def load_env():
    for line in open(os.path.join(HERE, "..", ".env")):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def to_frame(jpeg: bytes) -> str:
    """A 640x360 camera image as Dream's example sends it: 320x176, JPEG q85, base64."""
    img = Image.open(io.BytesIO(jpeg)).convert("RGB").resize(dc.FRAME_SIZE, Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Station:
    """The YAM station as Viam serves it."""

    def __init__(self, robot):
        self.world = Generic.from_robot(robot, "mj-world")
        self.cams = {name: Camera.from_robot(robot, comp) for name, comp in CAMERAS.items()}
        self.windows = {name: deque(maxlen=dc.FRAMES) for name in CAMERAS}

    async def capture(self):
        shots = await asyncio.gather(*(cam.get_images() for cam in self.cams.values()))
        for name, (images, _) in zip(self.cams, shots):
            self.windows[name].append(to_frame(images[0].data))

    async def state(self):
        return (await self.world.do_command({"yam_state": True}))["yam_state"]

    async def command(self, left, right):
        await self.world.do_command({"yam_set": {"left": list(left), "right": list(right)}})

    async def pause(self, on):
        await self.world.do_command({"pause": on})

    async def objects(self):
        return (await self.world.do_command({"objects": True}))["objects"]


def observation(state, task):
    """The station keys Dream's example sends a model that reads hand poses (dream-1-beta)."""
    obs = {"prompt": task}
    for side in ("left", "right"):
        obs[f"state.{dc.STATION_SLOTS[side]}.joint_pos"] = list(state[side][:6])
        obs[f"state.{dc.STATION_SLOTS[side]}.gripper_pos"] = list(state[side][6:])
    obs.update(dc.STATION_ROBOT)
    return obs


def feed(policy, station, state, task, steps, whole=False):
    """Hand the client the newest frame of each window (or, for a new session, every frame, oldest first)."""
    count = len(station.windows["top"])
    for i in (range(count) if whole else [count - 1]):
        frames = {f"image_window.{name}": station.windows[name][i] for name in CAMERAS}
        seconds = (steps - (count - 1 - i) * dc.FRAME_EVERY_STEPS) * ROW_S
        policy.set_observation(frames, observation(state, task), timestamp=max(seconds, 0.0) + i * 1e-6)


def clip_rows(reply, rows):
    left, right = (reply["q_l_chunk"], reply["q_r_chunk"])
    out = []
    for chunk in (left, right):
        out.append([[float(v) for v in row[:6]] + [min(max(float(row[6]), 0.0), 1.0)] for row in list(chunk)[:rows]])
    return out


async def run(task, seconds, rows, max_calls, check, model):
    load_env()
    robot = await RobotClient.at_address(ADDRESS, RobotClient.Options.with_api_key(
        api_key=os.environ["SO101_API_KEY"], api_key_id=os.environ["SO101_API_KEY_ID"]))
    st = Station(robot)
    policy = None
    try:
        if (await st.world.do_command({"get_scene": True})).get("scene") != "yam":
            print("loading Dream's YAM scene")
            await st.world.do_command({"load_scene": "yam"})
        await st.world.do_command({"reset": True})
        await st.pause(False)
        for _ in range(dc.FRAMES):           # a full window before the first call: nine frames, 0.2 s apart
            await st.capture()
            await asyncio.sleep(dc.FRAME_EVERY_STEPS * ROW_S)
        state = await st.state()
        print("through Viam: state", {s: [round(v, 2) for v in state[s]] for s in state},
              "| frames", {n: len(w) for n, w in st.windows.items()})
        print("objects:", await st.objects())
        if check:
            print("--check: no model call made")
            return
        config = dc.policy_config(model, None, True)
        policy = dream.DreamClient()
        policy.policy(config)
        steps = dc.FRAMES * dc.FRAME_EVERY_STEPS
        feed(policy, st, state, task, steps, whole=True)
        deadline = time.monotonic() + seconds
        calls = 0
        print(f"Running: {task} · {model}")
        while time.monotonic() < deadline and calls < max_calls:
            await st.pause(True)            # Dream's example holds the simulation while the call is pending
            t0 = time.monotonic()
            try:
                reply = await asyncio.to_thread(policy.get_action)
            except dream.DreamSessionExpired:
                await st.pause(False)
                feed(policy, st, await st.state(), task, steps, whole=True)
                continue
            finally:
                await st.pause(False)
            calls += 1
            left, right = clip_rows(reply, rows)
            print(f"  Dream · {time.monotonic() - t0:.2f}s RTT · {calls} requests · {steps} steps")
            for l, r in zip(left, right):
                if time.monotonic() >= deadline:
                    break
                tick = time.monotonic()
                await st.command(l, r)
                steps += 1
                if steps % dc.FRAME_EVERY_STEPS == 0:
                    await st.capture()
                    feed(policy, st, await st.state(), task, steps)
                await asyncio.sleep(max(0.0, ROW_S - (time.monotonic() - tick)))
        objs = await st.objects()
        bowl = objs.get("bowl")
        print(f"done: {calls} model calls")
        for name in ("apple", "orange", "banana"):
            if name in objs and bowl:
                d = math.dist(objs[name][:2], bowl[:2])
                print(f"  {name}: {d:.0f} mm from the bowl's center{'  <- in the bowl' if d < 80 else ''}")
    finally:
        if policy is not None:
            policy.close()
        await robot.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default="put apple into the bowl")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--rows", type=int, default=48, help="rows of each reply to play (Dream's /actions)")
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--model", default=dc.BETA_MODEL)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args.task, args.seconds, args.rows, args.max_calls, args.check, args.model))


if __name__ == "__main__":
    main()
