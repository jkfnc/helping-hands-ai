"""The demo's robot side, all through Viam on machine so101: Dream-1 tasks on Dream's YAM scene,
and act 2, the SO-101 fetching a cup of water and handing it over.

    tasks = RobotTasks(robot)            # robot: a connected viam RobotClient
    result = await tasks.yam_task("put apple into the bowl", abort=asyncio.Event())
    await tasks.fetch_water()

A YAM task ends as soon as the named fruit rests in its target (checked from the simulator after
every reply, so a quick success spends few Dream calls), on abort, or at the time limit; then the
arms go home and the episode is reset, as Dream's docs ask between tasks.
"""
from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass

import numpy as np

import dream
import yam_viam as yv
from yam_viam import Station, clip_rows, dc, feed

FRUITS = ("apple", "orange", "banana")
TARGETS = {"bowl": 80.0, "plate": 60.0}     # how close (mm, in the table plane) to the target's center counts
RESTING_Z_MM = 810.0                        # below this the fruit is down, not still in a gripper
HOME = [0.0] * 6 + [1.0]


def task_text(fruit, target):
    """Dream's own phrasing for its recommended tasks."""
    return f"put {fruit} into the bowl" if target == "bowl" else f"put {fruit} on the plate"


def parse_task(text):
    """(fruit, target) named in free text, or None."""
    t = text.lower()
    fruit = next((f for f in FRUITS if re.search(rf"\b{f}s?\b", t)), None)
    target = next((g for g in TARGETS if re.search(rf"\b{g}s?\b", t)), None)
    if fruit is None:
        return None
    return fruit, target or ("plate" if fruit == "banana" else "bowl")


@dataclass
class Result:
    task: str
    success: bool
    calls: int
    seconds: float
    reason: str


class RobotTasks:
    def __init__(self, robot, model=dc.BETA_MODEL, say=print):
        self.robot = robot
        self.model = model
        self.say = say
        self.st = Station(robot)
        self.policy = None

    # ---- scenes ------------------------------------------------------------------------
    async def use_scene(self, scene):
        if (await self.st.world.do_command({"get_scene": True})).get("scene") != scene:
            await self.st.world.do_command({"load_scene": scene})

    async def reset_yam(self):
        await self.use_scene("yam")
        await self.st.world.do_command({"reset": True})
        await self.st.pause(False)

    # ---- YAM ---------------------------------------------------------------------------
    async def home(self, seconds=2.0):
        """Ease both arms home (joints zero, grippers open), as Dream's /reset_robot does."""
        state = await self.st.state()
        start = {s: np.asarray(state[s], float) for s in ("left", "right")}
        n = int(seconds / yv.ROW_S)
        for i in range(1, n + 1):
            a = i / n
            a = a * a * (3 - 2 * a)
            await self.st.command(*(list(start[s] + (np.asarray(HOME) - start[s]) * a) for s in ("left", "right")))
            await asyncio.sleep(yv.ROW_S)

    async def hold(self):
        state = await self.st.state()
        await self.st.command(state["left"], state["right"])

    async def placed(self, fruit, target):
        objs = await self.st.objects()
        f, g = objs.get(fruit), objs.get(target)
        if not f or not g:
            return False
        return math.dist(f[:2], g[:2]) < TARGETS[target] and f[2] < RESTING_Z_MM

    async def yam_task(self, task, abort=None, seconds=60.0, max_calls=30, rows=48):
        """Run one Dream-1 task on the YAM scene through Viam."""
        abort = abort or asyncio.Event()
        spec = parse_task(task)
        await self.use_scene("yam")
        await self.st.pause(False)
        if self.policy is None:
            self.policy = dream.DreamClient()
            self.policy.policy(dc.policy_config(self.model, None, True))
        else:
            self.policy.reset_episode()
        for w in self.st.windows.values():
            w.clear()
        for _ in range(dc.FRAMES):
            await self.st.capture()
            await asyncio.sleep(dc.FRAME_EVERY_STEPS * yv.ROW_S)
        steps = dc.FRAMES * dc.FRAME_EVERY_STEPS
        feed(self.policy, self.st, await self.st.state(), task, steps, whole=True)
        t0 = time.monotonic()
        calls, reason = 0, "time limit"
        try:
            while time.monotonic() - t0 < seconds and calls < max_calls:
                if abort.is_set():
                    reason = "stopped"
                    break
                await self.st.pause(True)
                try:
                    reply = await asyncio.to_thread(self.policy.get_action)
                except dream.DreamSessionExpired:
                    await self.st.pause(False)
                    feed(self.policy, self.st, await self.st.state(), task, steps, whole=True)
                    continue
                finally:
                    await self.st.pause(False)
                calls += 1
                left, right = clip_rows(reply, rows)
                for l, r in zip(left, right):
                    if abort.is_set() or time.monotonic() - t0 >= seconds:
                        break
                    tick = time.monotonic()
                    await self.st.command(l, r)
                    steps += 1
                    if steps % dc.FRAME_EVERY_STEPS == 0:
                        await self.st.capture()
                        feed(self.policy, self.st, await self.st.state(), task, steps)
                    await asyncio.sleep(max(0.0, yv.ROW_S - (time.monotonic() - tick)))
                if spec and await self.placed(*spec):
                    reason = "done"
                    break
            else:
                if calls >= max_calls:
                    reason = "call limit"
        finally:
            if abort.is_set():
                await self.hold()
        if reason != "stopped":
            await self.home()
        success = bool(spec) and await self.placed(*spec)
        return Result(task, success, calls, time.monotonic() - t0, reason)

    async def objects(self):
        return await self.st.objects()

    # ---- act 2: the SO-101 brings a cup of water -----------------------------------------
    async def fetch_water(self):
        import fetch_water
        await self.use_scene("cup")
        await self.robot.refresh()   # the resource list the client took at connect can be stale
        return await fetch_water.Fetcher(self.robot).run(reset=True)

    def close(self):
        if self.policy is not None:
            self.policy.close()
            self.policy = None
