"""Dream's YAM MuJoCo example, unchanged, as a world the Viam components run against.

dream_examples.yam_mujoco.YamEnv is the scene and the physics Dream's own `dreamcli example
yam-mujoco` runs: two YAM arms, three cameras, an apple, a banana, an orange, a bowl and a plate.
YamSim steps it on its own thread at its own rate (one 50 ms step() of position targets, 20 a
second), so it runs in real time behind Viam, and serves its state, targets and camera images
to the components. pause() holds the physics, as Dream's example does while a call is pending.
"""
from __future__ import annotations

import concurrent.futures
import io
import queue
import threading
import time

import numpy as np
from PIL import Image

SIDES = ("left", "right")
STEP_S = 0.05          # YamEnv.step() runs 50 ms of physics
CAMERAS = ("top", "left", "right")


class YamSim:
    scene = "yam"

    def __init__(self):
        import mujoco
        from dream_examples.yam_mujoco.yam_env import HOME, YamEnv

        self._mujoco = mujoco
        self.env = YamEnv()
        self.home = HOME
        self.target = {s: HOME.copy() for s in SIDES}
        self.paused = False
        self._running = True
        self._work = queue.Queue()
        self._renderer = None
        self._thread = threading.Thread(target=self._loop, name="yam", daemon=True)
        self._thread.start()

    def call(self, fn, *args):
        fut = concurrent.futures.Future()
        self._work.put((fn, args, fut))
        return fut

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)

    def _loop(self):
        next_step = time.monotonic()
        while self._running:
            while True:
                try:
                    fn, args, fut = self._work.get_nowait()
                except queue.Empty:
                    break
                try:
                    fut.set_result(fn(*args))
                except Exception as e:  # noqa: BLE001
                    fut.set_exception(e)
            now = time.monotonic()
            if now >= next_step:
                if not self.paused:
                    self.env.step(self.target["left"], self.target["right"])
                next_step = max(next_step + STEP_S, now - STEP_S)
            time.sleep(0.002)
        # the renderer's GL context belongs to this thread
        if self._renderer is not None:
            self._renderer.close()
        self.env.close()

    # ---- operations, run on the physics thread -------------------------------------
    def op_reset(self):
        state = self.env.reset()
        self.target = {s: self.home.copy() for s in SIDES}
        return {s: state[s].tolist() for s in SIDES}

    def op_state(self):
        state = self.env.state()
        return {s: state[s].tolist() for s in SIDES}

    def op_set_targets(self, targets):
        """targets: {side: [6 joints rad, grip 0..1]}, either side or both; clipped as YamEnv.step() does."""
        for side, values in targets.items():
            v = np.asarray(values, float)
            if v.shape != (7,) or not np.isfinite(v).all():
                raise ValueError(f"{side}: seven finite values (six joints in rad, grip 0..1)")
            v[6] = min(max(v[6], 0.0), 1.0)
            self.target[side] = v
        return True

    def op_set_grip(self, side, grip):
        self.target[side] = self.target[side].copy()
        self.target[side][6] = min(max(float(grip), 0.0), 1.0)
        return True

    def op_objects(self):
        """World positions (mm) of the fruit, the bowl and the plate."""
        out = {}
        for name in ("apple", "banana", "orange", "bowl", "plate"):
            try:
                out[name] = [round(float(v) * 1000, 1) for v in self.env.data.body(name).xpos]
            except KeyError:
                pass
        return out

    def op_pause(self, paused):
        self.paused = bool(paused)
        return self.paused

    def op_render(self, camera, width, height, quality):
        """A camera as Dream's example renders it: 640x480, cropped to the middle 640x360 (the station's crop)."""
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(self.env.model, height=480, width=640)
        self._renderer.update_scene(self.env.data, camera=camera)
        img = Image.fromarray(self._renderer.render()[60:420])
        if img.size != (width, height):
            img = img.resize((width, height), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
