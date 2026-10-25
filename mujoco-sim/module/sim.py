"""The MuJoCo world behind the Viam components: one physics thread, shared by all of them.

Physics, rendering and every read of the model happen on that thread (the GL context
belongs to it); the Viam side hands it work through a queue and awaits the result.
"""
from __future__ import annotations

import concurrent.futures
import io
import math
import os
import queue
import threading
import time

import mujoco
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCENE = os.path.join(HERE, "..", "scene_cup.xml")

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_OPEN = 1.0     # rad on the gripper joint: jaws wide enough for the 40 mm cup
GRIPPER_CLOSED = -0.17  # rad: the bottom of the range, so the jaws squeeze what they hold
CUP_HOME = (0.41, -0.07, 0.04)


def grip_to_rad(grip):
    """A policy's grip, 0 closed to 1 open, as the gripper joint's target."""
    return GRIPPER_CLOSED + min(max(grip, 0.0), 1.0) * (GRIPPER_OPEN - GRIPPER_CLOSED)


def rad_to_grip(rad):
    return min(max((rad - GRIPPER_CLOSED) / (GRIPPER_OPEN - GRIPPER_CLOSED), 0.0), 1.0)

# Rotation from MuJoCo's "gripper" body frame to the Viam arm's end frame (constant,
# both are fixed to the wrist roll link). Measured by comparing the two at one pose.
R_MJ_TO_VIAM = None  # filled in by calibrate_end_frame()


def ov_from_matrix(R):
    """Viam orientation vector (ox, oy, oz, theta_deg) of a rotation matrix.

    Viam's OV is R = Rz(lon) @ Ry(lat) @ Rz(theta); its z axis is the pointing direction.
    """
    z = R[:, 2]
    lat = math.acos(max(-1.0, min(1.0, z[2])))
    lon = math.atan2(z[1], z[0]) if 1 - abs(z[2]) > 1e-6 else 0.0
    M = _ry(lat).T @ _rz(lon).T @ R
    theta = math.atan2(M[1, 0], M[0, 0])
    return float(z[0]), float(z[1]), float(z[2]), math.degrees(theta)


def matrix_from_ov(ox, oy, oz, theta_deg):
    n = math.sqrt(ox * ox + oy * oy + oz * oz)
    ox, oy, oz = ox / n, oy / n, oz / n
    lat = math.acos(max(-1.0, min(1.0, oz)))
    lon = math.atan2(oy, ox) if 1 - abs(oz) > 1e-6 else 0.0
    return _rz(lon) @ _ry(lat) @ _rz(math.radians(theta_deg))


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


SCENES = ("cup", "dream", "yam")


class Sim:
    def __init__(self, scene=DEFAULT_SCENE, speed_deg_s=60.0, gripper_speed_rad_s=3.0):
        """`scene` is an MJCF path, "cup" (scene_cup.xml) or "dream" (Dream's YAM table, scene_dream.py)."""
        self.scene = "dream" if scene == "dream" else "cup"
        if scene == "dream":
            import sys
            sys.path.insert(0, os.path.join(HERE, ".."))
            import scene_dream
            self.model = scene_dream.build_model()
            self.cup_home = (scene_dream.CUP_HOME[0], scene_dream.CUP_HOME[1], scene_dream.TABLE_TOP + 0.04)
            pos, quat = scene_dream.arm_root_pose()
            self.base_pos, self.base_R = np.asarray(pos), np.zeros(9)
            mujoco.mju_quat2Mat(self.base_R, np.asarray(quat, float))
            self.base_R = self.base_R.reshape(3, 3)
        else:
            self.model = mujoco.MjModel.from_xml_path(DEFAULT_SCENE if scene in (None, "cup") else scene)
            self.cup_home = CUP_HOME
            self.base_pos, self.base_R = np.zeros(3), np.eye(3)
        self.data = mujoco.MjData(self.model)
        self._running = True
        self.arm_qadr = [self.model.joint(n).qposadr[0] for n in ARM_JOINTS]
        self.arm_act = [self.model.actuator(n).id for n in ARM_JOINTS]
        self.grip_qadr = self.model.joint("gripper").qposadr[0]
        self.grip_act = self.model.actuator("gripper").id
        self.cup_qadr = self.model.joint("cup").qposadr[0]
        self.speed = math.radians(speed_deg_s)
        self.grip_speed = gripper_speed_rad_s
        self.target = np.zeros(6)      # where each actuator is heading (rad)
        self._work = queue.Queue()
        self._renderers = {}
        self._reset()
        self._thread = threading.Thread(target=self._loop, name="mujoco", daemon=True)
        self._thread.start()

    # ---- called from the Viam side; each returns a concurrent Future -----------------
    def call(self, fn, *args):
        fut = concurrent.futures.Future()
        self._work.put((fn, args, fut))
        return fut

    # ---- physics thread ----------------------------------------------------------------
    def _loop(self):
        dt = self.model.opt.timestep
        wall0, sim0 = time.monotonic(), self.data.time
        while self._running:
            # run queued work between steps
            while True:
                try:
                    fn, args, fut = self._work.get_nowait()
                except queue.Empty:
                    break
                try:
                    fut.set_result(fn(*args))
                except Exception as e:  # noqa: BLE001
                    fut.set_exception(e)
            # keep sim time in step with the wall clock
            behind = (time.monotonic() - wall0) - (self.data.time - sim0)
            steps = int(behind / dt)
            if steps > 200:  # fell far behind (e.g. a slow render): skip ahead
                wall0, sim0, steps = time.monotonic(), self.data.time, 1
            for _ in range(steps):
                self._ramp(dt)
                mujoco.mj_step(self.model, self.data)
            time.sleep(0.002)
        # stopping: the renderers' GL contexts belong to this thread, so free them here
        # (freed from another thread they crash the process)
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    def _ramp(self, dt):
        """Move each actuator's setpoint toward its target at the configured speed."""
        ctrl = self.data.ctrl
        for k, a in enumerate(self.arm_act):
            ctrl[a] += np.clip(self.target[k] - ctrl[a], -self.speed * dt, self.speed * dt)
        g = self.grip_act
        ctrl[g] += np.clip(self.target[5] - ctrl[g], -self.grip_speed * dt, self.grip_speed * dt)

    def _reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.target[:] = 0.0
        self.target[5] = GRIPPER_OPEN
        self.data.ctrl[:] = self.target
        self.data.qpos[self.grip_qadr] = GRIPPER_OPEN
        self._place_cup(*self.cup_home)
        mujoco.mj_forward(self.model, self.data)

    def stop(self):
        """End the physics thread (a scene switch builds a new Sim)."""
        self._running = False
        self._thread.join(timeout=2.0)

    def _place_cup(self, x, y, z=None):
        z = self.cup_home[2] if z is None else z
        a = self.cup_qadr
        self.data.qpos[a:a + 7] = [x, y, z, 1, 0, 0, 0]
        self.data.qvel[self.model.joint("cup").dofadr[0]:][:6] = 0

    # ---- operations (run on the physics thread via call()) ----------------------------
    def op_reset(self):
        self._reset()
        return True

    def op_place_cup(self, x, y):
        self._place_cup(x, y)
        mujoco.mj_forward(self.model, self.data)
        return True

    def op_set_arm_target(self, rad):
        for k, a in enumerate(self.arm_act):
            lo, hi = self.model.actuator_ctrlrange[a]
            self.target[k] = float(np.clip(rad[k], lo, hi))
        return True

    def op_set_gripper_target(self, rad):
        self.target[5] = float(rad)
        return True

    def op_hold_position(self):
        """Stop: aim every actuator at where it is now."""
        for k, a in enumerate(self.arm_act):
            self.target[k] = self.data.ctrl[a] = self.data.qpos[self.arm_qadr[k]]
        self.target[5] = self.data.ctrl[self.grip_act]
        return True

    def op_arm_state(self):
        q = np.array([self.data.qpos[i] for i in self.arm_qadr])
        ctrl = np.array([self.data.ctrl[a] for a in self.arm_act])
        return q, ctrl, self.target[:5].copy()

    def op_gripper_state(self):
        return float(self.data.qpos[self.grip_qadr]), float(self.data.ctrl[self.grip_act]), float(self.target[5])

    def op_end_pose(self):
        """Arm end frame (what Viam's arm reports), in mm and Viam OV, relative to the arm's base."""
        b = self.data.body("gripper")
        R = self.base_R.T @ b.xmat.reshape(3, 3) @ R_MJ_TO_VIAM
        p = self.base_R.T @ (b.xpos - self.base_pos) * 1000.0
        return (float(p[0]), float(p[1]), float(p[2])), ov_from_matrix(R)

    def op_cup_held(self):
        """True when the cup touches both the fixed jaw and the moving jaw."""
        cup = self.model.geom("cup").id
        fixed = self.model.body("gripper").id
        moving = self.model.body("moving_jaw_so101_v1").id
        touching = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if cup not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == cup else c.geom1
            touching.add(self.model.geom_bodyid[other])
        return fixed in touching and moving in touching

    def op_cup_pose(self):
        b = self.data.body("cup")
        return [float(v) for v in b.xpos * 1000.0], [float(v) for v in b.xquat]

    def op_render(self, camera, width, height, quality):
        key = (width, height)
        r = self._renderers.get(key)
        if r is None:
            r = self._renderers[key] = mujoco.Renderer(self.model, height, width)
        r.update_scene(self.data, camera=camera)
        buf = io.BytesIO()
        Image.fromarray(r.render()).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def calibrate_end_frame():
    """R_MJ_TO_VIAM from one pose where Viam's simulated SO-101 reported its end orientation.

    At joints [20, 10, 0, 30, 0] deg Viam's arm-1 reported OV (0.71985, -0.26199, -0.64278, 0).
    """
    global R_MJ_TO_VIAM
    m = mujoco.MjModel.from_xml_path(DEFAULT_SCENE)
    d = mujoco.MjData(m)
    for k, n in enumerate(ARM_JOINTS):
        d.qpos[m.joint(n).qposadr[0]] = math.radians([20, 10, 0, 30, 0][k])
    mujoco.mj_forward(m, d)
    R_mj = d.body("gripper").xmat.reshape(3, 3)
    R_viam = matrix_from_ov(0.7198547551051414, -0.2619949361844602, -0.6427812885937316, 0.0)
    R_MJ_TO_VIAM = R_mj.T @ R_viam


calibrate_end_frame()

_SIM = None
_SIM_LOCK = threading.Lock()


def get_sim(scene=None) -> Sim:
    """The one Sim per module process, created by whichever component starts first."""
    global _SIM
    with _SIM_LOCK:
        if _SIM is None:
            _SIM = Sim(scene or DEFAULT_SCENE)
        return _SIM


def switch_scene(scene) -> Sim:
    """Replace the running world with another scene ("cup" or "dream"); every component follows."""
    global _SIM
    if scene not in SCENES:
        raise ValueError(f"scene must be one of {SCENES}")
    with _SIM_LOCK:
        if _SIM is not None:
            _SIM.stop()
        if scene == "yam":
            from yam_sim import YamSim
            _SIM = YamSim()
        else:
            _SIM = Sim(scene)
        return _SIM
