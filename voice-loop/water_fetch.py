"""Fetch the cup and hand it over, on pi-arm3's follower arm.

Recorded from the session that worked on 2026-10-23. The arm must be calibrated
(zero pose homing + range sweep) or none of these joint angles mean anything.

How it works:
  1. READY pose: gripper tilted 60 deg down so the wrist camera looks at the table.
  2. Find the colored marker (tape band on the cup rim) in the wrist camera and
     steer the base angle and reach until it sits at the same pixel it had during
     the run that worked (ALIGN_PX). It learns how pixels move per degree / per mm
     by nudging the arm, so no camera calibration is needed.
  3. Fixed approach relative to that aligned pose, close the gripper, lift, swing
     the base to the handover side, wait for the release, return.

Safety:
  * --dry-run (the default) prints every command and uses a simulated cup.
  * Joint-space for big moves (tracked within 1 deg), small Cartesian moves that
    keep the current orientation for approach/lift (changing orientation sent the
    arm the wrong way). Joint targets are clamped to the module's limits.
  * ABORT stops before the next move; stop() cannot interrupt a move in progress.

Env for a real run: VIAM_ADDRESS, VIAM_API_KEY_ID, VIAM_API_KEY
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import random
import threading
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("water-fetch")


def _load_env():
    """Read KEY=VALUE lines from .env next to this file or one folder up (no extra dependency).
    Variables already set in the environment win."""
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here, os.path.dirname(here)):
        path = os.path.join(folder, ".env")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

RELEASE = threading.Event()   # hand the cup over now (e.g. a spoken "release")
ABORT = threading.Event()     # stop before the next move
_running = threading.Lock()

ARM = "follower-arm"
GRIPPER = "follower-gripper"
WRIST_CAM = "icspringcameraicspringcamerusbxhcihcd02-0"
DEFAULT_ADDRESS = "pi-arm3-main.1pabvgfolw.viam.cloud"

# Joint order: base, shoulder, elbow, wrist flex, wrist roll (degrees).
READY = [3.69, -3.43, 5.71, 57.27, -59.16]   # gripper tilted 60 deg down
# The "grasp-ready" pose: gripper tilted 60 deg down, the cup's near wall just beside the fixed
# jaw, one short step from closing. Measured live on 2026-10-23 (3 reads: 486/490/487, 248/248/249)
# with the green marker; the detector sees the ring's far inner arc. ALIGN_Z is the tip height
# there. Re-measure with `--live --print-marker` at that pose if the marker or camera changes.
ALIGN_PX = (488.0, 248.0)
ALIGN_Z = 149.3
ALIGN_TOL_PX = 20.0
ALIGN_TOL_LN_AREA = 0.15
# Two separate loops (position and reach do not mix in the image): the marker's left-right
# position steers the base, its apparent size steers the reach (moving along the camera's
# viewing direction mostly changes size, barely position).
ALIGN_AREA = 1690.0           # marker pixel count at the grasp-ready pose (3 live reads, +-1.5%)
PAN_PX_PER_DEG = -14.0        # marker x moves this many px per +deg of base (live: -13.3..-14.7)
LN_AREA_PER_MM = 0.02         # d ln(area) / d(reach mm), from two live frames ~40 mm apart (rough)
APPROACH = [(5.1, -9.8)]      # (radial mm outward, dz mm) from the grasp-ready pose; the grasp
CARRY_Z = 230.0               # 246 was refused by the planner; 238 worked at one heading
HANDOVER_BASE = -90.0         # negative = left; calibrated base range is about -109..+114
SWING_STEP = 20.0
JOINT_LIMITS = [(-105.0, 110.0), (-60.0, 95.0), (-100.0, 90.0), (-94.9, 94.9), (-180.0, 180.0)]

# Marker colour presets for find_marker. "green" is tuned on a dark-green marker pen
# (measured hue 152, sat 0.64, but brightness only 0.10), so val_min is low; a brighter
# highlighter works too. Warm colours (orange/red) are NOT usable in the wrist camera:
# the cup's inside looks warm there and matches them.
MARKERS = {
    "green": dict(green_min=0.10, min_sum=45, min_area=200),
    "red": dict(red_min=0.24, min_sum=150, min_area=150),   # band on the cup's outer wall
    "magenta": dict(hue=(290.0, 340.0), sat_min=0.45, val_min=0.25, min_area=400),
}


# ---------------------------------------------------------------- vision

def rgb_to_hsv(img: np.ndarray):
    """img: HxWx3 uint8 RGB. Returns hue (deg), sat, val as float arrays."""
    f = img.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx, mn = f.max(-1), f.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-6
    rc = nz & (mx == r)
    gc = nz & (mx == g) & ~rc
    bc = nz & ~rc & ~gc
    h[rc] = (60.0 * ((g - b)[rc] / d[rc])) % 360.0
    h[gc] = 60.0 * ((b - r)[gc] / d[gc]) + 120.0
    h[bc] = 60.0 * ((r - g)[bc] / d[bc]) + 240.0
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return h, s, mx


def find_marker(img: np.ndarray, hue=(80.0, 160.0), sat_min=0.45, val_min=0.25, min_area=400,
                green_min=None, red_min=None, min_sum=45):
    """Centre (x, y) in pixels of the marker colour's extent, plus its area, or None."""
    if red_min is not None:
        # Same idea for a red/pink band on the cup's white outer wall (not inside the cup,
        # where the cup looks warm-tinted and matches red).
        f = img.astype(np.float32)
        total = f.sum(-1)
        mask = ((f[..., 0] - 0.5 * (f[..., 1] + f[..., 2])) / (total + 1e-3) >= red_min) & (total >= min_sum)
    elif green_min is not None:
        # Brightness-independent "how green": a dark-green marker pen sits near the brightness
        # cutoff, so HSV thresholds flickered with the camera's auto-exposure; this does not.
        f = img.astype(np.float32)
        total = f.sum(-1)
        mask = ((f[..., 1] - 0.5 * (f[..., 0] + f[..., 2])) / (total + 1e-3) >= green_min) & (total >= min_sum)
    else:
        h, s, v = rgb_to_hsv(img)
        mask = (h >= hue[0]) & (h <= hue[1]) & (s >= sat_min) & (v >= val_min)
    ys, xs = np.nonzero(mask)
    if xs.size < min_area:
        return None
    for _ in range(2):  # drop stray pixels: keep points near the median
        mx, my = np.median(xs), np.median(ys)
        sx, sy = xs.std() + 1e-6, ys.std() + 1e-6
        keep = (np.abs(xs - mx) < 2.5 * sx) & (np.abs(ys - my) < 2.5 * sy)
        if keep.sum() < min_area:
            break
        xs, ys = xs[keep], ys[keep]
    # Centre of the extent, not the mean: when only part of the ring passes the threshold
    # (the dark arc), the mean is pulled toward it, the extent centre much less so.
    x0, x1 = np.percentile(xs, [2, 98])
    y0, y1 = np.percentile(ys, [2, 98])
    return float((x0 + x1) / 2), float((y0 + y1) / 2), int(xs.size)


# ---------------------------------------------------------------- hardware wrappers

@dataclass
class Pose:
    x: float
    y: float
    z: float
    o_x: float
    o_y: float
    o_z: float
    theta: float


class RealArm:
    def __init__(self, arm):
        from viam.proto.common import Pose as ViamPose
        from viam.proto.component.arm import JointPositions

        self._arm, self._ViamPose, self._JP = arm, ViamPose, JointPositions

    async def get_joint_positions(self):
        return list((await self._arm.get_joint_positions()).values)

    async def move_to_joint_positions(self, joints):
        await self._arm.move_to_joint_positions(self._JP(values=list(joints)))

    async def get_end_position(self):
        p = await self._arm.get_end_position()
        return Pose(p.x, p.y, p.z, p.o_x, p.o_y, p.o_z, p.theta)

    async def move_to_position(self, pose: Pose):
        await self._arm.move_to_position(
            self._ViamPose(x=pose.x, y=pose.y, z=pose.z, o_x=pose.o_x, o_y=pose.o_y, o_z=pose.o_z, theta=pose.theta)
        )

    async def stop(self):
        await self._arm.stop()


class RealCamera:
    def __init__(self, cam):
        self._cam = cam

    async def frame(self) -> np.ndarray:
        from viam.media.utils.pil import viam_to_pil_image

        images, _ = await self._cam.get_images()   # viam-sdk 0.72: get_images() replaces get_image()
        return np.asarray(viam_to_pil_image(images[0]).convert("RGB"))


class SimWorld:
    """Dry-run world: a cup at a random spot, and a camera whose marker pixel moves with the arm.

    Pixel rates are from the live probe on 2026-10-23.
    """

    JP = (-14.0, -12.7)   # px per +deg of base
    JR = (-1.3, 1.7)      # px per mm outward (position barely changes)
    K = 0.01              # ring radius grows this fraction per mm of reach (area ~ 2K)

    def __init__(self, seed=None):
        rnd = random.Random(seed)
        self.rnd = rnd
        self.pan, self.r = READY[0], 255.0
        self.z = 176.0
        self.joints = list(READY)
        self.orient = (0.494, -0.036, -0.869, 59.0)
        self.cup_pan = -6.0 + rnd.uniform(-6, 6)
        self.cup_r = 255.0 + rnd.uniform(-20, 20)

    # arm
    async def get_joint_positions(self):
        self.joints[0] = self.pan
        return list(self.joints)

    async def move_to_joint_positions(self, joints):
        log.info("[dry] joints -> %s", [round(j, 1) for j in joints])
        self.joints = list(joints)
        self.pan = joints[0]

    async def get_end_position(self):
        a = math.radians(self.pan)
        return Pose(self.r * math.cos(a), -self.r * math.sin(a), self.z, *self.orient)

    async def move_to_position(self, pose: Pose):
        log.info("[dry] tip -> (%.1f, %.1f, %.1f)", pose.x, pose.y, pose.z)
        x = pose.x + self.rnd.uniform(-4, 4)      # the real arm lands a few mm off
        y = pose.y + self.rnd.uniform(-5, 5)
        self.r = math.hypot(x, y)
        self.pan = math.degrees(math.atan2(-y, x))
        self.z = pose.z + self.rnd.uniform(-5, 1)
        self.orient = (pose.o_x, pose.o_y, pose.o_z, pose.theta)

    async def stop(self):
        log.info("[dry] stop")

    # gripper
    async def open(self):
        log.info("[dry] gripper open")

    async def grab(self):
        log.info("[dry] gripper grab")
        return True

    # camera
    async def frame(self) -> np.ndarray:
        d_pan, d_r = self.pan - self.cup_pan, self.r - self.cup_r
        px = ALIGN_PX[0] + self.JP[0] * d_pan + self.JR[0] * d_r
        py = 248.0 + self.JP[1] * d_pan + self.JR[1] * d_r
        scale = math.exp(self.K * d_r)
        outer, inner = 45.0 * scale, 38.6 * scale          # ring area = ALIGN_AREA at d_r = 0
        img = np.full((480, 640, 3), 235, np.uint8)
        yy, xx = np.mgrid[0:480, 0:640]
        dist2 = (xx - px) ** 2 + (yy - py) ** 2
        img[(dist2 < outer ** 2) & (dist2 > inner ** 2)] = (30, 90, 45)   # dark green
        return img


# ---------------------------------------------------------------- motion helpers

def clamp_joints(joints):
    out = []
    for i, (j, (lo, hi)) in enumerate(zip(joints, JOINT_LIMITS)):
        if not lo <= j <= hi:
            log.warning("joint %d target %.1f clamped to [%.1f, %.1f]", i + 1, j, lo, hi)
        out.append(min(max(j, lo), hi))
    return out


def check_abort():
    if ABORT.is_set():
        raise RuntimeError("aborted")


async def joints_to(arm, target, label):
    check_abort()
    log.info("%s", label)
    await arm.move_to_joint_positions(clamp_joints(target))


async def cart_to(arm, xyz, label):
    """Move the tip to xyz, keeping whatever orientation it has right now."""
    check_abort()
    cur = await arm.get_end_position()
    log.info("%s (tip now %.0f, %.0f, %.0f)", label, cur.x, cur.y, cur.z)
    await arm.move_to_position(Pose(xyz[0], xyz[1], xyz[2], cur.o_x, cur.o_y, cur.o_z, cur.theta))
    new = await arm.get_end_position()
    err = max(abs(new.x - xyz[0]), abs(new.y - xyz[1]), abs(new.z - xyz[2]))
    if err > 15.0:
        raise RuntimeError(f"{label}: tip landed {err:.0f} mm from target; stopping")


async def move_radial(arm, dr, dz=0.0, label="", z_abs=None):
    """Move the tip outward along the line from the base (dr mm) and up/down: dz mm from
    where it is, or to the absolute height z_abs. Each Cartesian move lands a few mm low,
    so alignment moves use z_abs and the error cannot add up."""
    cur = await arm.get_end_position()
    r = math.hypot(cur.x, cur.y)
    k = (r + dr) / r
    await cart_to(arm, (cur.x * k, cur.y * k, z_abs if z_abs is not None else cur.z + dz), label)


async def swing_base(arm, target_deg):
    j = await arm.get_joint_positions()
    cur = j[0]
    step = SWING_STEP if target_deg > cur else -SWING_STEP
    while abs(target_deg - cur) > SWING_STEP:
        cur += step
        await joints_to(arm, [cur] + j[1:], f"base -> {cur:.0f}")
    await joints_to(arm, [target_deg] + j[1:], f"base -> {target_deg:.0f}")


# ---------------------------------------------------------------- finding the cup

async def marker_obs(cam, spec, frames=3):
    """(x, y, area): median over a few frames, or None if the marker is never seen."""
    obs = []
    for _ in range(frames + 2):
        m = find_marker(await cam.frame(), **spec)
        if m:
            obs.append(m)
        if len(obs) >= frames:
            break
        await asyncio.sleep(0.2)
    if not obs:
        return None
    return tuple(float(np.median([o[k] for o in obs])) for k in range(3))


async def align_to_cup(arm, cam, spec, max_iters=14):
    """Steer to the grasp-ready pose using the marker: left-right position -> base angle,
    apparent size -> reach. One correction at a time, small steps, because Cartesian moves
    land a few mm off sideways and in height and the camera is close to the cup."""
    await move_radial(arm, 0.0, label="set height", z_abs=ALIGN_Z)
    m = await marker_obs(cam, spec)
    if m is None:  # sweep left/right to look for it
        j = await arm.get_joint_positions()
        for d in (-10, 10, -20, 20):
            await joints_to(arm, [j[0] + d] + j[1:], f"searching for the cup marker ({d:+d} deg)")
            m = await marker_obs(cam, spec)
            if m is not None:
                break
        if m is None:
            raise RuntimeError("cup marker not found in the wrist camera")
    best, stale = None, 0
    for i in range(max_iters):
        x, y, area = m
        ex = ALIGN_PX[0] - x
        ea = math.log(ALIGN_AREA / max(area, 1.0))        # >0: marker too small, cup too far
        log.info("align %d: marker x=%.0f (want %.0f), area=%d (want %d)", i + 1, x, ALIGN_PX[0], area, ALIGN_AREA)
        if abs(ex) < ALIGN_TOL_PX and abs(ea) < ALIGN_TOL_LN_AREA:
            log.info("aligned")
            return
        score = abs(ex) / ALIGN_TOL_PX + abs(ea) / ALIGN_TOL_LN_AREA
        if best is None or score < best * 0.85:
            best, stale = score, 0
        else:
            stale += 1
            if stale >= 5:
                raise RuntimeError("alignment is not improving; stopping")
        if abs(ex) >= ALIGN_TOL_PX:
            dpan = float(np.clip(0.5 * ex / PAN_PX_PER_DEG, -3.0, 3.0))
            log.info("   base %+.1f deg", dpan)
            j = await arm.get_joint_positions()
            await joints_to(arm, [j[0] + dpan] + j[1:], "align base")
        else:
            dr = float(np.clip(0.5 * ea / LN_AREA_PER_MM, -8.0, 8.0))
            log.info("   reach %+.1f mm", dr)
            await move_radial(arm, dr, label="align reach", z_abs=ALIGN_Z)
        m = await marker_obs(cam, spec)
        if m is None:
            raise RuntimeError("lost the marker while aligning")
    raise RuntimeError("did not converge")


# ---------------------------------------------------------------- the job

STAGES = ["align", "approach", "grab", "lift"]


async def fetch(arm, gripper, cam, spec, release, stop_after=None):
    """Runs the job; stop_after (one of STAGES) leaves the arm there, for commissioning.
    Returns True if it ran to the end."""
    await gripper.open()
    await joints_to(arm, READY, "ready pose")
    await align_to_cup(arm, cam, spec)
    if stop_after == "align":
        return False
    for i, (dr, dz) in enumerate(APPROACH, 1):
        await move_radial(arm, dr, dz, f"approach {i}/{len(APPROACH)}")
    if stop_after == "approach":
        return False
    await gripper.grab()
    if stop_after == "grab":
        return False
    cur = await arm.get_end_position()
    await cart_to(arm, (cur.x, cur.y, CARRY_Z), "lift")
    if stop_after == "lift":
        return False
    await swing_base(arm, HANDOVER_BASE)
    await release()
    await gripper.open()
    await asyncio.sleep(1.0)
    return True


async def retract(arm):
    await swing_base(arm, READY[0])
    await joints_to(arm, READY, "ready pose")


async def run(dry_run=True, release_after=10.0, color="green", seed=None, stop_after=None):
    """Whole job, no prompts. Waits up to `release_after` s at the handover point (or
    until RELEASE is set), then releases and returns. release_after=None waits for Enter."""

    async def release():
        if release_after is None:
            await asyncio.to_thread(input, "Hold your hand under the cup, press Enter to release: ")
            return
        log.info("at handover point; releasing in up to %.0f s (say 'release' to drop it now)", release_after)
        await asyncio.to_thread(RELEASE.wait, release_after)

    spec = MARKERS[color]
    robot = None
    if dry_run:
        world = SimWorld(seed)
        log.info("[dry] simulated cup at base %.1f deg, reach %.0f mm", world.cup_pan, world.cup_r)
        arm = gripper = cam = world
    else:
        from viam.components.arm import Arm
        from viam.components.camera import Camera
        from viam.components.gripper import Gripper
        from viam.robot.client import RobotClient

        opts = RobotClient.Options.with_api_key(
            api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"]
        )
        robot = await RobotClient.at_address(os.environ.get("VIAM_ADDRESS", DEFAULT_ADDRESS), opts)
        arm = RealArm(Arm.from_robot(robot, ARM))
        gripper = Gripper.from_robot(robot, GRIPPER)
        cam = RealCamera(Camera.from_robot(robot, WRIST_CAM))
    try:
        if await fetch(arm, gripper, cam, spec, release, stop_after):
            await retract(arm)
        else:
            log.info("stopped after '%s' as asked; the arm is left where it is", stop_after)
    except (KeyboardInterrupt, asyncio.CancelledError):
        await arm.stop()
        raise
    finally:
        if robot is not None:
            await robot.close()


async def print_marker(color):
    """Live: grab one wrist-camera frame and report where the marker is. Use it to check
    the colour threshold, and to set ALIGN_PX: put the arm in the pose where a grasp works
    (gripper at the cup rim), run this, and copy the centroid into ALIGN_PX."""
    from viam.components.camera import Camera
    from viam.robot.client import RobotClient

    opts = RobotClient.Options.with_api_key(api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"])
    robot = await RobotClient.at_address(os.environ.get("VIAM_ADDRESS", DEFAULT_ADDRESS), opts)
    try:
        frame = await RealCamera(Camera.from_robot(robot, WRIST_CAM)).frame()
        m = find_marker(frame, **MARKERS[color])
        print("marker:", "not found" if m is None else f"centroid=({m[0]:.0f}, {m[1]:.0f}) px, {m[2]} px area")
    finally:
        await robot.close()


def start_in_background(dry_run=True, release_after=10.0, color="green"):
    """Fire-and-forget for the voice loop. Returns False if a fetch is already running."""
    if not _running.acquire(blocking=False):
        return False
    RELEASE.clear()
    ABORT.clear()

    def job():
        try:
            asyncio.run(run(dry_run=dry_run, release_after=release_after, color=color))
        except Exception:
            log.exception("water fetch failed")
        finally:
            _running.release()

    threading.Thread(target=job, daemon=True, name="water-fetch").start()
    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true", help="drive the real arm (default is a dry run)")
    ap.add_argument("--color", choices=sorted(MARKERS), default="green", help="marker colour preset")
    ap.add_argument("--release-after", type=float, default=10.0, help="seconds to wait at the handover point (default 10)")
    ap.add_argument("--wait-for-enter", action="store_true", help="release on Enter instead of after a timer")
    ap.add_argument("--seed", type=int, help="dry-run: seed for the simulated cup position")
    ap.add_argument("--stop-after", choices=STAGES, help="commissioning: stop (and leave the arm) after this stage")
    ap.add_argument("--print-marker", action="store_true", help="live: report where the wrist camera sees the marker, then exit")
    args = ap.parse_args()
    if args.print_marker:
        asyncio.run(print_marker(args.color))
        return
    asyncio.run(run(dry_run=not args.live, release_after=None if args.wait_for_enter else args.release_after,
                    color=args.color, seed=args.seed, stop_after=args.stop_after))


if __name__ == "__main__":
    main()
