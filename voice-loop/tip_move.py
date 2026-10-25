"""Move the follower's tip to a target (reach r, height z, pitch) using the arm's own
GetEndPosition readout, not a kinematic model: measure how r, z and pitch respond to the
shoulder, elbow and wrist flex right here, then take damped Newton steps.

    python tip_move.py --r 295 --z 62 --pitch 0

Safety: each step moves a joint at most MAX_STEP deg, joints are clamped to their limits,
and it refuses to go below MIN_Z. Base angle and wrist roll are left alone.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os

import numpy as np

import water_fetch as wf   # loads .env

MAX_STEP = 6.0
MAX_REACH_STEP = 12.0   # mm of reach per iteration
MIN_Z = 40.0
LIMITS = [(-55.0, 95.0), (-105.0, 34.0), (-94.0, 94.0)]    # shoulder, elbow, wrist flex


async def connect():
    from viam.components.arm import Arm
    from viam.robot.client import RobotClient

    opts = RobotClient.Options.with_api_key(api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"])
    robot = await RobotClient.at_address(os.environ.get("VIAM_ADDRESS", wf.DEFAULT_ADDRESS), opts)
    return robot, wf.RealArm(Arm.from_robot(robot, wf.ARM))


async def state(arm):
    j = await arm.get_joint_positions()
    p = await arm.get_end_position()
    return np.array(j), np.array([math.hypot(p.x, p.y), p.z, math.degrees(math.asin(max(-1.0, min(1.0, -p.o_z))))])


async def move(arm, j, q):
    new = list(j)
    for k in range(3):
        lo, hi = LIMITS[k]
        new[1 + k] = float(min(max(q[k], lo), hi))
    await arm.move_to_joint_positions(new)


async def go_to(arm, target, tol=(4.0, 4.0, 4.0), max_iter=12, probe=3.0):
    j, t0 = await state(arm)
    print(f"start: joints {np.round(j[1:4], 1)}  tip r={t0[0]:.0f} z={t0[1]:.0f} pitch={t0[2]:.1f}")
    # finite-difference Jacobian d(r, z, pitch) / d(shoulder, elbow, flex)
    J = np.zeros((3, 3))
    for k in range(3):
        q = j[1:4].copy()
        q[k] += probe
        await move(arm, j, q)
        _, t1 = await state(arm)
        J[:, k] = (t1 - t0) / probe
        await move(arm, j, j[1:4])
    print("Jacobian (rows r,z,pitch; cols shoulder,elbow,flex):\n", np.round(J, 2))
    for i in range(max_iter):
        j, t = await state(arm)
        err = np.array(target) - t
        print(f"iter {i}: tip r={t[0]:.0f} z={t[1]:.0f} pitch={t[2]:.1f}  error {np.round(err, 1)}")
        if all(abs(e) <= tl for e, tl in zip(err, tol)):
            print("reached")
            return True
        step_err = err.copy()
        step_err[0] = np.clip(err[0], -MAX_REACH_STEP, MAX_REACH_STEP)   # small reach steps
        dq = np.linalg.lstsq(J, step_err, rcond=None)[0] * 0.7
        dq = np.clip(dq, -MAX_STEP, MAX_STEP)
        # shrink the step until the current estimate keeps the tip above the floor
        scale = 1.0
        while t[1] + (J @ (dq * scale))[1] < MIN_Z + 4.0 and scale > 0.1:
            scale *= 0.6
        if t[1] + (J @ (dq * scale))[1] < MIN_Z:
            raise SystemExit(f"stopping: even a tiny step would take the tip below z={MIN_Z}")
        dq = dq * scale
        q = j[1:4] + dq
        await move(arm, j, q)
        _, t2 = await state(arm)
        # Broyden update so the Jacobian follows the arm as it moves
        d = t2 - t
        if np.linalg.norm(dq) > 1e-6:
            J += np.outer(d - J @ dq, dq) / (dq @ dq)
    print("did not reach within the iteration limit")
    return False


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--z", type=float, required=True)
    ap.add_argument("--pitch", type=float, default=0.0, help="degrees the jaws point below horizontal")
    args = ap.parse_args()
    robot, arm = await connect()
    try:
        await go_to(arm, (args.r, args.z, args.pitch))
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
