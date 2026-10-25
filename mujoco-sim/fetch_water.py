"""Fetch the cup in the MuJoCo sim and hand it to the person behind the arm, through Viam.

    .venv/bin/python fetch_water.py            # reset the scene, then fetch
    .venv/bin/python fetch_water.py --no-reset # fetch from wherever the cup is now
    .venv/bin/python fetch_water.py --cup 400 -40   # put the cup at (x, y) mm first

The motion planner moves mj-gripper; the physics decides whether the cup comes along.
The cup's position comes from the simulator (mj-world get_cup) -- the stand-in for
perception, which Dream-1 or a vision model would replace.

Positions below are in the MuJoCo cell's own frame (arm base at the origin, mm). The
cell sits at MJ_OFFSET in Viam's world frame so it does not overlap arm-1.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os

from viam.components.arm import Arm
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.proto.component.arm import JointPositions
from viam.proto.common import (
    GeometriesInFrame,
    Geometry,
    Pose,
    PoseInFrame,
    RectangularPrism,
    Vector3,
    WorldState,
)
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

ADDRESS = os.environ.get("SO101_ADDRESS", "so101-main.xjrc4czns9.viam.cloud")
ARM, GRIPPER, WORLD, MOTION = "mj-arm", "mj-gripper", "mj-world", "builtin"
MJ_OFFSET = (0.0, 1500.0, 0.0)   # must match mj-arm's frame translation
REST = [0, -80, 80, -20, -90]    # joints (deg): jaws raised ~220 mm, clear of everything

ROLL_THETA = 90.0        # OV theta that gives wrist roll -90: jaws close sideways, camera on top
CUP_R, CUP_H = 20.0, 80.0
GRASP_IN = 6.0           # tip this far past the cup's axis
APPROACH = 70.0          # back off this far before going in
HOVER = 80.0             # height above the approach point for the first move
HANDOVER_LIFT = 40.0     # cup bottom above the palm while carrying
HAND = (-1.6, 307.4, 140.0)    # handover site: cup center over the palm (scene_cup.xml)
PALM_TOP = 96.0


def load_env():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# A 5-joint arm can only point along its own vertical plane. With the wrist rolled -90 the
# tip sits SIDE_OFFSET mm to the side of that plane, and the plane turns about PAN_AXIS, so
# the jaws must point slightly off the line from the base to the target (measured in MuJoCo).
PAN_AXIS = (38.84, 0.0)
SIDE_OFFSET = -12.15


def heading_to(x, y):
    """Compass heading (rad) of the arm plane that puts the tip at (x, y)."""
    px, py = x - PAN_AXIS[0], y - PAN_AXIS[1]
    reach = math.sqrt(max(px * px + py * py - SIDE_OFFSET ** 2, 1.0))
    return math.atan2(py, px) - math.atan2(SIDE_OFFSET, reach)


def pose(x, y, z):
    """Tip at (x, y, z) in the cell frame, jaws level, pointing out along the arm plane."""
    h = heading_to(x, y)
    return Pose(x=x + MJ_OFFSET[0], y=y + MJ_OFFSET[1], z=z + MJ_OFFSET[2],
                o_x=math.cos(h), o_y=math.sin(h), o_z=0.0, theta=ROLL_THETA)


def box(label, x, y, z, dx, dy, dz):
    return GeometriesInFrame(reference_frame="world", geometries=[Geometry(
        center=Pose(x=x + MJ_OFFSET[0], y=y + MJ_OFFSET[1], z=z + MJ_OFFSET[2], o_z=1),
        box=RectangularPrism(dims_mm=Vector3(x=dx, y=dy, z=dz)), label=label)])


class Fetcher:
    def __init__(self, robot):
        self.robot = robot
        self.motion = MotionClient.from_robot(robot, MOTION)
        self.arm = Arm.from_robot(robot, ARM)
        self.gripper = Gripper.from_robot(robot, GRIPPER)
        self.world = Generic.from_robot(robot, WORLD)

    def obstacles(self, cup=None):
        obs = [box("table", 150, 0, -6, 900, 900, 10),
               box("palm", HAND[0], HAND[1], 90, 90, 90, 12)]
        if cup is not None:
            obs.append(box("cup", cup[0], cup[1], CUP_H / 2, 2 * CUP_R + 6, 2 * CUP_R + 6, CUP_H + 4))
        return WorldState(obstacles=obs)

    async def move(self, label, p, obstacles=None, linear=False):
        print(f"-> {label}")
        last = None
        for attempt in range(3):
            # a straight-line move first; the planner often finds no straight path for this
            # 5-joint arm, and these moves are short, so fall back to a free plan
            constraints = (Constraints(linear_constraint=[LinearConstraint(line_tolerance_mm=5)])
                           if linear and attempt == 0 else None)
            try:
                await self.motion.move(component_name=GRIPPER, destination=PoseInFrame(reference_frame="world", pose=p),
                                       world_state=obstacles, constraints=constraints,
                                       extra={"timeout": 15.0}, timeout=90)
                return
            except Exception as e:  # noqa: BLE001
                last = e
                print(f"   plan attempt {attempt + 1} failed: {str(e).splitlines()[0][:160]}")
        raise last

    async def cup(self):
        c = (await self.world.do_command({"get_cup": True}))["cup"]
        return c["x"], c["y"], c["z"]

    async def run(self, reset=True, place=None):
        if reset:
            await self.world.do_command({"reset": True})
        if place:
            await self.world.do_command({"place_cup": {"x": place[0], "y": place[1]}})
        await asyncio.sleep(0.5)
        cx, cy, cz = await self.cup()
        print(f"cup at ({cx:.0f}, {cy:.0f}, {cz:.0f}) mm")
        heading = math.atan2(cy, cx)
        ux, uy = math.cos(heading), math.sin(heading)
        gx, gy = cx + GRASP_IN * ux, cy + GRASP_IN * uy
        px, py = gx - APPROACH * ux, gy - APPROACH * uy

        await self.gripper.open()
        await self.move("above the approach point", pose(px, py, cz + HOVER), self.obstacles((cx, cy)))
        await self.move("approach point", pose(px, py, cz), self.obstacles((cx, cy)))
        await self.move("in around the cup", pose(gx, gy, cz), self.obstacles(), linear=True)
        held = await self.gripper.grab()
        print(f"   grab: holding={held}")
        if not held:
            raise SystemExit("the jaws closed on nothing; stopping")
        await self.move("lift", pose(gx, gy, cz + 70), self.obstacles(), linear=True)

        hx, hy, hz = HAND
        hh = math.atan2(hy, hx)
        hux, huy = math.cos(hh), math.sin(hh)
        await self.move("carry toward the person", pose(hx - 60 * hux, hy - 60 * huy, hz + HANDOVER_LIFT), self.obstacles())
        await self.move("over the palm", pose(hx - GRASP_IN * hux, hy - GRASP_IN * huy, hz + 10), self.obstacles())
        await self.move("onto the palm", pose(hx - GRASP_IN * hux, hy - GRASP_IN * huy, hz), self.obstacles(), linear=True)
        await self.gripper.open()
        # back off in joint space: tilt the shoulder back (jaws rise and pull away), then rest
        print("-> back off")
        j = list((await self.arm.get_joint_positions()).values)
        j[1] -= 25
        await self.arm.move_to_joint_positions(JointPositions(values=j))
        await self.arm.move_to_joint_positions(JointPositions(values=[j[0]] + REST[1:]))
        await self.arm.move_to_joint_positions(JointPositions(values=REST))

        cx, cy, cz = await self.cup()
        on_palm = math.hypot(cx - hx, cy - hy) < 45 and abs((cz - CUP_H / 2) - PALM_TOP) < 10
        print(f"cup now at ({cx:.0f}, {cy:.0f}, {cz:.0f}) mm -> {'on the palm, handed over' if on_palm else 'NOT on the palm'}")
        return on_palm


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--cup", nargs=2, type=float, metavar=("X", "Y"))
    args = ap.parse_args()
    load_env()
    robot = await RobotClient.at_address(ADDRESS, RobotClient.Options.with_api_key(
        api_key=os.environ["SO101_API_KEY"], api_key_id=os.environ["SO101_API_KEY_ID"]))
    try:
        await Fetcher(robot).run(reset=not args.no_reset, place=args.cup)
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
