"""palletizer.py for the simulated SO-101 (machine so101).

A port of the palletizer-101 tutorial: same verbs, scaled from a UR5e with a
vacuum cup and 200 x 150 x 100 mm boxes to a 5-joint SO-101 with a jaw gripper
and 25 mm cubes.

    python palletizer.py            -> "run": stack 8 cubes, two layers of 2x2
    python palletizer.py pick       -> pick one cube and lift it
    python palletizer.py tower 170 100  -> one column at x=170, y=100 until out of reach

Differences from the UR5e version:
- Poses come from constants in helpers.py (no pick-station / pallet components).
- The jaws close around the middle of the cube, so the gripper tip goes to the
  cube's center height, not its top face.
- The pack-sequencer only draws the cubes in the 3D view; the planner sees
  placed and held cubes through the obstacle set built here.
- The simulated gripper never reports holding anything; grab() returns False.
"""

import asyncio
import sys

import helpers
from helpers import CUBE, HOVER_Z, connect
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import (
    GeometriesInFrame,
    Geometry,
    Pose,
    PoseInFrame,
    RectangularPrism,
    Vector3,
    WorldState,
)
from viam.proto.component.arm import JointPositions
from viam.services.motion import MotionClient


def down_pose(x, y, z) -> Pose:
    """A target pose at (x, y, z) with the gripper pointing straight down."""
    return Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=0)


def _box(label, frame, x, y, z, dims):
    return GeometriesInFrame(
        reference_frame=frame,
        geometries=[Geometry(
            center=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=1, theta=0),
            box=RectangularPrism(dims_mm=Vector3(x=dims[0], y=dims[1], z=dims[2])),
            label=label,
        )],
    )


class Palletizer:
    def __init__(self, robot):
        self.robot = robot
        self.arm = Arm.from_robot(robot, helpers.ARM)
        self.gripper = Gripper.from_robot(robot, helpers.GRIPPER)
        self.placed = []    # centers of cubes already placed

    def obstacles(self, held=False):
        """Table top plus every placed cube, and optionally the cube in the jaws."""
        # Top face 1 mm below z=0 so a cube resting on the table does not touch it.
        obs = [_box("table", "world", 0, 0, -6, (1000, 1000, 10))]
        obs += [_box(f"placed-{i}", "world", x, y, z, (CUBE,) * 3)
                for i, (x, y, z) in enumerate(self.placed)]
        if held:
            obs.append(_box("held", helpers.GRIPPER, 0, 0, 0, (CUBE,) * 3))
        return WorldState(obstacles=obs)

    async def resources(self):
        """List the resources the machine exposes (a first, safe call)."""
        for name in sorted(rn.name for rn in self.robot.resource_names):
            print(" ", name)

    async def wave(self):
        """Move the arm by commanding its 5 joint angles directly."""
        await self.arm.move_to_joint_positions(JointPositions(values=[20, 10, 0, 30, 0]))

    async def zero(self):
        """All joints to 0: the arm stretched out forward."""
        await self.arm.move_to_joint_positions(JointPositions(values=[0, 0, 0, 0, 0]))

    async def move_gripper(self, pose, obstacles=None):
        """Move the gripper tip to `pose`, routing around `obstacles`."""
        motion = MotionClient.from_robot(self.robot, helpers.MOTION)
        destination = PoseInFrame(reference_frame="world", pose=pose)
        return await helpers.retry("move", lambda: motion.move(
            component_name=helpers.GRIPPER,
            destination=destination,
            world_state=obstacles,
            extra={"timeout": 15.0},
            timeout=60,
        ))

    async def try_move(self, pose, obstacles=None):
        """One quick planning attempt; True if the gripper got there."""
        motion = MotionClient.from_robot(self.robot, helpers.MOTION)
        try:
            ok = await motion.move(
                component_name=helpers.GRIPPER,
                destination=PoseInFrame(reference_frame="world", pose=pose),
                world_state=obstacles,
                extra={"timeout": 5.0},
                timeout=30,
            )
        except Exception as e:  # noqa: BLE001
            print(f"   [plan failed] {type(e).__name__}: {str(e).replace(chr(10), ' ')[:200]}")
            return False
        return ok is not False

    async def move(self):
        """Send the gripper above the pick spot, pointing down."""
        await self.move_gripper(down_pose(*helpers.PICK_XY, HOVER_Z))

    async def pick(self, seq=0):
        """Pick the cube at the pick spot and lift it (cube ends up held)."""
        x, y = helpers.PICK_XY
        await self.gripper.open()
        await self.move_gripper(down_pose(x, y, HOVER_Z), self.obstacles())
        await helpers.show_box(self.robot, seq, x, y, CUBE / 2)
        await self.move_gripper(down_pose(x, y, CUBE / 2), self.obstacles())
        await self.gripper.grab()
        await helpers.attach_box(self.robot, seq)
        await self.move_gripper(down_pose(x, y, HOVER_Z), self.obstacles())
        print(f"  picked cube {seq}")

    def _place_pose(self, i):
        """Cube i -> (x, y, z of its center): two layers of a 2x2 grid."""
        layer, slot = i // 4, i % 4
        col, row = slot % 2, slot // 2
        cx, cy = helpers.PALLET_XY
        x = cx + (col - 0.5) * helpers.SLOT_PITCH
        y = cy + (row - 0.5) * helpers.SLOT_PITCH
        z = (layer + 0.5) * CUBE
        return x, y, z

    async def place(self):
        """One full pick-and-place of the next cube onto the pallet."""
        seq = len(self.placed)
        x, y, z = self._place_pose(seq)
        await self.pick(seq)
        # over the slot with the held cube modeled, so it cannot drag through the stack
        await self.move_gripper(down_pose(x, y, HOVER_Z), self.obstacles(held=True))
        # straight down into the slot
        await self.move_gripper(down_pose(x, y, z), self.obstacles())
        await self.gripper.open()
        await helpers.show_box(self.robot, seq, x, y, z)
        self.placed.append((x, y, z))
        await self.move_gripper(down_pose(x, y, HOVER_Z), self.obstacles())
        print(f"  placed cube {seq} at ({x:.0f}, {y:.0f}, {z:.1f})")

    async def run(self):
        """Stack 8 cubes: two layers of four."""
        await helpers.clear_boxes(self.robot)
        self.placed = []
        for _ in range(8):
            await self.place()
        print(f"packed {len(self.placed)} cubes")

    async def tower(self, x=None, y=None):
        """Stack cubes in one column until the arm can't reach the next level."""
        cx, cy = helpers.PALLET_XY
        cx = cx if x is None else x
        cy = cy if y is None else y
        await helpers.clear_boxes(self.robot)
        self.placed = []
        print(f"building a tower at x={cx:.0f}, y={cy:.0f}")
        for seq in range(10):
            z = (seq + 0.5) * CUBE
            hover = max(HOVER_Z, z + CUBE + 10)
            print(f"-> cube {seq + 1}: center at z={z:.1f}")
            # check the level with an empty gripper first, so we never pick a cube we can't place
            if not await self.try_move(down_pose(cx, cy, hover), self.obstacles()):
                print("   can't reach above that level, stopping")
                break
            await self.pick(seq)
            if not await self.try_move(down_pose(cx, cy, hover), self.obstacles(held=True)):
                print("   can't get above that level while holding the cube, stopping")
                break
            await self.move_gripper(down_pose(cx, cy, z), self.obstacles())
            await self.gripper.open()
            await helpers.show_box(self.robot, seq, cx, cy, z)
            self.placed.append((cx, cy, z))
        print(f"tower at ({cx:.0f}, {cy:.0f}): {len(self.placed)} cubes, {len(self.placed) * CUBE:.0f} mm tall")


# verb -> method. One entry per capability.
STEPS = {
    "resources": Palletizer.resources,
    "wave": Palletizer.wave,
    "zero": Palletizer.zero,
    "move": Palletizer.move,
    "pick": Palletizer.pick,
    "place": Palletizer.place,
    "run": Palletizer.run,
    "tower": Palletizer.tower,
}


async def main(verb, args=()):
    robot = await connect()
    palletizer = Palletizer(robot)
    try:
        step = STEPS.get(verb)
        if step is None:
            print(f"unknown step '{verb}'. steps: {', '.join(STEPS)}")
            return
        await step(palletizer, *args)
    finally:
        await robot.close()


if __name__ == "__main__":
    verb = sys.argv[1] if len(sys.argv) > 1 else "run"
    extra = [float(a) for a in sys.argv[2:]]
    asyncio.run(main(verb, extra))
