"""Plumbing for the SO-101 palletizer: connecting and retrying.

The palletizer-101 tutorial asked a pick-station and pallet component (and the
pack-sequencer) for poses. The so101 sim has none of those, so the cell layout
lives here as constants instead, sized for the SO-101's ~250 mm reach.

Credentials come from ../.env (or the environment):
    SO101_API_KEY_ID=...
    SO101_API_KEY=...
Create the key in the Viam app: so101 -> CONNECT -> API keys.
"""

import asyncio
import os
import sys

from viam.robot.client import RobotClient
from viam.services.worldstatestore import WorldStateStore

MACHINE_ADDRESS = os.environ.get("SO101_ADDRESS", "so101-main.xjrc4czns9.viam.cloud")

# Resource names on so101.
ARM = "arm-1"
GRIPPER = "gripper-1"
MOTION = "builtin"
PACK_SEQUENCER = "pack-sequencer"   # draws the cubes in the 3D view

# Cell layout, world frame, mm. Pointing straight down, the planner reaches
# z ~70 at up to ~220 mm out, so everything sits 170-215 mm from the base.
CUBE = 25.0                    # cube edge
PICK_XY = (170.0, -100.0)      # where cubes are presented
PALLET_XY = (170.0, 100.0)     # center of the 2x2 stack area
SLOT_PITCH = 30.0              # center-to-center spacing on the pallet
HOVER_Z = 70.0                 # safe travel height for the gripper tip


def _load_env():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()


async def connect() -> RobotClient:
    """Connect to so101 with the machine API key and return a client."""
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["SO101_API_KEY"], api_key_id=os.environ["SO101_API_KEY_ID"]
    )
    return await RobotClient.at_address(MACHINE_ADDRESS, opts)


async def retry(what, factory, attempts=3, delay=1.0):
    """Run an RPC, riding out the occasional connection blip."""
    last = None
    for k in range(attempts):
        try:
            return await factory()
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  [retry] {what}: attempt {k + 1}/{attempts} ({type(e).__name__})")
            await asyncio.sleep(delay)
    raise last


async def _set_box(robot, seq, parent, x, y, z):
    store = WorldStateStore.from_robot(robot, PACK_SEQUENCER)
    try:
        await retry(
            "set_box_transform",
            lambda: store.do_command(
                {"set_box_transform": {
                    "seq": seq, "parent": parent, "x": x, "y": y, "z": z,
                    "o_x": 0, "o_y": 0, "o_z": 1, "theta": 0}}
            ),
            attempts=2,
        )
    except Exception as e:  # viz is cosmetic; never let it block motion
        print(f"  [viz] set_box_transform failed (non-fatal): {type(e).__name__}")


async def show_box(robot, seq, x, y, z):
    """Draw cube `seq` resting at the world position (x, y, z) (its center)."""
    await _set_box(robot, seq, "world", x, y, z)


async def attach_box(robot, seq):
    """Draw cube `seq` between the jaws; it is centered on the gripper tip."""
    await _set_box(robot, seq, GRIPPER, 0, 0, 0)


async def clear_boxes(robot, count=10):
    """Remove the drawn cubes."""
    store = WorldStateStore.from_robot(robot, PACK_SEQUENCER)
    for i in range(count):
        try:
            await store.do_command({"clear_box_transform": {"seq": i}})
        except Exception:  # noqa: BLE001
            pass


# `python helpers.py` proves the connection.
if __name__ == "__main__":
    if not os.environ.get("SO101_API_KEY") or not os.environ.get("SO101_API_KEY_ID"):
        print("not connected: add SO101_API_KEY_ID and SO101_API_KEY to ../.env")
        sys.exit(1)

    async def _check():
        robot = await connect()
        print(f"connected to {MACHINE_ADDRESS}")
        await robot.close()

    try:
        asyncio.run(_check())
    except Exception as e:  # noqa: BLE001
        print(f"not connected: could not reach {MACHINE_ADDRESS} ({type(e).__name__}). "
              "Check the key and that viam-server is running.")
        sys.exit(1)
