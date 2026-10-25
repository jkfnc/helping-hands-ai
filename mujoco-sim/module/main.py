"""Viam module: an SO-101 simulated in MuJoCo, as an arm, a gripper, cameras and a world.

Models (all share one MuJoCo world in this process):
    jayakumar:mujoco-so101:arm      5 joints, same kinematics as devrel:so101:simulated
    jayakumar:mujoco-so101:gripper  jaw gripper; grab() is True only if the cup is between the jaws
    jayakumar:mujoco-so101:camera   rendered images; attribute "camera": top | front | wrist_cam
    jayakumar:mujoco-so101:world    DoCommand: reset, place_cup, get_cup
"""
from __future__ import annotations

import asyncio
import math
import os
import time
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.media.video import CameraMimeType, NamedImage
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import (
    Geometry,
    KinematicsFileFormat,
    Mesh,
    Pose,
    ResourceName,
    ResponseMetadata,
)
from viam.proto.component.arm import GetPropertiesResponse as ArmProperties
from viam.proto.component.arm import JointPositions, MoveOptions
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

import sim as simlib

FAMILY = ModelFamily("jayakumar", "mujoco-so101")
KIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kinematics")
JOINT_TOL = math.radians(2.0)


def _kinematics(name):
    with open(os.path.join(KIN_DIR, name), "rb") as f:
        return KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA, f.read()


async def _run(fn, *args):
    return await asyncio.wrap_future(simlib.get_sim().call(fn, *args))


class MujocoArm(Arm, EasyResource):
    MODEL: ClassVar[Model] = Model(FAMILY, "arm")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = super().new(config, dependencies)
        attrs = struct_to_dict(config.attributes)
        simlib.get_sim(attrs.get("scene"))
        self.timeout = float(attrs.get("move_timeout_s", 20))
        return self

    async def _wait_until_reached(self, timeout=None):
        sim = simlib.get_sim()
        deadline = time.monotonic() + (timeout or self.timeout)
        while time.monotonic() < deadline:
            q, ctrl, target = await _run(sim.op_arm_state)
            if (abs(ctrl - target) < 1e-6).all() and (abs(q - target) < JOINT_TOL).all():
                return
            await asyncio.sleep(0.02)
        raise TimeoutError("arm did not reach its target (blocked, or pushing on something)")

    async def move_to_joint_positions(self, positions: JointPositions, *, extra=None, timeout=None, **kwargs):
        sim = simlib.get_sim()
        await _run(sim.op_set_arm_target, [math.radians(v) for v in positions.values])
        await self._wait_until_reached(timeout)

    async def move_through_joint_positions(self, positions: List[JointPositions], options: Optional[MoveOptions] = None,
                                           *, extra=None, timeout=None, **kwargs):
        for p in positions:
            await self.move_to_joint_positions(p, timeout=timeout)

    async def move_through_joint_positions_streamed(self, batches, *, extra=None, timeout=None, **kwargs):
        raise NotImplementedError("use move_through_joint_positions")

    async def move_to_position(self, pose: Pose, *, extra=None, timeout=None, **kwargs):
        raise NotImplementedError("use the motion service to move to a pose")

    async def get_joint_positions(self, *, extra=None, timeout=None, **kwargs) -> JointPositions:
        q, _, _ = await _run(simlib.get_sim().op_arm_state)
        return JointPositions(values=[math.degrees(v) for v in q])

    async def get_end_position(self, *, extra=None, timeout=None, **kwargs) -> Pose:
        (x, y, z), (ox, oy, oz, th) = await _run(simlib.get_sim().op_end_pose)
        return Pose(x=x, y=y, z=z, o_x=ox, o_y=oy, o_z=oz, theta=th)

    async def stop(self, *, extra=None, timeout=None, **kwargs):
        await _run(simlib.get_sim().op_hold_position)

    async def is_moving(self) -> bool:
        q, ctrl, target = await _run(simlib.get_sim().op_arm_state)
        return not ((abs(ctrl - target) < 1e-6).all() and (abs(q - target) < JOINT_TOL).all())

    async def get_kinematics(self, *, include_collision_meshes=False, include_visual_meshes=False,
                             extra=None, timeout=None, **kwargs):
        return _kinematics("so101_arm.json")

    async def get_3d_models(self, *, extra=None, timeout=None, **kwargs) -> Mapping[str, Mesh]:
        return {}

    async def get_properties(self, *, extra=None, timeout=None, **kwargs) -> ArmProperties:
        return ArmProperties()

    async def get_manual_mode(self, *, extra=None, timeout=None, **kwargs) -> bool:
        return False

    async def set_manual_mode(self, manual_mode: bool, enabled_for: int = 0, *, extra=None, timeout=None, **kwargs):
        raise NotImplementedError("the simulated arm has no manual mode")

    async def get_geometries(self, *, extra=None, timeout=None) -> List[Geometry]:
        return []

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout=None, **kwargs) -> Mapping[str, ValueTypes]:
        """Streaming control for a policy loop, which cannot wait for each move to finish.

        {"set_target": {"joints_deg": [5 floats], "grip": 0..1}}  aim the arm (and jaws: 0 closed, 1 open); returns at once
        {"get_state": true}  joints (deg) and grip (0..1) as they are now
        """
        sim = simlib.get_sim()
        out: Dict[str, Any] = {}
        if "set_target" in command:
            t = command["set_target"]
            if "joints_deg" in t:
                await _run(sim.op_set_arm_target, [math.radians(v) for v in t["joints_deg"]])
            if "grip" in t:
                await _run(sim.op_set_gripper_target, simlib.grip_to_rad(float(t["grip"])))
            out["set_target"] = True
        if command.get("get_state"):
            q, _, _ = await _run(sim.op_arm_state)
            g, _, _ = await _run(sim.op_gripper_state)
            out["joints_deg"] = [math.degrees(v) for v in q]
            out["grip"] = simlib.rad_to_grip(g)
        return out


class MujocoGripper(Gripper, EasyResource):
    MODEL: ClassVar[Model] = Model(FAMILY, "gripper")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = super().new(config, dependencies)
        simlib.get_sim(struct_to_dict(config.attributes).get("scene"))
        return self

    async def _settle(self, seconds=1.5):
        """Wait until the jaw stops moving (it stops early when it closes on the cup)."""
        sim = simlib.get_sim()
        deadline = time.monotonic() + seconds
        last = None
        while time.monotonic() < deadline:
            q, ctrl, target = await _run(sim.op_gripper_state)
            if abs(ctrl - target) < 1e-6 and last is not None and abs(q - last) < 1e-4:
                return
            last = q
            await asyncio.sleep(0.05)

    async def open(self, *, extra=None, timeout=None, **kwargs):
        await _run(simlib.get_sim().op_set_gripper_target, simlib.GRIPPER_OPEN)
        await self._settle()

    async def grab(self, *, extra=None, timeout=None, **kwargs) -> bool:
        await _run(simlib.get_sim().op_set_gripper_target, simlib.GRIPPER_CLOSED)
        await self._settle()
        return await _run(simlib.get_sim().op_cup_held)

    async def is_holding_something(self, *, extra=None, timeout=None, **kwargs) -> Gripper.HoldingStatus:
        return Gripper.HoldingStatus(await _run(simlib.get_sim().op_cup_held))

    async def stop(self, *, extra=None, timeout=None, **kwargs):
        await _run(simlib.get_sim().op_hold_position)

    async def is_moving(self) -> bool:
        q, ctrl, target = await _run(simlib.get_sim().op_gripper_state)
        return abs(ctrl - target) > 1e-6

    async def get_kinematics(self, *, include_collision_meshes=False, include_visual_meshes=False,
                             extra=None, timeout=None, **kwargs):
        return _kinematics("so101_gripper.json")

    async def get_current_inputs(self, *, extra=None, timeout=None, **kwargs) -> List[float]:
        return []

    async def go_to_inputs(self, values: List[float], *, extra=None, timeout=None, **kwargs):
        return None

    async def get_geometries(self, *, extra=None, timeout=None) -> List[Geometry]:
        return []

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout=None, **kwargs) -> Mapping[str, ValueTypes]:
        if "get_position" in command:
            q, _, _ = await _run(simlib.get_sim().op_gripper_state)
            return {"radians": q}
        return {}


def _yam():
    """The running world, which must be Dream's YAM scene (mj-world {"load_scene": "yam"})."""
    s = simlib.get_sim()
    if getattr(s, "scene", None) != "yam":
        raise RuntimeError('the YAM components need the yam scene: mj-world {"load_scene": "yam"}')
    return s


def _yam_urdf():
    """yam.urdf with its meshes taken out: the kinematic chain only, for Viam's frame system."""
    import xml.etree.ElementTree as ET
    from dream_examples import yam_mujoco
    root = ET.parse(os.path.join(list(yam_mujoco.__path__)[0], "assets", "yam.urdf")).getroot()
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)
    return ET.tostring(root)


class YamArm(Arm, EasyResource):
    """One of the two YAM arms of Dream's example (attribute "side": left | right), six joints."""
    MODEL: ClassVar[Model] = Model(FAMILY, "yam-arm")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = super().new(config, dependencies)
        self.side = struct_to_dict(config.attributes).get("side", "right")
        return self

    async def get_joint_positions(self, *, extra=None, timeout=None, **kwargs) -> JointPositions:
        s = _yam()
        state = await asyncio.wrap_future(s.call(s.op_state))
        return JointPositions(values=[math.degrees(v) for v in state[self.side][:6]])

    async def move_to_joint_positions(self, positions: JointPositions, *, extra=None, timeout=None, **kwargs):
        s = _yam()
        state = await asyncio.wrap_future(s.call(s.op_state))
        target = [math.radians(v) for v in positions.values] + [state[self.side][6]]
        await asyncio.wrap_future(s.call(s.op_set_targets, {self.side: target}))
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            q = (await asyncio.wrap_future(s.call(s.op_state)))[self.side][:6]
            if max(abs(a - b) for a, b in zip(q, target[:6])) < JOINT_TOL:
                return
            await asyncio.sleep(0.05)

    async def move_through_joint_positions(self, positions, options=None, *, extra=None, timeout=None, **kwargs):
        for p in positions:
            await self.move_to_joint_positions(p)

    async def move_through_joint_positions_streamed(self, batches, *, extra=None, timeout=None, **kwargs):
        raise NotImplementedError("use move_through_joint_positions")

    async def move_to_position(self, pose: Pose, *, extra=None, timeout=None, **kwargs):
        raise NotImplementedError("use joint positions")

    async def get_end_position(self, *, extra=None, timeout=None, **kwargs) -> Pose:
        raise NotImplementedError("no end pose for the YAM arms")

    async def stop(self, *, extra=None, timeout=None, **kwargs):
        s = _yam()
        state = await asyncio.wrap_future(s.call(s.op_state))
        await asyncio.wrap_future(s.call(s.op_set_targets, {self.side: state[self.side]}))

    async def is_moving(self) -> bool:
        return False

    async def get_kinematics(self, *, include_collision_meshes=False, include_visual_meshes=False,
                             extra=None, timeout=None, **kwargs):
        return KinematicsFileFormat.KINEMATICS_FILE_FORMAT_URDF, _yam_urdf()

    async def get_3d_models(self, *, extra=None, timeout=None, **kwargs) -> Mapping[str, Mesh]:
        return {}

    async def get_properties(self, *, extra=None, timeout=None, **kwargs) -> ArmProperties:
        return ArmProperties()

    async def get_manual_mode(self, *, extra=None, timeout=None, **kwargs) -> bool:
        return False

    async def set_manual_mode(self, manual_mode: bool, enabled_for: int = 0, *, extra=None, timeout=None, **kwargs):
        raise NotImplementedError("the simulated arm has no manual mode")

    async def get_geometries(self, *, extra=None, timeout=None) -> List[Geometry]:
        return []


class YamGripper(Gripper, EasyResource):
    """A YAM arm's gripper in Dream's example (attribute "side")."""
    MODEL: ClassVar[Model] = Model(FAMILY, "yam-gripper")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = super().new(config, dependencies)
        self.side = struct_to_dict(config.attributes).get("side", "right")
        return self

    async def _set(self, grip):
        s = _yam()
        await asyncio.wrap_future(s.call(s.op_set_grip, self.side, grip))
        await asyncio.sleep(1.0)
        return (await asyncio.wrap_future(s.call(s.op_state)))[self.side][6]

    async def open(self, *, extra=None, timeout=None, **kwargs):
        await self._set(1.0)

    async def grab(self, *, extra=None, timeout=None, **kwargs) -> bool:
        return await self._set(0.0) > 0.05   # stopped short of closed: something is between the fingers

    async def is_holding_something(self, *, extra=None, timeout=None, **kwargs) -> Gripper.HoldingStatus:
        s = _yam()
        return Gripper.HoldingStatus((await asyncio.wrap_future(s.call(s.op_state)))[self.side][6] > 0.05)

    async def stop(self, *, extra=None, timeout=None, **kwargs):
        return None

    async def is_moving(self) -> bool:
        return False

    async def get_kinematics(self, *, include_collision_meshes=False, include_visual_meshes=False,
                             extra=None, timeout=None, **kwargs):
        return KinematicsFileFormat.KINEMATICS_FILE_FORMAT_UNSPECIFIED, b""

    async def get_current_inputs(self, *, extra=None, timeout=None, **kwargs) -> List[float]:
        return []

    async def go_to_inputs(self, values: List[float], *, extra=None, timeout=None, **kwargs):
        return None

    async def get_geometries(self, *, extra=None, timeout=None) -> List[Geometry]:
        return []


class MujocoCamera(Camera, EasyResource):
    MODEL: ClassVar[Model] = Model(FAMILY, "camera")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = super().new(config, dependencies)
        attrs = struct_to_dict(config.attributes)
        simlib.get_sim(attrs.get("scene"))
        self.camera = attrs.get("camera", "front")
        self.width = int(attrs.get("width", 640))
        self.height = int(attrs.get("height", 480))
        self.quality = int(attrs.get("jpeg_quality", 85))
        return self

    async def get_images(self, *, filter_source_names: Optional[Sequence[str]] = None, extra=None, timeout=None,
                         **kwargs) -> Tuple[Sequence[NamedImage], ResponseMetadata]:
        jpeg = await _run(simlib.get_sim().op_render, self.camera, self.width, self.height, self.quality)
        return [NamedImage(self.camera, jpeg, CameraMimeType.JPEG)], ResponseMetadata()

    async def get_point_cloud(self, *, extra=None, timeout=None, **kwargs) -> Tuple[bytes, str]:
        raise NotImplementedError("no point clouds from the MuJoCo camera")

    async def get_properties(self, *, timeout=None, **kwargs) -> Camera.Properties:
        return Camera.Properties(supports_pcd=False, mime_types=[CameraMimeType.JPEG])

    async def get_geometries(self, *, extra=None, timeout=None) -> List[Geometry]:
        return []


class MujocoWorld(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(FAMILY, "world")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = super().new(config, dependencies)
        simlib.get_sim(struct_to_dict(config.attributes).get("scene"))
        return self

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout=None, **kwargs) -> Mapping[str, ValueTypes]:
        out: Dict[str, Any] = {}
        if "load_scene" in command:
            # off the event loop: building Dream's scene takes a few seconds
            await asyncio.to_thread(simlib.switch_scene, str(command["load_scene"]))
            out["load_scene"] = command["load_scene"]
        if command.get("get_scene"):
            out["scene"] = simlib.get_sim().scene
        sim = simlib.get_sim()
        if getattr(sim, "scene", None) == "yam":
            # Dream's YAM scene: both arms in one call, so a 20 Hz policy loop is not two round trips a row
            if command.get("reset"):
                out["reset"] = await _run(sim.op_reset)
            if "pause" in command:
                out["paused"] = await _run(sim.op_pause, bool(command["pause"]))
            if "yam_set" in command:
                out["yam_set"] = await _run(sim.op_set_targets, dict(command["yam_set"]))
            if command.get("yam_state"):
                out["yam_state"] = await _run(sim.op_state)
            if command.get("objects"):
                out["objects"] = await _run(sim.op_objects)
            return out
        if command.get("reset"):
            out["reset"] = await _run(sim.op_reset)
        if "place_cup" in command:
            p = command["place_cup"]
            out["place_cup"] = await _run(sim.op_place_cup, float(p["x"]) / 1000.0, float(p["y"]) / 1000.0)
        if command.get("get_cup"):
            pos, quat = await _run(sim.op_cup_pose)
            out["cup"] = {"x": pos[0], "y": pos[1], "z": pos[2], "quat": quat}
        if command.get("cup_held"):
            out["cup_held"] = await _run(sim.op_cup_held)
        return out

    async def get_geometries(self, *, extra=None, timeout=None) -> List[Geometry]:
        return []


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
