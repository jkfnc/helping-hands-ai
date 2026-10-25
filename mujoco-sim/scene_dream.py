"""Dream's own YAM MuJoCo scene, with our SO-101 standing where Dream's zero-shot SO-101 stands.

The dream library ships the scene Dream-1 runs best in (dream_examples.yam_mujoco): an oak
table, a bowl, a plate, and the station's calibrated top camera. This builds it, takes out the
two YAM arms and the fruit, and attaches the MuJoCo Menagerie SO-101 at Dream's placement for a
zero-shot SO-101: its arm root 0.1775 m forward, 0.125 m left and 3 mm down from the right YAM
arm's base (so101_right.urdf's arm_root_joint), which is the frame its hand poses are read in
(the arm file's `served_base`). A cup like the one in scene_cup.xml goes on the table in reach.

    .venv/bin/python scene_dream.py      # build, settle, save camera frames to check the layout
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SO101_XML = os.path.join(HERE, "menagerie", "robotstudio_so101", "so101.xml")

RIGHT_BASE_POS = np.array([-0.305, 0.24, 0.772])       # yam_env: the right arm's base
RIGHT_BASE_QUAT = np.array([0.7071067811865476, 0, 0, -0.7071067811865476])
ARM_ROOT_IN_BASE = np.array([0.1775, 0.125, -0.003])   # so101_right.urdf arm_root_joint
CUP_HOME = (-0.12, -0.10, 0.0)                          # cup, world x, y (z set from the table top)
TABLE_TOP = 0.76


def _rotate(quat, v):
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, float), np.asarray(quat, float))
    return out


def arm_root_pose():
    """World position and orientation of the SO-101's root (its base link)."""
    return RIGHT_BASE_POS + _rotate(RIGHT_BASE_QUAT, ARM_ROOT_IN_BASE), RIGHT_BASE_QUAT


def build_spec() -> mujoco.MjSpec:
    from dream_examples.yam_mujoco import yam_env

    spec = mujoco.MjSpec.from_string(yam_env.build_xml())
    # the two YAM arms and everything that refers to them, and the fruit
    yam_joints = set()
    for side in ("left", "right"):
        body = spec.body(f"{side}_base")
        for j in body.find_all(mujoco.mjtObj.mjOBJ_JOINT):
            yam_joints.add(j.name)
        spec.delete(body)
    for a in list(spec.actuators):
        if a.target in yam_joints or a.target.startswith(("left", "right")):
            spec.delete(a)
    for e in list(spec.equalities):
        spec.delete(e)
    for e in list(spec.excludes):
        spec.delete(e)
    for k in list(spec.keys):
        spec.delete(k)
    for s in list(spec.sensors):
        spec.delete(s)
    for t in list(spec.tendons):
        spec.delete(t)
    for name in ("apple", "banana", "orange"):
        b = spec.body(name)
        if b is not None:
            spec.delete(b)

    # the SO-101, unprefixed so the module's joint, actuator and body names hold
    so101 = mujoco.MjSpec.from_file(SO101_XML)
    pos, quat = arm_root_pose()
    frame = spec.worldbody.add_frame(pos=pos, quat=quat)
    frame.attach_body(so101.body("base"), "", "")
    # classes, meshes and materials come over with the attach; the SO-101's actuators too
    for name in ("timestep",):
        pass

    # the cup: same as scene_cup.xml
    cup = spec.worldbody.add_body(name="cup", pos=[CUP_HOME[0], CUP_HOME[1], TABLE_TOP + 0.04])
    cup.add_freejoint(name="cup")
    cup.add_geom(name="cup", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, 0.04, 0], mass=0.05,
                 rgba=[0.95, 0.95, 0.97, 1], friction=[1.5, 0.01, 0.001], condim=4, solref=[0.01, 1])
    cup.add_geom(name="cup_band", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0205, 0.008, 0], pos=[0, 0, 0.01],
                 rgba=[0.85, 0.1, 0.15, 1], contype=0, conaffinity=0, mass=0)
    return spec


def build_model() -> mujoco.MjModel:
    return build_spec().compile()


if __name__ == "__main__":
    from PIL import Image

    m = build_model()
    d = mujoco.MjData(m)
    rest = [-0.106, -0.838, 0.74, 1.145, 0.0]
    for k, n in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]):
        d.qpos[m.joint(n).qposadr[0]] = rest[k]
        d.ctrl[m.actuator(n).id] = rest[k]
    for _ in range(300):
        mujoco.mj_step(m, d)
    print("cameras:", [m.camera(i).name for i in range(m.ncam)])
    print("arm root", np.round(arm_root_pose()[0], 3), "cup", np.round(d.body("cup").xpos, 3),
          "bowl", np.round(d.body("bowl").xpos, 3), "plate", np.round(d.body("plate").xpos, 3))
    print("tip", np.round(d.site("gripperframe").xpos, 3))
    out = os.environ.get("OUT", "/tmp")
    r = mujoco.Renderer(m, 360, 640)
    for cam in ("top", "wrist_cam"):
        r.update_scene(d, camera=cam)
        Image.fromarray(r.render()).save(os.path.join(out, f"dream_scene_{cam}.png"))
