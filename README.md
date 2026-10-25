# Helping Hands AI

**Physical AI for Good: Build AI that moves, with Viam** (Civic AI Hackathon, Oct 2026)

A robot helper you simply talk to. Say *"put the apple in the bowl"* or *"could you bring me some
water?"* and the robot arms do it, and tell you out loud when they're done. Built for older adults who
live alone and people with shaky or weak hands, who struggle with everyday reaching and gripping.

**Demo video (2 min):** https://drive.google.com/file/d/1B76BCV6_OEkv3cr5U5IfTMqt1bhvI0ce/view?usp=sharing

```
 voice ──► Gemini Live ──tools──► robot_tasks ──► Dream-1 (how to move) ──► Viam (runs the arms)
 (or local Whisper)                    └────────► Viam motion planner (SO-101 water handover)
```

## What works (in simulation)
- **Talk to it.** Gemini Live holds the conversation and calls robot tools (`put_fruit`, `stop`,
  `reset_table`, `get_water`); an offline fallback uses local Whisper + macOS `say`.
- **Dream-1 moves the arms**, zero-shot, on Dream Labs' own MuJoCo YAM scene, served through Viam.
  Apple → bowl is reliable; orange → bowl works about half the time; banana → plate does not yet.
- **"Stop!" freezes the arms** within a second, mid-task.
- **Water handover:** an SO-101 arm picks up a cup and sets it in a person's hand, planned by Viam's
  motion service.
- **Everything goes through Viam**: a MuJoCo simulation runs inside a Viam module and appears as
  normal Viam arms, grippers and cameras, so you can watch it live in the Viam app.

## Layout
| Path | What |
|---|---|
| `mujoco-sim/module/` | Viam module: MuJoCo worlds as Viam components (`jayakumar:mujoco-so101:*`) |
| `mujoco-sim/gemini_demo.py` | the live demo: Gemini Live voice conversation driving the robot |
| `mujoco-sim/demo.py` | offline voice demo: push-to-talk local Whisper |
| `mujoco-sim/robot_tasks.py` | the robot tasks (Dream-1 fruit tasks, water handover) |
| `mujoco-sim/yam_viam.py` | Dream's YAM example, driven through Viam |
| `mujoco-sim/fetch_water.py` | SO-101 cup fetch and handover with Viam's motion planner |
| `mujoco-sim/dream_sim.py`, `scene_dream.py` | Dream-1 zero-shot on the SO-101 (experimental) |
| `mujoco-sim/reliability.py` | runs each task N times and reports success rates |
| `mujoco-sim/demo_record.py`, `gemini_record.py`, `voices/` | how the demo videos were made |
| `voice-loop/ears.py` | local Whisper helper process |
| `so101-palletizer/` | Viam 101 palletizer tutorial ported to the SO-101 sim |

## Setup (macOS, Apple silicon)
1. A Viam machine with `viam-server` (`viam-server -disable-mdns -config <machine config>`).
2. `cd mujoco-sim && python3.12 -m venv .venv && .venv/bin/pip install mujoco viam-sdk numpy pillow dreamlabs dreamlabs-client google-genai sounddevice`
3. The SO-101 model: `git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie menagerie && git -C menagerie sparse-checkout set robotstudio_so101`
4. Dream's SO-101 URDF (for `dream_sim.py`): copy `dream_ik/assets/so101_right.urdf` from the installed
   `dreamlabs` package into `mujoco-sim/dream/`.
5. Add the local module (`mujoco-sim/module/run.sh`) and its components to the machine's config.
6. Keys go in `.env` at the repo root (never committed): `SO101_API_KEY_ID`, `SO101_API_KEY`,
   `GEMINI_API_KEY`; the Dream Labs key in `~/.dreamlabs/credentials`.

Then: `.venv/bin/python gemini_demo.py --headphones` and talk.

## Next
Run it on the real SO-101 arm, fine-tune Dream-1 on recordings of our arm for more household objects,
and use a camera to find the person's hand for a safer handover.
