"""Talk to the robot with Gemini Live: a spoken conversation, and Gemini drives the robot by
calling tools. The tools run the same tested tasks as demo.py: Dream-1 on Dream's YAM scene and
the SO-101 water handover, all through Viam on machine so101.

    .venv/bin/python gemini_demo.py                 # mic and speakers (the mic mutes while the robot talks)
    .venv/bin/python gemini_demo.py --headphones    # mic always on, so you can talk over the robot
    .venv/bin/python gemini_demo.py --wav a.wav --wav b.wav   # speak recorded lines instead of the mic (testing)

Needs GEMINI_API_KEY in ../.env. Your voice goes to Google's Gemini Live API.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import time
import wave

import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types
from viam.robot.client import RobotClient

import yam_viam as yv
from robot_tasks import RobotTasks, task_text

MODEL = "gemini-3.8-live"
IN_RATE, OUT_RATE = 16000, 24000
BLOCK = 1600                     # 100 ms of mic audio per message
MUTE_TAIL_S = 0.5                # keep the mic muted this long after the robot stops talking

SYSTEM = """You are Robo, the voice of a helper robot at a kitchen table, for people who find reaching \
or gripping hard. Two robot arms stand at the table with an apple, an orange and a banana, a bowl and a \
plate. In another corner a small arm can fetch a cup of water and hand it to the person.
Act through your tools; never claim you did something a tool did not report.
- put_fruit moves one fruit. The apple and the orange into the bowl work best. The banana is hard: \
say you'll try. Before moving the orange after another fruit is already in the bowl, call reset_table first.
- When a task starts, say exactly one short sentence, then stay quiet while it runs. You will get a "[robot]" message when it finishes: \
tell the person the result in one short sentence.
- stop freezes the arms at once: call it immediately whenever the person says stop, wait, or sounds alarmed.
- One task at a time. If the robot is busy, say so and offer to stop it.
Speak briefly and warmly, like a helpful friend. Keep every reply to one or two short sentences."""

TOOLS = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="put_fruit", description="Move one fruit with the robot arms (takes about 30 seconds).",
        parameters={"type": "object", "required": ["fruit", "target"], "properties": {
            "fruit": {"type": "string", "enum": ["apple", "orange", "banana"]},
            "target": {"type": "string", "enum": ["bowl", "plate"]}}}),
    types.FunctionDeclaration(name="stop", description="Freeze the robot arms immediately."),
    types.FunctionDeclaration(name="reset_table", description="Put all the fruit back where it started."),
    types.FunctionDeclaration(name="get_water", description="The small arm fetches a cup of water and hands it to the person (about 20 seconds)."),
])]


class Speaker:
    """Plays Gemini's 24 kHz audio; knows whether the robot is talking (to mute the mic)."""

    def __init__(self):
        self.buf = bytearray()
        self.last_audio = 0.0
        self.stream = sd.RawOutputStream(samplerate=OUT_RATE, channels=1, dtype="int16", callback=self._cb)
        self.stream.start()

    def _cb(self, outdata, frames, t, status):
        n = frames * 2
        chunk = bytes(self.buf[:n])
        del self.buf[:n]
        if chunk:
            self.last_audio = time.monotonic()
        outdata[:] = chunk + b"\0" * (n - len(chunk))

    def play(self, pcm):
        self.buf.extend(pcm)
        if os.environ.get("GEMINI_SAVE_AUDIO"):
            with open(os.environ["GEMINI_SAVE_AUDIO"], "ab") as f:
                f.write(pcm)

    def clear(self):
        self.buf.clear()

    def talking(self):
        return bool(self.buf) or time.monotonic() - self.last_audio < MUTE_TAIL_S


class GeminiRobot:
    def __init__(self, session, rt: RobotTasks, speaker: Speaker):
        self.s, self.rt, self.speaker = session, rt, speaker
        self.busy: asyncio.Task | None = None
        self.abort = asyncio.Event()
        self.turn_done = asyncio.Event()     # set each time Gemini finishes a turn

    def is_busy(self):
        return self.busy is not None and not self.busy.done()

    async def report(self, text):
        print(f"  [robot] {text}", flush=True)
        await self.s.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=f"[robot] {text}")]),
                                         turn_complete=True)

    async def run_fruit(self, fruit, target):
        r = await self.rt.yam_task(task_text(fruit, target), abort=self.abort, seconds=50, max_calls=15)
        print(f"  [{r.task}: {'done' if r.success else r.reason}, {r.calls} Dream calls, {r.seconds:.0f}s]", flush=True)
        where = "in the bowl" if target == "bowl" else "on the plate"
        if r.reason == "stopped":
            return
        await self.report(f"The {fruit} is now {where}." if r.success else f"Could not get the {fruit} {where} this time.")

    async def run_water(self):
        ok = await self.rt.fetch_water()
        await self.report("The cup of water is in the person's hand." if ok else "The cup was dropped.")

    def start(self, coro):
        self.abort.clear()

        async def guarded():
            try:
                await coro
            except Exception as e:  # noqa: BLE001
                print(f"  [error] {type(e).__name__}: {e}", flush=True)
                await self.report("Something went wrong with that task.")
        self.busy = asyncio.create_task(guarded())

    async def call(self, name, args):
        if name == "stop":
            if self.is_busy():
                self.abort.set()
                return {"result": "stopped: the arms are frozen"}
            return {"result": "nothing was moving"}
        if self.is_busy():
            return {"result": "busy with another task; offer to stop it"}
        if name == "put_fruit":
            self.start(self.run_fruit(args.get("fruit", "apple"), args.get("target", "bowl")))
            return {"result": "started; a [robot] message will say when it is done"}
        if name == "reset_table":
            await self.rt.reset_yam()
            return {"result": "done: all the fruit is back where it started"}
        if name == "get_water":
            self.start(self.run_water())
            return {"result": "started; a [robot] message will say when it is done"}
        return {"result": f"unknown tool {name}"}


async def mic_loop(session, speaker, headphones):
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def cb(indata, frames, t, status):
        loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

    with sd.RawInputStream(samplerate=IN_RATE, channels=1, dtype="int16", blocksize=BLOCK, callback=cb):
        while True:
            block = await q.get()
            if not headphones and speaker.talking():
                block = b"\0" * len(block)          # silence, so the robot does not hear itself
            await session.send_realtime_input(audio=types.Blob(data=block, mime_type=f"audio/pcm;rate={IN_RATE}"))


async def send_line(session, path, manual):
    """One recorded line (16 kHz mono WAV) as mic audio, paced in real time, then the end of the turn."""
    pcm = wave.open(path).readframes(10 ** 9)
    if manual:
        await session.send_realtime_input(activity_start=types.ActivityStart())
    for i in range(0, len(pcm), BLOCK * 2):
        await session.send_realtime_input(audio=types.Blob(data=pcm[i:i + BLOCK * 2], mime_type=f"audio/pcm;rate={IN_RATE}"))
        await asyncio.sleep(BLOCK / IN_RATE)
    if manual:
        await session.send_realtime_input(activity_end=types.ActivityEnd())
    else:
        await session.send_realtime_input(audio_stream_end=True)


async def wav_loop(session, paths, robot: GeminiRobot, speaker, manual=False):
    """Speak recorded lines (16 kHz mono) as if from the mic, waiting for the robot between them.

    A line whose file name starts with "stop" is spoken while the robot is still busy (to stop it).
    """
    async def turn(timeout):
        try:
            await asyncio.wait_for(robot.turn_done.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        robot.turn_done.clear()

    await turn(20)                                   # the greeting
    for path in paths:
        stop_line = os.path.basename(path).startswith("stop")
        while speaker.talking() or (robot.is_busy() and not stop_line):
            await asyncio.sleep(0.2)
        await asyncio.sleep(1.0)
        robot.turn_done.clear()
        print(f"(playing {os.path.basename(path)})", flush=True)
        await send_line(session, path, manual)
        await turn(20)                               # Gemini's answer
        await asyncio.sleep(1.0)
    while robot.is_busy() or speaker.talking():      # the last task, and what Gemini says about it
        await asyncio.sleep(0.5)
    await turn(15)
    while speaker.talking():
        await asyncio.sleep(0.3)


async def receive_loop(session, robot: GeminiRobot, speaker: Speaker):
    try:
        await _receive(session, robot, speaker)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"  [receive stopped] {type(e).__name__}: {e}", flush=True)
        raise


async def _receive(session, robot, speaker):
    heard, said = "", ""
    debug = os.environ.get("GEMINI_DEBUG")
    while True:
        async for msg in session.receive():
            if debug:
                sc = msg.server_content
                print("  [msg]", "tool_call" if msg.tool_call else "",
                      f"in={sc.input_transcription.text!r}" if sc and sc.input_transcription else "",
                      "turn_complete" if sc and sc.turn_complete else "", "model_turn" if sc and sc.model_turn else "",
                      "interrupted" if sc and sc.interrupted else "", "go_away" if msg.go_away else "", flush=True)
            if msg.tool_call:
                for fc in msg.tool_call.function_calls:
                    args = dict(fc.args or {})
                    print(f"  [tool] {fc.name}({', '.join(f'{k}={v}' for k, v in args.items())})", flush=True)
                    result = await robot.call(fc.name, args)
                    await session.send_tool_response(function_responses=[
                        types.FunctionResponse(id=fc.id, name=fc.name, response=result)])
            sc = msg.server_content
            if not sc:
                continue
            if sc.interrupted:
                speaker.clear()
            if sc.input_transcription and sc.input_transcription.text:
                heard += sc.input_transcription.text
            if sc.output_transcription and sc.output_transcription.text:
                said += sc.output_transcription.text
            if sc.model_turn:
                if heard.strip():
                    print(f"you: {heard.strip()}", flush=True)
                    heard = ""
                for p in sc.model_turn.parts:
                    if p.inline_data and p.inline_data.data:
                        speaker.play(p.inline_data.data)
            if sc.turn_complete:
                robot.turn_done.set()
                if heard.strip():
                    print(f"you: {heard.strip()}", flush=True)
                    heard = ""
                said = re.sub(r"<[^>]*>|\{[^}]*\}", "", said).strip()   # <no speech>, {pause} markers
                if said:
                    print(f"robot: {said}", flush=True)
                said = ""


async def main(args):
    yv.load_env()
    robot = await RobotClient.at_address(yv.ADDRESS, RobotClient.Options.with_api_key(
        api_key=os.environ["SO101_API_KEY"], api_key_id=os.environ["SO101_API_KEY_ID"]))
    rt = RobotTasks(robot)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"], tools=TOOLS, system_instruction=SYSTEM,
        input_audio_transcription={}, output_audio_transcription={},
        # its reasoning must never reach the speaker or the transcript
        thinking_config=types.ThinkingConfig(include_thoughts=False),
        # recorded lines mark their own turns (Gemini's voice detection misses some synthetic voices)
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)) if args.manual_turns else None,
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=args.voice))))
    speaker = Speaker()
    try:
        await rt.reset_yam()
        async with client.aio.live.connect(model=args.model, config=config) as session:
            gr = GeminiRobot(session, rt, speaker)
            print(f"connected to {args.model}. Talk to the robot (ctrl-c to quit).", flush=True)
            await session.send_client_content(turns=types.Content(role="user", parts=[types.Part(
                text="[robot] Ready. Greet the person in one short sentence and say what you can do.")]), turn_complete=True)
            recv = asyncio.create_task(receive_loop(session, gr, speaker))
            src = (wav_loop(session, args.wav, gr, speaker, args.manual_turns) if args.wav
                   else mic_loop(session, speaker, args.headphones))
            try:
                await src
            finally:
                recv.cancel()
                if gr.is_busy():
                    gr.abort.set()
                    await gr.busy
    finally:
        rt.close()
        speaker.stream.stop()
        await robot.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headphones", action="store_true", help="never mute the mic (no speaker echo to worry about)")
    ap.add_argument("--wav", action="append", default=[], help="speak these 16 kHz mono recordings instead of the mic")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--voice", default="Puck")
    ap.add_argument("--manual-turns", action="store_true", help="with --wav: mark each line as a turn yourself")
    try:
        asyncio.run(main(ap.parse_args()))
    except KeyboardInterrupt:
        print("bye")
