"""The hackathon demo: talk to the robot. Local Whisper hears you, Dream-1 decides the motion,
Viam runs the robot (machine so101; watch it in the Viam app's Control tab).

    .venv/bin/python demo.py

Press Enter and speak (recording ends when you stop talking), or type a command and press Enter.

    "put the apple in the bowl"      a Dream-1 task on Dream's YAM scene (apple and orange are reliable)
    "clean up the table"             apple, then orange, into the bowl
    "stop"                           freezes the arms at once, even mid-task
    "reset"                          puts the fruit back
    "get me water"                   act 2: the SO-101 brings a cup and hands it over
    "goodbye"                        quit
"""
from __future__ import annotations

import asyncio
import os
import re
import sys

from viam.robot.client import RobotClient

import yam_viam as yv
from robot_tasks import RobotTasks, parse_task, task_text

EARS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "voice-loop", "run-ears.sh")
CLEANUP = [("apple", "bowl"), ("orange", "bowl")]   # the reliable ones (reliability.py)


def has(text, *words):
    return any(re.search(rf"\b{w}\b", text) for w in words)


class Demo:
    def __init__(self, rt: RobotTasks):
        self.rt = rt
        self.abort = asyncio.Event()
        self.busy = None              # the running task, if any
        self.quit = asyncio.Event()
        self.ears = None

    # ---- voice out -------------------------------------------------------------------
    async def say(self, text):
        print(f"robot: {text}", flush=True)
        p = await asyncio.create_subprocess_exec("say", "-r", "190", text)
        await p.wait()

    # ---- voice in ----------------------------------------------------------------------
    async def start_ears(self):
        self.ears = await asyncio.create_subprocess_exec(
            EARS, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        while True:
            line = (await self.ears.stdout.readline()).decode().strip()
            if line == "READY" or not line:
                return line == "READY"

    async def listen(self):
        self.ears.stdin.write(b"listen\n")
        await self.ears.stdin.drain()
        while True:
            line = (await self.ears.stdout.readline()).decode().strip()
            if line == "LISTENING":
                print("  (listening... speak now)", flush=True)
            elif line.startswith("HEARD:"):
                return line[6:].strip()
            elif not line:
                return ""

    # ---- commands ---------------------------------------------------------------------
    async def handle(self, text):
        t = text.lower().strip(" .!?")
        if not t:
            return
        if has(t, "stop", "halt", "freeze", "wait"):
            if self.busy and not self.busy.done():
                self.abort.set()
                await self.say("Stopping.")
            else:
                await self.say("I'm already still.")
            return
        if has(t, "goodbye", "bye", "quit", "exit"):
            self.abort.set()
            await self.say("Goodbye!")
            self.quit.set()
            return
        if self.busy and not self.busy.done():
            await self.say("I'm busy. Say stop first.")
            return
        if has(t, "water", "drink", "thirsty"):
            self.start(self.water())
        elif has(t, "reset", "start over"):
            self.start(self.reset())
        elif has(t, "clean", "tidy", "clear"):
            self.start(self.cleanup())
        elif parse_task(t):
            fruit, target = parse_task(t)
            self.start(self.one_task(fruit, target))
        else:
            await self.say("Sorry, I can put the apple or the orange in the bowl, clean up the table, or get you water.")

    def start(self, coro):
        self.abort.clear()
        self.busy = asyncio.create_task(self.guarded(coro))

    async def guarded(self, coro):
        """A task that fails says so, rather than dying silently."""
        try:
            await coro
        except Exception as e:  # noqa: BLE001
            print(f"  [error: {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}]", flush=True)
            await self.say("Something went wrong. Say reset to start over.")

    async def one_task(self, fruit, target, announce=True):
        if announce:
            hint = " I'll try, the banana is tricky." if fruit == "banana" else ""
            await self.say(f"Putting the {fruit} {'in the bowl' if target == 'bowl' else 'on the plate'}.{hint}")
        r = await self.rt.yam_task(task_text(fruit, target), abort=self.abort, seconds=50, max_calls=15)
        print(f"  [{r.task}: {'done' if r.success else r.reason}, {r.calls} Dream calls, {r.seconds:.0f}s]", flush=True)
        if r.reason == "stopped":
            return r
        await self.say(f"Done. The {fruit} is {'in the bowl' if target == 'bowl' else 'on the plate'}." if r.success
                       else f"Sorry, I couldn't get the {fruit} there this time.")
        return r

    async def cleanup(self):
        await self.say("Cleaning up the table.")
        missed = []
        for fruit, target in CLEANUP:
            if await self.rt.placed(fruit, target):
                continue
            r = await self.one_task(fruit, target, announce=False)
            if r.reason == "stopped":
                return
            if not r.success:
                missed.append(fruit)
        await self.say("The table is clean." if not missed else
                       f"I tidied what I could. The {' and the '.join(missed)} still needs a hand.")

    async def reset(self):
        await self.rt.reset_yam()
        await self.say("Everything is back on the table.")

    async def water(self):
        await self.say("Getting you some water.")
        ok = await self.rt.fetch_water()
        await self.say("Here's your water." if ok else "Sorry, I dropped the cup.")

    # ---- main loop ---------------------------------------------------------------------
    async def run(self):
        print("starting local Whisper...", flush=True)
        voice = await self.start_ears()
        if not voice:
            print("  (Whisper did not start: type commands instead)", flush=True)
        await self.rt.reset_yam()
        await self.say("Hi, I'm ready. Press enter and tell me what to do.")
        loop = asyncio.get_running_loop()
        while not self.quit.is_set():
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            text = line.strip()
            if not text and voice:
                text = await self.listen()
                print(f"you: {text!r}", flush=True)
                if not text:
                    await self.say("I didn't catch that.")
                    continue
            await self.handle(text)
        if self.busy and not self.busy.done():
            self.abort.set()
            await self.busy


async def main():
    yv.load_env()
    robot = await RobotClient.at_address(yv.ADDRESS, RobotClient.Options.with_api_key(
        api_key=os.environ["SO101_API_KEY"], api_key_id=os.environ["SO101_API_KEY_ID"]))
    rt = RobotTasks(robot)
    demo = Demo(rt)
    try:
        await demo.run()
    finally:
        rt.close()
        if demo.ears:
            demo.ears.kill()
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
