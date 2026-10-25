"""How often does each demo task work? Runs every task N times through Viam and prints a table.

    .venv/bin/python reliability.py --trials 3              # Dream's three recommended tasks
    .venv/bin/python reliability.py --trials 2 "put apple on the plate"

Every trial starts from a reset scene. A trial stops as soon as the fruit rests in its target,
so successes are cheap; failures cost up to --max-calls Dream calls.
"""
from __future__ import annotations

import argparse
import asyncio
import os

from viam.robot.client import RobotClient

import yam_viam as yv
from robot_tasks import RobotTasks

DEFAULT_TASKS = ["put apple into the bowl", "put orange into the bowl", "put banana on the plate"]


async def main(tasks, trials, max_calls, seconds):
    yv.load_env()
    robot = await RobotClient.at_address(yv.ADDRESS, RobotClient.Options.with_api_key(
        api_key=os.environ["SO101_API_KEY"], api_key_id=os.environ["SO101_API_KEY_ID"]))
    rt = RobotTasks(robot)
    rows, total_calls = [], 0
    try:
        for task in tasks:
            for i in range(trials):
                await rt.reset_yam()
                r = await rt.yam_task(task, seconds=seconds, max_calls=max_calls)
                total_calls += r.calls
                print(f"{task!r} trial {i + 1}: {'OK ' if r.success else 'MISS'} "
                      f"{r.calls} calls, {r.seconds:.0f}s ({r.reason})", flush=True)
                rows.append((task, r))
    finally:
        rt.close()
        await robot.close()
    print("\nsummary")
    for task in tasks:
        rs = [r for t, r in rows if t == task]
        ok = sum(r.success for r in rs)
        print(f"  {task:28s} {ok}/{len(rs)} ok, avg {sum(r.calls for r in rs) / len(rs):.0f} calls")
    print(f"  total Dream calls: {total_calls}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--max-calls", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=45)
    args = ap.parse_args()
    asyncio.run(main(args.tasks, args.trials, args.max_calls, args.seconds))
