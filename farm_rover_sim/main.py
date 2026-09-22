#!/usr/bin/env python3
"""
Autonomous fruit-harvesting farm rover -- entry point.

    python3 main.py                          # headless, print results
    python3 main.py --fruit 6                # stop after six fruit
    python3 main.py --video out/harvest.mp4  # render a chase-camera video
    python3 main.py --video out/wrist.mp4 --camera wrist

Switching to real hardware
--------------------------
Nothing in control/ imports mujoco. To drive a physical rover, implement the
five interfaces in hardware/interfaces.py (skeletons are in hardware/real.py)
and build the harvester against those objects instead of the simulated ones:

    from hardware.real import (RealDrive, RealCamera, RealArm,
                               RealGripper, RealStemCutter)
    from control.harvester import FruitHarvester, HarvestConfig

    harvester = FruitHarvester(
        drive=RealDrive("/dev/ttyUSB0"),
        camera=RealCamera(device_index=0, model_path="fruit_detector.onnx"),
        arm=RealArm(),
        gripper=RealGripper(),
        stem=RealStemCutter(),
        wrist_camera=RealCamera(device_index=1),
        config=HarvestConfig(),
    )
    while not harvester.finished:
        harvester.tick(1.0 / CONTROL_HZ)
        time.sleep(1.0 / CONTROL_HZ)

The state machine, navigation, kinematics and manipulation code are unchanged.
"""

from __future__ import annotations
import argparse
import sys

from sim_runner import FarmSim, SimOptions


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autonomous fruit-harvesting rover simulation")
    p.add_argument("--fruit", type=int, default=20,
                   help="stop after this many fruit are in the crate (default: 20)")
    p.add_argument("--duration", type=float, default=650.0,
                   help="max simulated seconds (default: 650)")
    p.add_argument("--video", type=str, default=None,
                   help="write an MP4 to this path (implies rendering)")
    p.add_argument("--camera", choices=("track", "wrist", "onboard", "free"),
                   default="track", help="viewpoint for video (default: track)")
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=450)
    p.add_argument("--capture-fps", type=int, default=15,
                   help="frames captured per simulated second")
    p.add_argument("--playback-fps", type=int, default=30,
                   help="video frame rate; higher than capture-fps speeds playback up")
    p.add_argument("--quiet", action="store_true", help="suppress state-machine log")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    options = SimOptions(
        duration=args.duration,
        render=args.video is not None,
        width=args.width,
        height=args.height,
        render_fps=args.capture_fps,
        playback_fps=args.playback_fps,
        camera=args.camera,
        video_path=args.video,
        max_fruit=args.fruit,
        verbose=not args.quiet,
    )

    sim = FarmSim(options)
    try:
        stats = sim.run()
    except RuntimeError as exc:
        print(f"\nSIMULATION FAILED: {exc}", file=sys.stderr)
        return 1

    in_crate = sim.fruit_in_crate()
    total = len(sim.fruit_names)

    print("\n" + "=" * 58)
    print(f"  fruit in the field       : {total}")
    print(f"  picked from the stem     : {len(stats.picked)}")
    print(f"  confirmed in the crate   : {len(in_crate)}")
    print(f"  dropped while carrying   : {len(stats.dropped)} {stats.dropped or ''}")
    print(f"  unreachable / failed     : {len(stats.failed)} {stats.failed or ''}")
    print(f"  simulated time           : {sim.data.time:.0f} s")
    warning = sim.check_stability()
    print(f"  solver warnings          : {warning or 'none'}")
    print("=" * 58)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
