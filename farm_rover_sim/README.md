# Autonomous Fruit-Harvesting Farm Rover

A MuJoCo simulation of a four-wheel rover that drives crop rows, finds fruit
hanging on the plant, and picks it: positions a 5-DOF arm, closes a caging
gripper around the fruit, cuts the stem, and carries it to an onboard
collection crate. Verified result on the bundled farm layout: **20/20 fruit
picked, 20/20 confirmed in the crate, 0 dropped, 0 failed, no solver
warnings** (see "Verifying the numbers" below to reproduce).

## Quick start

```bash
pip install mujoco numpy imageio imageio-ffmpeg

python3 main.py                              # headless, prints a summary
python3 main.py --fruit 6                    # stop after six fruit
python3 main.py --video out/harvest.mp4      # chase-camera video
python3 main.py --video out/wrist.mp4 --camera wrist   # gripper POV
```

The farm layout itself is generated, not hand-authored:

```bash
python3 tools/build_model.py     # (re)writes model/farm_world.xml
```

Edit the constants at the top of `tools/build_model.py` (row spacing, plant
count, fruit height/inset) and re-run it to change the layout.

## Project layout

```
model/farm_world.xml       generated MJCF -- the world in tools/build_model.py, materialized
tools/build_model.py       generates model/farm_world.xml (plants/fruit are repetitive -> code)

hardware/interfaces.py     the five abstract interfaces control/ talks to
hardware/simulated.py      MuJoCo-backed implementations of those interfaces
hardware/real.py           skeleton real-hardware implementations (TODOs)

control/kinematics.py      closed-form forward/inverse kinematics for the arm
control/navigation.py      waypoint navigation for the differential-drive base
control/manipulation.py    Cartesian/joint-space arm motion, carry-arc, crate geometry
control/harvester.py       the state machine tying it all together

sim_runner.py               MuJoCo simulation host (stepping, rendering, video)
main.py                     command-line entry point
```

## Architecture: the sim/hardware seam

Everything in `control/` imports only from `hardware/interfaces.py` — never
`mujoco`. Five abstract interfaces define the boundary:

| Interface | Real-hardware analogue |
|---|---|
| `DriveInterface` | motor controllers + wheel encoders / IMU odometry |
| `CameraInterface` | RGB camera + object-detection model |
| `ArmInterface` | joint controller (e.g. MoveIt2, a servo bus) |
| `GripperInterface` | gripper servo/pneumatics + current or force sensing |
| `StemReleaseInterface` | a cutter or twist mechanism |

`hardware/simulated.py` implements all five against a MuJoCo model+data
pair. `hardware/real.py` has the matching classes with `TODO`-marked stubs
showing exactly what to wire up. To move from simulation to a physical
rover:

```python
from hardware.real import RealDrive, RealCamera, RealArm, RealGripper, RealStemCutter
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
    harvester.tick(dt)
    time.sleep(dt)
```

Nothing in `control/kinematics.py`, `navigation.py`, `manipulation.py`, or
`harvester.py` changes.

## How picking actually works

The arm is a pan joint (shoulder_pan, about z) followed by a planar 3R chain
(shoulder_lift, elbow, wrist_pitch, all about y). The tool point sits exactly
on the wrist_roll axis, so wrist roll never moves it — the arm is a 4-DOF
positioning mechanism with one redundant DOF, solved **in closed form**
(`control/kinematics.py`), verified against MuJoCo's own forward kinematics
to ~1e-16. No iterative solver, no convergence failures, no singularities.

The redundant DOF (the gripper's approach elevation) is resolved once per
move and held fixed for its duration — re-picking it every tick made the
wrist snap between solutions mid-motion and knock fruit off the branch (see
"Bugs found and fixed" below).

Two motion modes:
- **Joint-space transit** for free-space moves, where the tool path doesn't
  matter but the wrist must not thrash through a neighbouring fruit.
- **Cartesian / arc moves** for the final approach and for carrying a
  grasped fruit, where the tool path does matter — an arc keeps a held
  fruit level and at constant reach on the way to the crate.

Grasping is a real physics grasp, not a kinematic shortcut: the gripper has
two three-finger cages, each finger carrying five **spherical** pads
arranged on a shell of the fruit's radius about the tool point. Sphere-
sphere contact normals point along the line of centres, so the closed cage
gets true form closure from five directions rather than relying on friction
between flat pads (see below — this took three attempts to get right).
The stem itself is a MuJoCo `connect` equality constraint between the fruit
and its plant; "cutting" it is `data.eq_active[id] = 0`, matching the
`StemReleaseInterface.release()` call.

## Verifying the numbers

```
python3 main.py --fruit 20 --duration 650 --quiet
```

prints picked / dropped / failed / crate counts and any solver warning.
`sim_runner.FarmSim.check_stability()` inspects MuJoCo's own warning
counters every step; the simulation raises rather than continuing if the
solver ever diverges, because **MuJoCo silently resets the whole simulation
on divergence** and a corrupted run can still print plausible-looking
numbers (see below).

## Bugs found and fixed, in the order they were found

Building this surfaced enough real bugs that they're worth recording, since
several would have produced a plausible-looking but wrong simulation if left
unchecked.

1. **Wheels lying flat.** MuJoCo cylinders default to a z-axis; without an
   explicit `euler`, the wheels were discs resting on their faces and the
   chassis settled 5.5 cm too low. Fixed with `euler="1.5708 0 0"`.

2. **Stow pose wasn't stowed.** The first stow configuration put the tool
   35 cm in front of the shoulder at fruit height, so the rover ploughed
   through every plant while driving up to it, before picking even began. A
   second attempt folded the arm behind the bumper, which then made every
   far-row pick geometrically unreachable: interpolating in Cartesian space
   to a behind-the-shoulder stow forces the path through pan = π, outside
   the ±3.05 rad joint limit. Settled on a stow raised above the canopy at
   pan = 0, which clears the fruit and keeps both rows within a ±90°
   sweep.

3. **The wrist flailed through neighbouring fruit.** The arm's redundant DOF
   (approach elevation) was re-solved independently every control tick, so
   the solution jumped discontinuously mid-move, whipping the gripper
   sideways into fruit it wasn't even targeting. Fixed by locking the
   elevation for the duration of a move, and by routing free-space transit
   through joint-space interpolation rather than a Cartesian straight line
   (which says nothing about wrist orientation along the way).

4. **The solver was diverging and silently corrupting runs.** Contact
   `solref` time constants were only 2-3x the timestep, making contacts
   near-rigid; a finger merely grazing a fruit launched it from rest to
   90 m/s in one step. Because MuJoCo auto-resets the whole simulation on a
   NaN/Inf in qacc, this could restart a run from t=0 mid-harvest while
   still producing plausible-looking statistics. Fixed by halving the
   timestep to 1 ms, relaxing `solref` to 8-10 ms, and adding
   `check_stability()`, which now makes any solver warning a hard failure
   instead of a silent one.

5. **The gripper could not hold a sphere -- three attempts.** Flat parallel
   pads squeeze a smooth fruit out like a watermelon seed: closing force
   measured 9 N -> 0 N as the gap visibly narrowed and the fruit slid free.
   Moving to caging fingers fixed the *lift*, but the fruit then escaped
   *vertically* during the carry, because the pads were still flat boxes
   whose contact normals point sideways -- stacking more of them (a 3x2
   grid) does not add a vertical restoring force. The actual fix was
   **spherical** pads arranged on a shell of the fruit radius about the
   tool point, so sphere-sphere contact normals aim at the fruit centre
   from five directions and give true geometric form closure. This raised
   the full-batch result from 16/20 to 20/20 fruit landing in the crate.

6. **Fruit clipping the crate on the way in.** Even with a solid grasp,
   fruit was arriving at the wrong height: the fruit settles up to 3 cm low
   inside the cage, so clearance for the *tool point* wasn't clearance for
   the *fruit*, and the gripper's open width plus its lateral approach bias
   exceeded the crate's inner half-width, jamming it on the rim. Fixed by
   raising the carry height with margin for the fruit's offset and widening
   the crate.

7. **A wrist camera experiment that looked like it failed but wasn't the
   real fix.** After (5) was diagnosed but partially fixed (2D cage, not yet
   spherical), the last few losses were off-centre grasps from a blind final
   approach, so a wrist-mounted camera and a re-centring step were added.
   A/B test showed it made things *worse* (14/20 vs 13/20 without). The
   conclusion at the time -- that re-centring wasn't the fix -- was correct
   for the cage geometry in place, but incomplete: once the cage was
   switched to spherical pads, the same re-centring step became strictly
   better (20/20 vs 14/20 without, and ~40% faster), because it was never
   competing with the cage, it was waiting for the cage to be worth
   centring on.

## Known limitations

- **Perception is privileged.** `SimulatedCamera.detect_fruit()` reads
  ground-truth body poses from MuJoCo rather than running a detector on
  rendered pixels. `get_rgb_frame()` and the wrist camera both render real
  images (`mujoco.Renderer`), so a real detector could be dropped in and
  tested against them, but none is wired up. This is the one method
  `hardware/real.py` most needs filled in.
- **Ripeness is a stub.** Every fruit reports `ripeness=1.0`; there's no
  visual ripeness model.
- **One fruit per plant per pick pass**, in order along the row -- the
  harvester doesn't revisit a plant later in the same run.
- **The farm is flat and firm.** No terrain, no wheel slip modelling beyond
  MuJoCo's default friction cone, no uneven row spacing.
- Reported timings (`sim 339s wall ~90s` for 20 fruit) are for this
  container's CPU; treat them as relative, not absolute.
