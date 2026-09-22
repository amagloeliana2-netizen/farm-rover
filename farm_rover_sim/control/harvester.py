"""
Autonomous fruit-harvesting behaviour, as an explicit state machine.

Talks only to the five hardware interfaces, so the same logic runs against
the MuJoCo backend or a real rover. Call `tick(dt)` at a fixed control rate
(50 Hz by default in sim_runner.py).

Cycle per fruit:
    SEEK      drive the lane, watch for fruit in the camera
    ALIGN     position the base so the shoulder is level with the fruit
    APPROACH  move the open jaws to a stand-off point on the lane side
    REACH     close the last few centimetres onto the fruit
    GRASP     close the jaws, confirm grip force
    CUT       release the stem (blade / twist on real hardware)
    LIFT      raise to the carry radius, clear of the branch
    TRANSFER  pan to the onboard crate
    RELEASE   open the jaws, drop the fruit in
    STOW      fold the arm, resume the lane
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np

from hardware.interfaces import (
    DriveInterface, CameraInterface, ArmInterface,
    GripperInterface, StemReleaseInterface, FruitDetection,
)
from control.navigation import LaneNavigator
from control.manipulation import (
    ArmController, Waypoint, carry_point, _crate_polar,
    crate_approach_point, crate_release_point,
    STOW_POSE, CARRY_RADIUS, CARRY_HEIGHT,
)
from control.kinematics import world_to_arm_frame, SHOULDER_OFFSET


class State(Enum):
    INIT = auto()
    SEEK = auto()
    ALIGN = auto()
    APPROACH = auto()
    REACH = auto()
    RECENTER = auto()
    GRASP = auto()
    CUT = auto()
    LIFT = auto()
    TRANSFER = auto()
    DESCEND = auto()
    RELEASE = auto()
    RETRACT = auto()
    STOW = auto()
    DONE = auto()


@dataclass
class HarvestConfig:
    lane_end_x: float = 5.2
    max_fruit: int = 6
    standoff: float = 0.15          # metres back from the fruit, toward the lane
    align_tolerance: float = 0.04
    grasp_settle_time: float = 0.7
    release_time: float = 0.8
    max_grasp_attempts: int = 2
    recenter_tolerance: float = 0.008
    # Arm base sits this far ahead of the chassis origin along body +x.
    shoulder_forward_offset: float = float(SHOULDER_OFFSET[0])


@dataclass
class HarvestStats:
    picked: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


class FruitHarvester:
    def __init__(self,
                 drive: DriveInterface,
                 camera: CameraInterface,
                 arm: ArmInterface,
                 gripper: GripperInterface,
                 stem: StemReleaseInterface,
                 config: HarvestConfig | None = None,
                 wrist_camera: CameraInterface | None = None,
                 logger=print):
        self.drive = drive
        self.camera = camera
        self.arm = arm
        self.gripper = gripper
        self.stem = stem
        self.wrist_camera = wrist_camera
        self.cfg = config or HarvestConfig()
        self.log = logger

        self.nav = LaneNavigator(drive)
        self.arm_ctl = ArmController(arm)

        self.state = State.INIT
        self.stats = HarvestStats()

        self._target: FruitDetection | None = None
        self._target_pos = np.zeros(3)
        self._side = 1.0
        self._carry_pan = 0.0
        self._timer = 0.0
        self._attempts = 0
        self._claimed: set[str] = set()
        self._map: dict[str, FruitDetection] = {}

    # ------------------------------------------------------------- helpers --
    def _enter(self, state: State, msg: str = "") -> None:
        self.state = state
        self._timer = 0.0
        if msg:
            self.log(f"[{state.name}] {msg}")

    def _align_x(self) -> float:
        """Base x that puts the shoulder level with the target fruit."""
        return self._target_pos[0] - self.cfg.shoulder_forward_offset

    def _is_aligned(self) -> bool:
        p = self.drive.get_odometry()
        return (abs(p.x - self._align_x()) < self.cfg.align_tolerance
                and abs(p.y) < 0.08)

    def update_map(self) -> None:
        """
        Fold the current camera detections into a persistent landmark map.

        The camera looks forward with a ~62 degree cone, so once the rover
        pulls level with a plant its fruit fall outside the field of view.
        Without a map the rover would forget every fruit it had just driven
        up to. Accumulating detections is also what a real harvester does --
        it is a landmark map, refreshed whenever a fruit is seen again.
        """
        for f in self.camera.detect_fruit():
            self._map[f.id] = f
        if self.wrist_camera is not None:
            for f in self.wrist_camera.detect_fruit():
                self._map[f.id] = f

    def _select_target(self) -> FruitDetection | None:
        pose = self.drive.get_odometry()
        candidates = [
            f for f in self._map.values()
            if f.id not in self._claimed
            and f.ripeness > 0.5
            and f.world_pos[0] > pose.x - 0.10      # not already behind us
            and f.world_pos[0] < self.cfg.lane_end_x
        ]
        if not candidates:
            return None
        # Work down the lane in order, nearest plant first.
        candidates.sort(key=lambda f: (round(f.world_pos[0], 2), -f.world_pos[2]))
        return candidates[0]

    def _refresh_target(self) -> None:
        """Update the stored fruit position while it remains visible."""
        if self._target is None:
            return
        f = self._map.get(self._target.id)
        if f is not None:
            self._target_pos = f.world_pos.copy()

    def _standoff_point(self) -> np.ndarray:
        """World point `standoff` metres from the fruit, toward the lane centre."""
        p = self._target_pos.copy()
        p[1] -= self._side * self.cfg.standoff
        return p

    def _fruit_pan(self) -> float:
        origin, yaw = self.arm.get_base_frame()
        local = world_to_arm_frame(self._target_pos, origin, yaw)
        return float(np.arctan2(local[1], local[0]))

    # ---------------------------------------------------------------- tick --
    def tick(self, dt: float) -> None:
        self._timer += dt
        self.update_map()
        s = self.state

        if s is State.INIT:
            self.gripper.open()
            self.arm_ctl.hold_joints(STOW_POSE)
            if self._timer > 1.0:
                self._enter(State.SEEK, "scanning the lane for fruit")

        elif s is State.SEEK:
            self.arm_ctl.hold_joints(STOW_POSE)
            if len(self.stats.picked) >= self.cfg.max_fruit:
                self.nav.hold()
                self._enter(State.DONE, f"crate full ({len(self.stats.picked)} fruit)")
                return
            target = self._select_target()
            if target is not None:
                self._target = target
                self._target_pos = target.world_pos.copy()
                self._side = 1.0 if target.world_pos[1] >= 0 else -1.0
                self._attempts = 0
                self._claimed.add(target.id)
                self._enter(State.ALIGN,
                            f"{target.id} at "
                            f"({target.world_pos[0]:.2f}, {target.world_pos[1]:.2f}, "
                            f"{target.world_pos[2]:.2f})")
                return
            pose = self.drive.get_odometry()
            if pose.x >= self.cfg.lane_end_x:
                self.nav.hold()
                self._enter(State.DONE, "reached the end of the lane")
                return
            self.nav.go_to((min(pose.x + 1.0, self.cfg.lane_end_x), 0.0))

        elif s is State.ALIGN:
            self._refresh_target()
            self.arm_ctl.hold_joints(STOW_POSE)
            arrived = self.nav.go_to((self._align_x(), 0.0), final_heading=0.0)
            if (arrived and self._is_aligned()) or self._timer > 25.0:
                self.nav.hold()
                if not self.arm_ctl.begin_transit_to(
                        Waypoint(self._standoff_point(), "world", duration=2.2)):
                    self.stats.failed.append(self._target.id)
                    self._enter(State.STOW,
                                f"unreachable stand-off: {self.arm_ctl.ik_reason}")
                    return
                self._enter(State.APPROACH, "extending arm to stand-off")

        elif s is State.APPROACH:
            self.nav.hold()
            self.gripper.open()
            self.arm_ctl.tick(dt)
            if not self.arm_ctl.ik_ok:
                self.stats.failed.append(self._target.id)
                self._enter(State.STOW, f"unreachable: {self.arm_ctl.ik_reason}")
                return
            if self.arm_ctl.move_done() or self._timer > 7.0:
                self._refresh_target()
                if not self.arm_ctl.begin_cartesian_move(
                        Waypoint(self._target_pos.copy(), "world", duration=1.8)):
                    self.stats.failed.append(self._target.id)
                    self._enter(State.STOW,
                                f"unreachable fruit: {self.arm_ctl.ik_reason}")
                    return
                self._enter(State.REACH, "closing onto the fruit")

        elif s is State.REACH:
            self.nav.hold()
            self.arm_ctl.tick(dt)
            if self.arm_ctl.move_done() or self._timer > 5.0:
                # Re-observe from the wrist before committing to the grasp.
                # The caging jaws only hold a well-centred fruit, and the
                # fruit may have been nudged during the approach.
                self._refresh_target()
                error = float(np.linalg.norm(
                    self._target_pos - self.arm.get_tool_position()))
                if error > self.cfg.recenter_tolerance:
                    if self.arm_ctl.begin_cartesian_move(
                            Waypoint(self._target_pos.copy(), "world", duration=0.8)):
                        self._enter(State.RECENTER,
                                    f"re-centring on fruit ({error * 1000:.0f} mm off)")
                        return
                self._carry_pan = self._fruit_pan()
                self.gripper.close()
                self._enter(State.GRASP, "closing jaws")

        elif s is State.RECENTER:
            self.nav.hold()
            self.arm_ctl.tick(dt)
            if self.arm_ctl.move_done() or self._timer > 2.5:
                self._carry_pan = self._fruit_pan()
                self.gripper.close()
                self._enter(State.GRASP, "closing jaws")

        elif s is State.GRASP:
            self.nav.hold()
            self.arm_ctl.tick(dt)
            if self._timer > self.cfg.grasp_settle_time:
                if self.gripper.is_holding():
                    self._enter(State.CUT, "grip confirmed, cutting stem")
                else:
                    self._attempts += 1
                    if self._attempts >= self.cfg.max_grasp_attempts:
                        self.stats.failed.append(self._target.id)
                        self.gripper.open()
                        self._enter(State.STOW, "grasp failed, skipping")
                    else:
                        self.gripper.open()
                        self.arm_ctl.begin_transit_to(
                            Waypoint(self._standoff_point(), "world", duration=1.4))
                        self._enter(State.APPROACH, "retrying grasp")

        elif s is State.CUT:
            self.nav.hold()
            self.arm_ctl.tick(dt)
            if self._timer > 0.25:
                self.stem.release(self._target.id)
                self.arm_ctl.begin_arc_move(self._carry_pan, CARRY_RADIUS,
                                            CARRY_HEIGHT, duration=2.2)
                self._enter(State.LIFT, "stem cut, lifting clear")

        elif s is State.LIFT:
            self.nav.hold()
            self.arm_ctl.tick(dt)
            if self.arm_ctl.move_done() or self._timer > 5.0:
                if not self.gripper.is_holding():
                    self.stats.dropped.append(self._target.id)
                    self._enter(State.STOW, "fruit slipped during lift")
                    return
                pan, radius = _crate_polar(self._side)
                self.arm_ctl.begin_arc_move(pan, radius, CARRY_HEIGHT, duration=3.2)
                self._enter(State.TRANSFER, "carrying to crate")

        elif s is State.TRANSFER:
            self.nav.hold()
            self.arm_ctl.tick(dt)
            if self.arm_ctl.move_done() or self._timer > 6.0:
                self.arm_ctl.begin_cartesian_move(
                    Waypoint(crate_release_point(self._side), "arm", duration=1.2))
                self._enter(State.DESCEND, "lowering into crate")

        elif s is State.DESCEND:
            self.nav.hold()
            self.arm_ctl.tick(dt)
            if self.arm_ctl.move_done() or self._timer > 3.0:
                self.gripper.open()
                self._enter(State.RELEASE, "releasing into crate")

        elif s is State.RELEASE:
            self.nav.hold()
            self.arm_ctl.tick(dt)
            if self._timer > self.cfg.release_time:
                self.stats.picked.append(self._target.id)
                self.log(f"    harvested {self._target.id} "
                         f"({len(self.stats.picked)} in crate)")
                self.arm_ctl.begin_cartesian_move(
                    Waypoint(crate_approach_point(self._side), "arm", duration=1.2))
                self._enter(State.RETRACT, "lifting clear of the crate")

        elif s is State.RETRACT:
            self.nav.hold()
            self.gripper.open()
            self.arm_ctl.tick(dt)
            if self.arm_ctl.move_done() or self._timer > 3.0:
                self._enter(State.STOW, "stowing arm")

        elif s is State.STOW:
            self.nav.hold()
            self.gripper.open()
            self.arm_ctl.hold_joints(STOW_POSE)
            if self._timer > 1.2:
                self._target = None
                self._enter(State.SEEK, "resuming lane scan")

        elif s is State.DONE:
            self.nav.hold()
            self.arm_ctl.hold_joints(STOW_POSE)

    @property
    def finished(self) -> bool:
        return self.state is State.DONE
