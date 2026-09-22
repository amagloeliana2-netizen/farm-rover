"""
Cartesian and joint-space arm control, built on the closed-form IK in
control/kinematics.py.

Three things here are load-bearing, and each fixes a failure seen in testing:

1. **Transit moves go through joint space, not Cartesian space.** A straight
   line between two tool positions says nothing about wrist orientation
   along the way, so the gripper can sweep sideways through a neighbouring
   fruit while merely travelling to its stand-off point. Free-space motion
   is interpolated joint-to-joint; task space is used only for the short
   final approach, where the tool path actually matters.

2. **The redundant degree of freedom is locked for the duration of a move.**
   The arm has one spare DOF, resolved by choosing an approach elevation.
   Re-picking "the flattest feasible elevation" independently on every tick
   makes the solution jump mid-move -- the wrist snaps, the gripper flails,
   and it knocks fruit off the branch. The elevation is solved once at the
   goal and held fixed for the whole motion.

3. **Closed-loop base tracking.** The arm sits on a compliant rover that
   pitches when the arm extends. Re-solving against the *measured* base
   frame each tick removes ~15 mm of error that open-loop IK leaves behind.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from hardware.interfaces import ArmInterface
from control.kinematics import (
    inverse_kinematics, solve_with_fallback, world_to_arm_frame,
    MAX_REACH, JOINT_LIMITS, _default_elevations,
)


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


@dataclass
class Waypoint:
    """A tool-point goal, in either world or arm-base coordinates."""
    position: np.ndarray
    frame: str = "world"       # "world" or "arm"
    duration: float = 1.5
    wrist_roll: float = 0.0


class ArmController:
    def __init__(self, arm: ArmInterface, position_tolerance: float = 0.012,
                 joint_tolerance: float = 0.03):
        self.arm = arm
        self.position_tolerance = position_tolerance
        self.joint_tolerance = joint_tolerance

        self._mode: str | None = None          # "cartesian" | "joint" | None
        self._goal: Waypoint | None = None
        self._start_local: np.ndarray | None = None
        self._q_start: np.ndarray | None = None
        self._q_goal: np.ndarray | None = None
        self._elevation: float = 0.0
        self._duration: float = 1.0
        self._elapsed = 0.0
        self._arc: tuple | None = None
        self._failed = False
        self._reason = ""

    # ------------------------------------------------------------- frames --
    def tool_in_arm_frame(self) -> np.ndarray:
        origin, yaw = self.arm.get_base_frame()
        return world_to_arm_frame(self.arm.get_tool_position(), origin, yaw)

    def to_arm_frame(self, position: np.ndarray, frame: str) -> np.ndarray:
        if frame == "arm":
            return np.asarray(position, dtype=float)
        origin, yaw = self.arm.get_base_frame()
        return world_to_arm_frame(position, origin, yaw)

    def goal_in_arm_frame(self) -> np.ndarray:
        if self._goal is None:
            return self.tool_in_arm_frame()
        return self.to_arm_frame(self._goal.position, self._goal.frame)

    # -------------------------------------------------------------- solve --
    def solve_at(self, local: np.ndarray, wrist_roll: float = 0.0,
                 elevation: float | None = None):
        """
        Solve IK at `local`. If `elevation` is given, only that elevation is
        tried, which keeps wrist orientation continuous along a move.
        """
        if elevation is not None:
            res = None
            for elbow_up in (True, False):
                res = inverse_kinematics(local, elevation, elbow_up, wrist_roll)
                if res.ok and res.residual < 1e-6:
                    return res
            return res
        return solve_with_fallback(local, wrist_roll=wrist_roll)

    def plan_elevation(self, local: np.ndarray, wrist_roll: float = 0.0) -> float | None:
        """Flattest elevation that solves at `local`, or None if unreachable."""
        for elev in _default_elevations():
            for elbow_up in (True, False):
                res = inverse_kinematics(local, elev, elbow_up, wrist_roll)
                if res.ok and res.residual < 1e-6:
                    return float(elev)
        return None

    def reachable(self, position: np.ndarray, frame: str = "world",
                  wrist_roll: float = 0.0) -> bool:
        return self.plan_elevation(self.to_arm_frame(position, frame), wrist_roll) is not None

    # -------------------------------------------------------------- moves --
    def begin_cartesian_move(self, wp: Waypoint) -> bool:
        """
        Start an interpolated tool-space move. Returns False if the goal is
        unreachable, in which case nothing is commanded.
        """
        goal_local = self.to_arm_frame(wp.position, wp.frame)
        elev = self.plan_elevation(goal_local, wp.wrist_roll)
        if elev is None:
            self._failed, self._reason = True, "goal unreachable at any elevation"
            self._mode = None
            return False
        self._mode = "cartesian"
        self._goal = wp
        self._elevation = elev
        self._duration = max(wp.duration, 1e-3)
        self._start_local = self.tool_in_arm_frame()
        self._elapsed = 0.0
        self._failed, self._reason = False, ""
        return True

    def begin_transit_to(self, wp: Waypoint) -> bool:
        """
        Move to the configuration that reaches `wp`, interpolating in JOINT
        space. Use for free-space transit, where the tool path does not
        matter but the wrist must not thrash through nearby fruit.
        """
        goal_local = self.to_arm_frame(wp.position, wp.frame)
        elev = self.plan_elevation(goal_local, wp.wrist_roll)
        if elev is None:
            self._failed, self._reason = True, "transit goal unreachable"
            self._mode = None
            return False
        res = self.solve_at(goal_local, wp.wrist_roll, elev)
        if res is None or not res.ok:
            self._failed, self._reason = True, "transit IK failed"
            self._mode = None
            return False
        self._elevation = elev
        return self.begin_joint_move(res.q, wp.duration)

    def begin_arc_move(self, pan_goal: float, radius_goal: float,
                       height_goal: float, duration: float = 2.0,
                       wrist_roll: float = 0.0) -> bool:
        """
        Sweep the tool along an arc at (interpolated) constant radius and
        height, varying the pan angle.

        Carrying fruit from the plant to the crate is close to a pure pan
        rotation. Interpolating that in joint space swings the tool enough to
        fling a friction-held fruit out of the jaws, and a Cartesian straight
        line cuts a chord that dives toward the chassis. An arc keeps the
        tool level, at constant reach, on a predictable path.
        """
        start = self.tool_in_arm_frame()
        pan_start = float(np.arctan2(start[1], start[0]))
        radius_start = float(np.hypot(start[0], start[1]))
        height_start = float(start[2])

        # Unwrap so the sweep takes the short way round and never crosses pi.
        while pan_goal - pan_start > np.pi:
            pan_goal -= 2.0 * np.pi
        while pan_goal - pan_start < -np.pi:
            pan_goal += 2.0 * np.pi

        goal_local = carry_point(pan_goal, radius_goal, height_goal)
        elev = self.plan_elevation(goal_local, wrist_roll)
        if elev is None:
            self._failed, self._reason = True, "arc goal unreachable"
            self._mode = None
            return False

        self._mode = "arc"
        self._goal = Waypoint(goal_local, "arm", duration, wrist_roll)
        self._arc = (pan_start, pan_goal, radius_start, radius_goal,
                     height_start, height_goal)
        self._elevation = elev
        self._duration = max(duration, 1e-3)
        self._elapsed = 0.0
        self._failed, self._reason = False, ""
        return True

    def begin_joint_move(self, q_goal: np.ndarray, duration: float = 1.5) -> bool:
        self._mode = "joint"
        self._goal = None
        self._q_start = self.arm.get_state().positions.copy()
        self._q_goal = np.clip(np.asarray(q_goal, dtype=float),
                               JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        self._duration = max(duration, 1e-3)
        self._elapsed = 0.0
        self._failed, self._reason = False, ""
        return True

    def hold_joints(self, q: np.ndarray) -> None:
        """Command a fixed configuration directly, cancelling any move."""
        self._mode = None
        self._goal = None
        self.arm.set_joint_targets(np.asarray(q, dtype=float))

    # --------------------------------------------------------------- tick --
    def tick(self, dt: float) -> float:
        """Advance the active move and command the arm. Returns error."""
        if self._mode is None:
            return 0.0
        self._elapsed += dt
        alpha = smoothstep(self._elapsed / self._duration)

        if self._mode == "joint":
            q = self._q_start + alpha * (self._q_goal - self._q_start)
            self.arm.set_joint_targets(q)
            return float(np.max(np.abs(self._q_goal - self.arm.get_state().positions)))

        if self._mode == "arc":
            p0, p1, r0, r1, h0, h1 = self._arc
            pan = p0 + alpha * (p1 - p0)
            radius = r0 + alpha * (r1 - r0)
            height = h0 + alpha * (h1 - h0)
            goal_local = carry_point(p1, r1, h1)
            target = carry_point(pan, radius, height)
        else:
            # Cartesian: re-solve against the measured base frame every tick,
            # but at the elevation locked in when the move started.
            goal_local = self.goal_in_arm_frame()
            target = self._start_local + alpha * (goal_local - self._start_local)

        r = float(np.linalg.norm(target))
        if r > MAX_REACH * 0.995:
            target = target * (MAX_REACH * 0.995 / r)

        res = self.solve_at(target, self._goal.wrist_roll, self._elevation)
        if res is None or not res.ok:
            res = self.solve_at(target, self._goal.wrist_roll)
        if res is not None and res.ok:
            self.arm.set_joint_targets(res.q)
            self._failed, self._reason = False, ""
        else:
            self._failed = True
            self._reason = res.reason if res is not None else "no IK solution"

        return float(np.linalg.norm(goal_local - self.tool_in_arm_frame()))

    def move_done(self) -> bool:
        if self._mode is None:
            return True
        if self._elapsed < self._duration:
            return False
        state = self.arm.get_state()
        if self._mode == "joint":
            return bool(np.max(np.abs(self._q_goal - state.positions)) < self.joint_tolerance
                        and np.all(np.abs(state.velocities) < 0.25))
        goal = (carry_point(*[self._arc[1], self._arc[3], self._arc[5]])
                if self._mode == "arc" else self.goal_in_arm_frame())
        err = float(np.linalg.norm(goal - self.tool_in_arm_frame()))
        return bool(err < self.position_tolerance
                    and np.all(np.abs(state.velocities) < 0.25))

    @property
    def ik_ok(self) -> bool:
        return not self._failed

    @property
    def ik_reason(self) -> str:
        return self._reason

    @property
    def elevation(self) -> float:
        return self._elevation


# ------------------------------------------------------------ carry poses --
# The tool is carried at a fixed radius/height in the arm frame so that
# swinging from the plant to the onboard crate is close to a pure pan
# rotation, clearing the deck, the solar panel and the crate rim.
CARRY_RADIUS = 0.40
# Arm-frame z of the carry path; chassis-local z = 0.430.
#
# The fruit does not sit exactly on the tool point -- it settles up to ~30 mm
# low inside the cage -- so the carry height needs margin over the crate rim
# (chassis-local 0.330), not just clearance for the tool itself. At 0.430 the
# worst-case fruit underside rides 38 mm above the rim.
CARRY_HEIGHT = 0.165

# Stow pose: arm raised above the canopy, pan centred on the lane.
#
# Two constraints pin this down.
#
# 1. It must clear the fruit. A stow pose that leaves the gripper reaching
#    forward at fruit height turns the rover into a plough -- it knocks fruit
#    off the branch while driving up to the plant. This puts the tool at
#    world z=0.615, above the tallest fruit at z=0.433.
#
# 2. Pan must be 0, not 180 degrees. A stow behind the shoulder forces the
#    path to the far row through pan = pi, outside the +/-3.05 joint limit,
#    which makes every pick on that side unsolvable.
STOW_POSE = np.array([0.0, -0.5008, -2.3157, 2.1165, 0.0])

# Crate geometry, in the chassis frame (see model/farm_world.xml).
CRATE_CENTRE_X = -0.135
CRATE_RIM_Z = 0.330
CRATE_FLOOR_Z = 0.200
SHOULDER_Z = 0.265
SHOULDER_X = 0.255

# Release just above the crate floor. Dropping from the carry height is a
# 17 cm fall, which bounces fruit back out of the crate.
CRATE_RELEASE_Z = 0.250


def carry_point(pan: float,
                radius: float = CARRY_RADIUS,
                height: float = CARRY_HEIGHT) -> np.ndarray:
    """A tool point at the given pan angle, in arm-base coordinates."""
    return np.array([radius * np.cos(pan), radius * np.sin(pan), height])


def _crate_polar(side_sign: float) -> tuple[float, float]:
    """Pan angle and radius of the crate centre in the arm frame."""
    dx = CRATE_CENTRE_X - SHOULDER_X
    dy = 0.045 * side_sign
    return float(np.arctan2(dy, dx)), float(np.hypot(dx, dy))


def crate_approach_point(side_sign: float) -> np.ndarray:
    """
    Arm-frame point above the crate rim.

    `side_sign` (+1 for the +y row, -1 for the -y row) biases the approach to
    that side so the arm pans the short way round, and keeps the whole swing
    on one side of pan = pi, which would otherwise break the joint limit.
    """
    pan, radius = _crate_polar(side_sign)
    return carry_point(pan, radius=radius, height=CARRY_HEIGHT)


def crate_release_point(side_sign: float) -> np.ndarray:
    """Arm-frame point just above the crate floor, where the jaws open."""
    pan, radius = _crate_polar(side_sign)
    return carry_point(pan, radius=radius, height=CRATE_RELEASE_Z - SHOULDER_Z)
