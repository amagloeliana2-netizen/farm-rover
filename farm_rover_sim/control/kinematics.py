"""
Closed-form kinematics for the 5-DOF harvesting arm.

The arm is a yaw joint (shoulder_pan, about z) followed by a planar 3R chain
(shoulder_lift, elbow, wrist_pitch -- all about y), followed by a roll joint
(wrist_roll, about the tool axis). Because the tool point sits exactly on the
roll axis, wrist_roll does not move the tool point at all. That leaves a
4-DOF positioning problem with one redundant degree of freedom, which we
resolve by letting the caller specify the approach elevation of the gripper.

This is solved analytically rather than iteratively: no convergence
failures, no singularity blow-ups, and it is exact (verified against MuJoCo
forward kinematics to ~1e-16). The solver is pure numpy and has no MuJoCo
dependency, so it runs unchanged on real hardware.

Joint-angle sign convention matches MuJoCo: lift/elbow/wrist_pitch rotate
about +y, so a positive angle tilts the link downward. The elevation of a
link is therefore the negative of the accumulated joint angle.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Link lengths, metres. These mirror the body offsets in model/farm_world.xml.
L1 = 0.170   # shoulder_lift -> elbow
L2 = 0.150   # elbow -> wrist_pitch
L3 = 0.187   # wrist_pitch -> tool point (grip_site), through wrist_roll
MAX_REACH = L1 + L2 + L3          # 0.507 m
MIN_REACH = 0.06

# Shoulder (lift joint) position in the chassis body frame.
SHOULDER_OFFSET = np.array([0.255, 0.0, 0.265])

JOINT_LIMITS = np.array([
    [-3.05, 3.05],   # shoulder_pan
    [-1.75, 1.75],   # shoulder_lift
    [-2.70, 2.70],   # elbow
    [-2.20, 2.20],   # wrist_pitch
    [-3.05, 3.05],   # wrist_roll
])


@dataclass
class IKResult:
    ok: bool
    q: np.ndarray              # (5,) joint angles
    reason: str = ""
    residual: float = 0.0      # |FK(q) - target|, metres


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Tool point position in the *arm base frame* (origin at the shoulder)."""
    pan, lift, elbow, wpitch = q[0], q[1], q[2], q[3]
    a1 = -lift
    a2 = -(lift + elbow)
    a3 = -(lift + elbow + wpitch)
    rho = L1 * np.cos(a1) + L2 * np.cos(a2) + L3 * np.cos(a3)
    h = L1 * np.sin(a1) + L2 * np.sin(a2) + L3 * np.sin(a3)
    return np.array([rho * np.cos(pan), rho * np.sin(pan), h])


def inverse_kinematics(target_local: np.ndarray,
                       approach_elevation: float = 0.0,
                       elbow_up: bool = True,
                       wrist_roll: float = 0.0) -> IKResult:
    """
    Solve for joint angles placing the tool point at `target_local`
    (arm base frame, origin at the shoulder).

    `approach_elevation` is the elevation angle of the gripper's approach
    axis in radians: 0 means the gripper points horizontally, negative means
    it points downward. This is the redundancy resolution knob -- for
    picking a hanging fruit, a shallow horizontal approach keeps the fingers
    clear of the stem above.
    """
    tx, ty, tz = float(target_local[0]), float(target_local[1]), float(target_local[2])

    pan = np.arctan2(ty, tx)
    rho = float(np.hypot(tx, ty))
    h = tz

    # Back off along the approach axis to find the wrist_pitch joint centre.
    wr = rho - L3 * np.cos(approach_elevation)
    wh = h - L3 * np.sin(approach_elevation)

    dist_sq = wr * wr + wh * wh
    dist = float(np.sqrt(dist_sq))
    if dist > L1 + L2:
        return IKResult(False, np.zeros(5),
                        f"target out of reach (wrist dist {dist:.3f} > {L1 + L2:.3f})")
    if dist < abs(L1 - L2):
        return IKResult(False, np.zeros(5),
                        f"target too close (wrist dist {dist:.3f} < {abs(L1 - L2):.3f})")

    # Planar 2R elbow solution.
    cos_e = (dist_sq - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_e = float(np.clip(cos_e, -1.0, 1.0))
    elbow_planar = np.arccos(cos_e) * (1.0 if elbow_up else -1.0)
    lift_planar = np.arctan2(wh, wr) - np.arctan2(
        L2 * np.sin(elbow_planar), L1 + L2 * np.cos(elbow_planar)
    )

    # Convert planar elevations to MuJoCo joint-angle signs.
    q_lift = -lift_planar
    q_elbow = -elbow_planar
    q_wpitch = -approach_elevation - q_lift - q_elbow

    q = np.array([pan, q_lift, q_elbow, q_wpitch, wrist_roll])

    lo, hi = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    if np.any(q < lo - 1e-9) or np.any(q > hi + 1e-9):
        bad = [ArmJointName[i] for i in range(5) if q[i] < lo[i] - 1e-9 or q[i] > hi[i] + 1e-9]
        return IKResult(False, q, f"joint limits exceeded: {', '.join(bad)}")

    residual = float(np.linalg.norm(forward_kinematics(q) - np.array([tx, ty, tz])))
    return IKResult(True, q, "", residual)


ArmJointName = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_pitch", "wrist_roll")


def _default_elevations() -> tuple[float, ...]:
    """Elevations to try, ordered flattest-first then alternating sign."""
    out = [0.0]
    for step in np.arange(0.1, 1.45, 0.1):
        out.extend([-float(step), float(step)])
    return tuple(out)


def solve_with_fallback(target_local: np.ndarray,
                        elevations: tuple[float, ...] | None = None,
                        wrist_roll: float = 0.0) -> IKResult:
    """
    Try a series of approach elevations and return the first that solves.

    Reaching a fruit is usually possible at several approach angles; this
    picks the flattest feasible one, which keeps the gripper clear of the
    stem and the foliage above the fruit.
    """
    if elevations is None:
        elevations = _default_elevations()
    last = IKResult(False, np.zeros(5), "no elevation attempted")
    for elev in elevations:
        for elbow_up in (True, False):
            res = inverse_kinematics(target_local, elev, elbow_up, wrist_roll)
            if res.ok and res.residual < 1e-6:
                return res
            last = res
    return last


def world_to_arm_frame(world_pos: np.ndarray,
                       base_origin: np.ndarray,
                       base_yaw: float) -> np.ndarray:
    """Rotate a world-frame point into the arm base frame."""
    d = np.asarray(world_pos, dtype=float) - np.asarray(base_origin, dtype=float)
    c, s = np.cos(-base_yaw), np.sin(-base_yaw)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])


def arm_to_world_frame(local_pos: np.ndarray,
                       base_origin: np.ndarray,
                       base_yaw: float) -> np.ndarray:
    """Inverse of `world_to_arm_frame`."""
    c, s = np.cos(base_yaw), np.sin(base_yaw)
    p = np.asarray(local_pos, dtype=float)
    return np.asarray(base_origin, dtype=float) + np.array(
        [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]
    )
