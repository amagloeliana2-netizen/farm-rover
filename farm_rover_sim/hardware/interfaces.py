"""
Abstract hardware interfaces -- the seam between control logic and hardware.

Everything under control/ talks ONLY to these five interfaces, never to
MuJoCo directly. `hardware/simulated.py` implements them against a MuJoCo
model+data pair; `hardware/real.py` is a skeleton you fill in with real
drivers. Swapping backends is a one-line change in main.py and requires no
edits anywhere in control/.

Conventions used throughout:
  * All positions are metres, world frame, numpy arrays of shape (3,).
  * All joint angles are radians.
  * Quaternions are (w, x, y, z).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# --------------------------------------------------------------- data types --

@dataclass
class Pose2D:
    """Rover pose on the ground plane."""
    x: float
    y: float
    theta: float  # heading, radians


@dataclass
class FruitDetection:
    """A ripe fruit as reported by the perception layer."""
    id: str
    world_pos: np.ndarray       # (3,) estimated fruit centre
    distance: float             # metres from the rover
    ripeness: float = 1.0       # 0..1; the picker ignores anything below a threshold
    stem_dir: np.ndarray = field(  # unit vector from fruit toward its stem attachment
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )


@dataclass
class ArmState:
    positions: np.ndarray       # (5,) measured joint angles
    velocities: np.ndarray      # (5,) measured joint velocities
    at_target: bool             # controller reports it converged on its setpoint


# --------------------------------------------------------------- interfaces --

class DriveInterface(ABC):
    """Four-wheel skid-steer / differential drive base."""

    @abstractmethod
    def set_velocity(self, linear: float, angular: float) -> None:
        """Command body-frame linear (m/s) and angular (rad/s) velocity."""

    @abstractmethod
    def get_odometry(self) -> Pose2D:
        """Best current estimate of rover pose in world frame."""

    @abstractmethod
    def get_wheel_velocities(self) -> np.ndarray:
        """(4,) measured wheel angular velocities [fl, fr, rl, rr], rad/s."""

    @abstractmethod
    def stop(self) -> None:
        """Command zero velocity."""


class CameraInterface(ABC):
    """Forward-facing RGB camera plus fruit perception."""

    @abstractmethod
    def get_rgb_frame(self) -> Optional[np.ndarray]:
        """Latest H x W x 3 uint8 RGB frame, or None if rendering is disabled."""

    @abstractmethod
    def detect_fruit(self) -> list[FruitDetection]:
        """
        Fruit currently visible, nearest first.

        This is the one method whose *implementation* differs most between
        sim and reality. The simulated backend reads privileged body poses
        from mjData as a stand-in for perception. On a real rover this is
        where a detector runs on `get_rgb_frame()` and back-projects each
        detection to world coordinates using camera intrinsics/extrinsics
        plus a depth source. The return type is identical either way, so
        the picking logic above it is unaffected.
        """


class ArmInterface(ABC):
    """5-DOF harvesting arm, joint-space position controlled."""

    NUM_JOINTS = 5
    JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow",
                   "wrist_pitch", "wrist_roll")

    @abstractmethod
    def set_joint_targets(self, targets: np.ndarray) -> None:
        """Command target joint angles, shape (5,)."""

    @abstractmethod
    def get_state(self) -> ArmState:
        """Measured joint positions/velocities and convergence flag."""

    @abstractmethod
    def get_tool_position(self) -> np.ndarray:
        """(3,) world position of the point between the gripper fingers."""

    @abstractmethod
    def get_base_frame(self) -> tuple[np.ndarray, float]:
        """
        (origin[3], yaw) of the arm's shoulder frame in world coordinates.

        The IK solver needs this to convert a world-frame fruit position
        into the arm's own frame. On a real rover it comes from the URDF
        mounting transform composed with the base odometry.
        """


class GripperInterface(ABC):
    """Parallel-jaw gripper sized for fruit."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def get_opening(self) -> float:
        """Current jaw gap in metres."""

    @abstractmethod
    def is_holding(self) -> bool:
        """
        Best-effort 'something is gripped' signal.

        Sim: contact force on the gripper touch site plus a jaw-gap check.
        Real: motor current / force-torque threshold, or a jaw encoder
        stalling short of the fully-closed position.
        """


class StemReleaseInterface(ABC):
    """
    Detaching fruit from its stem.

    Real harvesters do this with a cutter, a thumb-roller, or a twist-and-pull
    motion. The state machine just asks for 'release the fruit I am holding'
    and this interface decides how. In simulation the stem is a MuJoCo
    `connect` equality constraint and releasing it means deactivating that
    constraint; on hardware you would fire a blade solenoid or command the
    wrist twist here instead.
    """

    @abstractmethod
    def release(self, fruit_id: str) -> bool:
        """Detach `fruit_id` from its stem. Returns True on success."""

    @abstractmethod
    def is_attached(self, fruit_id: str) -> bool:
        """Whether the fruit is still joined to the plant."""
