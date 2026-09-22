"""
Skeleton real-hardware backend.

These classes implement the SAME interfaces as hardware/simulated.py, so
control/navigation.py, control/manipulation.py and control/harvester.py work
unmodified against either backend. Fill in the TODOs with real drivers;
nothing else in the project needs to know it happened.

Swap it in via main.py-style wiring:

    from hardware.real import (RealDrive, RealCamera, RealArm,
                               RealGripper, RealStemCutter)
    drive  = RealDrive(port="/dev/ttyUSB0")
    camera = RealCamera(device_index=0, model_path="fruit_detector.onnx")
    arm    = RealArm()
    gripper = RealGripper()
    stem   = RealStemCutter()
    wrist_camera = RealCamera(device_index=1)   # optional, for the final approach
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from hardware.interfaces import (
    DriveInterface, CameraInterface, ArmInterface, GripperInterface,
    StemReleaseInterface, Pose2D, FruitDetection, ArmState,
)


class RealDrive(DriveInterface):
    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200):
        # TODO: open serial/CAN connection to the motor controllers.
        # TODO: initialize wheel encoder counters and an EKF/complementary
        #       filter fusing encoders + IMU for get_odometry().
        raise NotImplementedError("Wire up your motor controller driver here.")

    def set_velocity(self, linear: float, angular: float) -> None:
        # TODO: convert body-frame (linear, angular) to left/right wheel
        #       speeds using your actual track width and wheel radius, then
        #       send them to the motor controllers.
        raise NotImplementedError

    def get_odometry(self) -> Pose2D:
        # TODO: return fused pose estimate (wheel odometry + IMU, or
        #       GPS/RTK if available outdoors).
        raise NotImplementedError

    def get_wheel_velocities(self) -> np.ndarray:
        # TODO: read back the four wheel encoders, rad/s.
        raise NotImplementedError

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0)


class RealCamera(CameraInterface):
    """
    Used for both the mast (row-scanning) camera and, if fitted, a second
    instance for the wrist camera used during the final approach.
    """

    def __init__(self, device_index: int = 0, model_path: str | None = None,
                 camera_extrinsics: np.ndarray | None = None):
        # TODO: open the physical camera (e.g. cv2.VideoCapture, or a
        #       ROS image topic subscriber).
        # TODO: load a real fruit-detection model (e.g. a small YOLO/ONNX
        #       model fine-tuned on ripe fruit).
        # TODO: store camera_extrinsics (camera pose relative to the rover
        #       base, or relative to the gripper for the wrist camera) and
        #       intrinsics, needed to back-project pixel detections to
        #       world coordinates.
        raise NotImplementedError("Wire up your camera + detector here.")

    def get_rgb_frame(self) -> Optional[np.ndarray]:
        # TODO: grab the latest frame from the physical camera.
        raise NotImplementedError

    def detect_fruit(self) -> list[FruitDetection]:
        # TODO: run the detector on get_rgb_frame(), then back-project each
        #       detection's pixel centroid to a world-frame (x, y, z) using
        #       intrinsics + extrinsics + a depth source (stereo, a depth
        #       camera, or a known approach distance for the wrist camera):
        #         ray = intrinsics_inverse @ [u, v, 1]
        #         world_pos = base_pose * camera_extrinsics * (ray * depth)
        #       Populate `stem_dir` if your detector or a secondary pass can
        #       estimate which way the peduncle runs -- the sim backend
        #       reports it but the harvester does not currently use it.
        raise NotImplementedError


class RealArm(ArmInterface):
    def __init__(self, controller_type: str = "moveit"):
        # TODO: connect to the real arm's joint controller (e.g. MoveIt2,
        #       a Dynamixel/serial servo bus, or a manufacturer SDK).
        raise NotImplementedError("Wire up your arm controller here.")

    def set_joint_targets(self, targets: np.ndarray) -> None:
        # TODO: send joint position (or trajectory) commands, shape (5,).
        raise NotImplementedError

    def get_state(self) -> ArmState:
        # TODO: read back joint encoder positions/velocities and report
        #       whether the controller considers itself at its setpoint.
        raise NotImplementedError

    def get_tool_position(self) -> np.ndarray:
        # TODO: forward-kinematics on measured joint angles (you can reuse
        #       control/kinematics.forward_kinematics if your link lengths
        #       match, or query the controller's own FK/TF).
        raise NotImplementedError

    def get_base_frame(self) -> tuple[np.ndarray, float]:
        # TODO: world position and yaw of the shoulder joint -- the rover's
        #       fused base pose composed with the arm's fixed mounting
        #       transform.
        raise NotImplementedError


class RealGripper(GripperInterface):
    def __init__(self, pwm_channel: int = 0):
        # TODO: initialize the servo/pneumatic driver for the gripper.
        raise NotImplementedError("Wire up your gripper driver here.")

    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def get_opening(self) -> float:
        # TODO: jaw gap in metres, from a servo/encoder position.
        raise NotImplementedError

    def is_holding(self) -> bool:
        # TODO: current-sense, force/torque, or a jaw encoder stalling short
        #       of fully closed -- whatever your gripper hardware exposes.
        raise NotImplementedError


class RealStemCutter(StemReleaseInterface):
    """
    Real fruit release is a physical action (a blade solenoid, a
    thumb-roller, or a wrist-twist-and-pull motion), not a constraint
    toggle. `release()` is where you trigger it.
    """

    def __init__(self, actuator_channel: int = 0):
        # TODO: initialize whatever fires the cutter (solenoid, servo, or
        #       just a flag your RealArm checks to command a twist motion).
        raise NotImplementedError("Wire up your stem-release mechanism here.")

    def release(self, fruit_id: str) -> bool:
        # TODO: fire the cutter / twist motion. `fruit_id` is informational
        #       here (real hardware acts on "whatever is in the gripper
        #       right now", not a named body), useful for logging.
        raise NotImplementedError

    def is_attached(self, fruit_id: str) -> bool:
        # TODO: real hardware generally cannot query this directly; a
        #       reasonable approximation is "return False once release()
        #       has been called for this pick", tracked locally.
        raise NotImplementedError
