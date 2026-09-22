"""
Simulated hardware backend, implemented against a MuJoCo model + data pair.

Each class here implements one interface from hardware/interfaces.py. This is
the ONLY module in the project (besides sim_runner.py) that imports mujoco.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import mujoco

from hardware.interfaces import (
    DriveInterface, CameraInterface, ArmInterface, GripperInterface,
    StemReleaseInterface, Pose2D, FruitDetection, ArmState,
)

WHEEL_RADIUS = 0.10
TRACK_WIDTH = 0.51          # lateral distance between left and right wheels
JAW_OPEN = 0.0              # slide joint value for fully open
JAW_CLOSED = 0.045          # slide joint value for fully closed
JAW_GAP_OPEN = 0.116        # metres between the mid pads when fully open


def _quat_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _sensor(model, data, name: str, dim: int) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = model.sensor_adr[sid]
    return data.sensordata[adr:adr + dim]


# ------------------------------------------------------------------- drive --

class SimulatedDrive(DriveInterface):
    def __init__(self, model, data, max_wheel_rate: float = 10.0):
        self.model, self.data = model, data
        self.max_wheel_rate = max_wheel_rate
        self._act = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ("drive_left_front", "drive_left_rear",
                      "drive_right_front", "drive_right_rear")
        ]

    def set_velocity(self, linear: float, angular: float) -> None:
        # Differential-drive kinematics -> per-side wheel angular rates.
        v_left = (linear - angular * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
        v_right = (linear + angular * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
        v_left = float(np.clip(v_left, -self.max_wheel_rate, self.max_wheel_rate))
        v_right = float(np.clip(v_right, -self.max_wheel_rate, self.max_wheel_rate))
        self.data.ctrl[self._act[0]] = v_left
        self.data.ctrl[self._act[1]] = v_left
        self.data.ctrl[self._act[2]] = v_right
        self.data.ctrl[self._act[3]] = v_right

    def get_odometry(self) -> Pose2D:
        pos = _sensor(self.model, self.data, "rover_pos", 3)
        quat = _sensor(self.model, self.data, "rover_quat", 4)
        return Pose2D(float(pos[0]), float(pos[1]), _quat_to_yaw(quat))

    def get_wheel_velocities(self) -> np.ndarray:
        return np.array([
            float(_sensor(self.model, self.data, n, 1)[0])
            for n in ("enc_fl", "enc_fr", "enc_rl", "enc_rr")
        ])

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0)


# ------------------------------------------------------------------ camera --

class SimulatedCamera(CameraInterface):
    """
    Forward RGB camera plus a stand-in perception layer.

    `detect_fruit` reads ground-truth fruit poses from mjData and filters
    them by the camera's field of view and range, rather than running a
    detector on pixels. Replace this one method (see hardware/real.py) to
    move to real perception; everything downstream is unchanged.
    """

    def __init__(self, model, data, fruit_names: list[str],
                 fov_deg: float = 62.0, max_range: float = 2.5,
                 renderer=None, site_name: str = "camera_site",
                 mount_body: str = "chassis", camera_name: str = "front_rgb",
                 planar_bearing: bool = True):
        self.model, self.data = model, data
        self.fruit_names = list(fruit_names)
        self.half_fov = np.radians(fov_deg) / 2.0
        self.max_range = max_range
        self.renderer = renderer
        self.camera_name = camera_name
        # `planar_bearing` compares only the horizontal bearing, which suits a
        # mast camera scanning a row. The wrist camera points along the
        # approach axis, so it needs a true 3-D cone test instead.
        self.planar_bearing = planar_bearing
        self._cam_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self._mount = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mount_body)
        self._fruit_ids = {
            n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in self.fruit_names
        }

    def get_rgb_frame(self) -> Optional[np.ndarray]:
        if self.renderer is None:
            return None
        self.renderer.update_scene(self.data, camera=self.camera_name)
        return self.renderer.render()

    def detect_fruit(self) -> list[FruitDetection]:
        cam_pos = self.data.site_xpos[self._cam_site].copy()
        forward = self.data.xmat[self._mount].reshape(3, 3)[:, 0]  # mount +x

        out: list[FruitDetection] = []
        for name, bid in self._fruit_ids.items():
            fpos = self.data.xpos[bid].copy()
            rel = fpos - cam_pos
            dist = float(np.linalg.norm(rel))
            if dist < 1e-6 or dist > self.max_range:
                continue
            if self.planar_bearing:
                ray = np.array([rel[0], rel[1], 0.0])
            else:
                ray = rel
            n = np.linalg.norm(ray)
            if n < 1e-9:
                continue
            cosang = float(np.dot(ray / n, forward))
            if np.arccos(np.clip(cosang, -1.0, 1.0)) > self.half_fov:
                continue
            out.append(FruitDetection(id=name, world_pos=fpos, distance=dist,
                                      ripeness=1.0,
                                      stem_dir=np.array([0.0, 0.0, 1.0])))
        out.sort(key=lambda f: f.distance)
        return out


# --------------------------------------------------------------------- arm --

class SimulatedArm(ArmInterface):
    def __init__(self, model, data, tolerance: float = 0.02):
        self.model, self.data = model, data
        self.tolerance = tolerance
        self._qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                      for n in self.JOINT_NAMES]
        self._dadr = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                      for n in self.JOINT_NAMES]
        self._act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{n}")
                     for n in self.JOINT_NAMES]
        self._tool_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        self._chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self._targets = np.zeros(5)

    def set_joint_targets(self, targets: np.ndarray) -> None:
        t = np.asarray(targets, dtype=float).reshape(5)
        self._targets = t
        for a, v in zip(self._act, t):
            self.data.ctrl[a] = v

    def get_state(self) -> ArmState:
        pos = np.array([self.data.qpos[a] for a in self._qadr])
        vel = np.array([self.data.qvel[a] for a in self._dadr])
        converged = bool(np.all(np.abs(pos - self._targets) < self.tolerance)
                         and np.all(np.abs(vel) < 0.15))
        return ArmState(pos, vel, converged)

    def get_tool_position(self) -> np.ndarray:
        return self.data.site_xpos[self._tool_site].copy()

    def get_base_frame(self) -> tuple[np.ndarray, float]:
        """Shoulder origin and yaw in world coordinates."""
        from control.kinematics import SHOULDER_OFFSET
        cpos = self.data.xpos[self._chassis].copy()
        cmat = self.data.xmat[self._chassis].reshape(3, 3)
        origin = cpos + cmat @ SHOULDER_OFFSET
        yaw = _quat_to_yaw(self.data.xquat[self._chassis])
        return origin, yaw


# ----------------------------------------------------------------- gripper --

class SimulatedGripper(GripperInterface):
    def __init__(self, model, data):
        self.model, self.data = model, data
        self._act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                     for n in ("act_finger_left", "act_finger_right")]
        self._qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                      for n in ("finger_left_joint", "finger_right_joint")]
        self._commanded_closed = False

    def open(self) -> None:
        for a in self._act:
            self.data.ctrl[a] = JAW_OPEN
        self._commanded_closed = False

    def close(self) -> None:
        for a in self._act:
            self.data.ctrl[a] = JAW_CLOSED
        self._commanded_closed = True

    def get_opening(self) -> float:
        travel = sum(float(self.data.qpos[a]) for a in self._qadr)
        return max(0.0, JAW_GAP_OPEN - travel)

    def get_grip_force(self) -> float:
        """Total normal force reported by both finger pads, newtons."""
        return (float(_sensor(self.model, self.data, "grip_touch_left", 1)[0])
                + float(_sensor(self.model, self.data, "grip_touch_right", 1)[0]))

    def is_holding(self) -> bool:
        if not self._commanded_closed:
            return False
        # An object is held if the jaws stalled short of fully closed AND
        # both pads report contact force. On real hardware this is the same
        # test against motor current and a jaw encoder.
        return self.get_grip_force() > 0.5 and self.get_opening() > 0.015


# ------------------------------------------------------------ stem release --

class SimulatedStemRelease(StemReleaseInterface):
    """
    Stems are MuJoCo `connect` equality constraints named `stem_<fruit_id>`.
    Picking = deactivating that constraint, which is the simulation analogue
    of cutting or snapping the peduncle.
    """

    def __init__(self, model, data, fruit_names: list[str]):
        self.model, self.data = model, data
        self._eq = {}
        for n in fruit_names:
            eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"stem_{n}")
            if eid < 0:
                raise ValueError(f"no stem equality constraint for fruit '{n}'")
            self._eq[n] = eid

    def release(self, fruit_id: str) -> bool:
        eid = self._eq.get(fruit_id)
        if eid is None:
            return False
        self.data.eq_active[eid] = 0
        return True

    def is_attached(self, fruit_id: str) -> bool:
        eid = self._eq.get(fruit_id)
        return eid is not None and bool(self.data.eq_active[eid])

    def reset_all(self) -> None:
        for eid in self._eq.values():
            self.data.eq_active[eid] = 1
