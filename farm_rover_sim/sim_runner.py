"""
MuJoCo simulation host.

Builds the model, instantiates the *simulated* hardware backend, hands those
interfaces to the backend-agnostic FruitHarvester, and steps physics at a
fixed rate with the controller running at a slower control rate.

This module and hardware/simulated.py are the only places `mujoco` is
imported. main.py chooses which backend to build.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np
import mujoco

from hardware.simulated import (
    SimulatedDrive, SimulatedCamera, SimulatedArm,
    SimulatedGripper, SimulatedStemRelease,
)
from control.harvester import FruitHarvester, HarvestConfig
from control.manipulation import STOW_POSE

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "farm_world.xml")


def fruit_body_names(model) -> list[str]:
    """Every body in the model whose name starts with 'fruit_'."""
    names = []
    for i in range(model.nbody):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if n and n.startswith("fruit_"):
            names.append(n)
    return names


@dataclass
class SimOptions:
    duration: float = 120.0
    control_hz: float = 50.0
    render: bool = False
    render_fps: int = 20
    playback_fps: int = 30
    width: int = 1280
    height: int = 720
    camera: str = "track"
    video_path: str | None = None
    max_fruit: int = 6
    verbose: bool = True


class FarmSim:
    def __init__(self, options: SimOptions | None = None,
                 model_path: str = MODEL_PATH):
        self.opt = options or SimOptions()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.fruit_names = fruit_body_names(self.model)

        self.renderer = None
        if self.opt.render:
            self.renderer = mujoco.Renderer(self.model, height=self.opt.height,
                                            width=self.opt.width)

        # ---- simulated hardware backend ----
        self.drive = SimulatedDrive(self.model, self.data)
        self.camera = SimulatedCamera(self.model, self.data, self.fruit_names,
                                      renderer=self.renderer)
        self.wrist_camera = SimulatedCamera(
            self.model, self.data, self.fruit_names,
            fov_deg=70.0, max_range=0.45, renderer=self.renderer,
            site_name="wrist_cam_site", mount_body="gripper_base",
            camera_name="wrist_rgb", planar_bearing=False)
        self.arm = SimulatedArm(self.model, self.data)
        self.gripper = SimulatedGripper(self.model, self.data)
        self.stem = SimulatedStemRelease(self.model, self.data, self.fruit_names)

        # ---- backend-agnostic controller ----
        self.harvester = FruitHarvester(
            self.drive, self.camera, self.arm, self.gripper, self.stem,
            HarvestConfig(max_fruit=self.opt.max_fruit),
            wrist_camera=self.wrist_camera,
            logger=(print if self.opt.verbose else (lambda *a, **k: None)),
        )

        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.arm.set_joint_targets(STOW_POSE)
        self.gripper.open()
        # Let the chassis settle onto its wheels before control starts.
        for _ in range(400):
            mujoco.mj_step(self.model, self.data)

    def fruit_in_crate(self) -> list[str]:
        """Fruit whose position lies inside the onboard crate volume."""
        chassis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        cpos = self.data.xpos[chassis]
        cmat = self.data.xmat[chassis].reshape(3, 3)
        out = []
        for name in self.fruit_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            local = cmat.T @ (self.data.xpos[bid] - cpos)
            if (-0.250 < local[0] < -0.020 and abs(local[1]) < 0.175
                    and 0.195 < local[2] < 0.340):
                out.append(name)
        return out

    def check_stability(self) -> str | None:
        """
        Return a description of any solver warning, else None.

        MuJoCo silently resets the whole simulation when the solver diverges
        (NaN/Inf in qacc). Without this check a run can restart from t=0
        mid-harvest and still report plausible-looking statistics, so any
        warning is treated as a hard failure rather than ignored.
        """
        for i in range(mujoco.mjtWarning.mjNWARNING):
            if self.data.warning[i].number > 0:
                name = mujoco.mjtWarning(i).name
                return f"{name} x{self.data.warning[i].number}"
        return None

    def run(self):
        dt = self.model.opt.timestep
        control_every = max(1, int(round(1.0 / (self.opt.control_hz * dt))))
        control_dt = control_every * dt

        frames = []
        frame_every = None
        if self.opt.render and self.opt.video_path:
            frame_every = max(1, int(round(1.0 / (self.opt.render_fps * dt))))

        n_steps = int(self.opt.duration / dt)
        for i in range(n_steps):
            if i % control_every == 0:
                self.harvester.tick(control_dt)
            mujoco.mj_step(self.model, self.data)

            warning = self.check_stability()
            if warning is not None:
                raise RuntimeError(
                    f"solver diverged at t={self.data.time:.2f}s ({warning}). "
                    "MuJoCo auto-resets on divergence, so the run is void.")

            if frame_every and i % frame_every == 0:
                frames.append(self._render_frame())

            if self.harvester.finished and self.harvester.state.name == "DONE":
                # Let the last motion settle, then stop.
                if self.data.time > 2.0:
                    remaining = int(1.5 / dt)
                    for j in range(remaining):
                        mujoco.mj_step(self.model, self.data)
                        if frame_every and (i + j) % frame_every == 0:
                            frames.append(self._render_frame())
                    break

        if frames and self.opt.video_path:
            self._write_video(frames)
        return self.harvester.stats

    # ------------------------------------------------------------ rendering --
    def _render_frame(self):
        cam = mujoco.MjvCamera()
        if self.opt.camera == "wrist":
            self.renderer.update_scene(self.data, camera="wrist_rgb")
            return self.renderer.render()
        if self.opt.camera == "track":
            chassis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = chassis
            cam.distance = 2.1
            cam.azimuth = 142
            cam.elevation = -17
            cam.lookat[:] = [0.15, 0.0, 0.30]
        elif self.opt.camera == "onboard":
            self.renderer.update_scene(self.data, camera="front_rgb")
            return self.renderer.render()
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = 6.0
            cam.azimuth = 130
            cam.elevation = -25
            cam.lookat[:] = [2.5, 0.0, 0.4]
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render()

    def _write_video(self, frames) -> None:
        import imageio
        os.makedirs(os.path.dirname(self.opt.video_path) or ".", exist_ok=True)
        imageio.mimsave(self.opt.video_path, frames, fps=self.opt.playback_fps,
                        quality=7, macro_block_size=1)
        print(f"wrote {self.opt.video_path} ({len(frames)} frames)")
       ## this is a new comment
        
