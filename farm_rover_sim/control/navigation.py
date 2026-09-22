"""
Lane-following waypoint navigation for the skid-steer base.

Pure control law over DriveInterface -- no MuJoCo, no hardware specifics,
so it runs identically in sim and on the real rover.
"""

from __future__ import annotations
import numpy as np
from hardware.interfaces import DriveInterface, Pose2D


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class LaneNavigator:
    """
    Drives to (x, y) waypoints with a simple heading-then-approach policy.

    Between crop rows the rover follows the lane centre line; `cross_track_gain`
    pulls it back toward y=0 so it does not drift into the plants.
    """

    def __init__(self, drive: DriveInterface,
                 position_tolerance: float = 0.025,
                 max_linear: float = 0.55,
                 max_angular: float = 1.6,
                 heading_gain: float = 2.2,
                 cross_track_gain: float = 1.4,
                 turn_in_place_threshold: float = 0.45,
                 approach_gain: float = 1.2):
        self.drive = drive
        self.position_tolerance = position_tolerance
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.heading_gain = heading_gain
        self.cross_track_gain = cross_track_gain
        self.turn_in_place_threshold = turn_in_place_threshold
        self.approach_gain = approach_gain

    def distance_to(self, target_xy) -> float:
        p = self.drive.get_odometry()
        return float(np.hypot(target_xy[0] - p.x, target_xy[1] - p.y))

    def go_to(self, target_xy, final_heading: float | None = None) -> bool:
        """
        One control tick toward `target_xy`. Returns True once arrived
        (and, if `final_heading` is given, once aligned to it).
        """
        pose: Pose2D = self.drive.get_odometry()
        dx = target_xy[0] - pose.x
        dy = target_xy[1] - pose.y
        distance = float(np.hypot(dx, dy))

        if distance < self.position_tolerance:
            if final_heading is None:
                self.drive.stop()
                return True
            err = wrap_angle(final_heading - pose.theta)
            if abs(err) < 0.05:
                self.drive.stop()
                return True
            self.drive.set_velocity(
                0.0, float(np.clip(self.heading_gain * err,
                                   -self.max_angular, self.max_angular)))
            return False

        desired = np.arctan2(dy, dx)
        heading_err = wrap_angle(desired - pose.theta)

        if abs(heading_err) > self.turn_in_place_threshold:
            # Too far off bearing to make useful forward progress -- rotate first.
            self.drive.set_velocity(
                0.0, float(np.clip(self.heading_gain * heading_err,
                                   -self.max_angular, self.max_angular)))
            return False

        linear = float(np.clip(self.approach_gain * distance, 0.0, self.max_linear))
        angular = self.heading_gain * heading_err - self.cross_track_gain * pose.y
        angular = float(np.clip(angular, -self.max_angular, self.max_angular))
        self.drive.set_velocity(linear, angular)
        return False

    def hold(self) -> None:
        self.drive.stop()
