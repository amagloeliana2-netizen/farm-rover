"""
Generates model/farm_world.xml.

The plants and fruit are highly repetitive, so they are emitted
programmatically. Edit the constants below and re-run to change the farm
layout:

    python3 tools/build_model.py
"""

from __future__ import annotations
import math
import os

# ----------------------------------------------------------------- layout ---
ROW_Y = 0.62                 # crop rows at y = +ROW_Y and y = -ROW_Y
PLANT_X = [1.1, 1.9, 2.7, 3.5, 4.3]
FRUIT_HEIGHTS = [0.40, 0.31]  # two fruit per plant, at these world heights
FRUIT_INSET = 0.26           # how far the fruit hangs toward the lane centre
FRUIT_RADIUS = 0.033

# --------------------------------------------------------------- materials ---
HEADER = """<mujoco model="farm_harvest_rover">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001" integrator="implicitfast" gravity="0 0 -9.81"
          cone="pyramidal" solver="Newton" iterations="150" ls_iterations="50"/>
  <size njmax="800" nconmax="300"/>

  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5" specular="0.2 0.2 0.2"/>
    <rgba haze="0.75 0.82 0.88 1"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.02" zfar="60" shadowclip="3"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.45 0.62 0.85" rgb2="0.85 0.91 0.97"
             width="512" height="512"/>
    <texture type="2d" name="soil_tex" builtin="checker" rgb1="0.31 0.225 0.15"
             rgb2="0.295 0.215 0.142" width="512" height="512" mark="random" markrgb="0.35 0.26 0.18"/>
    <material name="soil_mat" texture="soil_tex" texrepeat="260 170" specular="0.05" shininess="0.05"/>

    <texture type="2d" name="metal_tex" builtin="flat" rgb1="0.26 0.30 0.36"
             width="64" height="64"/>
    <material name="body_mat"   rgba="0.20 0.26 0.34 1" specular="0.55" shininess="0.55" reflectance="0.12"/>
    <material name="accent_mat" rgba="0.85 0.52 0.10 1" specular="0.4"  shininess="0.4"/>
    <material name="panel_mat"  rgba="0.07 0.10 0.18 1" specular="0.9"  shininess="0.9" reflectance="0.35"/>
    <material name="tire_mat"   rgba="0.09 0.09 0.10 1" specular="0.1"  shininess="0.1"/>
    <material name="hub_mat"    rgba="0.62 0.64 0.68 1" specular="0.8"  shininess="0.8" reflectance="0.25"/>
    <material name="arm_mat"    rgba="0.80 0.81 0.83 1" specular="0.7"  shininess="0.7" reflectance="0.2"/>
    <material name="joint_mat"  rgba="0.30 0.33 0.38 1" specular="0.6"  shininess="0.6"/>
    <material name="grip_mat"   rgba="0.15 0.16 0.18 1" specular="0.3"  shininess="0.3"/>
    <material name="pad_mat"    rgba="0.55 0.16 0.14 1" specular="0.15" shininess="0.15"/>
    <material name="crate_mat"  rgba="0.72 0.73 0.75 1" specular="0.35" shininess="0.35"/>
    <material name="lens_mat"   rgba="0.05 0.08 0.14 1" specular="1.0"  shininess="1.0" reflectance="0.5"/>

    <material name="stem_mat"   rgba="0.30 0.42 0.18 1" specular="0.12" shininess="0.2"/>
    <material name="stem_dark"  rgba="0.24 0.33 0.14 1" specular="0.1"  shininess="0.15"/>
    <material name="leaf_a_mat" rgba="0.20 0.44 0.16 1" specular="0.22" shininess="0.35"/>
    <material name="leaf_b_mat" rgba="0.26 0.52 0.20 1" specular="0.22" shininess="0.35"/>
    <material name="fruit_mat"  rgba="0.82 0.14 0.11 1" specular="0.62" shininess="0.75" reflectance="0.12"/>
    <material name="calyx_mat"  rgba="0.28 0.52 0.20 1" specular="0.2"  shininess="0.3"/>
    <material name="mound_mat"  rgba="0.30 0.21 0.14 1" specular="0.05" shininess="0.05"/>
  </asset>

  <default>
    <default class="wheel">
      <joint type="hinge" axis="0 1 0" damping="0.6" armature="0.02"/>
      <geom type="cylinder" size="0.10 0.045" euler="1.5708 0 0" material="tire_mat"
            friction="1.6 0.02 0.001" condim="4" contype="1" conaffinity="1"
            mass="0.9" priority="2" solref="0.010 1"/>
    </default>
    <default class="armlink">
      <joint type="hinge" damping="6.0" armature="0.02" frictionloss="0.2"/>
      <geom material="arm_mat" contype="0" conaffinity="0" density="600"/>
    </default>
    <default class="finger">
      <joint type="slide" damping="12.0" armature="0.005" frictionloss="0.4"/>
      <geom type="box" material="grip_mat" contype="1" conaffinity="1"
            friction="3.0 0.08 0.004" condim="4" priority="3" solref="0.008 1" density="900"/>
    </default>
    <!-- Stems collide only with the chassis (bit 2), never with fruit or wheels. -->
    <default class="stem">
      <geom type="capsule" material="stem_mat" contype="2" conaffinity="2" density="400"/>
    </default>
    <!-- Leaves are decorative only. -->
    <default class="leaf">
      <geom type="ellipsoid" material="leaf_a_mat" contype="0" conaffinity="0" density="100"/>
    </default>
    <default class="fruit">
      <geom type="sphere" material="fruit_mat" contype="1" conaffinity="1"
            friction="1.4 0.03 0.002" condim="4" solref="0.008 1"/>
    </default>
  </default>

  <worldbody>
    <light name="sun" directional="true" pos="2 -3 6" dir="-0.3 0.45 -1"
           diffuse="0.85 0.83 0.78" specular="0.25 0.25 0.25" castshadow="true"/>
    <light name="fill" directional="true" pos="-3 3 5" dir="0.4 -0.4 -1"
           diffuse="0.25 0.27 0.32" specular="0 0 0" castshadow="false"/>

    <geom name="ground" type="plane" size="12 8 0.1" material="soil_mat"
          friction="1.4 0.02 0.001" condim="4" contype="1" conaffinity="3"/>
"""

FOOTER_START = """
    <!-- ==================== ROVER ==================== -->
    <body name="chassis" pos="0 0 0.10">
      <freejoint name="chassis_free"/>
      <inertial pos="0 -0.0 0.08" mass="14.0" diaginertia="0.45 0.60 0.75"/>

      <!-- main hull -->
      <geom name="hull" type="box" size="0.30 0.20 0.055" pos="0 0 0.105"
            material="body_mat" contype="2" conaffinity="3" friction="0.5 0.01 0.001"/>
      <geom name="skid_plate" type="box" size="0.26 0.18 0.012" pos="0 0 0.045"
            material="panel_mat" contype="2" conaffinity="3"/>
      <geom name="deck" type="box" size="0.28 0.19 0.012" pos="0 0 0.172"
            material="body_mat" contype="0" conaffinity="0"/>
      <geom name="solar_panel" type="box" size="0.17 0.165 0.008" pos="-0.06 0 0.190"
            material="panel_mat" contype="0" conaffinity="0"/>
      <geom name="front_bumper" type="box" size="0.02 0.20 0.035" pos="0.315 0 0.085"
            material="accent_mat" contype="2" conaffinity="3"/>
      <geom name="rear_bumper" type="box" size="0.02 0.20 0.035" pos="-0.315 0 0.085"
            material="accent_mat" contype="2" conaffinity="3"/>
      <geom name="rail_l" type="capsule" size="0.012" fromto="-0.26 0.185 0.195 0.24 0.185 0.195"
            material="hub_mat" contype="0" conaffinity="0"/>
      <geom name="rail_r" type="capsule" size="0.012" fromto="-0.26 -0.185 0.195 0.24 -0.185 0.195"
            material="hub_mat" contype="0" conaffinity="0"/>

      <!-- sensor mast + forward RGB camera head -->
      <geom name="mast" type="capsule" size="0.018" fromto="0.20 0 0.185 0.20 0 0.42"
            material="hub_mat" contype="0" conaffinity="0"/>
      <geom name="camera_head" type="box" size="0.035 0.055 0.030" pos="0.215 0 0.455"
            material="body_mat" contype="0" conaffinity="0"/>
      <geom name="camera_lens" type="cylinder" size="0.020 0.008" pos="0.252 0 0.455"
            euler="0 1.5708 0" material="lens_mat" contype="0" conaffinity="0"/>
      <camera name="front_rgb" mode="fixed" pos="0.255 0 0.455" euler="1.4 0 -1.5708" fovy="62"/>
      <site name="camera_site" pos="0.255 0 0.455" size="0.006" rgba="1 0 0 0.3"/>
      <site name="imu_site" pos="0 0 0.105" size="0.005" rgba="0 0 1 0.3"/>

      <!-- onboard harvest crate (the collection container) -->
      <!-- The crate must be wider than the open gripper plus its lateral
           approach bias, or the fingers jam on the rim while descending and
           flick the fruit back out. Gripper half-width 0.083 + bias 0.045
           = 0.128, so the inner half-width is 0.167. -->
      <geom name="crate_floor" type="box" size="0.115 0.175 0.008" pos="-0.135 0 0.192"
            material="crate_mat" contype="1" conaffinity="1" friction="1.2 0.02 0.001"/>
      <geom name="crate_wx_p" type="box" size="0.008 0.175 0.065" pos="-0.020 0 0.265"
            material="crate_mat" contype="1" conaffinity="1"/>
      <geom name="crate_wx_n" type="box" size="0.008 0.175 0.065" pos="-0.250 0 0.265"
            material="crate_mat" contype="1" conaffinity="1"/>
      <geom name="crate_wy_p" type="box" size="0.115 0.008 0.065" pos="-0.135 0.175 0.265"
            material="crate_mat" contype="1" conaffinity="1"/>
      <geom name="crate_wy_n" type="box" size="0.115 0.008 0.065" pos="-0.135 -0.175 0.265"
            material="crate_mat" contype="1" conaffinity="1"/>
      <site name="crate_site" pos="-0.135 0 0.235" size="0.01" rgba="0 1 0 0.3"/>

      <!-- suspension arms (visual) -->
      <geom name="susp_fl" type="capsule" size="0.016" fromto="0.20 0.20 0.06 0.26 0.255 0.0"
            material="joint_mat" contype="0" conaffinity="0"/>
      <geom name="susp_fr" type="capsule" size="0.016" fromto="0.20 -0.20 0.06 0.26 -0.255 0.0"
            material="joint_mat" contype="0" conaffinity="0"/>
      <geom name="susp_rl" type="capsule" size="0.016" fromto="-0.20 0.20 0.06 -0.26 0.255 0.0"
            material="joint_mat" contype="0" conaffinity="0"/>
      <geom name="susp_rr" type="capsule" size="0.016" fromto="-0.20 -0.20 0.06 -0.26 -0.255 0.0"
            material="joint_mat" contype="0" conaffinity="0"/>
"""

WHEELS = ""
for wname, wx, wy in [("fl", 0.26, 0.255), ("fr", 0.26, -0.255),
                      ("rl", -0.26, 0.255), ("rr", -0.26, -0.255)]:
    hub_off = 0.046 if wy > 0 else -0.046
    WHEELS += f"""
      <body name="wheel_{wname}" pos="{wx} {wy} 0">
        <joint name="wheel_{wname}_joint" class="wheel"/>
        <geom class="wheel"/>
        <geom type="cylinder" size="0.045 0.012" pos="0 {hub_off} 0" euler="1.5708 0 0"
              material="hub_mat" contype="0" conaffinity="0" mass="0.05"/>
        <geom type="box" size="0.085 0.012 0.008" pos="0 {hub_off*0.6} 0" material="hub_mat"
              contype="0" conaffinity="0" mass="0.01"/>
        <geom type="box" size="0.012 0.012 0.085" pos="0 {hub_off*0.6} 0" material="hub_mat"
              contype="0" conaffinity="0" mass="0.01"/>
      </body>"""

ARM = """
      <!-- ============ 5-DOF HARVEST ARM (front mounted) ============ -->
      <body name="arm_base" pos="0.255 0 0.185" gravcomp="1">
        <geom type="cylinder" size="0.045 0.020" pos="0 0 0.012" material="joint_mat"
              contype="0" conaffinity="0"/>

        <body name="link_pan" pos="0 0 0.038" gravcomp="1">
          <joint name="shoulder_pan" class="armlink" axis="0 0 1" range="-3.05 3.05"/>
          <geom class="armlink" type="cylinder" size="0.038 0.028" pos="0 0 0.010"
                material="joint_mat"/>

          <body name="link_lift" pos="0 0 0.042" gravcomp="1">
            <joint name="shoulder_lift" class="armlink" axis="0 1 0" range="-1.75 1.75"/>
            <geom class="armlink" type="cylinder" size="0.034 0.042" euler="1.5708 0 0"
                  material="joint_mat"/>
            <geom class="armlink" type="capsule" size="0.028" fromto="0 0 0 0.170 0 0"/>
            <geom class="armlink" type="box" size="0.075 0.016 0.034" pos="0.085 0 0"
                  material="arm_mat"/>

            <body name="link_elbow" pos="0.170 0 0" gravcomp="1">
              <joint name="elbow" class="armlink" axis="0 1 0" range="-2.70 2.70"/>
              <geom class="armlink" type="cylinder" size="0.030 0.036" euler="1.5708 0 0"
                    material="joint_mat"/>
              <geom class="armlink" type="capsule" size="0.023" fromto="0 0 0 0.150 0 0"/>

              <body name="link_wrist_pitch" pos="0.150 0 0" gravcomp="1">
                <joint name="wrist_pitch" class="armlink" axis="0 1 0" range="-2.20 2.20"/>
                <geom class="armlink" type="cylinder" size="0.024 0.028" euler="1.5708 0 0"
                      material="joint_mat"/>
                <geom class="armlink" type="capsule" size="0.019" fromto="0 0 0 0.085 0 0"/>

                <body name="link_wrist_roll" pos="0.085 0 0" gravcomp="1">
                  <joint name="wrist_roll" class="armlink" axis="1 0 0" range="-3.05 3.05"/>
                  <geom class="armlink" type="cylinder" size="0.021 0.022" euler="0 1.5708 0"
                        pos="0.014 0 0" material="joint_mat"/>

                  <!-- ============ GRIPPER ============ -->
                  <!-- Concave three-point fingers. Flat parallel pads cannot
                       hold a sphere: friction alone lets the fruit squeeze
                       out under the inertial load of the carry motion. Each
                       finger carries three pads whose inner faces lie on a
                       circle of the fruit's radius about the tool point, so
                       the closed jaws cage the fruit fore-and-aft instead of
                       merely pinching it. -->
                  <body name="gripper_base" pos="0.040 0 0" gravcomp="1">
                    <geom type="box" size="0.022 0.085 0.020" material="grip_mat"
                          contype="0" conaffinity="0" density="700"/>
                    <!-- Wrist-mounted RGB camera looking down the approach
                         axis. The chassis camera loses the fruit once the
                         rover pulls level with the plant, leaving the last
                         15 cm of the approach blind; this keeps the fruit in
                         view right up to the moment the jaws close. -->
                    <geom type="cylinder" size="0.010 0.006" pos="0.024 0 0.024"
                          euler="0 1.5708 0" material="lens_mat"
                          contype="0" conaffinity="0" density="100"/>
                    <camera name="wrist_rgb" mode="fixed" pos="0.026 0 0.026"
                            euler="1.5708 0 -1.5708" fovy="70"/>
                    <site name="wrist_cam_site" pos="0.026 0 0.026" size="0.005"
                          rgba="1 0 0 0.3"/>
                    <site name="grip_site" pos="0.062 0 0" size="0.008" rgba="1 0.9 0 0.35"/>

                    <body name="finger_left" pos="0.024 0.075 0" gravcomp="1">
                      <joint name="finger_left_joint" class="finger" axis="0 -1 0"
                             range="0 0.045"/>
                      <geom type="box" size="0.034 0.008 0.020" pos="0.038 0 0"
                            material="grip_mat" contype="0" conaffinity="0" density="900"/>
                      <geom class="finger" type="sphere" size="0.0080" pos="0.0380 -0.0090 0.0000" material="pad_mat"/>
                      <geom class="finger" type="sphere" size="0.0080" pos="0.0626 -0.0172 0.0000" material="pad_mat"/>
                      <geom class="finger" type="sphere" size="0.0080" pos="0.0134 -0.0172 0.0000" material="pad_mat"/>
                      <geom class="finger" type="sphere" size="0.0080" pos="0.0380 -0.0172 0.0246" material="pad_mat"/>
                      <geom class="finger" type="sphere" size="0.0080" pos="0.0380 -0.0172 -0.0246" material="pad_mat"/>
                      <site name="touch_left" type="box" size="0.048 0.020 0.032"
                            pos="0.038 -0.018 0" rgba="0 0 0 0"/>
                    </body>
                    <body name="finger_right" pos="0.024 -0.075 0" gravcomp="1">
                      <joint name="finger_right_joint" class="finger" axis="0 1 0"
                             range="0 0.045"/>
                      <geom type="box" size="0.034 0.008 0.020" pos="0.038 0 0"
                            material="grip_mat" contype="0" conaffinity="0" density="900"/>
                      <geom class="finger" type="sphere" size="0.0080" pos="0.0380 0.0090 0.0000" material="pad_mat"/>
                      <geom class="finger" type="sphere" size="0.0080" pos="0.0626 0.0172 0.0000" material="pad_mat"/>
                      <geom class="finger" type="sphere" size="0.0080" pos="0.0134 0.0172 0.0000" material="pad_mat"/>
                      <geom class="finger" type="sphere" size="0.0080" pos="0.0380 0.0172 0.0246" material="pad_mat"/>
                      <geom class="finger" type="sphere" size="0.0080" pos="0.0380 0.0172 -0.0246" material="pad_mat"/>
                      <site name="touch_right" type="box" size="0.048 0.020 0.032"
                            pos="0.038 0.018 0" rgba="0 0 0 0"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
"""


def plant(name: str, px: float, py: float, sign: int) -> str:
    """One bushy plant. `sign` is +1 for the row at +y, -1 for the row at -y.
    Branches that carry fruit lean toward the lane (i.e. toward -sign*y)."""
    s = f'    <body name="{name}" pos="{px} {py} 0">\n'
    s += ('      <geom class="stem" type="ellipsoid" size="0.20 0.20 0.055" pos="0 0 0.02"'
          ' material="mound_mat" contype="0" conaffinity="0"/>\n')
    # trunk, in two tapering sections
    s += '      <geom class="stem" size="0.026 0.020" fromto="0 0 0.03 0.01 0 0.26" material="stem_dark"/>\n'
    s += '      <geom class="stem" size="0.017" fromto="0.01 0 0.26 0.02 0 0.50"/>\n'
    # side branches
    branch_specs = [
        (0.16, 0.22, 0.30, 0.16 * sign),
        (-0.15, 0.26, 0.34, -0.18 * sign),
        (0.06, 0.34, 0.48, 0.20 * sign),
        (-0.10, 0.40, 0.52, -0.14 * sign),
    ]
    for i, (bx, bz0, bz1, by) in enumerate(branch_specs):
        s += (f'      <geom class="stem" size="0.010" '
              f'fromto="0.01 0 {bz0:.3f} {bx:.3f} {by:.3f} {bz1:.3f}"/>\n')
    # fruit-bearing branches, leaning into the lane
    for i, fz in enumerate(FRUIT_HEIGHTS):
        fy = -sign * FRUIT_INSET
        s += (f'      <geom class="stem" size="0.009" '
              f'fromto="0.01 0 {fz + 0.10:.3f} {0.0:.3f} {fy:.3f} {fz + 0.055:.3f}"/>\n')
    # leaves
    leaf_specs = [
        (0.150, 0.045, 0.295, 0.35, 0.95, 0.20), (-0.140, -0.035, 0.335, -0.25, 1.00, -0.45),
        (0.075, 0.140 * sign, 0.465, 0.40, 0.70, 0.90), (-0.100, -0.115 * sign, 0.505, -0.50, 0.80, -0.80),
        (0.035, -0.130 * sign, 0.395, 0.20, 1.10, 0.50), (-0.028, 0.125 * sign, 0.245, -0.30, 0.90, 1.20),
        (0.105, -0.070 * sign, 0.540, 0.10, 0.60, -1.10), (-0.062, 0.090 * sign, 0.565, 0.50, 0.70, 0.30),
        (0.085, 0.085 * sign, 0.360, -0.40, 0.85, 0.65), (-0.080, -0.075 * sign, 0.415, 0.45, 1.05, -0.25),
        (0.045, 0.060 * sign, 0.520, 0.15, 0.75, 1.40), (-0.035, -0.055 * sign, 0.300, -0.20, 1.15, -1.30),
        (0.125, -0.030 * sign, 0.430, 0.30, 0.65, 0.05), (-0.115, 0.040 * sign, 0.470, -0.35, 0.90, 0.95),
    ]
    for i, (lx, ly, lz, ex, ey, ez) in enumerate(leaf_specs):
        mat = "leaf_a_mat" if i % 2 == 0 else "leaf_b_mat"
        w = 0.042 + 0.010 * ((i * 7) % 3)
        s += (f'      <geom class="leaf" size="{w:.3f} {w * 0.62:.3f} 0.004" '
              f'pos="{lx:.3f} {ly:.3f} {lz:.3f}" euler="{ex} {ey} {ez}" material="{mat}"/>\n')
    s += '    </body>\n'
    return s


def fruit(name: str, px: float, py: float, pz: float) -> str:
    return (
        f'    <body name="{name}" pos="{px:.4f} {py:.4f} {pz:.4f}">\n'
        f'      <freejoint name="{name}_free"/>\n'
        f'      <geom class="fruit" size="{FRUIT_RADIUS}" mass="0.120"/>\n'
        f'      <geom type="sphere" size="0.012" pos="0 0 {FRUIT_RADIUS - 0.004:.4f}"\n'
        f'            material="calyx_mat" contype="0" conaffinity="0" mass="0.001"/>\n'
        f'      <geom type="capsule" size="0.005"\n'
        f'            fromto="0 0 {FRUIT_RADIUS:.4f} 0 0 {FRUIT_RADIUS + 0.020:.4f}"\n'
        f'            material="stem_dark" contype="0" conaffinity="0" mass="0.001"/>\n'
        f'      <site name="{name}_site" pos="0 0 0" size="0.004" rgba="1 1 0 0.2"/>\n'
        f'    </body>\n'
    )


def build() -> str:
    body_xml = ""
    fruit_xml = ""
    equality = '  <equality>\n'
    fruit_names: list[str] = []

    for row_i, sign in enumerate([1, -1]):
        row = "n" if sign > 0 else "s"
        for pi, px in enumerate(PLANT_X):
            pname = f"plant_{row}{pi}"
            py = sign * ROW_Y
            body_xml += plant(pname, px, py, sign)
            for fi, fz in enumerate(FRUIT_HEIGHTS):
                fname = f"fruit_{row}{pi}_{fi}"
                fy = py - sign * FRUIT_INSET
                fruit_xml += fruit(fname, px, fy, fz)
                fruit_names.append(fname)
                # The peduncle at the top of the fruit is pinned to the branch
                # tip. Deactivating this equality is what "picking" means.
                equality += (
                    f'    <connect name="stem_{fname}" body1="{fname}" body2="{pname}"\n'
                    f'             anchor="0 0 {FRUIT_RADIUS + 0.020:.4f}" '
                    f'solref="0.05 1" solimp="0.80 0.92 0.01"/>\n'
                )
    equality += '  </equality>\n'

    actuator = """
  <actuator>
    <velocity name="drive_left_front"  joint="wheel_fl_joint" kv="14" ctrlrange="-12 12"/>
    <velocity name="drive_left_rear"   joint="wheel_rl_joint" kv="14" ctrlrange="-12 12"/>
    <velocity name="drive_right_front" joint="wheel_fr_joint" kv="14" ctrlrange="-12 12"/>
    <velocity name="drive_right_rear"  joint="wheel_rr_joint" kv="14" ctrlrange="-12 12"/>

    <position name="act_shoulder_pan"  joint="shoulder_pan"  kp="180" kv="18" ctrlrange="-3.05 3.05"/>
    <position name="act_shoulder_lift" joint="shoulder_lift" kp="180" kv="18" ctrlrange="-1.75 1.75"/>
    <position name="act_elbow"         joint="elbow"         kp="140" kv="14" ctrlrange="-2.70 2.70"/>
    <position name="act_wrist_pitch"   joint="wrist_pitch"   kp="80"  kv="8"  ctrlrange="-2.20 2.20"/>
    <position name="act_wrist_roll"    joint="wrist_roll"    kp="40"  kv="4"  ctrlrange="-3.05 3.05"/>

    <position name="act_finger_left"  joint="finger_left_joint"  kp="300" kv="16" ctrlrange="0 0.045"/>
    <position name="act_finger_right" joint="finger_right_joint" kp="300" kv="16" ctrlrange="0 0.045"/>
  </actuator>
"""

    sensor = """
  <sensor>
    <jointvel name="enc_fl" joint="wheel_fl_joint"/>
    <jointvel name="enc_fr" joint="wheel_fr_joint"/>
    <jointvel name="enc_rl" joint="wheel_rl_joint"/>
    <jointvel name="enc_rr" joint="wheel_rr_joint"/>

    <framepos  name="rover_pos"  objtype="site" objname="imu_site"/>
    <framequat name="rover_quat" objtype="site" objname="imu_site"/>
    <gyro name="imu_gyro" site="imu_site"/>
    <accelerometer name="imu_accel" site="imu_site"/>

    <framepos  name="grip_pos"  objtype="site" objname="grip_site"/>
    <framequat name="grip_quat" objtype="site" objname="grip_site"/>
    <touch name="grip_touch_left"  site="touch_left"/>
    <touch name="grip_touch_right" site="touch_right"/>

    <jointpos name="jp_shoulder_pan"  joint="shoulder_pan"/>
    <jointpos name="jp_shoulder_lift" joint="shoulder_lift"/>
    <jointpos name="jp_elbow"         joint="elbow"/>
    <jointpos name="jp_wrist_pitch"   joint="wrist_pitch"/>
    <jointpos name="jp_wrist_roll"    joint="wrist_roll"/>
    <jointpos name="jp_finger_left"   joint="finger_left_joint"/>
    <jointpos name="jp_finger_right"  joint="finger_right_joint"/>
  </sensor>
"""

    xml = (HEADER
           + "\n    <!-- ==================== CROP ROWS ==================== -->\n"
           + body_xml
           + "\n    <!-- ==================== FRUIT ==================== -->\n"
           + fruit_xml
           + FOOTER_START + WHEELS + ARM
           + equality + actuator + sensor
           + "</mujoco>\n")
    return xml, fruit_names


if __name__ == "__main__":
    xml, names = build()
    out = os.path.join(os.path.dirname(__file__), "..", "model", "farm_world.xml")
    out = os.path.normpath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(xml)
    print(f"wrote {out}  ({len(xml)} chars, {len(names)} fruit)")
