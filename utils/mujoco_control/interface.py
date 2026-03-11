#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# HealTeleopSimulator
#
# Keyboard-based teleoperation for a HEAL robotic arm with a Robotiq 2F-85 gripper
# using the new MuJoCo Python bindings. The gripper stays vertically aligned by
# default, and users can control motion and gripper twist via keyboard input.
#
# Features:
#   - Cartesian motion using arrow keys (X, Y, Z axes) and 'z'/'v' for forward/back.
#   - Gripper twist with 'o'/'p', tilt with 'y'/'u', roll with 'e'/'s' (when unlocked).
#   - Toggle vertical-orientation lock with 'l'.
#   - Open/close gripper with 'm'/'n'.
#
# Controls remain live while the MuJoCo viewer is running.
# ------------------------------------------------------------------------------

import os
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard
from pynput.keyboard import Key


class HealTeleopSimulator:
    """
    Teleoperation class for Heal + Robotiq gripper in MuJoCo.
    """

    def __init__(self):
        # Load and compile the MuJoCo model
        self.model = self._construct_model()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Gripper actuator configuration
        self.GRIPPERS = [
            {"actuator_id": 6, "open_cmd": 0.0, "close_cmd": 255.0},
        ]

        # Teleop state
        self.site_name = "right_center"
        self.vel_scale = 1.0
        self.running = True

        self.vel_cmd = np.zeros(6)
        self.gripper_closed = False
        self.lock_vertical_orientation = True
        self.key_lock = threading.Lock()
        self.active_keys = set()

        # Setup keyboard listeners
        self._setup_key_control()

    def _construct_model(self) -> mujoco.MjModel:
        """
        Build the MjModel by merging the HEAL arm and Robotiq gripper specs,
        adding a manipulable cube, and compiling.
        """
        def find_repo_root(target="robot_descriptions"):
            cwd = os.path.abspath(os.path.dirname(__file__))
            while cwd != os.path.sep:
                if os.path.isdir(os.path.join(cwd, target)):
                    return cwd
                cwd = os.path.dirname(cwd)
            raise FileNotFoundError(f"'{target}' not found in any parent directory.")

        repo_root = find_repo_root()
        desc_dir = os.path.join(repo_root, "robot_descriptions")

        arm_path = os.path.join(desc_dir, "single_arm_heal_effort_actuation_velocity.xml")
        grip_path = os.path.join(desc_dir, "robotiq_2f85_v4", "2f85.xml")

        # Load and merge specs
        arm_spec = mujoco.MjSpec.from_file(arm_path)
        grip_spec = mujoco.MjSpec.from_file(grip_path)

        arm_spec.compiler.inertiafromgeom = True
        arm_spec.attach(grip_spec, prefix="gripper/", site=arm_spec.site("right_center"))

        # Add a small red cube to the world
        cube_size = 0.02
        spawn_z = 0.2
        cube = arm_spec.worldbody.add_body(name="cube", pos=[0.7, 0.0, spawn_z])
        cube.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[cube_size] * 3,
            rgba=[1.0, 0.0, 0.0, 1.0],
            mass=0.1
        )
        cube.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="cube_free")

        return arm_spec.compile()

    def compensate_gravity(self):
        """
        Apply gravity bias compensation to all actuators except the gripper.
        """
        skip_ids = {g["actuator_id"] for g in self.GRIPPERS}
        for aid in range(self.model.nu):
            if aid not in skip_ids:
                joint = self.model.actuator_trnid[aid][0]
                self.data.ctrl[aid] = self.data.qfrc_bias[joint]

    def _setup_key_control(self):
        """
        Launch a background thread to listen for key presses/releases
        and update self.active_keys, gripper state, and orientation lock.
        """
        # Mapping keys to 6D velocity increments
        v_lin = 2.0
        v_twist = 5.0
        self.key_mapping = {
            'z': [ v_lin,  0,     0,    0,    0,    0],  # forward
            'v': [-v_lin,  0,     0,    0,    0,    0],  # backward
            Key.right: [0,  v_lin, 0,    0,    0,    0],  # right
            Key.left:  [0, -v_lin, 0,    0,    0,    0],  # left
            Key.up:    [0,  0,     v_lin,0,    0,    0],  # up
            Key.down:  [0,  0,    -v_lin,0,    0,    0],  # down
            'y': [0, 0, 0,  15,   0,    0],             # tilt CW
            'u': [0, 0, 0, -15,   0,    0],             # tilt CCW
            'e': [0, 0, 0,  0,   15,    0],             # roll CW
            's': [0, 0, 0,  0,  -15,    0],             # roll CCW
            'o': [0, 0, 0,  0,    0,  v_twist],         # twist CW
            'p': [0, 0, 0,  0,    0, -v_twist],         # twist CCW
        }

        def on_press(key):
            with self.key_lock:
                # Track velocity keys
                if (hasattr(key, 'char') and key.char in self.key_mapping) or key in self.key_mapping:
                    self.active_keys.add(key)
                # Gripper and lock toggles
                elif hasattr(key, 'char'):
                    if key.char == 'n':
                        self.gripper_closed = True
                    elif key.char == 'm':
                        self.gripper_closed = False
                    elif key.char == 'l':
                        self.lock_vertical_orientation = not self.lock_vertical_orientation
                        status = 'ON' if self.lock_vertical_orientation else 'OFF'
                        print(f"Orientation lock: {status}")

        def on_release(key):
            with self.key_lock:
                self.active_keys.discard(key)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()

    def _compute_velocity_command(self) -> np.ndarray:
        """
        Sum up all active key mappings into a single 6D velocity command.
        """
        with self.key_lock:
            cmd = np.zeros(6)
            for key in self.active_keys:
                mapping = self.key_mapping.get(key) or self.key_mapping.get(getattr(key, 'char', None))
                if mapping:
                    cmd += np.array(mapping)
            return self.vel_scale * cmd

    def step(self):
        """
        One control cycle:
          1. Apply gripper open/close commands.
          2. Compute the 6×nv Jacobian at the end-effector site.
          3. Form the desired velocity command.
          4. Optionally enforce vertical orientation lock.
          5. Solve a damped least-squares for joint increments.
        """
        # 1) Gripper control
        for g in self.GRIPPERS:
            cmd = g["close_cmd"] if self.gripper_closed else g["open_cmd"]
            self.data.ctrl[g["actuator_id"]] = cmd

        # 2) Jacobian at the site
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.site_name)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        J = np.vstack([jacp, jacr])

        # 3) Desired task-space velocity
        v_cmd = self._compute_velocity_command()

        # 4) Vertical lock adjustment
        if self.lock_vertical_orientation:
            ee_mat = self.data.site_xmat[site_id].reshape(3, 3)
            z_axis = ee_mat[:, 2]
            desired_z = np.array([0, 0, -1.0])
            tilt_err = np.cross(z_axis, desired_z)
            # Remove yaw component
            tilt_err -= desired_z * np.dot(tilt_err, desired_z)
            v_cmd[3:6] = tilt_err + v_cmd[5] * desired_z

        # 5) Map to joint increments via damped LS
        if np.linalg.norm(v_cmd) > 0:
            damp = 1e-4
            JT = J.T
            # (J Jᵀ + λI)⁻¹ v_cmd  → dq
            dq = JT @ np.linalg.inv(J @ JT + damp * np.eye(6)) @ v_cmd
            # Apply only to the first nv actuators
            self.data.ctrl[: self.model.nu] += dq[: self.model.nu]

    def run(self):
        """
        Launch the MuJoCo viewer and run the teleop loop until closed.
        """
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and self.running:
                self.compensate_gravity()
                self.step()
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

if __name__ == "__main__":
    teleop = HealTeleopSimulator()
    teleop.run()
