'''
simulation_manager.py

This module provides the `SimulationManager` class, which serves as the high-level
orchestration layer for loading, configuring, and running MuJoCo simulations in the
`mujoco_control` framework (formerly `irl_control`).

Key responsibilities:
 1. **Load YAML configurations** specifying devices, robots, and controller gains.
 2. **Instantiate** the MuJoCo model and simulation via `load_model_from_path` and `MjSim`.
 3. **Create** `DeviceInterface` (formerly `Device`) instances and wrap them into
    `Robot` objects, organizing all hardware/software abstractions.
 4. **Provide utility methods** to:
    - Sleep the simulation thread (`sleep_for`).
    - Retrieve a named `Robot` object (`get_robot`).
    - Fetch controller configuration dictionaries (`get_controller_config`).
    - Directly set free-floating joint positions/quaternions for floating-base robots
      (`set_free_joint_qpos`).

In the overall architecture:
  - `SimulationManager` ties together configuration files, the MuJoCo engine,
    and the `DeviceInterface`/`Robot` classes.
  - Higher-level control loops (e.g. operational-space controllers, RL drivers)
    call into this manager to step the sim, query states, and send torques.
'''

import os
import time
import yaml
import numpy as np
from typing import Dict, Optional
import mujoco
from mujoco import MjModel, MjData
from .interface import DeviceInterface as Device
#from mujoco_control.robot     import Robot

# class SimulationManager:
#     """
#     High-level class for setting up and managing a MuJoCo simulation.

#     Attributes:
#         model (mujoco_py.MjModel): The loaded MuJoCo XML model.
#         sim (mujoco_py.MjSim): The simulation instance for stepping and data access.
#         devices (np.ndarray[Device]): Array of DeviceInterface objects.
#         config (dict): Parsed YAML configuration containing devices, robots,
#                        and controller parameters.
#         controller_configs (list): List of controller configuration dicts.
#     """
#     def __init__(
#         self,
#         robot_config_file: str,
#         scene_file: str,
#         use_sim: bool = True
#     ):
#         """
#         Initialize the simulation environment.

#         Args:
#             robot_config_file (str): Filename of the YAML with 'devices' and 'robots'.
#             scene_file (str): Filename of the MuJoCo XML scene to load.
#             use_sim (bool): If True, devices query sim.data directly; otherwise caching.
#         """
#         # Determine the package's root directory
#         main_dir = os.path.dirname(os.path.abspath(__file__))

#         # Build full paths for the scene XML and robot YAML
#         scene_path = os.path.join(main_dir, 'scenes', scene_file)
#         robot_cfg_path = os.path.join(main_dir, 'robot_configs', robot_config_file)

#         # Load the YAML configuration for devices, robots, and controllers
#         with open(robot_cfg_path, 'r') as file:
#             self.config = yaml.safe_load(file)

#         # Load the MuJoCo model and create the simulation
#         self.model = load_model_from_path(scene_path)
#         self.sim = MjSim(self.model)

#         # Instantiate DeviceInterface objects for each entry in config['devices']
#         self.devices = np.array([
#             Device(dev_yaml, self.model, self.sim, use_sim)
#             for dev_yaml in self.config['devices']
#         ])

#         # Group individual devices into Robot objects
#         self._create_robots(self.config.get('robots', []), use_sim)

#         # Store controller gain configurations
#         self.controller_configs = self.config.get('controller_configs', [])

#         # Flag to prevent concurrent sleeps
#         self._sleeping = False

#     def _create_robots(self, robot_entries: list, use_sim: bool):
#         """
#         Wrap subsets of devices into Robot objects based on YAML definitions.

#         Updates `self.devices` to include Robot instances in place of grouped devices.

#         Args:
#             robot_entries (list): Each dict has 'name' and 'device_ids' indices.
#             use_sim (bool): Passed through to Robot constructors.
#         """
#         robots = []  # new list mixing leftover Device and Robot objects
#         used_idxs = set()

#         # Instantiate each Robot from its device indices
#         for entry in robot_entries:
#             idxs = entry['device_ids']
#             used_idxs.update(idxs)
#             sub_devices = self.devices[idxs]
#             robot = Robot(sub_devices, entry['name'], self.sim, use_sim)
#             robots.append(robot)

#         # Append any standalone DeviceInterfaces not part of a Robot
#         for i, dev in enumerate(self.devices):
#             if i not in used_idxs:
#                 robots.append(dev)

#         # Overwrite devices list with unified elements
#         self.devices = np.array(robots, dtype=object)

#     def sleep_for(self, duration: float):
#         """
#         Sleep for the given duration without blocking simulation setup calls.

#         Args:
#             duration (float): Time in seconds to sleep.

#         Raises:
#             AssertionError: If called reentrantly.
#         """
#         assert not self._sleeping, "Sleep already in progress"
#         self._sleeping = True
#         time.sleep(duration)
#         self._sleeping = False

#     def get_robot(self, name: str) -> Optional[Robot]:
#         """
#         Retrieve a Robot by its configured name.

#         Args:
#             name (str): The robot name as defined in the YAML.

#         Returns:
#             Robot or None: Matching Robot instance, or None if not found.
#         """
#         for element in self.devices:
#             if isinstance(element, Robot) and element.name == name:
#                 return element
#         return None

#     def get_controller_config(self, name: str) -> Optional[Dict]:
#         """
#         Fetch the controller configuration dict for a given controller name.

#         Args:
#             name (str): Name of the controller (as in YAML).

#         Returns:
#             dict or None: The config entry, or None if no match.
#         """
#         for cfg in self.controller_configs:
#             if cfg.get('name') == name:
#                 return cfg
#         return None

#     def set_free_joint_qpos(
#         self,
#         joint_name: str,
#         quat: Optional[np.ndarray] = None,
#         pos: Optional[np.ndarray] = None
#     ):
#         """
#         Directly modify the qpos of a free-floating (free) joint in the MuJoCo simulation.

#         Useful for repositioning floating bases without resetting the entire sim.

#         Args:
#             joint_name (str): Name of the joint to modify.
#             quat (np.ndarray[4], optional): New quaternion [w,x,y,z].
#             pos (np.ndarray[3], optional): New position [x,y,z].
#         """
#         # Get joint ID and its qpos address offset
#         jid = self.sim.model.joint_name2id(joint_name)
#         qpos_off = self.sim.model.jnt_qposadr[jid]

#         # Write quaternion into sim.data.qpos
#         if quat is not None:
#             qidxs = np.arange(qpos_off+3, qpos_off+7)
#             self.sim.data.qpos[qidxs] = quat

#         # Write position into sim.data.qpos
#         if pos is not None:
#             pidxs = np.arange(qpos_off, qpos_off+3)
#             self.sim.data.qpos[pidxs] = pos

#         # Must call forward() to propagate changes
#         self.sim.forward()
