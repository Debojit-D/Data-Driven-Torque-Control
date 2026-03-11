import os
import mujoco
import numpy as np

from typing import Tuple, List, Dict, Optional
from pathlib import Path

class SceneBuilder:
    
    """
    A utility class to construct MuJoCo scenes for simulation involving a single HEAL arm,
    with optional gripper, table, cube, and tray. Also returns gripper configuration.
    """

    def __init__(self, robot: str = "heal" ,include_gripper: bool = None, include_table: bool = False,
                 randomize_cube: bool = False, cube_spawn_bounds: dict = None,
                 include_tray: bool = False, randomize_tray: bool = False, tray_spawn_bounds: dict = None,
                 include_plate: bool = False, randomize_plate: bool = False, plate_spawn_bounds: dict = None,
                 include_cube: bool = False, include_robot:bool=True, model_spec=None, robot_y =0, prefix=None):
        
        """
        Initializes the scene builder.
        """
        self.robot = robot
        self.include_robot = include_robot
        self.model_spec = model_spec
        self.robot_y = robot_y
        self.prefix = prefix
        self.include_gripper = True if include_gripper is None else include_gripper
        self.include_table = True if include_table is None else include_table
        self.include_tray = include_tray
        self.include_cube = include_cube
        self.include_plate = include_plate
        self.randomize_cube = randomize_cube
        self.randomize_tray = randomize_tray
        self.randomize_plate = randomize_plate

        self.table_extents = [0.22, 0.32, 0.025]
        self.table_z = 0.175
        self.table_top_z = self.table_z + self.table_extents[2]
        self.cube_half = 0.02

        self.cube_spawn_bounds = cube_spawn_bounds or {
            "x": [0.38 - self.table_extents[0], 0.38 + self.table_extents[0]],
            "y": [self.robot_y-self.table_extents[1],self.robot_y+ self.table_extents[1]],
            "z": [self.table_top_z + self.cube_half, self.table_top_z + self.cube_half],
        }

        self.tray_extents = [0.05, 0.05, 0.005]
        self.tray_wall_thickness = 0.001
        self.tray_wall_height = 0.025

        self.tray_spawn_bounds = tray_spawn_bounds or {
            "x": [0.38 - 0.15, 0.38 + 0.15],
            "y": [self.robot_y-0.1, self.robot_y+0.1],
            "z": [self.table_top_z + self.tray_extents[2] / 2, self.table_top_z + self.tray_extents[2] / 2],
        }

        self.plate_spawn_bounds = plate_spawn_bounds or {
            "x": [0.38 - 0.15, 0.38 + 0.15],
            "y": [self.robot_y-0.1, self.robot_y+0.1],
            "z": [self.table_top_z, self.table_top_z],
        }

        self.repo_root = self._find_repo_root()
        self.desc_dir = os.path.join(self.repo_root, 'robot_descriptions')

    def _find_repo_root(self, target="robot_descriptions") -> str:
        script_dir = os.path.abspath(os.path.dirname(__file__))
        current = script_dir
        while current != os.path.abspath(os.sep):
            if os.path.isdir(os.path.join(current, target)):
                return current
            current = os.path.abspath(os.path.join(current, '..'))
        raise FileNotFoundError(f"Could not find '{target}' in any parent folder.")

    def build_single_arm_robot_scene(self) -> tuple[mujoco.MjModel, list[dict]]:
        
        if self.include_robot:
            if self.robot == "heal":
                arm_path = os.path.join(self.desc_dir, 'single_arm_heal_effort_actuation_rs_mj.xml')
            elif self.robot == "franka":
                arm_path = os.path.join(self.desc_dir, 'franka','mjx_panda_nohand.xml')
                
            arm = mujoco.MjSpec.from_file(arm_path)
            arm.compiler.inertiafromgeom = True
        else:
            arm = self.model_spec
            arm.compiler.inertiafromgeom = True

        if self.include_gripper:
            
            if self.robot == "heal":
                grip_path = os.path.join(self.desc_dir, 'robotiq_2f85_v4', '2f85.xml')
            elif self.robot == "franka":
                grip_path = os.path.join(self.desc_dir, 'franka', 'hand.xml')

            grip = mujoco.MjSpec.from_file(grip_path)
            arm.attach(grip, prefix='gripper/', site=arm.site('right_center'))

        # Add table
        if self.include_table:
            table = arm.worldbody.add_body(name=f'{self.prefix or ""}table', pos=[0.38, self.robot_y, self.table_z])
            table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=self.table_extents,
                           rgba=[0.5, 0.4, 0.3, 1],
                           contype=1,
                           conaffinity=1)

        # Spawn tray and cube
        tray_pos = None
        if self.include_tray:
            while True:
                tray_pos = self._random_or_fixed_position(self.tray_spawn_bounds)
                cube_pos = self._random_or_fixed_position(self.cube_spawn_bounds)
                if np.linalg.norm(np.array(tray_pos[:2]) - np.array(cube_pos[:2])) > 0.08:
                    break

            # Fix tray Z to be on table
            tray_pos[2] = self.table_top_z + self.tray_extents[2] / 2

            tray = arm.worldbody.add_body(name=f'{self.prefix or ""}tray', pos=tray_pos)
            tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=self.tray_extents,
                          rgba=[0.7, 0.7, 0.7, 1],
                          contype=1,
                          conaffinity=1)

            # Add tray walls
            wall_size_x = [self.tray_wall_thickness, self.tray_extents[1], self.tray_wall_height]
            wall_size_y = [self.tray_extents[0], self.tray_wall_thickness, self.tray_wall_height]

            tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[self.tray_extents[0], 0, self.tray_wall_height],
                          size=wall_size_x, rgba=[0.3, 0.3, 0.3, 1])
            tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-self.tray_extents[0], 0, self.tray_wall_height],
                          size=wall_size_x, rgba=[0.3, 0.3, 0.3, 1])
            tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, self.tray_extents[1], self.tray_wall_height],
                          size=wall_size_y, rgba=[0.3, 0.3, 0.3, 1])
            tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -self.tray_extents[1], self.tray_wall_height],
                          size=wall_size_y, rgba=[0.3, 0.3, 0.3, 1])
            
        elif self.include_plate:
            while True:
                plate_pos = self._random_or_fixed_position(self.plate_spawn_bounds)
                cube_pos = self._random_or_fixed_position(self.cube_spawn_bounds)
                if np.linalg.norm(np.array(plate_pos[:2]) - np.array(cube_pos[:2])) > 0.3:
                    break

            plate_dir = os.path.join(self.desc_dir,'plate_description', 'model.xml')
            
            # Set plate body position
            # plate = arm.worldbody.add_body(name='model', pos=plate_pos)
            plate = mujoco.MjSpec.from_file(plate_dir)
            plate.modelname = f'{self.prefix or ""}plate_body'
            attachment_frame = arm.worldbody.add_frame(pos=plate_pos, name=f'{self.prefix or ""}plate_frame')
            attachment_frame.attach_body(plate.body('plate_body'),prefix=self.prefix)
            
        if self.include_cube:
            cube_pos = self._random_or_fixed_position(self.cube_spawn_bounds)
            # Add cube
            cube = arm.worldbody.add_body(name=f'{self.prefix or ""}cube', pos=cube_pos)
            cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[self.cube_half] * 3,
                        rgba=[1, 0, 0, 1],
                        mass=0.1,
                        friction=[0.1, 0.005, 0.005])
            
            cube.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name=f'{self.prefix or ""}cube_free')

        # Compile model
        if self.include_robot:
            model = arm.compile()

            # Gripper config
            gripper_config = []
            if self.include_gripper:
                for i in range(model.nu):
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    if name and name.startswith("gripper/"):
                        gripper_config.append({
                            "actuator_id": i,
                            "open_cmd": 0.0,
                            "close_cmd": 255.0,
                        })
                    
                
            model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
            model.opt.impratio = 50
            model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
            model.opt.iterations = 50
            model.opt.noslip_iterations = 100

        if self.include_robot:
            return model, gripper_config
        else:
            return arm

    def _random_or_fixed_position(self, bounds):
        return [
            np.random.uniform(*bounds["x"]),
            np.random.uniform(*bounds["y"]),
            np.random.uniform(*bounds["z"]),
        ]

class DualSceneBuilder:
    """
    A utility class to construct MuJoCo scenes with two HEAL arms,
    each optionally fitted with a Robotiq gripper.  Default model/mesh paths
    are baked in, so no external path passing is needed.
    """

    def __init__(self, 
             l_include_gripper: bool = True,
             r_include_gripper: bool = True, 
             robotA: str = "heal", 
             robotB: str = "heal", 
             include_table: list[bool] = [False, False],
             randomize_cube: list[bool] = [False, False], 
             cube_spawn_bounds: list[dict] = [None, None],
             include_tray: list[bool] = [False, False], 
             randomize_tray: list[bool] = [False, False], 
             tray_spawn_bounds: list[dict] = [None, None],
             include_plate: list[bool] = [False, False], 
             randomize_plate: list[bool] = [False, False], 
             plate_spawn_bounds: list[dict] = [None, None],
             include_cube: list[bool] = [False, False]):
        """
        Initializes the dual-scene builder.

        Each config parameter is a list of two elements:
        - First element → Robot A
        - Second element → Robot B
        """

        self.l_include_gripper = l_include_gripper
        self.r_include_gripper = r_include_gripper
        self.robotA = robotA
        self.robotB = robotB

        # Store per-robot configs as lists
        self.include_table = include_table
        self.randomize_cube = randomize_cube
        self.cube_spawn_bounds = cube_spawn_bounds
        self.include_tray = include_tray
        self.randomize_tray = randomize_tray
        self.tray_spawn_bounds = tray_spawn_bounds
        self.include_plate = include_plate
        self.randomize_plate = randomize_plate
        self.plate_spawn_bounds = plate_spawn_bounds
        self.include_cube = include_cube


        # Locate the repository root containing 'robot_descriptions'
        self.repo_root = self._find_repo_root(target="robot_descriptions")
        self.desc_dir  = os.path.join(self.repo_root, 'robot_descriptions')

        # Default filenames for the two HEAL arms
        if self.robotA == "heal":
            self.left_xml        = 'single_arm_heal_effort_actuation_rs_mj.xml'
            self.l_gripper_dir     = 'robotiq_2f85_v4'
            self.l_gripper_xml     = '2f85.xml'
            
        elif self.robotA == "franka":
            self.left_xml        = 'franka/mjx_panda_nohand.xml'
            self.l_gripper_dir     = 'franka'
            self.l_gripper_xml     = 'hand.xml'
            
        if self.robotB == "heal":
            self.right_xml       = 'single_arm_heal_effort_actuation_rs.xml'
            self.r_gripper_dir     = 'robotiq_2f85_v4'
            self.r_gripper_xml     = '2f85.xml'
            
        elif self.robotB == "franka":
            self.right_xml       = 'franka/mjx_panda_nohand_noscene.xml'
            self.r_gripper_dir     = 'franka'
            self.r_gripper_xml     = 'hand.xml'
            
    def _find_repo_root(self, target="robot_descriptions") -> str:
        """
        Walk upward from this file's directory until finding `target`.
        Returns the path to that root directory.
        """
        script_dir = os.path.abspath(os.path.dirname(__file__))
        current    = script_dir
        while current and current != os.path.abspath(os.sep):
            if os.path.isdir(os.path.join(current, target)):
                return current
            current = os.path.abspath(os.path.join(current, '..'))
        raise FileNotFoundError(f"Could not find '{target}' in any parent folder.")

    def build(self) -> mujoco.MjModel:
        """
        Builds and compiles the dual-arm HEAL scene.

        Returns:
            mujoco.MjModel: The compiled dual-arm model with optional grippers.
        """
        # Create base MJCF spec
        root = mujoco.MjSpec()

        # World attachment sites for left and right arms
        left_site  = root.worldbody.add_site(name='l_attach',  pos=[0.0, -0.5, 0.0], group=1)
        right_site = root.worldbody.add_site(name='r_attach', pos=[0.0,0.5, 0.0], group=1)

        # Load and attach left arm
        left_path = os.path.join(self.desc_dir, self.left_xml)
        left_spec = mujoco.MjSpec.from_file(left_path)
        if self.l_include_gripper:
            grip_path = os.path.join(self.desc_dir, self.l_gripper_dir, self.l_gripper_xml)
            grip_spec = mujoco.MjSpec.from_file(grip_path)
            left_spec.attach(grip_spec, prefix='l_gripper/', site=left_spec.site('right_center'))
        left_spec.modelname = 'left_robot'
        root.attach(left_spec, site=left_site, prefix='left/')

        # Load and attach right arm
        right_path = os.path.join(self.desc_dir, self.right_xml)
        right_spec = mujoco.MjSpec.from_file(right_path)
        if self.r_include_gripper:
            grip_path = os.path.join(self.desc_dir, self.r_gripper_dir, self.r_gripper_xml)
            grip_spec = mujoco.MjSpec.from_file(grip_path)
            right_spec.attach(grip_spec, prefix='r_gripper/', site=right_spec.site('right_center'))
        right_spec.modelname = 'right_robot'
        root.attach(right_spec, site=right_site, prefix='right/')
        
        l_build = SceneBuilder(
                include_robot=False,
                model_spec=root,
                robot_y=-0.5,
                include_gripper=False,
                include_table=self.include_table[0],
                randomize_cube=self.randomize_cube[0],
                cube_spawn_bounds=self.cube_spawn_bounds[0],
                include_tray=self.include_tray[0],
                randomize_tray=self.randomize_tray[0],
                tray_spawn_bounds=self.tray_spawn_bounds[0],
                include_plate=self.include_plate[0],
                randomize_plate=self.randomize_plate[0],
                plate_spawn_bounds=self.plate_spawn_bounds[0],
                include_cube=self.include_cube[0],
                prefix="left/"
            )
        root = l_build.build_single_arm_robot_scene()
        
        r_build = SceneBuilder(
                include_robot=False,
                model_spec=root,
                robot_y=0.5,
                include_gripper=False,
                include_table=self.include_table[1],
                randomize_cube=self.randomize_cube[1],
                cube_spawn_bounds=self.cube_spawn_bounds[1],
                include_tray=self.include_tray[1],
                randomize_tray=self.randomize_tray[1],
                tray_spawn_bounds=self.tray_spawn_bounds[1],
                include_plate=self.include_plate[1],
                randomize_plate=self.randomize_plate[1],
                plate_spawn_bounds=self.plate_spawn_bounds[1],
                include_cube=self.include_cube[1],
                prefix="right/"
            )
        root = r_build.build_single_arm_robot_scene()
        
        model = root.compile()
        
        l_gripper_config = []
        r_gripper_config = []
        if self.l_include_gripper:
            for i in range(model.nu):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if name and name.startswith("left/l_gripper/"):
                    l_gripper_config.append({
                        "actuator_id": i,
                        "open_cmd": 0.0,
                        "close_cmd": 255.0,
                    })
        if self.r_include_gripper:            
            for i in range(model.nu):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if name and name.startswith("right/r_gripper/"):
                    r_gripper_config.append({
                        "actuator_id": i,
                        "open_cmd": 0.0,
                        "close_cmd": 255.0,
                    })
        
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.impratio = 50
        model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
        model.opt.iterations = 50
        model.opt.noslip_iterations = 100
        # Compile and return
        return model, l_gripper_config , r_gripper_config
    
class SceneBuilderPeg:
    """
    A utility class to construct MuJoCo scenes involving a single HEAL arm,
    a dual peg assembly, and optionally a gripper, table, and grommet.
    """

    def __init__(self, include_gripper=True, include_table=True,
                 randomize_peg=False, include_grommet=False):
        """
        Initializes the peg scene builder.
        """
        self.include_gripper = include_gripper
        self.include_table = include_table
        self.randomize_peg = randomize_peg
        self.include_grommet = include_grommet

        # Table dimensions and height
        self.table_extents = [0.22, 0.32, 0.025]
        self.table_z = 0.175
        self.table_top_z = self.table_z + self.table_extents[2]

        # Peg and grommet size
        self.peg_half = 0.02
        self.grommet_half = 0.02

        # Peg spawn bounds
        self.peg_spawn_bounds = {
            "x": [0.38 - self.table_extents[0] + 0.05, 0.38 + self.table_extents[0] - 0.05],
            "y": [-self.table_extents[1] + 0.05, self.table_extents[1] - 0.05],
            "z": [self.table_top_z] * 2,
        }

        # Grommet spawn bounds
        self.grommet_spawn_bounds = {
            "x": [0.38 - 0.15 + 0.05, 0.38 + 0.15 - 0.05],
            "y": [-0.1 + 0.05, 0.1 - 0.05],
            "z": [self.table_top_z + self.grommet_half] * 2,
        }

        # Locate repository root
        self.repo_root = self._find_repo_root()
        self.desc_dir = os.path.join(self.repo_root, 'robot_descriptions')

    def _find_repo_root(self, target="robot_descriptions") -> str:
        """
        Traverses parent directories to find the repository root containing the target folder.
        """
        script_dir = os.path.abspath(os.path.dirname(__file__))
        current = script_dir
        while current != os.path.abspath(os.sep):
            if os.path.isdir(os.path.join(current, target)):
                return current
            current = os.path.abspath(os.path.join(current, '..'))
        raise FileNotFoundError(f"Could not find '{target}' in any parent folder.")

    def _random_or_fixed_position(self, bounds):
        """
        Returns a random position within the specified bounds.
        """
        return [
            np.random.uniform(*bounds["x"]),
            np.random.uniform(*bounds["y"]),
            np.random.uniform(*bounds["z"]),
        ]
    
    def _random_x_quaternion(self):
        """
        Returns a quaternion representing a random roll (X-axis only).
        """
        theta = np.random.uniform(0, 2 * np.pi)
        return [np.sin(theta / 2), 0, 0, np.cos(theta / 2)]  # [x, y, z, w]

    def build_single_arm_heal_scene(self):
        """
        Builds and returns the MuJoCo model for the scene,
        along with the gripper configuration if included.
        """
        # Load model files
        arm_path = os.path.join(self.desc_dir, 'single_arm_heal_effort_actuation_rs_mj.xml')
        grip_path = os.path.join(self.desc_dir, 'robotiq_2f85_v4', '2f85.xml')
        dual_pegs_path = os.path.join(self.desc_dir, 'assets', 'dual_pegs', 'dual_pegs.xml')
        grommet_path = os.path.join(self.desc_dir, 'assets', 'dual_pegs', 'grommet_16mm.xml')

        # Load HEAL arm
        arm = mujoco.MjSpec.from_file(arm_path)
        arm.compiler.inertiafromgeom = True

        # Attach gripper if requested
        if self.include_gripper:
            grip = mujoco.MjSpec.from_file(grip_path)
            arm.attach(grip, prefix='gripper/', site=arm.site('right_center'))

        # Add table
        if self.include_table:
            table = arm.worldbody.add_body(name='table', pos=[0.38, 0.0, self.table_z])
            table.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=self.table_extents,
                rgba=[0.5, 0.4, 0.3, 1],
                contype=1,
                conaffinity=1
            )

        # Load peg and grommet assemblies
        dual_pegs = mujoco.MjSpec.from_file(dual_pegs_path)
        grommet = mujoco.MjSpec.from_file(grommet_path)

        # Attach peg and grommet to the world
        arm.attach(dual_pegs, frame=arm.frame("world_attach"))
        arm.attach(grommet, frame=arm.frame("world_attach"))

        # Ensure peg and grommet are spaced apart if grommet is included
        if self.include_grommet:
            while True:
                peg_pos = self._random_or_fixed_position(self.peg_spawn_bounds)
                grommet_pos = self._random_or_fixed_position(self.grommet_spawn_bounds)
                if np.linalg.norm(np.array(peg_pos[:2]) - np.array(grommet_pos[:2])) > 0.2:
                    break
        else:
            peg_pos = self._random_or_fixed_position(self.peg_spawn_bounds)
            grommet_pos = self._random_or_fixed_position(self.grommet_spawn_bounds)

        # Set peg and grommet positions
        peg_body = arm.body('dual_peg')
        peg_body.pos = peg_pos
        peg_body.quat = self._random_x_quaternion()

        grommet_body = arm.body('grommet_16mm')
        grommet_body.pos = grommet_pos
        grommet_body.quat = self._random_x_quaternion()

        # Print orientation of peg and grommet
        print("[DEBUG] Peg orientation (quaternion):", peg_body.quat)
        print("[DEBUG] Grommet orientation (quaternion):", grommet_body.quat)

        # Compile the final model
        model = arm.compile()

        # Gripper configuration
        gripper_cfg = []
        if self.include_gripper:
            for i in range(model.nu):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if name and name.startswith("gripper/"):
                    gripper_cfg.append({
                        "actuator_id": i,
                        "open_cmd": 0.0,
                        "close_cmd": 255.0,
                    })

        return model, gripper_cfg

class SceneBuilderStack:
    
    """
    A utility class to construct MuJoCo scenes for simulation involving a single HEAL arm,
    with optional gripper, table, cube, and tray. Also returns gripper configuration.
    """

    def __init__(self, robot: str = "heal" ,include_gripper: bool = None, include_table: bool = None,
                 randomize_cube: bool = False, cube_s_spawn_bounds: dict = None,cube_b_spawn_bounds: dict = None,
                 include_tray: bool = False, randomize_tray: bool = False, tray_spawn_bounds: dict = None,
                 include_plate: bool = False, randomize_plate: bool = False, plate_spawn_bounds: dict = None):
        
        """
        Initializes the scene builder.
        """
        self.robot = robot
        self.include_gripper = True if include_gripper is None else include_gripper
        self.include_table = True if include_table is None else include_table
        self.include_tray = include_tray
        self.include_plate = include_plate
        self.randomize_cube = randomize_cube
        self.randomize_tray = randomize_tray
        self.randomize_plate = randomize_plate

        self.table_extents = [0.22, 0.32, 0.025]
        self.table_z = 0.175
        self.table_top_z = self.table_z + self.table_extents[2]
        self.cube_s_half = 0.02
        self.cube_b_half = 0.025

        self.cube_s_spawn_bounds = cube_s_spawn_bounds or {
            "x": [0.38 - self.table_extents[0], 0.38 + self.table_extents[0]],
            "y": [-self.table_extents[1], self.table_extents[1]],
            "z": [self.table_top_z + self.cube_s_half, self.table_top_z + self.cube_s_half],
        }
        
        self.cube_b_spawn_bounds = cube_b_spawn_bounds or {
            "x": [0.38 - self.table_extents[0], 0.38 + self.table_extents[0]],
            "y": [-self.table_extents[1], self.table_extents[1]],
            "z": [self.table_top_z + self.cube_b_half, self.table_top_z + self.cube_b_half],
        }

        self.tray_extents = [0.05, 0.05, 0.005]
        self.tray_wall_thickness = 0.001
        self.tray_wall_height = 0.025

        self.tray_spawn_bounds = tray_spawn_bounds or {
            "x": [0.38 - 0.15, 0.38 + 0.15],
            "y": [-0.1, 0.1],
            "z": [self.table_top_z + self.tray_extents[2] / 2, self.table_top_z + self.tray_extents[2] / 2],
        }

        self.plate_spawn_bounds = plate_spawn_bounds or {
            "x": [0.38 - 0.15, 0.38 + 0.15],
            "y": [-0.1, 0.1],
            "z": [self.table_top_z, self.table_top_z],
        }

        self.repo_root = self._find_repo_root()
        self.desc_dir = os.path.join(self.repo_root, 'robot_descriptions')

    def _find_repo_root(self, target="robot_descriptions") -> str:
        script_dir = os.path.abspath(os.path.dirname(__file__))
        current = script_dir
        while current != os.path.abspath(os.sep):
            if os.path.isdir(os.path.join(current, target)):
                return current
            current = os.path.abspath(os.path.join(current, '..'))
        raise FileNotFoundError(f"Could not find '{target}' in any parent folder.")

    def build_single_arm_robot_scene(self) -> tuple[mujoco.MjModel, list[dict]]:
        if self.robot == "heal":
            arm_path = os.path.join(self.desc_dir, 'single_arm_heal_effort_actuation_rs_mj.xml')
        elif self.robot == "franka":
            arm_path = os.path.join(self.desc_dir, 'franka','mjx_panda_nohand.xml')
        arm = mujoco.MjSpec.from_file(arm_path)
        arm.compiler.inertiafromgeom = True

        if self.include_gripper:
            
            if self.robot == "heal":
                grip_path = os.path.join(self.desc_dir, 'robotiq_2f85_v4', '2f85.xml')
            elif self.robot == "franka":
                grip_path = os.path.join(self.desc_dir, 'franka', 'hand.xml')

            grip = mujoco.MjSpec.from_file(grip_path)
            arm.attach(grip, prefix='gripper/', site=arm.site('right_center'))

        # Add table
        if self.include_table:
            table = arm.worldbody.add_body(name='table', pos=[0.38, 0.0, self.table_z])
            table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=self.table_extents,
                           rgba=[0.5, 0.4, 0.3, 1],
                           contype=1,
                           conaffinity=1)

        # Spawn tray and cube
        tray_pos = None
        if self.include_tray:
            while True:
                tray_pos = self._random_or_fixed_position(self.tray_spawn_bounds)
                cube_s_pos = self._random_or_fixed_position(self.cube_s_spawn_bounds)
                cube_b_pos = self._random_or_fixed_position(self.cube_b_spawn_bounds)
                if (np.linalg.norm(np.array(tray_pos[:2]) - np.array(cube_s_pos[:2])) > 0.08) & (np.linalg.norm(np.array(tray_pos[:2]) - np.array(cube_b_pos[:2])) > 0.08) & (np.linalg.norm(np.array(cube_b_pos[:2]) - np.array(cube_s_pos[:2])) > 0.08):
                    break

            # Fix tray Z to be on table
            tray_pos[2] = self.table_top_z + self.tray_extents[2] / 2

            tray = arm.worldbody.add_body(name='tray', pos=tray_pos)
            tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=self.tray_extents,
                          rgba=[0.7, 0.7, 0.7, 1],
                          contype=1,
                          conaffinity=1)

            # Add tray walls
            wall_size_x = [self.tray_wall_thickness, self.tray_extents[1], self.tray_wall_height]
            wall_size_y = [self.tray_extents[0], self.tray_wall_thickness, self.tray_wall_height]

            tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[self.tray_extents[0], 0, self.tray_wall_height],
                          size=wall_size_x, rgba=[0.3, 0.3, 0.3, 1])
            tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-self.tray_extents[0], 0, self.tray_wall_height],
                          size=wall_size_x, rgba=[0.3, 0.3, 0.3, 1])
            tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, self.tray_extents[1], self.tray_wall_height],
                          size=wall_size_y, rgba=[0.3, 0.3, 0.3, 1])
            tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -self.tray_extents[1], self.tray_wall_height],
                          size=wall_size_y, rgba=[0.3, 0.3, 0.3, 1])
            
        elif self.include_plate:
            while True:
                plate_pos = self._random_or_fixed_position(self.plate_spawn_bounds)
                cube_s_pos = self._random_or_fixed_position(self.cube_s_spawn_bounds)
                cube_b_pos = self._random_or_fixed_position(self.cube_b_spawn_bounds)
                if (np.linalg.norm(np.array(plate_pos[:2]) - np.array(cube_s_pos[:2])) > 0.08) & (np.linalg.norm(np.array(plate_pos[:2]) - np.array(cube_b_pos[:2])) > 0.08) & (np.linalg.norm(np.array(cube_b_pos[:2]) - np.array(cube_s_pos[:2])) > 0.08):
                    break

            plate_dir = os.path.join(self.desc_dir,'plate_description', 'model.xml')
            

            # Set plate body position
            plate = mujoco.MjSpec.from_file(plate_dir)
            attachment_frame = arm.worldbody.add_frame(pos=plate_pos, name='plate_frame')
            attachment_frame.attach_body(plate.body('plate_body'))
            
        else:
            while True:
                cube_s_pos = self._random_or_fixed_position(self.cube_s_spawn_bounds)
                cube_b_pos = self._random_or_fixed_position(self.cube_b_spawn_bounds)
                if np.linalg.norm(np.array(cube_s_pos[:2]) - np.array(cube_b_pos[:2])) > 0.1:
                    break
            

        # Add cube
        cube_s = arm.worldbody.add_body(name='cube_small', pos=cube_s_pos)
        cube_s.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[self.cube_s_half] * 3,
                      rgba=[1, 0, 0, 1],
                      mass=0.1,
                      friction=[0.1, 0.005, 0.005])
        
        cube_s.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name='cube_free')
        
        cube_b = arm.worldbody.add_body(name='cube_big', pos=cube_b_pos)
        cube_b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[self.cube_b_half] * 3,
                      rgba=[0, 0, 1, 1],
                      mass=0.1,
                      friction=[0.1, 0.005, 0.005])
        
        cube_b.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name='cubeb_free')

        # Compile model
        model = arm.compile()

        # Gripper config
        gripper_config = []
        if self.include_gripper:
            for i in range(model.nu):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if name and name.startswith("gripper/"):
                    gripper_config.append({
                        "actuator_id": i,
                        "open_cmd": 0.0,
                        "close_cmd": 255.0,
                    })
                    
            
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.impratio = 50
        model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
        model.opt.iterations = 50
        model.opt.noslip_iterations = 100

        return model, gripper_config

    def _random_or_fixed_position(self, bounds):
        return [
            np.random.uniform(*bounds["x"]),
            np.random.uniform(*bounds["y"]),
            np.random.uniform(*bounds["z"]),
        ]

class SceneBuilderPeg11:
    """
    A utility class to construct MuJoCo scenes involving a single HEAL arm,
    a dual peg assembly, and optionally a gripper, table, and grommet.
    """

    def __init__(self, include_gripper=True, include_table=True,
                 randomize_peg=False, include_grommet=False):
        """
        Initializes the peg scene builder.
        """
        self.include_gripper = include_gripper
        self.include_table = include_table
        self.randomize_peg = randomize_peg
        self.include_grommet = include_grommet

        # Table dimensions and height
        self.table_extents = [0.22, 0.32, 0.025]
        self.table_z = 0.175
        self.table_top_z = self.table_z + self.table_extents[2]

        # Peg and grommet size
        self.peg_half = 0.02
        self.grommet_half = 0.02

        # Peg spawn bounds
        self.peg_spawn_bounds = {
            "x": [0.38 - self.table_extents[0] + 0.05, 0.38 + self.table_extents[0] - 0.05],
            "y": [-self.table_extents[1] + 0.05, self.table_extents[1] - 0.05],
            "z": [self.table_top_z] * 2,
        }

        # Grommet spawn bounds
        self.grommet_spawn_bounds = {
            "x": [0.38 - 0.15 + 0.05, 0.38 + 0.15 - 0.05],
            "y": [-0.1 + 0.05, 0.1 - 0.05],
            "z": [self.table_top_z + self.grommet_half] * 2,
        }

        # Locate repository root
        self.repo_root = self._find_repo_root()
        self.desc_dir = os.path.join(self.repo_root, 'robot_descriptions')

    def _find_repo_root(self, target="robot_descriptions") -> str:
        """
        Traverses parent directories to find the repository root containing the target folder.
        """
        script_dir = os.path.abspath(os.path.dirname(__file__))
        current = script_dir
        while current != os.path.abspath(os.sep):
            if os.path.isdir(os.path.join(current, target)):
                return current
            current = os.path.abspath(os.path.join(current, '..'))
        raise FileNotFoundError(f"Could not find '{target}' in any parent folder.")

    def _random_or_fixed_position(self, bounds):
        """
        Returns a random position within the specified bounds.
        """
        return [
            np.random.uniform(*bounds["x"]),
            np.random.uniform(*bounds["y"]),
            np.random.uniform(*bounds["z"]),
        ]
    
    def _random_x_quaternion(self):
        """
        Returns a quaternion representing a random roll (X-axis only).
        """
        theta = np.random.uniform(0, 2 * np.pi)
        return [np.sin(theta / 2), 0, 0, np.cos(theta / 2)]  # [x, y, z, w]

    def build_single_arm_heal_scene(self):
        """
        Builds and returns the MuJoCo model for the scene,
        along with the gripper configuration if included.
        """
        # Load model files
        arm_path = os.path.join(self.desc_dir, 'single_arm_heal_effort_actuation_rs_mj.xml')
        grip_path = os.path.join(self.desc_dir, 'robotiq_2f85_v4', '2f85.xml')
        dual_pegs_path = os.path.join(self.desc_dir, 'assets', 'dual_pegs', 'dual_pegs.xml')
        grommet_path = os.path.join(self.desc_dir, 'assets', 'dual_pegs', 'grommet_11mm.xml')

        # Load HEAL arm
        arm = mujoco.MjSpec.from_file(arm_path)
        arm.compiler.inertiafromgeom = True

        # Attach gripper if requested
        if self.include_gripper:
            grip = mujoco.MjSpec.from_file(grip_path)
            arm.attach(grip, prefix='gripper/', site=arm.site('right_center'))

        # Add table
        if self.include_table:
            table = arm.worldbody.add_body(name='table', pos=[0.38, 0.0, self.table_z])
            table.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=self.table_extents,
                rgba=[0.5, 0.4, 0.3, 1],
                contype=1,
                conaffinity=1
            )

        # Load peg and grommet assemblies
        dual_pegs = mujoco.MjSpec.from_file(dual_pegs_path)
        grommet = mujoco.MjSpec.from_file(grommet_path)

        # Attach peg and grommet to the world
        arm.attach(dual_pegs, frame=arm.frame("world_attach"))
        arm.attach(grommet, frame=arm.frame("world_attach"))

        # Ensure peg and grommet are spaced apart if grommet is included
        if self.include_grommet:
            while True:
                peg_pos = self._random_or_fixed_position(self.peg_spawn_bounds)
                grommet_pos = self._random_or_fixed_position(self.grommet_spawn_bounds)
                if np.linalg.norm(np.array(peg_pos[:2]) - np.array(grommet_pos[:2])) > 0.2:
                    break
        else:
            peg_pos = self._random_or_fixed_position(self.peg_spawn_bounds)
            grommet_pos = self._random_or_fixed_position(self.grommet_spawn_bounds)

        # Set peg and grommet positions
        peg_body = arm.body('dual_peg')
        peg_body.pos = peg_pos
        peg_body.quat = self._random_x_quaternion()

        grommet_body = arm.body('grommet_11mm')
        grommet_body.pos = grommet_pos
        grommet_body.quat = self._random_x_quaternion()

        # Print orientation of peg and grommet
        print("[DEBUG] Peg orientation (quaternion):", peg_body.quat)
        print("[DEBUG] Grommet orientation (quaternion):", grommet_body.quat)

        # Compile the final model
        model = arm.compile()

        # Gripper configuration
        gripper_cfg = []
        if self.include_gripper:
            for i in range(model.nu):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if name and name.startswith("gripper/"):
                    gripper_cfg.append({
                        "actuator_id": i,
                        "open_cmd": 0.0,
                        "close_cmd": 255.0,
                    })

        return model, gripper_cfg

class SceneBuilderHumanoid2:
    """
    Scene builder that loads the Unitree G1 humanoid (g1_with_hands.xml)
    and constructs a scene with table, tray and two cubes (different color).
    """

    def __init__(
        self,
        include_table: bool = True,
        include_tray: bool = False,
        randomize_tray: bool = True,
        include_cube: bool = True,
        randomize_cube: bool = True,
        include_second_cube: bool = True,
        cube_spawn_bounds: Optional[dict] = None,
        tray_spawn_bounds: Optional[dict] = None,
        robot_y: float = 0.0,
        prefix: Optional[str] = None,
        min_pairwise_dist: float = 0.12,   # minimum allowed distance between tray and cubes / between cubes
        humanoid_xml_relpath: str = "unitree_g1/g1_with_hands.xml",  # relative to robot_descriptions
    ):
        self.include_table = include_table
        self.include_tray = include_tray
        self.randomize_tray = randomize_tray
        self.include_cube = include_cube
        self.randomize_cube = randomize_cube
        self.include_second_cube = include_second_cube
        self.robot_y = robot_y
        self.prefix = prefix or ""
        self.min_pairwise_dist = min_pairwise_dist
        self.humanoid_xml_relpath = humanoid_xml_relpath

        # Table and cube geometry defaults (copied/adapted from your class)
        self.table_extents = [0.22, 0.32, 0.025]
        self.table_z = 0.175
        self.table_top_z = self.table_z + self.table_extents[2]
        self.cube_half = 0.02

        # spawn bounds (same style as your original)
        default_cube_bounds = {
            "x": [0.38 - self.table_extents[0], 0.38 + self.table_extents[0]],
            "y": [self.robot_y - self.table_extents[1], self.robot_y + self.table_extents[1]],
            "z": [self.table_top_z + self.cube_half, self.table_top_z + self.cube_half],
        }
        self.cube_spawn_bounds = cube_spawn_bounds or default_cube_bounds

        self.tray_extents = [0.05, 0.05, 0.005]
        self.tray_wall_thickness = 0.001
        self.tray_wall_height = 0.025

        default_tray_bounds = {
            "x": [0.38 - 0.15, 0.38 + 0.15],
            "y": [self.robot_y - 0.1, self.robot_y + 0.1],
            "z": [self.table_top_z + self.tray_extents[2] / 2, self.table_top_z + self.tray_extents[2] / 2],
        }
        self.tray_spawn_bounds = tray_spawn_bounds or default_tray_bounds

        # repo and desc dir resolution (same approach as your SceneBuilder)
        self.repo_root = self._find_repo_root()
        self.desc_dir = os.path.join(self.repo_root, "robot_descriptions")

    def _find_repo_root(self, target="robot_descriptions") -> str:
        script_dir = os.path.abspath(os.path.dirname(__file__))
        current = script_dir
        while current != os.path.abspath(os.sep):
            if os.path.isdir(os.path.join(current, target)):
                return current
            current = os.path.abspath(os.path.join(current, ".."))
        raise FileNotFoundError(f"Could not find '{target}' in any parent folder.")

    def _random_or_fixed_position(self, bounds):
        return [
            float(np.random.uniform(*bounds["x"])),
            float(np.random.uniform(*bounds["y"])),
            float(np.random.uniform(*bounds["z"])),
        ]

    def _pairwise_ok(self, positions, min_dist):
        # Check that every pair of positions (only XY considered) is farther than min_dist
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(np.array(positions[i][:2]) - np.array(positions[j][:2])) < min_dist:
                    return False
        return True

    def build_single_arm_robot_scene(self) -> Tuple[mujoco.MjModel, List[dict]]:
        """
        Loads the humanoid XML, optionally adds table/tray/cubes and compiles the model.

        Returns:
            model (mujoco.MjModel), gripper_config (list[dict]) -- gripper_config kept for API parity.
        """
        humanoid_path = os.path.join(self.desc_dir, self.humanoid_xml_relpath)

        if not os.path.exists(humanoid_path):
            raise FileNotFoundError(f"Humanoid xml not found at {humanoid_path}")

        # Load MjSpec for the humanoid
        arm_spec = mujoco.MjSpec.from_file(humanoid_path)

        # The humanoid xml already contains inertial tags. Don't infer inertia from geometry.
        arm_spec.compiler.inertiafromgeom = False

        # Add table if requested
        if self.include_table:
            table = arm_spec.worldbody.add_body(name=f'{self.prefix}table', pos=[0.38, self.robot_y, self.table_z])
            table.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=self.table_extents,
                rgba=[0.5, 0.4, 0.3, 1],
                contype=1,
                conaffinity=1,
            )

        # spawn tray and two cubes with distance constraints
        spawn_attempts = 0
        max_attempts = 200

        tray_pos = None
        cube1_pos = None
        cube2_pos = None

        if self.include_tray or self.include_cube or self.include_second_cube:
            while spawn_attempts < max_attempts:
                spawn_attempts += 1
                cand_positions = []

                if self.include_tray:
                    tray_pos = self._random_or_fixed_position(self.tray_spawn_bounds) if self.randomize_tray else [
                        (self.tray_spawn_bounds["x"][0] + self.tray_spawn_bounds["x"][1]) / 2,
                        (self.tray_spawn_bounds["y"][0] + self.tray_spawn_bounds["y"][1]) / 2,
                        self.tray_spawn_bounds["z"][0],
                    ]
                    # fix Z onto table top
                    tray_pos[2] = self.table_top_z + self.tray_extents[2] / 2
                    cand_positions.append(tray_pos)

                if self.include_cube:
                    cube1_pos = self._random_or_fixed_position(self.cube_spawn_bounds) if self.randomize_cube else [
                        (self.cube_spawn_bounds["x"][0] + self.cube_spawn_bounds["x"][1]) / 2,
                        (self.cube_spawn_bounds["y"][0] + self.cube_spawn_bounds["y"][1]) / 2,
                        self.cube_spawn_bounds["z"][0],
                    ]
                    cand_positions.append(cube1_pos)

                if self.include_second_cube:
                    # second cube uses same bounds by default (could be changed)
                    cube2_pos = self._random_or_fixed_position(self.cube_spawn_bounds) if self.randomize_cube else [
                        (self.cube_spawn_bounds["x"][0] + self.cube_spawn_bounds["x"][1]) / 2 + 0.08,
                        (self.cube_spawn_bounds["y"][0] + self.cube_spawn_bounds["y"][1]) / 2,
                        self.cube_spawn_bounds["z"][0],
                    ]
                    cand_positions.append(cube2_pos)

                # If some items are not included, cand_positions length will adapt
                if self._pairwise_ok([p for p in cand_positions if p is not None], self.min_pairwise_dist):
                    break

            if spawn_attempts >= max_attempts:
                raise RuntimeError("Failed to sample non-overlapping spawn positions after many attempts. "
                                   "Consider increasing scene area or lowering min_pairwise_dist.")

        # Create the tray and cube bodies in the spec
        if self.include_tray and tray_pos is not None:
            tray = arm_spec.worldbody.add_body(name=f'{self.prefix}tray', pos=tray_pos)
            tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=self.tray_extents,
                          rgba=[0.7, 0.7, 0.7, 1],
                          contype=1,
                          conaffinity=1)
            # tray walls (same as original)
            wall_size_x = [self.tray_wall_thickness, self.tray_extents[1], self.tray_wall_height]
            wall_size_y = [self.tray_extents[0], self.tray_wall_thickness, self.tray_wall_height]

            tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[self.tray_extents[0], 0, self.tray_wall_height],
                          size=wall_size_x, rgba=[0.3, 0.3, 0.3, 1])
            tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-self.tray_extents[0], 0, self.tray_wall_height],
                          size=wall_size_x, rgba=[0.3, 0.3, 0.3, 1])
            tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, self.tray_extents[1], self.tray_wall_height],
                          size=wall_size_y, rgba=[0.3, 0.3, 0.3, 1])
            tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -self.tray_extents[1], self.tray_wall_height],
                          size=wall_size_y, rgba=[0.3, 0.3, 0.3, 1])

        if self.include_cube and cube1_pos is not None:
            cube = arm_spec.worldbody.add_body(name=f'{self.prefix}cube', pos=cube1_pos)
            cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[self.cube_half] * 3,
                          rgba=[1.0, 0.0, 0.0, 1.0],  # red
                          mass=0.1,
                          friction=[0.1, 0.005, 0.005])
            cube.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name=f'{self.prefix}cube_free')

        if self.include_second_cube and cube2_pos is not None:
            cube2 = arm_spec.worldbody.add_body(name=f'{self.prefix}cube2', pos=cube2_pos)
            cube2.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[self.cube_half] * 3,
                           rgba=[0.0, 0.4, 1.0, 1.0],  # bluish
                           mass=0.1,
                           friction=[0.1, 0.005, 0.005])
            cube2.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name=f'{self.prefix}cube2_free')

        # Compile model and return
        try:
            model = arm_spec.compile()
        except Exception as e:
            # Give the user a helpful error message like your earlier runtime error
            raise RuntimeError(
                "Failed to compile model. Common causes: missing mesh assets (check xml meshdir/assets), "
                "or mass/inertia too small. Ensure the humanoid's mesh files exist relative to the xml's meshdir. "
                f"Original error: {e}"
            ) from e

        # no gripper config for humanoid, but return empty list for API parity
        gripper_config = []

        # set solver/options (copied from your previous model options)
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.impratio = 50
        model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
        model.opt.iterations = 50
        model.opt.noslip_iterations = 100

        return model, gripper_config

import os
import numpy as np
import mujoco


class SceneBuilderHumanoid:
    """
    Scene builder for Unitree G1 humanoid (without hands) + Robotiq 2F-85 grippers,
    with table, tray, cube etc., mimicking the HEAL `SceneBuilder` structure.
    """

    def __init__(
        self,
        include_table: bool = None,
        randomize_cube: bool = False,
        cube_spawn_bounds: dict = None,
        include_tray: bool = False,
        randomize_tray: bool = False,
        tray_spawn_bounds: dict = None,
        include_plate: bool = False,   # kept for API similarity, unused here
        randomize_plate: bool = False, # kept for API similarity, unused here
        plate_spawn_bounds: dict = None,
        include_cube: bool = False,
        robot_y: float = 0.0,
        prefix: str | None = None,
    ):
        self.include_table = True if include_table is None else include_table
        self.include_tray = include_tray
        self.include_cube = include_cube
        self.include_plate = include_plate   # not used, just kept for compatibility
        self.randomize_cube = randomize_cube
        self.randomize_tray = randomize_tray
        self.randomize_plate = randomize_plate
        self.robot_y = robot_y
        self.prefix = prefix

        # --- Same geometry constants as SceneBuilder -------------------
        self.table_extents = [0.22, 0.32, 0.025]
        self.table_z = 0.175
        self.table_top_z = self.table_z + self.table_extents[2]
        self.cube_half = 0.02

        self.cube_spawn_bounds = cube_spawn_bounds or {
            "x": [0.38 - self.table_extents[0], 0.38 + self.table_extents[0]],
            "y": [self.robot_y - self.table_extents[1], self.robot_y + self.table_extents[1]],
            "z": [self.table_top_z + self.cube_half, self.table_top_z + self.cube_half],
        }

        self.tray_extents = [0.05, 0.05, 0.005]
        self.tray_wall_thickness = 0.001
        self.tray_wall_height = 0.025

        self.tray_spawn_bounds = tray_spawn_bounds or {
            "x": [0.38 - 0.15, 0.38 + 0.15],
            "y": [self.robot_y - 0.1, self.robot_y + 0.1],
            "z": [self.table_top_z + self.tray_extents[2] / 2,
                  self.table_top_z + self.tray_extents[2] / 2],
        }

        self.plate_spawn_bounds = plate_spawn_bounds or {
            "x": [0.38 - 0.15, 0.38 + 0.15],
            "y": [self.robot_y - 0.1, self.robot_y + 0.1],
            "z": [self.table_top_z, self.table_top_z],
        }

        self.repo_root = self._find_repo_root()
        self.desc_dir = os.path.join(self.repo_root, "robot_descriptions")

    # ------------------------------------------------------------------
    # Helper: find repo root (same as SceneBuilder)
    # ------------------------------------------------------------------
    def _find_repo_root(self, target: str = "robot_descriptions") -> str:
        script_dir = os.path.abspath(os.path.dirname(__file__))
        current = script_dir
        while current != os.path.abspath(os.sep):
            if os.path.isdir(os.path.join(current, target)):
                return current
            current = os.path.abspath(os.path.join(current, ".."))
        raise FileNotFoundError(f"Could not find '{target}' in any parent folder.")

    # ------------------------------------------------------------------
    # Public API: build_scene (analogous to build_single_arm_robot_scene)
    # ------------------------------------------------------------------
    def build_scene(self) -> tuple[mujoco.MjModel, list[dict]]:
        # 1. Load humanoid WITHOUT hands
        humanoid_path = os.path.join(
            self.desc_dir, "unitree_g1", "g1_without_hands.xml"
        )
        spec = mujoco.MjSpec.from_file(humanoid_path)
        spec.compiler.inertiafromgeom = True

        # 2. Attach Robotiq 2F-85 to both wrists with SAME twisted-down orientation
        #    (90° about X, then 90° about Z) – same logic you used in the script.
        angle_x = np.pi / 2.0
        angle_z = np.pi / 2.0

        cx = float(np.cos(angle_x / 2.0))
        sx = float(np.sin(angle_x / 2.0))
        q_x = np.array([cx, sx, 0.0, 0.0])

        cz = float(np.cos(angle_z / 2.0))
        sz = float(np.sin(angle_z / 2.0))
        q_z = np.array([cz, 0.0, 0.0, sz])

        def quat_mul(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ])

        palm_quat = quat_mul(q_z, q_x)

        # Right wrist site
        right_wrist_body = spec.body("right_wrist_yaw_link")
        right_wrist_body.add_site(
            name="right_palm",
            pos=[0.0, 0.0, 0.0],
            quat=palm_quat.tolist(),
        )

        # Left wrist site
        left_wrist_body = spec.body("left_wrist_yaw_link")
        left_wrist_body.add_site(
            name="left_palm",
            pos=[0.0, 0.0, 0.0],
            quat=palm_quat.tolist(),
        )

        # Load gripper spec
        grip_path = os.path.join(self.desc_dir, "robotiq_2f85_v4", "2f85.xml")
        grip = mujoco.MjSpec.from_file(grip_path)

        # Attach to right wrist
        spec.attach(
            grip,
            prefix="right_gripper/",
            site=spec.site("right_palm"),
        )

        # Attach to left wrist (reload spec to avoid name conflicts)
        grip_left = mujoco.MjSpec.from_file(grip_path)
        spec.attach(
            grip_left,
            prefix="left_gripper/",
            site=spec.site("left_palm"),
        )

        # 3. Add table (same as SceneBuilder)
        if self.include_table:
            table = spec.worldbody.add_body(
                name=f"{self.prefix or ''}table",
                pos=[0.38, self.robot_y, self.table_z],
            )
            table.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=self.table_extents,
                rgba=[0.5, 0.4, 0.3, 1.0],
                contype=1,
                conaffinity=1,
            )

        # 4. Tray + cube (same logic as SceneBuilder)
        tray_pos = None
        if self.include_tray:
            # Sample tray and cube so they don't overlap too closely
            while True:
                tray_pos = self._random_or_fixed_position(self.tray_spawn_bounds)
                cube_pos = self._random_or_fixed_position(self.cube_spawn_bounds)
                if np.linalg.norm(np.array(tray_pos[:2]) - np.array(cube_pos[:2])) > 0.08:
                    break

            # Fix tray Z on table
            tray_pos[2] = self.table_top_z + self.tray_extents[2] / 2

            tray = spec.worldbody.add_body(
                name=f"{self.prefix or ''}tray", pos=tray_pos
            )
            tray.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=self.tray_extents,
                rgba=[0.7, 0.7, 0.7, 1.0],
                contype=1,
                conaffinity=1,
            )

            # Tray walls
            wall_size_x = [self.tray_wall_thickness,
                           self.tray_extents[1],
                           self.tray_wall_height]
            wall_size_y = [self.tray_extents[0],
                           self.tray_wall_thickness,
                           self.tray_wall_height]

            tray.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[self.tray_extents[0], 0, self.tray_wall_height],
                size=wall_size_x,
                rgba=[0.3, 0.3, 0.3, 1.0],
            )
            tray.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[-self.tray_extents[0], 0, self.tray_wall_height],
                size=wall_size_x,
                rgba=[0.3, 0.3, 0.3, 1.0],
            )
            tray.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0, self.tray_extents[1], self.tray_wall_height],
                size=wall_size_y,
                rgba=[0.3, 0.3, 0.3, 1.0],
            )
            tray.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0, -self.tray_extents[1], self.tray_wall_height],
                size=wall_size_y,
                rgba=[0.3, 0.3, 0.3, 1.0],
            )

        # Plate path kept for API parity, but *not* used here
        if self.include_plate:
            # You can mirror the plate logic from SceneBuilder if needed
            pass

        # 5. Cube
        if self.include_cube:
            cube_pos = self._random_or_fixed_position(self.cube_spawn_bounds)
            cube = spec.worldbody.add_body(
                name=f"{self.prefix or ''}cube",
                pos=cube_pos,
            )
            cube.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[self.cube_half] * 3,
                rgba=[1.0, 0.0, 0.0, 1.0],
                mass=0.1,
                friction=[0.1, 0.005, 0.005],
            )
            cube.add_joint(
                type=mujoco.mjtJoint.mjJNT_FREE,
                name=f"{self.prefix or ''}cube_free",
            )

        # 6. Compile model
        model = spec.compile()

        # Gripper config (both hands)
        gripper_config: list[dict] = []
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if not name:
                continue
            if name.startswith("left_gripper/") or name.startswith("right_gripper/"):
                gripper_config.append(
                    {
                        "actuator_id": i,
                        "open_cmd": 0.0,
                        "close_cmd": 255.0,
                    }
                )

        # Solver options same as SceneBuilder
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.impratio = 50
        model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
        model.opt.iterations = 50
        model.opt.noslip_iterations = 100

        return model, gripper_config

    # ------------------------------------------------------------------
    # Same helper as SceneBuilder
    # ------------------------------------------------------------------
    def _random_or_fixed_position(self, bounds):
        return [
            np.random.uniform(*bounds["x"]),
            np.random.uniform(*bounds["y"]),
            np.random.uniform(*bounds["z"]),
        ]