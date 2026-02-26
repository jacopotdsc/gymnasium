# Copyright 2025
# Licensed under the Apache License, Version 2.0
#
# MuJoCo Tita Wheel-Legged Robot Environment for Gymnasium

import sys
import time
#sys.path.insert(0, '/home/ubuntu/Desktop/repo_rl/Residual_Learning_TITA/TITA_MJ/compiled')
sys.path.insert(0, '/home/ubuntu/Desktop/repo_rl/TITA-dynamic-obstacle-avoidance/TITA_MJ/compiled')

import numpy as np
from typing import Any, Dict, Optional
import gymnasium as gym
from gymnasium import utils
from gymnasium import error, spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from ml_collections import config_dict
from collections import deque
import wm
import copy
import torch
from scipy.spatial.transform import Rotation
import transformations as tr

try:
    import mujoco
except ImportError:
    raise ImportError(
        "MuJoCo is required for TitaEnv. "
        "Install it with: pip install mujoco"
    )

'''
TODO: check frequency of data readings )
TODO: randomization doesn't work cause of Controller
TODO: set right frequency: 
        sim_dt * frame_skip = ctrl_dt
        1 / ctrl_dt = Controller frequency 

    - sim_dt how much time passes in sim for each mujoco step
    - frame_skip: how many mujoco steps per env step
    - sim_dt * frame_skip correspond to how much time
        passed for one call of step()
    - the frequency of call of the controller is 1/(sim_dt * frame_skip)

 Robot State:
    - position, it explodd with values, how handle it
    - orientation, quaternion
    - Linear velocity in body frame (x, y, z )
    - angular velocity in body frame ( Gyroscope readings )
    - Joint State
        - Joint positions
        - Joint velocities
        - joint torques 
    - Total force
    - contact points
    - Contact forces

Desired configuration
    - position
    - orientation
    - Linear velocity
    - angular velocity
    - Joint positions
    - Joint velocities
    - joint acceleration
    - com: position, velocity, acceleration
    - l_wheel and r_wheel: 
        SE3 pose ( position + orientation of the wheel )
        linear velocity
        angular velocity
        linear acceleration
        angular acceleration
    - base link:
        position
        angular velocity
        angular acceleration

MPC I/O
    - ddp_com
    - gravity vector
    - total force
    - p_zmp - p_con

    - p_com(t), v_com(t),a_com(t)
    - p_zmp(t), v_zmp(t), a_zmp(t)

WBC I/O
    - robot_state
    - desired_configuration


in get obs,

Note: MPC_torque_k + NN_Action_k = motor_target_k
        With k the timestamp

Idea: add delta( com - zmp)

Action Space: gaussian 0 mean, std 1.0 

Rewards: should be the same optimized by MPC 
         suggested by the paper: A Modular Residual Learning Framework to Enhance Model-Based Approach for Robust Locomotion

    - alive
    - tracking vx, vy, omega

    - stand still

'''


DEFAULT_CAMERA_CONFIG = {
    "distance": 10,
    "elevation": -20,
    "azimuth": 135,
}

def default_consts() -> config_dict.ConfigDict:
    return config_dict.create(
        FEET_SITES = [
            "left_leg_4_site",
            "right_leg_4_site",
        ],

        LEFT_FEET_SITES = "left_leg_4_site",
        RIGHT_FEET_SITES = "right_leg_4_site",
        LEFT_FEET_POS = "left_leg_4_site_pos",
        RIGHT_FEET_POS = "right_leg_4_site_pos",

        LEFT_FEET_GEOMS = [
            "left_leg_4_collision",
        ],

        RIGHT_FEET_GEOMS = [
            "right_leg_4_collision",
        ],

        FEET_POS_SENSOR =[
            "left_leg_4_site_pos",
            "right_leg_4_site_pos",
        ],
                        
        ROOT_BODY = "base_link",

        LOCAL_ANGVEL_SENSOR = "local_angvel",
        LOCAL_LINVEL_SENSOR = "local_linvel",
        LOCAL_LINACC_SENSOR = "local_linacc",

        UPVECTOR_SENSOR = "upvector",
        FORWARD_VECTOR = "forward_vector",

        GLOBAL_LINVEL_SENSOR = "global_linvel",
        GLOBAL_ANGVEL_SENSOR = "global_angvel",

        TITA_WHEEL_INDICES = np.array([3, 7]),

    )

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      #ctrl_dt=0.02,
      #sim_dt=0.004,
      use_controller = True,
      frame_skip=1,
      randomize_on_reset=False,
      frame_stack=1,
      action_repeat=1,
      action_scale=5.0,
      soft_joint_pos_limit_factor=0.95,
      min_height=0.33,
      max_height=0.49,
      soft_height_limit_factor=0.5,
      reset_noise_scale=0.01,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      reward_config=config_dict.create(
          total_scaling=0.002, #0.002,
          scales=config_dict.create(
              # Standard robotic-specific shaping reward
              #reward_tracking_pose=1.0,
              #reward_tracking_orientation=1.0,
              reward_tracking_lin_vel=1, 
              reward_tracking_ang_vel=0.5,
              reward_tracking_height=1,
              cost_lin_vel_z=-0,
              cost_ang_vel_xy=-0.01,
              #cost_joint_motion=-0.2,
              #cost_joint_torques=-0.00001,

              cost_action_nn=-0.0005,
              cost_action_rate=-0.005, # aumentare
              cost_action_rate_second_order=-0.0000,

              # Custom rewards
              cost_total_torque=-0.0001,
              cost_total_energy=-0.0001,
              cost_total_torque_smoother=0.0,
              cost_orientation=-5.0,
              cost_early_termination=-100,
              cost_joint_pos_home = -0.1,

              #cost_com_projection=-1,
              #cost_feet_height=-1,
              #cost_contact_forces=-0.000004,
              #reward_pose=0.0,
              #cost_stand_still=-0.01
              #reward_tracking_mpc_com_pos=0,
              #reward_tracking_mpc_com_vel=0.5,
              #reward_tracking_mpc_com_acc=0.1,
              #reward_tracking_mpc_feet_pos=0,
              #cost_touch_grund=-10.0,
              #cost_feet_air=-100.0,
              #cost_energy=-0.0000001,
              #collision=0.0,
              #cost_dof_pos_limits=-0.1,
              #cost_joint_effort_limits=-0.0,
          ),
          tracking_sigma=0.25,  
          base_height_target=0.4,
          home_config = [0.0, 0.48, -0.98, 0.0, 0.0, 0.48, -0.98, 0.0],
      ),
      pert_config=config_dict.create(
          enable=True,
          #max_force=80.0,
          max_force_x=120.0,
          max_force_y=120.0,
          max_force_z=0.0,
          velocity_kick=[0.0, 3.0],  # range for random sampling
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
          # Fixed-frame mode: when True, `fixed_wait_steps` and `fixed_duration_steps`
          # (expressed in simulation steps/frames) override the sampled seconds.
          fixed_perturbation=True,
          first_perturb = 100,
          fixed_wait_steps=200,
          fixed_duration_steps=200,
          perturbation_directions={
                "0": np.array([1, 0, 0]),   # forward
                "1": np.array([-1, 0, 0]),  # backward
                "2": np.array([0, 1, 0]),   # left
                "3": np.array([0, -1, 0]),  # right
          },
          max_perturbations=0,
      ),
      # Command on cartesian space velocities: vx, wz
      command_config=config_dict.create( 
          v_max = 0.5, # max linear velocity
          omega_max = 0.3, # max angular velocity
          vz_max = 0.1, # max vertical velocity

          a=[1.5, 0.0], # Uniform distribution for command amplitude.
          b=[0.9, 0.0], # Probability of not zeroing out new command.
      ),
  )

class TitaEnv(MujocoEnv, utils.EzPickle):
    """
    Gymnasium environment for the Tita Wheel-Legged Robot.
    
    The robot can be controlled to track joystick commands (linear and angular velocity).
    The environment provides sensor readings and rewards for task completion.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 500,
    }

    def __init__(
        self,
        xml_file: str = "/home/ubuntu/Desktop/repo_rl/TITA-dynamic-obstacle-avoidance/TITA_MJ/tita_mj_description/tita_world.xml", #"None",
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        config: Optional[config_dict.ConfigDict] = default_config(),
        consts: Optional[config_dict.ConfigDict] = default_consts(),
        **kwargs,
    ):
        """
        Initialize the Tita environment.
        
        Args:
            xml_file: Name of the MuJoCo XML model file
            frame_skip: Number of MuJoCo simulation steps per environment step
            default_camera_config: Camera configuration dict
            config: configuration of the enviroment
            consts: constant for the robot and sensors
        """
        self._config = config
        self._consts = consts

        utils.EzPickle.__init__(
            self,
            xml_file,
            default_camera_config,
            self._config,
            self._consts,
            **kwargs,
        )       
        self.action_scale = self._config.action_scale
        self._reset_noise_scale = self._config.reset_noise_scale

        #seed = 42
        #torch.manual_seed(seed) 
        #np.random.seed(seed)

        self.task_to_execute = kwargs.pop("task_to_execute")
        
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=self._config.frame_skip,
            observation_space=None,  # Will be set below
            default_camera_config=default_camera_config,
            **kwargs,
        )

        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        keep_idx = [0,1,2,4,5,6]  
        low, high = bounds[keep_idx].T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        v_max = abs(self._config.command_config.v_max)
        omega_max = abs(self._config.command_config.omega_max)
        vz_max = abs(self._config.command_config.vz_max)

        # mapping from task index to desired command
        map_task_list = [
            #{"v": 0.0,       "omega": 0.0,         "vz": 0.0, },       # stand still
            {"v": v_max,     "omega": 0.0,         "vz": 0.0, },       # forward
            #{"v": -v_max,    "omega": 0.0,         "vz": 0.0, },       # backward
            #{"v": 0.0,       "omega": omega_max,   "vz": 0.0, },       # turn on place
            #{"v": v_max,     "omega": omega_max,   "vz": 0.0, },       # turn right
            #{"v": v_max,     "omega": -omega_max,  "vz": 0.0, },       # turn left
            #{"v": 0.0,       "omega": 0.0,         "vz": -vz_max, },   # crouch 
            #{"v": v_max,     "omega": 0.0,         "vz": -vz_max, },   # crouch forward
            #{"v": -v_max,    "omega": 0.0,         "vz": vz_max, },    # crouch backward
        ]

        self.map_task = {i: task for i, task in enumerate(map_task_list)}
        for task in self.map_task.values():
            task["v"]     = np.clip(task["v"], -v_max, v_max)
            task["omega"] = np.clip(task["omega"], -omega_max, omega_max)
            task["vz"]    = np.clip(task["vz"], -vz_max, vz_max)

            task.setdefault("jump_info", {"perform_jump_routine": False, "h_jump": 0.15, "start_jump_at": 100})
            task.setdefault("terrain", "floor")
        
        self.n_frame = 0
        np.set_printoptions(precision=3, suppress=True)

        #print(f"MuJoCo Timestep: {self.model.opt.timestep}")
        #print(f"Frame Skip: {self.frame_skip}")
        #print(f"Environment DT: {self.dt}")
        #print(f"Command Frequency: {1/self.dt} Hz")

        self.check_scales_sign()
        self._post_init()
        #print(self.model.opt.timestep ) # 0.002
        #self.model.opt.timestep = self._config.sim_dt

        self.reset_model()

    # =========== Sensor readings ===========

    def check_scales_sign(self):
        for k, v in self._config.reward_config.scales.items():
            if k.startswith("reward_") and v < 0:
                raise ValueError(f"Reward with negative scale: {k}, {v}")
            elif k.startswith("cost_") and v > 0:
                raise ValueError(f"Cost with positive scale: {k}, {v}")
            elif not (k.startswith("reward_") or k.startswith("cost_")):
                raise ValueError(f"Invalid reward/cost name: {k}, {v}. Should start with 'reward_' or 'cost_'")

    def get_config(self) -> config_dict.ConfigDict:
        return self._config

    def get_num_tasks(self) -> int:
        return len(list(self.map_task.keys()))

    def get_obs_info(self) -> Any:
        obs_slices = {}
        start_idx = 0
        for key, val in self.obs_dict.items():
            arr = np.asarray(val)
            obs_slices[key] = slice(start_idx, start_idx + arr.size)
            start_idx += arr.size

        return self.obs_dict, obs_slices

    def get_sensor_data(
        self, model: mujoco.MjModel, data: mujoco.MjData, sensor_name: str
    ) -> np.ndarray:
        """Gets sensor data given sensor name."""
        sensor_id = model.sensor(sensor_name).id
        sensor_adr = model.sensor_adr[sensor_id]
        sensor_dim = model.sensor_dim[sensor_id]
        return data.sensordata[sensor_adr : sensor_adr + sensor_dim]

    def get_upvector(self, data: mujoco.MjData) -> np.ndarray:
        return self.get_sensor_data(self.model, data, self._consts.UPVECTOR_SENSOR)
    
    def get_gravity(self, data: mujoco.MjData) -> np.ndarray:
        return data.site_xmat[self._imu_site_id].reshape(3,3).T @ np.array([0, 0, 1])

    def get_global_linvel(self, data: mujoco.MjData) -> np.ndarray:
        return self.get_sensor_data(
            self.model, data, self._consts.GLOBAL_LINVEL_SENSOR
        )

    def get_global_angvel(self, data: mujoco.MjData) -> np.ndarray:
        return self.get_sensor_data(
            self.model, data, self._consts.GLOBAL_ANGVEL_SENSOR
        )

    def get_local_linvel(self, data: mujoco.MjData) -> np.ndarray:
        return self.get_sensor_data(
            self.model, data, self._consts.LOCAL_LINVEL_SENSOR
        )

    def get_accelerometer(self, data: mujoco.MjData) -> np.ndarray:
        return self.get_sensor_data(
            self.model, data, self._consts.ACCELEROMETER_SENSOR
        )

    def get_gyro(self, data: mujoco.MjData) -> np.ndarray:
        return self.get_sensor_data(self.model, data, self._consts.LOCAL_ANGVEL_SENSOR)
    
    def get_imu_rotation_matrix_body_to_world(self, data: mujoco.MjData) -> np.ndarray:
        return data.site_xmat[self._imu_site_id].reshape(3,3)
    
    def get_feet_site_state(self, feet_site_id):
        pos = self.data.site_xpos[feet_site_id].copy() 
        v_buf, a_buf = np.zeros(6), np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, feet_site_id, v_buf, 0)
        mujoco.mj_objectAcceleration(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, feet_site_id, a_buf, 0)
        
        vel, acc = np.concatenate([v_buf[3:], v_buf[:3]]), np.concatenate([a_buf[3:], a_buf[:3]])
        return pos, vel, acc

    def compute_tita_controller_torque(self, data: mujoco.MjData) -> np.ndarray:
        robot_state = wm.robot_state_from_mujoco(self.model._address, self.data._address)
        result_update = self._walking_manager.update(robot_state )
        torque = result_update.torque
        joint_acc = result_update.acc
        mpc_solution = result_update.solution
        pinocchio_state = result_update.pinocchio_info

        torque_sorted = []
        joint_acc_sorted = []
        for joint_name in self._actuated_joint_names:
            val = torque[joint_name]
            torque_sorted.append(val)
            
            if joint_name in joint_acc:
                val_acc = joint_acc[joint_name]
            else:
                val_acc = -999.0
            joint_acc_sorted.append(val_acc)

        return np.array(torque_sorted), np.array(joint_acc_sorted), mpc_solution, pinocchio_state
    
    def _quat_mul(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Multiplies two quaternions.

        Args:
            u: (4,) quaternion (w,x,y,z)
            v: (4,) quaternion (w,x,y,z)

        Returns:
            A quaternion u * v.
        """
        return np.array([
            u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
            u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
            u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
            u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
        ])

    # =========== Environment methods ===========
    
    def _tita_controller_init(self) -> None:
        robot_state = wm.robot_state_from_mujoco(self.model._address, self.data._address)
        armatures = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            dof_adr = self.model.jnt_dofadr[i]
            if name and dof_adr >= 0:
                val = self.model.dof_armature[dof_adr]
            armatures[name] = val

        chosen_task = self.task_to_execute % len(self.map_task)

        dt = 0.002
        v_des = self.map_task[chosen_task]["v"]
        omega_des = self.map_task[chosen_task]["omega"]
        vz_des = self.map_task[chosen_task]["vz"]
        z0 = self._config.reward_config.base_height_target
        z_min = 0.25
        z_max = 0.49

        self._walking_manager = wm.WalkingManager()

        try:
            self._walking_planner = wm.WalkingPlanner(dt, v_des, omega_des, vz_des, z0, z_min, z_max)
        except Exception:
            try:
                self._walking_planner = wm.WalkingPlanner(v_des, omega_des, vz_des, z0, z_min, z_max)
            except Exception:
                raise RuntimeError("Failed to initialize the walking planner with the given parameters.")


        #command0 = self._get_current_planner_command(self.n_frame)
        command0 = {
            "v_des": v_des,
            "omega_des": omega_des,
            "vz_des": vz_des
        }

        perform_jump_routine = self.map_task[chosen_task]["jump_info"]["perform_jump_routine"]
        h_jump = self.map_task[chosen_task]["jump_info"]["h_jump"]
        start_jump_at = self.map_task[chosen_task]["jump_info"]["start_jump_at"]

        try:
            res_init = self._walking_manager.init(robot_state, armatures, self._walking_planner, perform_jump_routine, h_jump, start_jump_at)
        except Exception:
            try:
                res_init = self._walking_manager.init(robot_state, armatures, self._walking_planner)
            except Exception:
                raise RuntimeError("Failed to initialize the walking manager with the given robot state and planner.")
            
        self._walking_planner = self._walking_manager.get_walking_planner()
        self.info_planner = self._walking_planner.get_variables()

        return res_init, command0
    
    def _get_current_planner_command(self, current_frame) -> np.ndarray:
        time_ms = int( current_frame * self.dt * 1000 )
        command = self._walking_planner.get_uref_at_time_ms(time_ms).ravel()
        vx = command[0]
        omega = command[1]
        vz = command[2]
        return np.array([vx, omega, vz])

    def _post_init(self) -> None:

        self._init_model = copy.deepcopy(self.model)
        self._init_qpos = np.array(self.model.keyframe("home").qpos.copy())
        self._default_pose = np.array(self.model.keyframe("home").qpos[7:].copy())

        self._actuated_joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.actuator_trnid[i, 0])
            for i in range(self.model.nu)
        ]

        self.robot_mass = 0.0
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            mass = self.model.body_mass[i]

            if name == "world": continue
            self.robot_mass += mass
        
        # Setup Indices and IDs
        self._torso_body_id = self.model.body(self._consts.ROOT_BODY).id
        self._torso_mass = self.model.body_subtreemass[self._torso_body_id]
    
        self._imu_site_id = self.model.site("imu").id
        self._left_feet_site_id = self.model.site(self._consts.LEFT_FEET_SITES).id
        self._right_feet_site_id = self.model.site(self._consts.RIGHT_FEET_SITES).id
        self._feet_site_id = np.array( [self.model.site(name).id for name in self._consts.FEET_SITES])
        
        self._torso_geom_id = self.model.geom("base_link_collision").id
        self._floor_geom_id = self.model.geom("floor").id
        self._feet_geom_id = np.array( [self.model.geom(name).id for name in (self._consts.LEFT_FEET_GEOMS + self._consts.RIGHT_FEET_GEOMS)] )
        
        try:
            self._pert_geom_id = self.model.geom('perturbation_visual').id
        except Exception:
            self._pert_geom_id = None

        # Limit DOFs, exclude body ( free joint ) and wheels
        jnt_range_reduced = np.delete(self.model.jnt_range, np.array([0, 4, 8]), axis=0)

        self._lowers, self._uppers = jnt_range_reduced.T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        max_height = self._config.max_height
        min_height = self._config.min_height
        height_mean = (max_height + min_height) / 2
        height_range = max_height - min_height
        self._height_lower_bound = height_mean - 0.5 * height_range * self._config.soft_height_limit_factor
        self._height_upper_bound = height_mean + 0.5 * height_range * self._config.soft_height_limit_factor

        left_initial_pos = self.get_feet_site_state(self._left_feet_site_id)[0]
        right_initial_pos = self.get_feet_site_state(self._right_feet_site_id)[0]
        self._initial_feet_distance = np.linalg.norm(left_initial_pos - right_initial_pos)

        # Sensors
        foot_linvel_sensor_adr = []
        for site in self._consts.FEET_SITES:
            sensor_id = self.model.sensor(f"{site}_global_linvel").id
            sensor_adr = self.model.sensor_adr[sensor_id]
            sensor_dim = self.model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = np.array(foot_linvel_sensor_adr)

        self._cmd_a = np.array(self._config.command_config.a)
        self._cmd_b = np.array(self._config.command_config.b)

        obs_size = 46 #* self._config.frame_stack
        self.observation_space = Box( low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32 )

        self.history_obs = deque(maxlen=self._config.frame_stack )
        self.history_act = deque(maxlen=self._config.frame_stack )

        res_init, command0 = self._tita_controller_init()

        # info used to be propagated
        self.info = {
            "command": command0,
            "pinocchio_state":{
                "p_com": res_init.pinocchio_info.p_com,
                "v_com": res_init.pinocchio_info.v_com,
                "a_com": res_init.pinocchio_info.a_com,
                "right_rcp": res_init.pinocchio_info.right_rcp,
                "left_rcp": res_init.pinocchio_info.left_rcp,
                "right_contact": res_init.pinocchio_info.right_contact,
                "left_contact": res_init.pinocchio_info.left_contact,
            },
            "prev_nn_act": np.zeros(self.model.nu),
            "prev_motor_act": np.zeros(self.model.nu),
            "prev_prev_nn_act": np.zeros(self.model.nu),
            "prev_prev_motor_act": np.zeros(self.model.nu),
            "tita_controller_output" : np.zeros(self.model.nu),
            "total_applied_torque" : np.zeros(self.model.nu),
            "mpc_sol_com_pos" : np.zeros(3),
            "mpc_sol_com_vel" : np.zeros(3),
            "mpc_sol_com_acc" : np.zeros(3),
            "mpc_sol_pl_pos" : np.zeros(3),
            "mpc_sol_pl_vel" : np.zeros(3),
            "mpc_sol_pl_acc" : np.zeros(3),
            "mpc_sol_pr_pos" : np.zeros(3),
            "mpc_sol_pr_vel" : np.zeros(3),
            "mpc_sol_pr_acc" : np.zeros(3),
            "mpc_sol_theta" : 0.0,
            "mpc_sol_omega" : 0.0,
            "mpc_sol_alpha" : 0.0,
            "mpc_sol_contact_force_left" : np.zeros(3),
            "mpc_sol_contact_force_right" : np.zeros(3),
            "steps_until_next_pert": 0,
            "pert_duration_seconds": 0,
            "pert_duration_steps": 0,
            "steps_since_last_pert": 0,
            "pert_steps": 0,
            "pert_dir": np.array([0.0, 0.0, 0.0]),
            "pert_mag": 0,
            "n_perturbations_applied": 0,
            "perturbing": False,
            'lin_vel_xyz_error': np.zeros(3),
            'omega_vel_error': np.zeros(3),
            'gravity_xy_error': np.zeros(2),
        }

        self._init_info = copy.deepcopy(self.info.copy())
    
    def reset_model(self):
        """Reset the environment model."""
        self.n_frame = 0
        self.model = copy.deepcopy(self._init_model)
        
        qpos = self._init_qpos.copy()
        qvel = np.zeros(self.model.nv)

        # ----- Apply randomization -----
        if self._config.randomize_on_reset == True:
            print("randomizing environment on reset...")
            # State: base position randomization
            state_base_noise = 0.01
            qpos[0:2] += self.np_random.uniform(low=-state_base_noise, high=state_base_noise, size=2)
            
            # State :base orientation randomization
            state_orientation_noise = (np.pi / 180)*15 # 15 degrees noise
            roll = self.np_random.uniform(-state_orientation_noise, state_orientation_noise)
            pitch = self.np_random.uniform(-state_orientation_noise, state_orientation_noise)
            yaw = self.np_random.uniform(-state_orientation_noise*2, state_orientation_noise*2)
            quat_roll  = np.array([ np.cos(roll/2),  np.sin(roll/2), 0, 0])
            quat_pitch = np.array([ np.cos(pitch/2), 0,   np.sin(pitch/2), 0])
            quat_yaw   = np.array([ np.cos(yaw/2),   0, 0, np.sin(yaw/2)])
            q_noise = self._quat_mul( self._quat_mul(quat_yaw, quat_pitch), quat_roll )

            qpos[3:7] = self._quat_mul(qpos[3:7], q_noise)
            
            # State: velocity randomization
            state_vel_noise = 0.01
            qvel[0:6] = state_vel_noise * self.np_random.standard_normal(6) # mean 0, std 1

            # Center of mass randomization: +U(-0.05, 0.05)
            d_com_offset = 0.05
            d_com = self.np_random.uniform(low=-d_com_offset, high=d_com_offset, size=3)
            #self.model.body_ipos[self._torso_body_id] += d_com

            # Mass: torso mass randomization: +U(-1, 1)
            d_torso_mass_offset = 1
            d_torso_mass = self.np_random.uniform(low=-d_torso_mass_offset, high=d_torso_mass_offset)
            self.model.body_mass[self._torso_body_id] += d_torso_mass

            # Mass: link mass randomization: *U(-0.1, 0.1)
            d_link_mass_percentage = 0.1
            d_link_mass = self.np_random.uniform(low=-d_link_mass_percentage, high=d_link_mass_percentage, size=(self.model.nbody-1))
            self.model.body_mass[1:] *= (1 + d_link_mass)

            # Mass: armature randomization: +U(-0.05, 0.05)
            d_armature_offset = 0.05
            d_armature = self.np_random.uniform(low=-d_armature_offset, high=d_armature_offset, size=(self.model.nv-6))
            #self.model.dof_armature[6:] += d_armature

            # Joint: frictionloss randomization: +U(-0.1, 0.1)
            d_frictionloss_offset = 0.1
            d_frictionloss = self.np_random.uniform(low=-d_frictionloss_offset, high=d_frictionloss_offset, size=(self.model.nv - 6))
            #self.model.dof_frictionloss[6:] *= (1 + d_frictionloss)

            # Friction: floor friction randomization: +U(-0.15, 0.15)
            d_floor_friction_offset = 0.15
            d_floor_friction = self.np_random.uniform(low=-d_floor_friction_offset, high=d_floor_friction_offset, size=3)
            #self.model.geom_friction[self._floor_geom_id, 0:3] *= (1 + d_floor_friction)
        
        # ----- Reset history-----
        self.info = copy.deepcopy(self._init_info.copy())
        
        # ----- Set perturbation -----
        def get_random_value(config_param):
            if isinstance(config_param, (list, np.ndarray, tuple)):
                return self.np_random.uniform(low=config_param[0], high=config_param[1])
            return config_param

        time_until_next_pert = get_random_value(self._config.pert_config.kick_wait_times)
        self.info["steps_until_next_pert"] = np.round(time_until_next_pert / self.dt ).astype(int)

        pert_duration_seconds = get_random_value(self._config.pert_config.kick_durations)
        self.info["pert_duration_seconds"] = pert_duration_seconds

        pert_duration_steps = np.round(pert_duration_seconds / self.dt ).astype(int)
        self.info["pert_duration_steps"] = pert_duration_steps
        
        pert_mag = get_random_value(self._config.pert_config.velocity_kick)
        self.info["pert_mag"] = pert_mag

        # If fixed-frame perturbation mode is enabled, override the sampled
        # seconds-based values and use explicit frame counts provided in
        # the config (`fixed_wait_steps`, `fixed_duration_steps`). This
        # makes the perturbation start every N frames and last M frames.
        if getattr(self._config.pert_config, "fixed_perturbation", False):
            self.info["steps_until_next_pert"] = int(self._config.pert_config.fixed_wait_steps)
            self.info["pert_duration_steps"] = int(self._config.pert_config.fixed_duration_steps)
            self.info["pert_duration_seconds"] = float(self.info["pert_duration_steps"] * self.dt)

        # ----- Set state -----
        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)

        # ----- Initialize controller -----
        res_init, command0 = self._tita_controller_init()
        self.info['pinocchio_state'] = {
            "p_com": res_init.pinocchio_info.p_com,
            "v_com": res_init.pinocchio_info.v_com,
            "a_com": res_init.pinocchio_info.a_com,
            "right_rcp": res_init.pinocchio_info.right_rcp,
            "left_rcp": res_init.pinocchio_info.left_rcp,
            "right_contact": res_init.pinocchio_info.right_contact,
            "left_contact": res_init.pinocchio_info.left_contact,
        }

        self.info["command"] = command0

        self.history_obs.clear()
        self.history_act.clear()

        ob = self._get_obs(self.info)
        return ob

    def step(self, action):
        """Execute one step of the environment."""
        #print(f"-----------\nframe: {self.n_frame}")
        #if self.n_frame == 0:
        #    self.reset_model()

        if self._config.pert_config.enable == True:
            #print(f"{self.n_frame}, time since last pert: {self.info['steps_since_last_pert'] * self.dt:.3f} s, steps until next pert: {self.info['steps_until_next_pert']}, pert dur steps: {self.info['pert_duration_steps']}, pert dur s: {self.info['pert_duration_seconds']:.3f} s")
            self._maybe_apply_perturbation()

        tita_controller_torque, joint_acc, mpc_solution, pinocchio_state = self.compute_tita_controller_torque(self.data) #if self.n_frame >= self.frame_threshold else [0.0]*self.model.nu
        
        if action.shape != tita_controller_torque.shape:
            action = np.insert(action, 3,0.0)
            action = np.insert(action, 7,0.0)
        scaled_action = action * self._config.action_scale

        motor_targets = tita_controller_torque + scaled_action #if self.n_frame >= self.frame_threshold else tita_controller_torque
    
        #if norm_delta  >= 30.0 and self.n_frame > 1:
        #    tita_controller_torque = self.info["tita_controller_output"]
        #    motor_targets = tita_controller_torque + scaled_action

        out_of_bound_torque_wbc = any(abs(x) >= self.model.actuator_forcerange[i][1] for i, x in enumerate(tita_controller_torque))
        if np.isnan(motor_targets).any() or out_of_bound_torque_wbc:
            self.info["has_nan"] += 1
            tita_controller_torque = self.info["tita_controller_output"]
            motor_targets = tita_controller_torque + scaled_action
            
            #print("OUT OF BOUNDS TORQUE FROM WBC!!! Using last valid torque. Count:", self.info["has_nan"])
        else:
            self.info["has_nan"] = 0

        delta = tita_controller_torque - self.info['tita_controller_output']
        delta_max = 5.0
        need_clip = np.any( np.abs(delta) > delta_max )

        if need_clip and self.n_frame > 10:
            delta_clipped = np.clip(delta, -delta_max, delta_max)
            
            tita_controller_torque = self.info["tita_controller_output"]
            motor_targets = tita_controller_torque + delta_clipped + scaled_action

        self.do_simulation(motor_targets, self._config.frame_skip)

        #print(f"{self.n_frame}, {motor_targets}")
        # Store info
        self.info["current_nn_act"] = scaled_action
        self.info["tita_controller_output"] = tita_controller_torque
        self.info["total_applied_torque"] = motor_targets
        self.info["pinocchio_state"] = {
            "p_com": pinocchio_state.p_com,
            "v_com": pinocchio_state.v_com,
            "a_com": pinocchio_state.a_com,
            "right_rcp": pinocchio_state.right_rcp,
            "left_rcp": pinocchio_state.left_rcp,
            "right_contact": pinocchio_state.right_contact,
            "left_contact": pinocchio_state.left_contact,
        }

        # update visual debug geom for perturbation (if present in model)
        try:
            self._update_perturbation_visual()
        except Exception:
            pass
        # COM
        self.info["mpc_sol_com_pos"] = list(mpc_solution.com.pos)
        self.info["mpc_sol_com_vel"] = list(mpc_solution.com.vel)
        self.info["mpc_sol_com_acc"] = list(mpc_solution.com.acc)

        # Left wheel
        self.info["mpc_sol_pl_pos"] = list(mpc_solution.pl.pos)
        self.info["mpc_sol_pl_vel"] = list(mpc_solution.pl.vel)
        self.info["mpc_sol_pl_acc"] = list(mpc_solution.pl.acc)

        # Right wheel
        self.info["mpc_sol_pr_pos"] = list(mpc_solution.pr.pos)
        self.info["mpc_sol_pr_vel"] = list(mpc_solution.pr.vel)
        self.info["mpc_sol_pr_acc"] = list(mpc_solution.pr.acc)

        # Contact forces
        cf_left = np.array(mpc_solution.contact_force_left)
        cf_right = np.array(mpc_solution.contact_force_right)
        cf_left = np.nan_to_num(cf_left, nan=0.0)
        cf_right = np.nan_to_num(cf_right, nan=0.0)
        self.info["mpc_sol_contact_force_left"] = cf_left.tolist()
        self.info["mpc_sol_contact_force_right"] = cf_right.tolist()    

        # Other scalar parameters
        self.info["mpc_sol_theta"] = mpc_solution.theta
        self.info["mpc_sol_omega"] = mpc_solution.omega
        self.info["mpc_sol_alpha"] = mpc_solution.alpha

        observation = self._get_obs(self.info)

        if np.isnan(observation).any():
            print(observation)
            raise RuntimeError(f"ERROR: NaN in observation at frame {self.n_frame}!!!")

        #print(f"-------\nFrame: {self.n_frame},")
        #print(f"\tlin_vel: {self.get_local_linvel(self.data)}, norm: {np.linalg.norm(self.get_local_linvel(self.data))},  ")
        #print(f"\tang_vel: {self.get_gyro(self.data)}, norm: {np.linalg.norm(self.get_gyro(self.data))} ")

        self.dbg = 0
        reward, reward_info = self._get_rew(motor_torque=motor_targets, action=action, scaled_action=scaled_action, data=self.data, info=self.info)
        terminated = self._is_terminated(motor_targets, self.data)
        info_reward = {
            **reward_info,
        }

        if self.dbg == 1:# and (self.n_frame % 100 == 0 or ( (1 + self.n_frame) % 100 == 0) ):
            print("------------------------------------------")
            print(f"frame: {self.n_frame}, {action}")
            print({k: f"{v:.3f}" for k, v in reward_info.items()})
            print(f"motor_targets: {motor_targets}")

        if self.render_mode == "human":
            self.render()

        self.info["prev_prev_nn_act"] = self.info["prev_nn_act"]
        self.info["prev_prev_motor_act"] = self.info["prev_motor_act"]
        self.info["prev_nn_act"] = action
        self.info["prev_motor_act"] = motor_targets

        info_reward['n_frame'] = self.n_frame

        if 'info' not in info_reward:
            info_reward['info'] = {}

        info_reward['info'].update(copy.deepcopy(self.info))
        self.n_frame += 1

        return observation, reward, terminated, False, info_reward

    def _get_obs(self, info: dict[str, Any],) -> np.ndarray:
        """Get the current observation."""
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # Extract gravity from the IMU frame
        imu_xmat = self.data.site_xmat[self._imu_site_id].reshape(3, 3)

        # Variable definition for readability
        com_position = self.info['pinocchio_state']['p_com']  #self.data.subtree_com[0].copy()
        com_height_wrt_ground = com_position[2].copy().reshape(1,)

        orientation = qpos[3:7]

        gravity_vector = self.get_gravity(self.data)
        gravity_body_frame = gravity_vector/np.linalg.norm(gravity_vector)

        linvel = self.get_sensor_data(self.model, self.data, self._consts.LOCAL_LINVEL_SENSOR)
        linacc = self.get_sensor_data(self.model, self.data, self._consts.LOCAL_LINACC_SENSOR)
        angvel = self.get_sensor_data(self.model, self.data, self._consts.LOCAL_ANGVEL_SENSOR)
        command = np.array([info["command"]['v_des'], info["command"]['omega_des'], info["command"]['vz_des']]).reshape(3,)

        joint_angles = qpos[7:]
        joint_vel = qvel[6:]

        mpc_sol_com_pos = self.info["mpc_sol_com_pos"] 
        mpc_sol_com_vel = self.info["mpc_sol_com_vel"] 
        mpc_sol_com_acc = self.info["mpc_sol_com_acc"] 
        mpc_sol_pl_pos = self.info["mpc_sol_pl_pos"]
        mpc_sol_pl_vel = self.info["mpc_sol_pl_vel"]
        mpc_sol_pl_acc = self.info["mpc_sol_pl_acc"]
        mpc_sol_pr_pos = self.info["mpc_sol_pr_pos"]
        mpc_sol_pr_vel = self.info["mpc_sol_pr_vel"]
        mpc_sol_pr_acc = self.info["mpc_sol_pr_acc"]

        if np.linalg.norm(self.info["mpc_sol_contact_force_left"]) > 0:
            mpc_contact_force_left = self.info["mpc_sol_contact_force_left"]/np.linalg.norm(self.info["mpc_sol_contact_force_left"])
        else:
            mpc_contact_force_left = self.info["mpc_sol_contact_force_left"]
        
        if np.linalg.norm(self.info["mpc_sol_contact_force_right"]) > 0:
            mpc_contact_force_right = self.info["mpc_sol_contact_force_right"]/np.linalg.norm(self.info["mpc_sol_contact_force_right"])
        else:
            mpc_contact_force_right = self.info["mpc_sol_contact_force_right"]
        
        feet_state  = self.get_feet_site_state(self._left_feet_site_id)[0]
        feet_state  = self.get_feet_site_state(self._right_feet_site_id)[0]

        left_feet_pos = self.get_feet_site_state(self._left_feet_site_id)[0]
        left_feet_linvel = self.get_feet_site_state(self._left_feet_site_id)[1][:3]
        left_feet_angvel = self.get_feet_site_state(self._left_feet_site_id)[1][3:]
        left_feet_linacc = self.get_feet_site_state(self._left_feet_site_id)[2][:3]
        left_feet_angacc = self.get_feet_site_state(self._left_feet_site_id)[2][3:]

        right_feet_pos = self.get_feet_site_state(self._right_feet_site_id)[0]
        right_feet_linvel = self.get_feet_site_state(self._right_feet_site_id)[1][:3]
        right_feet_angvel = self.get_feet_site_state(self._right_feet_site_id)[1][3:]
        right_feet_linacc = self.get_feet_site_state(self._right_feet_site_id)[2][:3]
        right_feet_angacc = self.get_feet_site_state(self._right_feet_site_id)[2][3:]

        mpc_sol_theta = np.array(self.info["mpc_sol_theta"]).reshape(1,)
        mpc_sol_omega = np.array(self.info["mpc_sol_omega"]).reshape(1,)
        mpc_sol_alpha = np.array(self.info["mpc_sol_alpha"]).reshape(1,)
        joint_torque_controller_normalized = info["tita_controller_output"]  / abs(self.model.actuator_forcerange[:, 1])
        total_applied_torque = info["total_applied_torque"] / abs(self.model.actuator_forcerange[:, 1])
        prev_nn_act = info["prev_nn_act"]
        
        self.obs_dict = {
            "com_height_wrt_ground" : com_height_wrt_ground,
            "gravity_body_frame" : gravity_body_frame,
            "linvel" : linvel,
            "angvel" : angvel,
            #"linacc" : linacc,
            "joint_angles" : joint_angles,
            "joint_vel" : joint_vel,
            "command" : command,

            #"delta_mpc_sol_com_pos" : mpc_sol_com_pos,
            "mpc_sol_com_vel" : mpc_sol_com_vel, # - linvel,
            #"mpc_sol_com_acc" : mpc_sol_com_acc,
            #"mpc_sol_pl_pos" : mpc_sol_pl_pos,
            "mpc_sol_pl_vel" : mpc_sol_pl_vel,
            #"mpc_sol_pl_acc" : mpc_sol_pl_acc,
            #"mpc_sol_pr_pos" : mpc_sol_pr_pos,
            "mpc_sol_pr_vel" : mpc_sol_pr_vel,
            #"mpc_sol_pr_acc" : mpc_sol_pr_acc,
            #"mpc_contact_force_left" : mpc_contact_force_left,
            #"mpc_contact_force_right" : mpc_contact_force_right,
            #"mpc_sol_theta" : np.array(mpc_sol_theta).reshape(1,),
            #"mpc_sol_omega" : np.array(mpc_sol_omega).reshape(1,),
            #"mpc_sol_alpha" : np.array(mpc_sol_alpha).reshape(1,),

            #"joint_torque_controller_normalized" : info["tita_controller_output"]  / abs(self.model.actuator_forcerange[:, 1]),
            "prev_nn_act" : prev_nn_act
            #"total_applied_torque" : total_applied_torque,
        }

        observation = np.concatenate([v for v in self.obs_dict.values()]).astype(np.float32)

        if not np.isfinite(observation).all():
            print(f"NaN or Inf in observation, {self.n_frame} ")
            print(observation)
            return np.zeros_like(observation)
        
        return observation

    def _is_terminated(self, actuator_force: np.ndarray, data: mujoco.MjData) -> bool:
        """Check if episode should terminate."""
        
        # Fall termination: check if z-axis of IMU frame is pointing down
        imu_zaxis = self.data.site_xmat[self._imu_site_id].reshape(3, 3)[2, :]
        fall = imu_zaxis[2] < -0.2

        gravity_vector = self.get_gravity(self.data)
        gravity_threshold = 0.7
        fall_lateral = abs(gravity_vector[0]) >= gravity_threshold or abs(gravity_vector[1]) >= gravity_threshold
        
        # Base hit ground
        base_collision = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2

            if (g1 == self._torso_geom_id and g2 == self._floor_geom_id) or (g2 == self._torso_geom_id and g1 == self._floor_geom_id):
                base_collision = True
                break
        
        # Feet air
        feet_touching = set()
        feet_air = False
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2

            if g1 == self._floor_geom_id:
                if g2 in self._feet_geom_id:
                    feet_touching.add(g2)
        
            elif g2 == self._floor_geom_id:
                if g1 in self._feet_geom_id:
                    feet_touching.add(g1)
                    
        target_num_feet = len(self._feet_geom_id)
        if len(feet_touching) < target_num_feet:
            feet_air = True

        nan_threshold = 100
        nan_termination = self._has_nan() >= nan_threshold

        ang_vel_too_fast = np.linalg.norm( self.get_gyro(self.data) ) > 20.0 

        ep_terminated = fall or base_collision or nan_termination or fall_lateral
        #if ep_terminated:
        #    print(f"[TERMINATION @ frame {self.n_frame}] fall={fall}, fall_lateral={fall_lateral}, base_collision={base_collision}, has_nan={self.info['has_nan']}/{nan_threshold}")

        return ep_terminated
        #return False #fe et_air or base_collision or fall

    def _has_nan(self) -> int:
        return float((self.info['has_nan'] > 0))
        return (self.info['has_nan'] > 0)*self.info['has_nan']

    def _get_rew(self,
                motor_torque: np.ndarray,
                action: np.ndarray,
                scaled_action: np.ndarray,
                data: mujoco.MjData,
                info: dict[str, Any],
                ):

        # Standard robotic-specific shaping reward
        def _reward_tracking_lin_vel(
            self,
            commands: np.ndarray,
            local_vel: np.ndarray,
        ) -> np.ndarray:
            track_x_command = self.info["command"]['v_des']
            track_y_command = 0.0  # y vel is 0
            track_z_command = self.info["command"]['vz_des']

            error_x = track_x_command - local_vel[0]
            error_y = track_y_command - local_vel[1]
            error_z = track_z_command - local_vel[2]
            lin_vel_error = np.square(error_x) + np.square(error_y)
            r_precision = np.exp(-lin_vel_error / (0.25**2))
            
            self.info['lin_vel_xyz_error'] = np.array([error_x, error_y, error_z])

            return r_precision 

        def _reward_tracking_ang_vel(
            self,
            commands: np.ndarray,
            ang_vel: np.ndarray,
        ) -> np.ndarray:
            # Tracking of angular velocity commands (yaw).
            track_omega_command = self.info["command"]['omega_des']
            ang_vel_error = np.square(track_omega_command - ang_vel[2])
            r_precision = np.exp(-ang_vel_error / (0.25**2))

            self.info['omega_vel_error'] = ang_vel_error

            return r_precision 
        
        def _reward_tracking_height(self, body_height: np.ndarray) -> np.ndarray:
            current_height = self.info["pinocchio_state"]["p_com"][2]
            #print(f"frame: {self.n_frame}, current height: {current_height}")
            target_height = self._config.reward_config.base_height_target
            error = target_height - current_height
            r_precision = np.exp(-np.square(error) / (0.02**2))  

            return r_precision
        

        def _reward_tracking_feet(self) -> np.ndarray:

            left_pos, left_vel, left_acc = self.get_feet_site_state(self._left_feet_site_id)
            right_pos, right_vel, right_acc = self.get_feet_site_state(self._right_feet_site_id)

            left_foot_pos_error = np.array(self.info["mpc_sol_pl_pos"]) - left_pos
            right_foot_pos_error = np.array(self.info["mpc_sol_pr_pos"]) - right_pos

            left_foot_vel_error = np.array(self.info["mpc_sol_pl_vel"]) - left_vel
            right_foot_vel_error = np.array(self.info["mpc_sol_pr_vel"]) - right_vel

            r_precision_vel_l =  np.exp(-np.sum(np.square(left_foot_pos_error)) / (0.25*2))
            r_precision_vel_r =  np.exp(-np.sum(np.square(right_foot_pos_error)) / (0.25*2))
            r_precision = r_precision_vel_l + r_precision_vel_r

            return r_precision 

        def _cost_vel_feet(self, commands: np.ndarray, data: mujoco.MjData) -> float:
            left_foot_pos, left_foot_vel, _ = self.get_feet_site_state(self._left_feet_site_id)
            right_foot_pos, right_foot_vel, _ = self.get_feet_site_state(self._right_feet_site_id)

            vel_l_global = left_foot_vel[:3]
            vel_r_global = right_foot_vel[:3]

            R_world2body = self.get_imu_rotation_matrix_body_to_world(data).T
            vel_l_body = R_world2body @ vel_l_global
            vel_r_body = R_world2body @ vel_r_global

            command_body = np.array([commands[0], 0.0, 0.0])

            left_error = vel_l_body - command_body
            right_error = vel_r_body - command_body

            slip_error = np.exp(- (np.sum(np.square(left_error)) + np.sum(np.square(right_error))) / (0.2*2) )

            return 1.0 - slip_error

        def _reward_tracking_pos_mpc(self) -> np.ndarray:
            com_pos_error = np.array(self.info["mpc_sol_com_pos"]) - self.data.subtree_com[0]
            r_precision =  np.exp(-np.sum(np.square(com_pos_error)) / (0.3*2))
            r_penalty = np.sum(np.abs(com_pos_error))

            if self.dbg == 1: # and (self.n_frame % 100 == 0 or ( (1 + self.n_frame) % 100 == 0) ):
                print(f" MPC com pos target: {np.array(self.info['mpc_sol_com_pos'])}, actual: {self.data.subtree_com[0]}, error: {com_pos_error}, r_precision: {r_precision:.3f}, r_penalty: {r_penalty:.3f} ")

            return r_precision - r_penalty

        def _reward_tracking_vel_mpc(self) -> np.ndarray:
            com_vel_error = np.array(self.info["mpc_sol_com_vel"]) - self.data.qvel[0:3]
            r_precision =  np.exp(-np.sum(np.square(com_vel_error)) / (self._config.reward_config.tracking_sigma**2))
            r_penalty = np.sum(np.abs(com_vel_error))

            return r_precision - r_penalty

        def _reward_tracking_acc_mpc(selFf) -> np.ndarray:
            com_acc_error = np.array(self.info["mpc_sol_com_acc"]) - self.data.qacc[0:3]
            r_precision =  np.exp(-np.sum(np.abs(com_acc_error)) / (self._config.reward_config.tracking_sigma**2))
            r_penalty = np.sum(np.abs(com_acc_error))

            return r_precision - r_penalty

        def _cost_lin_vel_z(self, global_linvel: np.ndarray) -> np.ndarray:
            # Penalize z axis base linear velocity.*
            vz = self.info["command"]["vz_des"]
            lin_vel_z_error = abs(global_linvel[2] - vz )
            r_penalty = np.square(lin_vel_z_error)
            return r_penalty

        def _cost_ang_vel_xy(self, global_angvel: np.ndarray) -> np.ndarray:
            ang_vel_error = global_angvel[:2]
            r_penalty = np.sum(np.square(ang_vel_error))
            return r_penalty

        def _cost_joint_motion(self, qvel: np.ndarray, qacc: np.ndarray) -> np.ndarray:
            # Penalize joint motion (acceleration and velocity).
            return np.sqrt(np.sum(np.square(qacc)) + np.sum(np.square(qvel)))

        def _cost_joint_torques(self, torques: np.ndarray) -> np.ndarray:
            # Penalize torques: L2 and L1 norms.
            return np.sqrt(np.sum(np.square(torques))) + np.sum(np.abs(torques))

        def _cost_action_rate(self, act: np.ndarray) -> np.ndarray:

            prev_act = info["prev_nn_act"]
            delta_act = (act - prev_act) * self._config.action_scale

            r_penalty = np.sum(np.square(delta_act))
            return r_penalty if self.n_frame > 0 else 0.0
        
        def _cost_action_rate_second_order(self, act: np.ndarray) -> np.ndarray:

            prev_act = info["prev_nn_act"]
            prev_prev_act = info["prev_prev_nn_act"]

            term = (act - 2*prev_act + prev_prev_act) * self._config.action_scale
            r_penalty = np.sum(np.square(term))
            return r_penalty if self.n_frame > 1 else 0.0
        
        def _cost_collision(self, data: mujoco.MjData) -> np.ndarray:
            # Penalize collisions of feet with the torso.
            n_collision = 0
            for i in range(data.ncon):
                c = data.contact[i]
                g1, g2 = c.geom1, c.geom2

                if (g1 in self._feet_geom_id and g2 == self._torso_geom_id) or (g2 in self._feet_geom_id and g1 == self._torso_geom_id) or \
                    (g1 == self._torso_geom_id and g2 == self._floor_geom_id) or (g2 == self._torso_geom_id and g1 == self._floor_geom_id):
            
                    n_collision += 1.0
            return n_collision

        def _reward_feet_air_time(self, air_time: np.ndarray, first_contact: np.ndarray, commands: np.ndarray ) -> np.ndarray:
            # Reward air time.
            cmd_norm = np.linalg.norm(commands)
            rew_air_time = np.sum((air_time - 0.1) * first_contact)
            rew_air_time *= cmd_norm > 0.01  # No reward for zero commands.
            return rew_air_time

        # Other reward
        def _reward_orientation(
            self, current_up_vec: np.ndarray, target_up_vec: np.ndarray
        ) -> np.ndarray:
            z_distance = np.dot(current_up_vec, target_up_vec)
            normalized_z_distance = 1.0 - z_distance

            if self.dbg == 1 and (self.n_frame % 100 == 0 or ( (1 + self.n_frame) % 100 == 0) ):
                print(f"{current_up_vec}, {target_up_vec}, {z_distance:.3f}, {normalized_z_distance:.3f} ")

            return np.exp(-normalized_z_distance / 0.2)
        
            #return np.square(normalized_dist)
        
        def _cost_orientation(self) -> np.ndarray:
            gravity_vector = self.get_gravity(self.data)
            error = gravity_vector[0:2]
            r_penalty = np.sum( np.square(  error ) )

            self.info['gravity_xy_error'] = error
            return r_penalty
        
        def _cost_com_projection(self, data: mujoco.MjData) -> np.ndarray:
            # Penalize COM projection outside the support polygon.
            left_foot_pos = self.get_feet_site_state(self._left_feet_site_id)[0]
            right_foot_pos = self.get_feet_site_state(self._right_feet_site_id)[0]
            com_pos = data.subtree_com[0]

            foot_center = (left_foot_pos + right_foot_pos) / 2
            com_offset = com_pos[0:2] - foot_center[0:2]

            #r_precision_com = np.exp(-np.sum(np.linalg.norm(com_offset)) / 0.065)
            r_precision = np.square(np.linalg.norm(com_offset))
            return r_precision
        
        def _reward_com_projection(self, data: mujoco.MjData) -> np.ndarray:
            # Reward COM projection inside the support polygon.
            left_foot_pos = self.get_feet_site_state(self._left_feet_site_id)[0]
            right_foot_pos = self.get_feet_site_state(self._right_feet_site_id)[0]
            com_pos = data.subtree_com[0]

            foot_center = (left_foot_pos + right_foot_pos) / 2
            com_offset = com_pos[0:2] - foot_center[0:2]

            r_precision_com = np.exp(-np.sum(np.linalg.norm(com_offset)) / 0.065)
            return r_precision_com
        
        def _cost_contact_forces(self, data: mujoco.MjData) -> np.ndarray:
            contact_force_left = self.info["mpc_sol_contact_force_left"]
            contact_force_right = self.info["mpc_sol_contact_force_right"]
            total_contact_force = np.array(contact_force_left) + np.array(contact_force_right)

            max_safe_force = 400
            r_penalty = np.maximum(np.sum(np.linalg.norm(total_contact_force)) - max_safe_force, 0)
            return np.square(r_penalty)
        
        def _cost_contact_forces_simple(self) -> np.ndarray:
            contact_force_left = self.info["mpc_sol_contact_force_left"]
            contact_force_right = self.info["mpc_sol_contact_force_right"]
            #robot_mass = self.model.body_mass[self.model.body_name2id("base")]

            #print(f"----\n{self.n_frame}, \ncontact_force_left: {contact_force_left}, \ncontact_force_right: {contact_force_right}")
            #print("Robot mass:", self.robot_mass)
            #print("gravity:", self.model.opt.gravity)
            normalize_factor = 1.0 / (self.robot_mass * abs(self.model.opt.gravity[2]))
            normalized_contact = normalize_factor * ( np.array(contact_force_left) + np.array(contact_force_right)) 
            
            error_xy = normalized_contact[0:2]
            error_z = max(0.0, normalized_contact[2])
            r_penalty_xy = np.sum( np.square(  error_xy ) )
            r_penalty_z = np.square( error_z )

            r_penalty = r_penalty_z
            r_clipped = np.clip(r_penalty, a_min=0.0, a_max=3.0)
            
            #print("Normalized contact force:", normalized_contact)
            #print("Contact force error:", error_xy, error_z)
            #print(r_penalty_xy, r_penalty_z)

            return r_clipped
        
        def _cost_feet_height(self, data: mujoco.MjData) -> np.ndarray:
            left_foot_pos = self.get_feet_site_state(self._left_feet_site_id)[0]
            right_foot_pos = self.get_feet_site_state(self._right_feet_site_id)[0]

            wheel_radius = 0.0925
            left_foot_height_error = np.abs(left_foot_pos[2] - wheel_radius)
            right_foot_height_error = np.abs(right_foot_pos[2] - wheel_radius)

            return left_foot_height_error + right_foot_height_error
        
        # Energy related rewards.
        def _cost_energy(self, qvel: np.ndarray, qfrc_actuator: np.ndarray) -> np.ndarray:
            # Penalize energy consumption.
            return np.sum(np.abs(qvel) * np.abs(qfrc_actuator))

        def _cost_total_energy(self, motor_torque: np.ndarray, qvel: np.ndarray ) -> np.ndarray:
            return np.sum(np.abs(qvel) * np.abs(motor_torque))
        
        def _cost_total_torque(self, motor_torque: np.ndarray) -> np.ndarray:
            r_penalty = np.sum(np.square(motor_torque))
            return r_penalty

        def _cost_action_nn(self, action: np.ndarray) -> np.ndarray:
            r_penalty = np.sum(np.square(action*self._config.action_scale))
            return r_penalty
        
        def _cost_stand_still(self, commands: np.ndarray, action: np.ndarray,) -> np.ndarray:
            cmd_norm = np.linalg.norm(commands)
            return np.sum(np.square(action)) * (cmd_norm < 0.01)
        
        def _reward_is_alive(self, ep_terminated: np.ndarray) -> np.ndarray:
            if ep_terminated == True or (self._has_nan() > 0):
                return 0.0
            else:
                return 1.0
        
        def _cost_early_termination(self, done: np.ndarray) -> np.ndarray:
            return done

        def _cost_joint_effort_limits(self, joint_torques: np.ndarray) -> np.ndarray:
            # Penalize joints if they exceed effort limits
            upper_limits = self.model.actuator_ctrlrange[:, 1]
            lower_limits = self.model.actuator_ctrlrange[:, 0]

            out_of_limits = -np.clip(joint_torques - lower_limits, None, 0.0)
            out_of_limits += np.clip(joint_torques - upper_limits, 0.0, None)

            return np.sum(out_of_limits)

        def _cost_joint_pos_limits(self, qpos: np.ndarray) -> np.ndarray:
            # Penalize joints if they cross soft limits.
            qpos_reduced = np.delete(qpos, self._consts.TITA_WHEEL_INDICES)  # exclude wheels
            out_of_limits = -np.clip(qpos_reduced - self._soft_lowers, None, 0.0)
            out_of_limits += np.clip(qpos_reduced - self._soft_uppers, 0.0, None)
            return np.sum(out_of_limits)

        def _cost_joint_pos_home(self) -> np.ndarray:
            home_configuration = np.array( self._config.reward_config.home_config )
            current_configuration = np.array(self.data.qpos[7:])

            delta_config = home_configuration - current_configuration 
            delta_reduced = np.delete( delta_config, self._consts.TITA_WHEEL_INDICES)
            r_penalty = np.sum( np.abs( delta_reduced ) )
            
            return r_penalty

        def _cost_touch_grund(self, data: mujoco.MjData) -> np.ndarray:
            base_collision = False
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1, g2 = c.geom1, c.geom2

                if (g1 == self._torso_geom_id and g2 == self._floor_geom_id) or (g2 == self._torso_geom_id and g1 == self._floor_geom_id):
                    base_collision = True
                    break

            if base_collision == True:
                return 1.0
            else:
                return 0.0
            
        def _cost_feet_distance(self, data: mujoco.MjData) -> np.ndarray:
            current_left_foot_pos = self.get_feet_site_state(self._left_feet_site_id)[0]
            current_right_foot_pos = self.get_feet_site_state(self._right_feet_site_id)[0]
            current_feet_distance = np.linalg.norm(current_left_foot_pos - current_right_foot_pos)
            feet_distance_error = np.abs(current_feet_distance - self._initial_feet_distance)

            return feet_distance_error

        """Compute reward for current step."""
        imu_orientation = data.site_xmat[self._imu_site_id].reshape(3,3)
        z_world_vector = np.array([0.0, 0.0, 1.0])
        current_up = imu_orientation @ z_world_vector

        reward = {

            # Standard robotic-specific shaping reward
            "reward_tracking_lin_vel": _reward_tracking_lin_vel(self, info["command"], self.get_local_linvel(data)),
            "reward_tracking_ang_vel": _reward_tracking_ang_vel( self, info["command"], self.get_gyro(data) ),
            "reward_tracking_height": _reward_tracking_height(self, data.subtree_com[0, 2].copy()),
            "cost_lin_vel_z": _cost_lin_vel_z(self, self.get_global_linvel(data)),
            "cost_ang_vel_xy": _cost_ang_vel_xy(self, self.get_global_angvel(data)),
            #"cost_joint_motion": _cost_joint_motion(self, data.qvel[6:], data.qacc[6:]),
            #"cost_joint_torques": _cost_joint_torques(self, data.actuator_force),
            "cost_action_nn": _cost_action_nn(self, action),
            "cost_action_rate": _cost_action_rate(self, action),
            "cost_action_rate_second_order": _cost_action_rate_second_order(self, action),
            # Other reward
            #"reward_com_projection": _reward_com_projection(self, data),
            #"cost_vel_feet": _cost_vel_feet(self, info["command"], data),
            #"cost_contact_forces_simple": _cost_contact_forces_simple(self),

            "cost_total_torque" : _cost_total_torque(self, motor_torque),
            "cost_total_energy" : _cost_total_energy(self, motor_torque, data.qvel[6:]),
            "cost_orientation": _cost_orientation(self),
            #"reward_is_alive": _reward_is_alive(self, self._is_terminated(data.actuator_force, data)),
            "cost_early_termination": _cost_early_termination(self, self._is_terminated(data.actuator_force, data)),
            #"cost_feet_height": _cost_feet_height(self, data),
            #"cost_com_projection": _cost_com_projection(self, data),
            #"cost_contact_forces": _cost_contact_forces(self, data),
            "cost_joint_pos_home": _cost_joint_pos_home(self),
            #"has_nan": self._has_nan(),

            #"cost_height": _cost_height(self, data.qpos[2]),
            #"reward_orientation": _reward_orientation(self, current_up, z_world_vector),
            #"reward_vel_com_feet": _reward_vel_com_feet(self, self.get_local_linvel(data) ),
            #"cost_touch_grund": _cost_touch_grund(self, data),
            #"cost_stand_still": _cost_stand_still(self, info["command"], action),
            #"reward_tracking_mpc_com_pos": _reward_tracking_pos_mpc(self),
            #"reward_tracking_mpc_com_vel": _reward_tracking_vel_mpc(self),
            #"reward_tracking_mpc_com_acc": _reward_tracking_acc_mpc(self),
            #"reward_tracking_mpc_feet_pos": _reward_tracking_feet_pos_mpc(self),
            #"cost_dof_pos_limits": _cost_joint_pos_limits(self, data.qpos[7:]),
            #"cost_joint_effort_limits": _cost_joint_effort_limits(self, data.actuator_force),
            #"collision": _cost_collision(self, data),
        }

        reward_info = {
            k: v * self._config.reward_config.scales[k] for k, v in reward.items()
        }

        #self.info["reward_raw"] = {
        #    'raw_' + k: v for k, v in reward.items()
        #}

        #if self.n_frame % 100 == 0:
        #    print({k: f"{v:.3f}" for k, v in reward_info.items()})
        #reward_info['cost_action_rate'] = np.clip(reward_info['cost_action_rate'], -10.0, 10.0)
        #reward_info['cost_orientation'] = np.clip(reward_info['cost_orientation'], -10.0, 10.0)

        '''
        reward_info = {}
        for k, v in reward.items():
            if k.startswith("reward_") and self._config.reward_config.scales[k] >= 0:
                reward_info[k] = reward.get(k) * self._config.reward_config.scales[k]
            elif k.startswith("cost_") and  self._config.reward_config.scales[k] <= 0:
                reward_info[k] = reward.get(k) * self._config.reward_config.scales[k]
            else:
                if k.startswith("reward_") and self._config.reward_config.scales[k] <= 0:
                    raise(f"Reward with negative scale: {k}, {self._config.reward_config.scales[k]}")
                elif k.startswith("cost_") and  self._config.reward_config.scales[k] >= 0:
                    raise(f"Cost with positive scale: {k}, {self._config.reward_config.scales[k]}")
                else:
                    raise ValueError(f"Unknown reward component: {k}, {v} {self._config.reward_config.scales[k]}")
                
        '''

        total_reward = np.clip(sum(reward_info.values()), -10000.0, 10000.0) * self._config.reward_config.total_scaling
        #termination_penalty = 0.0
        #if self._is_terminated(data.actuator_force):
        #    termination_penalty = self._config.reward_config.scales["cost_early_termination"]
        
        total_reward = total_reward #+ termination_penalty

        return total_reward, reward_info
    
    def _maybe_apply_perturbation(self):
        def gen_dir() -> np.ndarray:
            #angle = self.np_random.uniform(low=0.0, high=np.pi * 2)
            #dir_force = np.array([np.cos(angle), np.sin(angle), 0.0])
            perturbation_directions = self._config.pert_config.perturbation_directions
            key = "0" #self.np_random.choice(list(perturbation_directions.keys()))
            return perturbation_directions[key]

            #return  dir_force

        def apply_pert():
            t = self.info["pert_steps"] * self.dt
            u_t = np.sin(np.pi * t / self.info["pert_duration_seconds"])

            max_force_x = abs(self._config.pert_config.max_force_x)
            max_force_y = abs(self._config.pert_config.max_force_y)
            max_force_z = abs(self._config.pert_config.max_force_z)
            max_force = np.array([max_force_x, max_force_y, max_force_z ])

            force = max_force * u_t
            self.data.xfrc_applied[self._torso_body_id, :3] = force * self.info["pert_dir"]
           
            if self.info["pert_steps"] >= self.info["pert_duration_steps"]:
                self.info["steps_since_last_pert"]  = 0
                self.info["n_perturbations_applied"] += 1

            self.info["pert_steps"] += 1
            self.info["perturb"] = self.data.xfrc_applied.copy()[self._torso_body_id, :3]

        def wait():
            self.info["steps_since_last_pert"] += 1
            self.data.xfrc_applied[self._torso_body_id, :3] = 0.0
            self.info["perturb"] = np.zeros(3)

            if self.info["steps_since_last_pert"] >= self.info["steps_until_next_pert"]:
                self.info['pert_steps'] = 0

        if (self.n_frame >= self._config.pert_config.first_perturb 
            and self.info["steps_since_last_pert"] >= self._config.pert_config.fixed_wait_steps
            and self.info["n_perturbations_applied"] < self._config.pert_config.max_perturbations
            ):
            if self.info['perturbing'] == False:
                self.info["pert_dir"] = gen_dir()
                self.info["pert_steps"] = 0

            self.info["perturbing"] = True
            apply_pert()
        else:
            self.info["perturbing"] = False
            wait()
            
        #self._update_perturbation_visual()

    def _update_perturbation_visual(self):
        if self._pert_geom_id is None or self.info["perturbing"] == False:
            return
        
        gid = self._pert_geom_id
        f = self.info["perturb"]
        mag = np.linalg.norm(f)

        if mag < 1e-6:
            self.model.geom_rgba[gid, 3] = 0.0
            return

        # Alpha proportional to perturbation magnitude (clamped)
        max_vis_force = getattr(self._config.pert_config, "max_force", 1.0)
        alpha = float(np.clip(mag / max_vis_force, 0.05, 0.9))
        self.model.geom_rgba[gid, 3] = alpha

        # Slightly vary color intensity with magnitude while keeping a red tint
        base_color = np.array([1.0, 0.2, 0.2])
        intensity = 1.0 # 0.2 + 0.8 * alpha
        self.model.geom_rgba[gid, :3] = (intensity * base_color)

        # DEBUG: force strong visibility and ensure size is large enough
        try:
            self.model.geom_rgba[gid, 3] = max(self.model.geom_rgba[gid, 3], 0.95)
            self.model.geom_size[gid, 0] = max(self.model.geom_size[gid, 0], 0.05)
            self.model.geom_size[gid, 1] = max(self.model.geom_size[gid, 1], length)
            self.model.geom_size[gid, 2] = max(self.model.geom_size[gid, 2], 0.05)
        except Exception:
            pass

        dir = f / mag
        base_world = self.data.xpos[self._torso_body_id].copy()
        length = 0.5  # same as geom_size[1]
        center_pos_world = base_world + dir * (length / 2)

        # Place geom relative to its body (model.geom_pos is body-local)
        body_id = int(self.model.geom_bodyid[gid])
        body_pos_world = self.data.xpos[body_id].copy()
        local_pos = center_pos_world - body_pos_world
        self.model.geom_pos[gid] = local_pos

        # Compute orientation in world frame then convert to body-local
        x_axis = dir
        up = np.array([0.0, 0.0, 1.0])
        if np.allclose(x_axis, up):
            up = np.array([0.0, 1.0, 0.0])
        y_axis = np.cross(up, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)

        rot_world = np.column_stack((x_axis, y_axis, z_axis))

        # Convert world rotation to body-local rotation
        body_rot_world = self.data.xmat[body_id].reshape(3, 3)
        rot_local = body_rot_world.T @ rot_world
        rot_iso = np.eye(4)
        rot_iso[:3, :3] = rot_local
        q_local = tr.quaternion_from_matrix(rot_iso)
        self.model.geom_quat[gid] = q_local

        #self.model.geom_size[gid, 0] = 0.05  
        #self.model.geom_size[gid, 1] = length
        
        mujoco.mj_forward(self.model, self.data)


def make_tita_env(
    render_mode: Optional[str] = None,
    **kwargs
) -> TitaEnv:
    """Factory function to create a Tita environment."""
    return TitaEnv(render_mode=render_mode, **kwargs)
