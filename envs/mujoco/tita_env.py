# Copyright 2025
# Licensed under the Apache License, Version 2.0
#
# MuJoCo Tita Wheel-Legged Robot Environment for Gymnasium

import sys
import time
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
      observation_state_only=True, # True only for sac
      randomize_on_reset=False,
      apply_perturbations=False,
      episode_length=1000,
      frame_stack=1,
      action_repeat=1,
      action_scale=100.0,
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
          scales=config_dict.create(
              # Standard robotic-specific shaping reward
              #reward_tracking_pose=1.0,
              #reward_tracking_orientation=1.0,
              #reward_tracking_lin_vel=0.5, 
              #reward_tracking_ang_vel=0.5, 
              #cost_lin_vel_z=-0.0,
              #cost_ang_vel_xy=-0.0,
              #cost_joint_motion=-0.2,
              #cost_joint_torques=-0.00001,
              cost_action_rate=-1.0, 

              # Other rewards
              reward_height=1.0,
              reward_orientation=1.0,
              #cost_orientation=-0.0,
              #cost_early_termination=-5000.0,
              #cost_has_nan=-10.0,
              #reward_pose=0.0,
              
              reward_is_alive=1.0,
              cost_action_nn=-0.1,
              #cost_stand_still=-0.02,

              #cost_touch_grund=-10.0,
              #cost_feet_air=-100.0,
              #cost_energy=-0.000001,
              #collision=0.0,
              #cost_dof_pos_limits=-0.1,
              #cost_joint_effort_limits=-0.0,
          ),
          tracking_sigma=0.25,  
          base_height_target=0.40,
      ),
      pert_config=config_dict.create(
          enable=False,
          velocity_kick=[0.0, 3.0],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      # Command on cartesian space velocities: vx, vy, wz
      command_config=config_dict.create( 
          a=[1.5, 0.8, 0.0],  # Uniform distribution for command amplitude.
          b=[0.9, 0.25, 0.0], # Probability of not zeroing out new command.
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
        xml_file: str = "/home/ubuntu/Desktop/repo_rl/TITA-dynamic-obstacle-avoidance/TITA_MJ/tita_mj_description/tita.xml", #"None",
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

        seed = 42
        torch.manual_seed(seed) 
        np.random.seed(seed)
        
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=self._config.frame_skip,
            observation_space=None,  # Will be set below
            default_camera_config=default_camera_config,
            **kwargs,
        )
        
        self.n_frame = 0
        self.frame_threshold = 0 # number of frames to wait before enabling controller
        np.set_printoptions(precision=3, suppress=True)

        self._post_init()
        #print(self.model.opt.timestep ) # 0.002
        #self.model.opt.timestep = self._config.sim_dt

        self.reset_model()

    # =========== Sensor readings ===========

    def get_config(self) -> config_dict.ConfigDict:
        return self._config

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
        return data.site_xmat[self._imu_site_id].T @ np.array([0, 0, 1])

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
        return self.get_sensor_data(self.model, data, self._consts.GYRO_SENSOR)

    def get_feet_pos(self, data: mujoco.MjData) -> np.ndarray:
        return np.vstack([
            self.get_sensor_data(self.model, data, sensor_name)
            for sensor_name in self._consts.FEET_POS_SENSOR
        ])

    def compute_tita_controller_torque(self, data: mujoco.MjData) -> np.ndarray:
        robot_state = wm.robot_state_from_mujoco(self.model, data)
        result_update = self._walking_manager.update(robot_state, np.array([0.0, 0.0, 0.40]))
        torque = result_update.cmd
        mpc_solution = result_update.solution

        torque_sorted = []
        for joint_name in self._actuated_joint_names:
            val = torque[joint_name]
            torque_sorted.append(val)
            
        return np.array(torque_sorted), mpc_solution
    
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
        robot_state = wm.robot_state_from_mujoco(self.model, self.data)
        armatures = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            dof_adr = self.model.jnt_dofadr[i]
            if name and dof_adr >= 0:
                val = self.model.dof_armature[dof_adr]
            armatures[name] = val

        self._walking_manager = wm.WalkingManager()
        self._walking_manager.init(robot_state, armatures)

    def _post_init(self) -> None:

        self._init_model = copy.deepcopy(self.model)
        self._init_qpos = np.array(self.model.keyframe("home").qpos.copy())
        self._default_pose = np.array(self.model.keyframe("home").qpos[7:].copy())

        self._actuated_joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.actuator_trnid[i, 0])
            for i in range(self.model.nu)
        ]
        
        # Setup Indices and IDs
        self._torso_body_id = self.model.body(self._consts.ROOT_BODY).id
        self._torso_mass = self.model.body_subtreemass[self._torso_body_id]
        
        self._imu_site_id = self.model.site("imu").id
        self._feet_site_id = np.array( [self.model.site(name).id for name in self._consts.FEET_SITES])
        
        self._torso_geom_id = self.model.geom("base_link_collision").id
        self._floor_geom_id = self.model.geom("floor").id
        self._feet_geom_id = np.array( [self.model.geom(name).id for name in (self._consts.LEFT_FEET_GEOMS + self._consts.RIGHT_FEET_GEOMS)] )

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

        obs_size =  49 #* self._config.frame_stack
        self.observation_space = Box( low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32 )

        self.history_obs = deque(maxlen=self._config.frame_stack )
        self.history_act = deque(maxlen=self._config.frame_stack )

        # info used to be propagated
        self.info = {
            "command": np.zeros(3),
            "last_nn_act": np.zeros(self.model.nu),
            "last_motor_act": np.zeros(self.model.nu),
            "last_last_nn_act": np.zeros(self.model.nu),
            "last_last_motor_act": np.zeros(self.model.nu),
            "tita_controller_output" : np.zeros(self.model.nu),
            "mpc_sol_com_pos": np.zeros(3),
            "mpc_sol_com_vel": np.zeros(3),
            "mpc_sol_com_acc": np.zeros(3),
            "mpc_sol_pc_pos": np.zeros(3),
            "mpc_sol_pc_vel": np.zeros(3),
            "mpc_sol_pc_acc": np.zeros(3),
            "steps_until_next_pert": 0,
            "pert_duration_seconds": 0,
            "pert_duration_steps": 0,
            "steps_since_last_pert": 0,
            "pert_steps": 0,
            "pert_dir": np.array([0.0, 0.0, 1.0]),
            "pert_mag": 0,
        }

        self._init_info = copy.deepcopy(self.info.copy())

        self._tita_controller_init()
    
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
            d_com = np.random.uniform(low=-d_com_offset, high=d_com_offset, size=3)
            #self.model.body_ipos[self._torso_body_id] += d_com

            # Mass: torso mass randomization: +U(-1, 1)
            d_torso_mass_offset = 1
            d_torso_mass = np.random.uniform(low=-d_torso_mass_offset, high=d_torso_mass_offset)
            self.model.body_mass[self._torso_body_id] += d_torso_mass

            # Mass: link mass randomization: *U(-0.1, 0.1)
            d_link_mass_percentage = 0.1
            d_link_mass = np.random.uniform(low=-d_link_mass_percentage, high=d_link_mass_percentage, size=(self.model.nbody-1))
            self.model.body_mass[1:] *= (1 + d_link_mass)

            # Mass: armature randomization: +U(-0.05, 0.05)
            d_armature_offset = 0.05
            d_armature = np.random.uniform(low=-d_armature_offset, high=d_armature_offset, size=(self.model.nv-6))
            #self.model.dof_armature[6:] += d_armature

            # Joint: frictionloss randomization: +U(-0.1, 0.1)
            d_frictionloss_offset = 0.1
            d_frictionloss = np.random.uniform(low=-d_frictionloss_offset, high=d_frictionloss_offset, size=(self.model.nv - 6))
            #self.model.dof_frictionloss[6:] *= (1 + d_frictionloss)

            # Friction: floor friction randomization: +U(-0.15, 0.15)
            d_floor_friction_offset = 0.15
            d_floor_friction = np.random.uniform(low=-d_floor_friction_offset, high=d_floor_friction_offset, size=3)
            #self.model.geom_friction[self._floor_geom_id, 0:3] *= (1 + d_floor_friction)
        
        # ----- Reset history-----
        self.info = copy.deepcopy(self._init_info.copy())
        
        # ----- Set perturbation -----
        time_until_next_pert = np.random.uniform(low=self._config.pert_config.kick_wait_times[0], high=self._config.pert_config.kick_wait_times[1])
        self.info["steps_until_next_pert"] = np.round(time_until_next_pert / self.dt ).astype(int)

        pert_duration_seconds = np.random.uniform(low=self._config.pert_config.kick_durations[0], high=self._config.pert_config.kick_durations[1])
        self.info["pert_duration_seconds"] = pert_duration_seconds

        pert_duration_steps = np.round(pert_duration_seconds / self.dt ).astype(int)
        self.info["pert_duration_steps"] = pert_duration_steps
        
        pert_mag = np.random.uniform(low=self._config.pert_config.velocity_kick[0], high=self._config.pert_config.velocity_kick[1])
        self.info["pert_mag"] = pert_mag

        # ----- Set state -----
        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)

        # ----- Initialize controller -----
        self._tita_controller_init()

        self.history_obs.clear()
        self.history_act.clear()

        ob = self._get_obs(self.info)
        return ob

    def step(self, action):
        """Execute one step of the environment."""
        #print(f"Step frame: {self.n_frame}")
        if self._config.apply_perturbations == True:
            self._maybe_apply_perturbation()

        self.frame_threshold = 1
        tita_controller_torque, mpc_solution = self.compute_tita_controller_torque(self.data) #if self.n_frame >= self.frame_threshold else [0.0]*self.model.nu
        scaled_action = action*self._config.action_scale
        motor_targets = tita_controller_torque + scaled_action if self.n_frame >= self.frame_threshold else tita_controller_torque
        #time.sleep(2)

        nan_mask = np.isnan(motor_targets)
        if np.isnan(motor_targets).any():
            self.info["has_nan"] = True
            sign = np.sign(self.info["last_motor_act"][nan_mask])
            motor_targets[nan_mask] = sign * 0.0
            tita_controller_torque[nan_mask] = sign * 0.0
        else:
            self.info["has_nan"] = False

        if np.isnan(tita_controller_torque).any():
            self.info["has_nan"] = True
            tita_controller_torque = np.array([0.0]*self.model.nu)
        else:
            self.info["has_nan"] = False
        
        self.do_simulation(motor_targets, self._config.frame_skip)
        
        observation = self._get_obs(self.info)

        if np.isnan(observation).any():
            raise RuntimeError(f"ERROR: NaN in observation {self.n_frame}!!!")

        

        self.dbg = 0
        reward, reward_info = self._get_rew(action, self.data, self.info)
        terminated = self._is_terminated(motor_targets, self.data)
        info_reward = {
            **reward_info,
        }

        if self.dbg == 1 and (self.n_frame % 100 == 0 or ( (1 + self.n_frame) % 100 == 0) ):
            print("------------------------------------------")
            print(f"frame: {self.n_frame}, {action}")
            print({k: f"{v:.3f}" for k, v in reward_info.items()})

        self.info["tita_controller_output"] = tita_controller_torque
        self.info["last_last_nn_act"] = self.info["last_nn_act"]
        self.info["last_last_motor_act"] = self.info["last_motor_act"]
        self.info["last_nn_act"] = action
        self.info["last_motor_act"] = motor_targets
        self.info["mpc_sol_com_pos"] = list(mpc_solution.com.pos)
        self.info["mpc_sol_com_vel"] = list(mpc_solution.com.vel)
        self.info["mpc_sol_com_acc"] = list(mpc_solution.com.acc)
        self.info["mpc_sol_pc_pos"] = list(mpc_solution.pc.pos) if self.n_frame > 0 else np.zeros(3) # first values is broken, high values
        self.info["mpc_sol_pc_vel"] = list(mpc_solution.pc.vel)
        self.info["mpc_sol_pc_acc"] = list(mpc_solution.pc.acc)

        if self.render_mode == "human":
            self.render()

        self.n_frame += 1

        return observation, reward, terminated, False, info_reward

    def _get_obs(self, info: dict[str, Any],) -> np.ndarray:
        """Get the current observation."""
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # Extract gravity from the IMU frame
        imu_xmat = self.data.site_xmat[self._imu_site_id].reshape(3, 3)

        # Variable definition for readability
        height = np.array(qpos[2]).reshape(1,)
        orientation = qpos[3:7]
        gravity_body_frame = imu_xmat.T @ np.array([0, 0, 1])
        linvel = self.get_sensor_data(self.model, self.data, self._consts.LOCAL_LINVEL_SENSOR)
        linacc = self.get_sensor_data(self.model, self.data, self._consts.LOCAL_LINACC_SENSOR)
        joint_angles = qpos[7:]
        joint_vel = qvel[6:]
        joint_torque_controller_normalized = info["tita_controller_output"]  / abs(self.model.actuator_forcerange[:, 1])
        last_nn_act = info["last_nn_act"]
        command = info["command"]
    
        # Observation
        observation = np.concatenate([
            height,
            orientation,
            gravity_body_frame,
            linvel,
            linacc,
            joint_angles,
            joint_vel,
            joint_torque_controller_normalized,
            last_nn_act,
            command
        ]).astype(np.float32)

        if not np.isfinite(observation).all():
            print(f"NaN or Inf in observation, {self.n_frame} ")
            print(observation)
            return np.zeros_like(observation)
        
        return observation

    def _is_terminated(self, actuator_force: np.ndarray, data: mujoco.MjData) -> bool:
        """Check if episode should terminate."""
        
        # Fall termination: check if z-axis of IMU frame is pointing down
        imu_zaxis = self.data.site_xmat[self._imu_site_id].reshape(3, 3)[2, :]
        fall = imu_zaxis[2] < 0.3
        
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

        return self._has_nan() or fall 
        #return False #feet_air or base_collision or fall

    def _has_nan(self) -> bool:
        return self.info['has_nan'] == True

    def _get_rew(self, 
                action: np.ndarray,
                data: mujoco.MjData,
                info: dict[str, Any],
                ):

        # Standard robotic-specific shaping reward
        def _reward_tracking_lin_vel(
            self,
            commands: np.ndarray,
            local_vel: np.ndarray,
        ) -> np.ndarray:
            # Tracking of linear velocity commands (xy axes).
            lin_vel_error = np.sum(np.square(commands[:2] - local_vel[:2]))
            return np.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

        def _reward_tracking_ang_vel(
            self,
            commands: np.ndarray,
            ang_vel: np.ndarray,
        ) -> np.ndarray:
            # Tracking of angular velocity commands (yaw).
            ang_vel_error = np.square(commands[2] - ang_vel[2])
            return np.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

        def _cost_lin_vel_z(self, global_linvel: np.ndarray) -> np.ndarray:
            # Penalize z axis base linear velocity.
            return np.square(global_linvel[2])

        def _cost_ang_vel_xy(self, global_angvel: np.ndarray) -> np.ndarray:
            # Penalize xy axes base angular velocity.
            return np.sum(np.square(global_angvel[:2]))

        def _cost_joint_motion(self, qvel: np.ndarray, qacc: np.ndarray) -> np.ndarray:
            # Penalize joint motion (acceleration and velocity).
            return np.sqrt(np.sum(np.square(qacc)) + np.sum(np.square(qvel)))

        def _cost_joint_torques(self, torques: np.ndarray) -> np.ndarray:
            # Penalize torques: L2 and L1 norms.
            return np.sqrt(np.sum(np.square(torques))) + np.sum(np.abs(torques))

        def _cost_action_rate(self, act: np.ndarray, last_act: np.ndarray) -> np.ndarray:
            return np.sum(np.square(act - last_act))
        
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
        
        def _cost_orientation(self, torso_zaxis: np.ndarray) -> np.ndarray:
            # Penalize non flat base orientation.
            return np.sum(np.square(torso_zaxis[:2]))

        def _reward_height(self, body_height: np.ndarray) -> np.ndarray:
            error = self.init_qpos[2] - body_height 
            return np.exp(-np.square(error) / 1.0)

        # Energy related rewards.
        def _cost_energy(self, qvel: np.ndarray, qfrc_actuator: np.ndarray) -> np.ndarray:
            # Penalize energy consumption.
            return np.sum(np.abs(qvel) * np.abs(qfrc_actuator))

        def _cost_action_nn(self, action: np.ndarray) -> np.ndarray:
            return np.sum(np.abs(action))
        
        def _cost_stand_still(self, commands: np.ndarray, action: np.ndarray,) -> np.ndarray:
            cmd_norm = np.linalg.norm(commands)
            return np.sum(np.abs(action)) * (cmd_norm < 0.01)

        def _reward_is_alive(self, ep_terminated: np.ndarray) -> np.ndarray:
            if ep_terminated == True:
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
            
        # Feet related rewards.
        def _cost_feet_air(self, data: mujoco.MjData) -> np.ndarray:
            feet_touching = set()

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
                return 1.0
            else:
                return 0.0

        """Compute reward for current step."""
        imu_orientation = data.site_xmat[self._imu_site_id].reshape(3,3)
        z_world_vector = np.array([0.0, 0.0, 1.0])
        current_up = imu_orientation @ z_world_vector


        reward = {

            # Standard robotic-specific shaping reward
            #"reward_tracking_lin_vel": _reward_tracking_lin_vel(self, info["command"], self.get_local_linvel(data)),
            #"reward_tracking_ang_vel": _reward_tracking_ang_vel( self, info["command"], self.get_gyro(data) ),
            #"cost_lin_vel_z": _cost_lin_vel_z(self, self.get_global_linvel(data)),
            #"cost_ang_vel_xy": _cost_ang_vel_xy(self, self.get_global_angvel(data)),
            #"cost_joint_motion": _cost_joint_motion(self, data.qvel[6:], data.qacc[6:]),
            #"cost_joint_torques": _cost_joint_torques(self, data.actuator_force),
            "cost_action_rate": _cost_action_rate(self, action, info["last_nn_act"]),

            # Other reward
            "cost_action_nn": _cost_action_nn(self, action),
            #"cost_orientation": _cost_orientation(self, self.get_upvector(data)),
            "reward_height": _reward_height(self, data.qpos[2]),
            "reward_orientation": _reward_orientation(self, current_up, z_world_vector),
            "reward_is_alive": _reward_is_alive(self, self._is_terminated(data.actuator_force, data)),
            #"cost_early_termination": _cost_early_termination(self, self._is_terminated(data.actuator_force, data)),
            #"cost_touch_grund": _cost_touch_grund(self, data),
            #"cost_stand_still": _cost_stand_still(self, info["command"], action),
            
            #"cost_has_nan": self._has_nan(),
            #"cost_energy": _cost_energy(self, data.qvel[6:], data.qacc[6:]),
            #"cost_dof_pos_limits": _cost_joint_pos_limits(self, data.qpos[7:]),
            #"cost_joint_effort_limits": _cost_joint_effort_limits(self, data.actuator_force),
            #"collision": _cost_collision(self, data),
        }

        reward_info = {
            k: v * self._config.reward_config.scales[k] for k, v in reward.items()
        }

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

        total_reward = np.clip(sum(reward_info.values()), -10000.0, 10000.0)
        #termination_penalty = 0.0
        #if self._is_terminated(data.actuator_force):
        #    termination_penalty = self._config.reward_config.scales["cost_early_termination"]
        
        total_reward = total_reward #+ termination_penalty

        return total_reward, reward_info
    
    def _maybe_apply_perturbation(self):
        def gen_dir() -> np.ndarray:
            angle = np.random.uniform(low=0.0, high=np.pi * 2)
            dir_force = np.array([np.cos(angle), np.sin(angle), 0.0])
            return  dir_force

        def apply_pert():
            t = self.info["pert_steps"] * self.dt
            u_t = np.sin(np.pi * t / self.info["pert_duration_seconds"])
            # kg * m/s * 1/s = m/s^2 = kg * m/s^2 (N).
            max_force = 170.0
            force = max_force * u_t

            # Lateral force vector, total latera magnitude:
            #   150 N: gentle push
            #   190 N: noticeable
            #   200 N: hard, bring to NaN
            # Front-back force vector, total longitudinal magnitude:
            #   200 N: gentle push, FEASIBLE
            # Top-down force vector, total vertical magnitude:
            #   10000 N: light 
            #   11000 N: gentle
            #   15000 N: noticeable 
            #   20000 N: moderate
            #   50000 N: pretty strong
            #print(self.info["pert_dir"], force)
            self.data.xfrc_applied[self._torso_body_id, :3] = force * self.info["pert_dir"]

            if self.info["pert_steps"] >= self.info["pert_duration_steps"]:
                self.info["steps_since_last_pert"]  = 0

            self.info["pert_steps"] += 1

        def wait():
            self.info["steps_since_last_pert"] += 1
            self.data.xfrc_applied[self._torso_body_id, :3] = 0.0

            if self.info["steps_since_last_pert"] >= self.info["steps_until_next_pert"]:
                self.info['pert_steps'] = 0
                
            if self.info["steps_since_last_pert"] >= self.info["steps_until_next_pert"]:
                self.info["pert_dir"] = gen_dir()

        if self.info["steps_since_last_pert"] >= self.info["steps_until_next_pert"]:
            apply_pert()
        else:
            wait()

def make_tita_env(
    render_mode: Optional[str] = None,
    **kwargs
) -> TitaEnv:
    """Factory function to create a Tita environment."""
    return TitaEnv(render_mode=render_mode, **kwargs)
