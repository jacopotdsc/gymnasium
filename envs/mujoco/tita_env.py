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



Observation vector: ( take a sequence of observation ?)
- Linear velocity in body frame (x, y, z )
- angular velocity in body frame ( Gyroscope readings )
- Joint angles
- Joint velocities
- joint accelerations ?
- joint torques ?

- Desired CoM state
- footstep placement ?
- Last action taken from neural network

- User command (vx, vy, omega)

base robotic reward:
- linear velocity tracking
- angular velocity tracking
- linear velocity penalty
- angular velocity penalty
- joint motion: acc è vel
- joint torques
- action rate
- collision
- feet air time
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

        UPVECTOR_SENSOR = "upvector",
        GLOBAL_LINVEL_SENSOR = "global_linvel",
        GLOBAL_ANGVEL_SENSOR = "global_angvel",
        LOCAL_LINVEL_SENSOR = "local_linvel",
        ACCELEROMETER_SENSOR = "accelerometer",
        GYRO_SENSOR = "gyro",

        TITA_NUM_FEET = 2,
        TITA_WHEEL_INDICES = np.array([3, 7]),
        TITA_LEG_INDICES = np.array([0, 1, 2, 4, 5, 6]),

    )

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      #ctrl_dt=0.02,
      #sim_dt=0.004,
      use_controller = True,
      frame_skip=1,
      observation_state_only=True, # True only for sac
      episode_length=1000,
      deque_length=5,
      action_repeat=1,
      action_scale=20.0,
      history_len=1,
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
        torque = self._walking_manager.update(robot_state)

        torque_sorted = []
        for joint_name in self._actuated_joint_names:
            val = torque[joint_name]
            torque_sorted.append(val)
            
        return np.array(torque_sorted)
    
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

        # Observation space: 
        #   (7) body pose
        #   (3) linvel
        #   (3) gyro   
        #   (8) joint_angles
        #   (8) joint_vel
        #   (8) last_nn_act
        #   (8) tita_controller_output
        #   (8) last_motor_act
        #   (3) command
        obs_size =  7 + 3 + 3 + 8 + 8 + 8 + 3
        self.observation_space = Box( low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32 )

        self.history_obs = deque(maxlen=self._config.deque_length )
        self.history_act = deque(maxlen=self._config.deque_length )

        # info used to be propagated
        self.info = {
            "command": np.zeros(3),
            "last_nn_act": np.zeros(self.model.nu),
            "last_motor_act": np.zeros(self.model.nu),
            "last_last_nn_act": np.zeros(self.model.nu),
            "last_last_motor_act": np.zeros(self.model.nu),
            #"steps_until_next_cmd": None, #steps_until_next_cmd,
            #"feet_air_time": np.zeros(2),
            #"last_contact": np.zeros(2, dtype=bool),
            #"swing_peak": np.zeros(2),
            #"steps_until_next_pert": None, #steps_until_next_pert,
            #"pert_duration_seconds": None, #pert_duration_seconds,
            #"pert_duration": None, #pert_duration_steps,
            #"steps_since_last_pert": 0,
            #"pert_steps": 0,
            #"pert_dir": np.zeros(3),
            #"pert_mag": None, #pert_mag,
        }

        self._tita_controller_init()

    def randomize(self):

        d_pose = self.np_random.uniform(low=-0.1, high=0.1, size=2)

        deg_10 = (np.pi / 180)*10
        roll = self.np_random.uniform(-deg_10, deg_10)
        pitch = self.np_random.uniform(-deg_10, deg_10)
        yaw = self.np_random.uniform(-deg_10, deg_10)

        quat_roll  = np.array([ np.cos(roll/2),  np.sin(roll/2), 0, 0])
        quat_pitch = np.array([ np.cos(pitch/2), 0,   np.sin(pitch/2), 0])
        quat_yaw   = np.array([ np.cos(yaw/2),   0, 0, np.sin(yaw/2)])

        d_orientation = self._quat_mul( self._quat_mul(quat_yaw, quat_pitch), quat_roll )
        d_vel = np.random.uniform(low=-0.1, high=0.1) * self.np_random.standard_normal(6)

        floor_friction = np.random.uniform(low=-0.15, high=0.15, size=3)

        #return d_pose, d_orientation, d_vel
    
    def reset_model(self):
        """Reset the environment model."""
        self.n_frame = 0
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        
        qpos = self._init_qpos.copy()
        qvel = np.zeros(self.model.nv)

        # Base position randomization
        qpos[0:2] += self.np_random.uniform(low=noise_low, high=noise_high, size=2)
        
        # Base orientation randomization
        #yaw = self.np_random.uniform(-np.pi/12, np.pi/12)
        #quat_yaw = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
        #qpos[3:7] = self._quat_mul(qpos[3:7], quat_yaw)

        deg_rnd = (np.pi / 180)*5
        roll = self.np_random.uniform(-deg_rnd, deg_rnd)
        pitch = self.np_random.uniform(-deg_rnd, deg_rnd)
        yaw = self.np_random.uniform(-deg_rnd*2, deg_rnd*2)
        quat_roll  = np.array([ np.cos(roll/2),  np.sin(roll/2), 0, 0])
        quat_pitch = np.array([ np.cos(pitch/2), 0,   np.sin(pitch/2), 0])
        quat_yaw   = np.array([ np.cos(yaw/2),   0, 0, np.sin(yaw/2)])
        q_noise = self._quat_mul( self._quat_mul(quat_yaw, quat_pitch), quat_roll )

        qpos[3:7] = self._quat_mul(qpos[3:7], q_noise)
        
        # Velocity randomization
        qvel[0:6] = self._reset_noise_scale * self.np_random.standard_normal(6)
        
        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data) 
        
        self.info = {
            "command": np.zeros(3),
            "last_nn_act": np.zeros(self.model.nu),
            "last_motor_act": np.zeros(self.model.nu),
            "last_last_nn_act": np.zeros(self.model.nu),
            "last_last_motor_act": np.zeros(self.model.nu),
            "tita_controller_output" : np.zeros(self.model.nu),
            #"steps_until_next_cmd": None, #steps_until_next_cmd,
            #"feet_air_time": np.zeros(2),
            #"last_contact": np.zeros(2, dtype=bool),
            #"swing_peak": np.zeros(2),
            #"steps_until_next_pert": None, #steps_until_next_pert,
            #"pert_duration_seconds": None, #pert_duration_seconds,
            #"pert_duration": None, #pert_duration_steps,
            #"steps_since_last_pert": 0,
            #"pert_steps": 0,
            #"pert_dir": np.zeros(3),
            #"pert_mag": None, #pert_mag,
        }

        self._tita_controller_init()

        return self._get_obs(self.info)

    def step(self, action):
        """Execute one step of the environment."""

        self.frame_threshold = 1
        tita_controller_torque = self.compute_tita_controller_torque(self.data) #if self.n_frame >= self.frame_threshold else [0.0]*self.model.nu
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

        self.history_obs.append(observation)
        self.history_act.append(action)

        if self.render_mode == "human":
            self.render()
        
        self.n_frame += 1
        return observation, reward, terminated, False, info_reward

    def _get_obs(self, info: dict[str, Any],) -> np.ndarray:
        """Get the current observation."""
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        gyro = self.get_sensor_data(self.model, self.data, self._consts.GYRO_SENSOR)
        linvel = self.get_sensor_data(self.model, self.data, self._consts.LOCAL_LINVEL_SENSOR)
        
        # Extract gravity from the IMU frame
        imu_xmat = self.data.site_xmat[self._imu_site_id].reshape(3, 3)
        gravity_body = imu_xmat.T @ np.array([0, 0, 1])
        
        joint_angles = qpos[7:]
        joint_vel = qvel[6:]
        
        # Observation
        observation = np.concatenate([
            qpos[0:3],
            qpos[3:7], 
            linvel,
            gyro,
            joint_angles,
            joint_vel,
            #info["last_nn_act"],
            info["tita_controller_output"],
            #info["last_motor_act"],
            info["command"],
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

    
        return False #feet_air or base_collision or fall

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
            # Penalize early termination.
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
    
def make_tita_env(
    render_mode: Optional[str] = None,
    **kwargs
) -> TitaEnv:
    """Factory function to create a Tita environment."""
    return TitaEnv(render_mode=render_mode, **kwargs)
