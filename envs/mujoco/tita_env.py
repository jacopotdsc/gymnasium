# Copyright 2025
# Licensed under the Apache License, Version 2.0
#
# MuJoCo Tita Wheel-Legged Robot Environment for Gymnasium

import numpy as np
from typing import Any, Dict, Optional
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from ml_collections import config_dict

try:
    import mujoco
except ImportError:
    raise ImportError(
        "MuJoCo is required for TitaEnv. "
        "Install it with: pip install mujoco"
    )

'''
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
- joint torquest
- action rate
- collision
- feet air time
'''


DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
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
      ctrl_dt=0.02,
      sim_dt=0.004,
      frame_skip=5,
      observation_state_only=True, # True only for sac
      episode_length=1000,
      action_repeat=1,
      action_scale=1.0,
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
              # tracking
              tracking_lin_vel=1.0, 
              tracking_ang_vel=1.0, 
              # base
              lin_vel_z=0.0,
              ang_vel_xy=0.0,
              orientation=1.0,
              base_height=0.0, # work with base_height_target
              # Other.
              dof_pos_limits=-1.0,
              pose=0.0,
              # Other.
              termination=-1.0,
              stand_still=-0.0,
              # Regularization.
              torques=-0.00005,
              action_rate=-0.001, 
              energy=-0.0001,
              # Feet.
              #feet_clearance=-2.0,
              #feet_height=-0.2,
              #feet_slip=-0.1,
              #feet_air_time=0.1,
          ),
          tracking_sigma=0.25,  
          max_foot_height=0.1,
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
          a=[1.5, 0.8, 1.2],  # Uniform distribution for command amplitude.
          b=[0.9, 0.25, 0.5], # Probability of not zeroing out new command.
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
        "render_fps": 100,
    }

    def __init__(
        self,
        xml_file: str = "tita_mjx.xml",
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
        
        self._post_init()

    # ------- Sensor readings. -------

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


    # ------- Environment methods. -------

    def _post_init(self) -> None:
        self._init_qpos = np.array(self.model.keyframe("home").qpos.copy())
        self._default_pose = np.array(self.model.keyframe("home").qpos[7:].copy())

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
        
        # Observation space: linvel(3) + gyro(3) + gravity(3) + joint_angles(8) + joint_vel(8) + last_act(8) + command(3)
        obs_size = 3 + 3 + 3 + 8 + 8 + 8 + 3
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize tracking variables
        self.last_act = np.zeros(8)
        self.last_contact = np.array([False, False])
        self.command = np.zeros(3)  # [vx, vy, omega]

        # info used to be propagated
        self.info = {
            "command": np.zeros(3),
            "steps_until_next_cmd": None, #steps_until_next_cmd,
            "last_act": np.zeros(self.model.nu),
            "last_last_act": np.zeros(self.model.nu),
            #"feet_air_time": np.zeros(2),
            "last_contact": np.zeros(2, dtype=bool),
            #"swing_peak": np.zeros(2),
            "steps_until_next_pert": None, #steps_until_next_pert,
            "pert_duration_seconds": None, #pert_duration_seconds,
            "pert_duration": None, #pert_duration_steps,
            "steps_since_last_pert": 0,
            "pert_steps": 0,
            "pert_dir": np.zeros(3),
            "pert_mag": None, #pert_mag,
        }

    def reset_model(self):
        """Reset the environment model."""
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        
        qpos = self._init_qpos.copy()
        qvel = np.zeros(self.model.nv)
        
        # Base position randomization
        qpos[0:2] += self.np_random.uniform(low=noise_low, high=noise_high, size=2)
        
        # Base orientation randomization
        yaw = self.np_random.uniform(-np.pi, np.pi)
        quat_yaw = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
        qpos[3:7] = self._quat_mul(qpos[3:7], quat_yaw)
        
        # Velocity randomization
        qvel[0:6] = self._reset_noise_scale * self.np_random.standard_normal(6)
        
        self.set_state(qpos, qvel)
        
        # Reset tracking variables
        self.last_act = np.zeros(8)
        self.last_contact = np.array([False, False])
        self.command = np.zeros(3)

        self.info = {
            "command": np.zeros(3),
            "steps_until_next_cmd": None, #steps_until_next_cmd,
            "last_act": np.zeros(self.model.nu),
            "last_last_act": np.zeros(self.model.nu),
            #"feet_air_time": np.zeros(2),
            "last_contact": np.zeros(2, dtype=bool),
            #"swing_peak": np.zeros(2),
            "steps_until_next_pert": None, #steps_until_next_pert,
            "pert_duration_seconds": None, #pert_duration_seconds,
            "pert_duration": None, #pert_duration_steps,
            "steps_since_last_pert": 0,
            "pert_steps": 0,
            "pert_dir": np.zeros(3),
            "pert_mag": None, #pert_mag,
        }
        
        return self._get_obs()

    def step(self, action):
        """Execute one step of the environment."""

        # Set motor targets
        motor_targets = self._default_pose + action * self.action_scale
        
        self.do_simulation(motor_targets, self._config.frame_skip)
        
        observation = self._get_obs()
        reward, reward_info = self._get_rew(action, self.data, self.info)
        terminated = self._is_terminated()
        info = {
            "base_height": self.data.qpos[2],
            **reward_info,
        }
        
        self.last_act = action
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, False, info

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

    def _get_obs(self):
        """Get the current observation."""
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Extract sensor data
        sensor_data = self.data.sensordata.copy()
        
        gyro = sensor_data[0:3]  # gyro
        linvel = sensor_data[3:6]  # local linear velocity
        
        # Extract gravity from the IMU frame
        imu_xmat = self.data.site_xmat[self._imu_site_id].reshape(3, 3)
        gravity_body = imu_xmat.T @ np.array([0, 0, 1])
        
        joint_angles = qpos[7:]
        joint_vel = qvel[6:]
        
        # Observation
        
        observation = np.concatenate([
            linvel,
            gyro,
            gravity_body,
            joint_angles,
            joint_vel,
            self.last_act,
            self.command,
        ]).astype(np.float32)
        
        return observation

    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Fall termination: check if z-axis of IMU frame is pointing down
        imu_zaxis = self.data.site_xmat[self._imu_site_id].reshape(3, 3)[2, :]
        fall = imu_zaxis[2] < 0.0
        
        # Base hit ground
        base_collision = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2

            if (g1 == self._torso_geom_id and g2 == self._floor_geom_id) or (g2 == self._torso_geom_id and g1 == self._floor_geom_id):
                base_collision = True
                break
    
        # NaN check
        has_nan = np.isnan(qpos).any() or np.isnan(qvel).any()
        
        return fall or base_collision or has_nan

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

        def _cost_joint_motion(
                self, qvel: np.ndarray, qacc: np.ndarray
            ) -> np.ndarray:
            # Penalize joint motion (acceleration and velocity).
            return np.sum(np.square(qacc)) + np.sum(np.square(qvel))

        def _cost_joint_torques(self, torques: np.ndarray) -> np.ndarray:
            # Penalize torques: L2 and L1 norms.
            return np.sqrt(np.sum(np.square(torques))) + np.sum(np.abs(torques))

        def _cost_action_rate(
            self, act: np.ndarray, last_act: np.ndarray, last_last_act: np.ndarray
        ) -> np.ndarray:
            del last_last_act  # Unused.
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

        def _reward_feet_air_time(
            self, air_time: np.ndarray, first_contact: np.ndarray, commands: np.ndarray
        ) -> np.ndarray:
            # Reward air time.
            cmd_norm = np.linalg.norm(commands)
            rew_air_time = np.sum((air_time - 0.1) * first_contact)
            rew_air_time *= cmd_norm > 0.01  # No reward for zero commands.
            return rew_air_time

        # Other reward
        def _reward_orientation(
            self, current_up_vec: np.ndarray, target_up_vec: np.ndarray
        ) -> np.ndarray:
            cos_dist = np.dot(current_up_vec, target_up_vec)
            normalized = 0.5 * cos_dist + 0.5
            return np.square(normalized)
        
        def _cost_orientation(self, torso_zaxis: np.ndarray) -> np.ndarray:
            # Penalize non flat base orientation.
            return np.sum(np.square(torso_zaxis[:2]))

        def _reward_height(self, body_height: np.ndarray) -> np.ndarray:
            error = self.init_qpos[2] - body_height 
            return np.exp(-error / 1.0)
        
        # define a reward height based on how much it is close to constraints
        def _cost_height(self, body_height: np.ndarray) -> np.ndarray:
            # Penalize height outside of bounds.
            out_of_limits = -np.clip(body_height - self._height_lower_bound, None, 0.0)
            out_of_limits += np.clip(body_height - self._height_upper_bound, 0.0, None)
            return np.sum(out_of_limits)

        # Energy related rewards.
        def _cost_energy(
            self, qvel: np.ndarray, qfrc_actuator: np.ndarray
        ) -> np.ndarray:
            # Penalize energy consumption.
            return np.sum(np.abs(qvel) * np.abs(qfrc_actuator))

        # Other rewards.
        def _reward_pose(self, qpos: np.ndarray) -> np.ndarray:
            # Stay close to the default pose.
            weight = np.array([1.0, 1.0, 1.0, 0.0] * 2)
            return np.exp(-np.sum(np.square(qpos - self._default_pose) * weight))

        def _cost_stand_still(
            self,
            commands: np.ndarray,
            qpos: np.ndarray,
        ) -> np.ndarray:
            cmd_norm = np.linalg.norm(commands)
            return np.sum(np.abs(qpos - self._default_pose)) * (cmd_norm < 0.01)

        def _cost_termination(self, done: np.ndarray) -> np.ndarray:
            # Penalize early termination.
            return done

        def cost_joint_effort_limits(self, joint_torques: np.ndarray) -> np.ndarray:
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

        # Feet related rewards.
        def _cost_feet_slip(
            self, data: mujoco.MjData, contact: np.ndarray, info: dict[str, Any]
        ) -> np.ndarray:
            cmd_norm = np.linalg.norm(info["command"])
            feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
            vel_xy = feet_vel[..., :2]
            vel_xy_norm_sq = np.sum(np.square(vel_xy), axis=-1)
            return np.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)

        def _cost_feet_clearance(self, data: mujoco.MjData) -> np.ndarray:
            feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
            vel_xy = feet_vel[..., :2]
            vel_norm = np.sqrt(np.linalg.norm(vel_xy, axis=-1))
            foot_pos = data.site_xpos[self._feet_site_id]
            foot_z = foot_pos[..., -1]
            delta = np.abs(foot_z - self._config.reward_config.max_foot_height)
            return np.sum(delta * vel_norm)

        def _cost_feet_height(
            self,
            swing_peak: np.ndarray,
            first_contact: np.ndarray,
            info: dict[str, Any],
        ) -> np.ndarray:
            cmd_norm = np.linalg.norm(info["command"])
            error = swing_peak / self._config.reward_config.max_foot_height - 1.0
            return np.sum(np.square(error) * first_contact) * (cmd_norm > 0.01)


        """Compute reward for current step."""
        imu_orientation = data.site_xmat[self._imu_site_id].reshape(3,3)
        current_up = imu_orientation @ np.array([0.0, 0.0, 1.0])
        reward = {

            # Standard robotic-specific shaping reward
            "tracking_lin_vel": _reward_tracking_lin_vel(self, info["command"], self.get_local_linvel(data)),
            "tracking_ang_vel": _reward_tracking_ang_vel( self, info["command"], self.get_gyro(data) ),
            "lin_vel_z": _cost_lin_vel_z(self, self.get_global_linvel(data)),
            "ang_vel_xy": _cost_ang_vel_xy(self, self.get_global_angvel(data)),
            "joint_motion": _cost_joint_motion(self, data.qvel[6:], data.qacc[6:]),
            "torques": _cost_joint_torques(self, data.actuator_force),
            "action_rate": _cost_action_rate( self, action, info["last_act"], info["last_last_act"]),

            # Other reward
            #"orientation": self._cost_orientation(self.get_upvector(data)),
            "base_height": _reward_height(self, data.qpos[2]),
            "orientation": _reward_orientation(self, current_up, np.array([0, 0, 1])),
            "termination": _cost_termination(self, self._is_terminated()),
            #"pose": _reward_pose(self, data.qpos[7:]),
            #"stand_still": _cost_stand_still(self, info["command"], data.qpos[7:]),
            
            "energy": _cost_energy(self, data.qvel[6:], data.qacc[6:]),
            #"feet_slip": self._cost_feet_slip(data, contact, info),
            #"feet_clearance": self._cost_feet_clearance(data),
            #"feet_height": self._cost_feet_height(
            #    info["swing_peak"], first_contact, info
            #),
            #"feet_air_time": self._reward_feet_air_time(
            #    info["feet_air_time"], first_contact, info["command"]
            #),
            "dof_pos_limits": _cost_joint_pos_limits(self, data.qpos[7:]),
        }

        reward_info = {
            k: v * self._config.reward_config.scales[k] for k, v in reward.items()
        }
        total_reward = np.clip(sum(reward_info.values()) * self.dt, 0.0, 10000.0)

        return total_reward, reward_info
    
    
    # done: post init, reset, termination, get reward

    # todo: get obs, step
    # Add apply perturbation

def make_tita_env(
    render_mode: Optional[str] = None,
    **kwargs
) -> TitaEnv:
    """Factory function to create a Tita environment."""
    return TitaEnv(render_mode=render_mode, **kwargs)


# Register the environment
gym.register(
    id="Tita-v0",
    entry_point="gymnasium.envs.mujoco:TitaEnv",
    max_episode_steps=1000,
)
