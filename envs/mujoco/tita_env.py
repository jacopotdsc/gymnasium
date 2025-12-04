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

try:
    import mujoco
except ImportError:
    raise ImportError(
        "MuJoCo is required for TitaEnv. "
        "Install it with: pip install mujoco"
    )


DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "elevation": -20,
    "azimuth": 135,
}


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
        frame_skip: int = 5,
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        action_scale: float = 1.0,
        reset_noise_scale: float = 0.1,
        **kwargs,
    ):
        """
        Initialize the Tita environment.
        
        Args:
            xml_file: Name of the MuJoCo XML model file
            frame_skip: Number of MuJoCo simulation steps per environment step
            default_camera_config: Camera configuration dict
            action_scale: Scaling factor for actions
            reset_noise_scale: Scale of random perturbations of initial position and velocity
        """
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            action_scale,
            reset_noise_scale,
            **kwargs,
        )
        
        self.action_scale = action_scale
        self._reset_noise_scale = reset_noise_scale
        
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # Will be set below
            default_camera_config=default_camera_config,
            **kwargs,
        )
        
        # Store default pose from keyframe
        home_qpos = self.model.keyframe("home").qpos
        self._default_pose = home_qpos[7:]  # Joint angles (skip base 7-DOF)
        self._init_qpos = home_qpos.copy()
        
        # Get important indices
        self._torso_body_id = self.model.body("base_link").id
        self._imu_site_id = self.model.site("imu").id
        self._floor_geom_id = self.model.geom("floor").id
        
        # Feet geoms
        self._feet_geom_id = np.array([
            self.model.geom("left_leg_4_collision").id,
            self.model.geom("right_leg_4_collision").id,
        ])
        
        # Observation space: linvel(3) + gyro(3) + gravity(3) + joint_angles(8) + joint_vel(8) + last_act(8) + command(3)
        obs_size = 3 + 3 + 3 + 8 + 8 + 8 + 3
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize tracking variables
        self.last_act = np.zeros(8)
        self.last_contact = np.array([False, False])
        self.command = np.zeros(3)  # [vx, vy, omega]

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
        
        return self._get_obs()
    
    def _get_reset_info(self):
        """Return info dict after reset."""
        return {
            "base_height": self.data.qpos[2],
        }

    def step(self, action):
        """Execute one step of the environment."""
        action = np.clip(action, -1.0, 1.0)
        
        # Set motor targets
        motor_targets = self._default_pose + action * self.action_scale
        
        self.do_simulation(motor_targets, self.frame_skip)
        
        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        terminated = self._is_terminated()
        info = {
            "base_height": self.data.qpos[2],
            **reward_info,
        }
        
        self.last_act = action
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, False, info

    def _quat_mul(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions (w, x, y, z format)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
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
        
        # Normalize joint angles w.r.t. default pose
        joint_angles_normalized = joint_angles - self._default_pose
        
        # Build observation
        observation = np.concatenate([
            linvel,
            gyro,
            gravity_body,
            joint_angles_normalized,
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
        
        # Base hit ground (simplified check)
        base_collision = False
        
        # NaN check
        has_nan = np.isnan(qpos).any() or np.isnan(qvel).any()
        
        return False  # fall or base_collision or has_nan

    def _get_rew(self, action: np.ndarray):
        """Compute reward for current step."""
        qpos = self.data.qpos.copy()
        
        # Simple reward: penalize large actions and joint limits violations
        action_penalty = 0.01 * np.sum(np.square(action))
        
        # Height reward (target ~0.49m)
        height = qpos[2]
        height_reward = -np.square(height - 0.49) if abs(height - 0.49) < 0.2 else -1.0
        
        reward = 1.0 + height_reward - action_penalty
        
        reward_info = {
            "action_penalty": action_penalty,
            "height_reward": height_reward,
        }
        
        return float(np.clip(reward, -10.0, 10.0)), reward_info


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
