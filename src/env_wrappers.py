import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from collections import deque
import cv2

class GrayScaleObservation(gym.ObservationWrapper):
    """
    Converts RGB image observations to grayscale.

    Assumes the input observation space is a 3D Box representing (H, W, 3) RGB images.
    The output observation space is a 2D Box representing (H, W) grayscale images.
    """
    def __init__(self, env):
        """Initializes the grayscale wrapper."""
        super().__init__(env)
        # Check if the input observation space is compatible (RGB image)
        assert len(env.observation_space.shape) == 3 and env.observation_space.shape[2] == 3, \
               f"GrayScaleObservation expects RGB image input (H, W, 3), got {env.observation_space.shape}"

        # Define the new grayscale observation space
        obs_shape = self.observation_space.shape[:2] # Height, Width
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Converts a single RGB observation to grayscale using OpenCV."""
        # Use OpenCV's conversion function
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return obs

class TimeLimit(gym.Wrapper):
    """
    Applies a maximum step limit to an environment's episode.

    This wrapper ensures that an episode terminates (by setting truncated=True)
    if it exceeds a specified number of steps, regardless of the environment's
    internal termination conditions.
    """
    def __init__(self, env, max_episode_steps: int = 1000):
        """
        Initializes the time limit wrapper.

        Args:
            env: The Gymnasium environment to wrap.
            max_episode_steps: The maximum number of steps allowed per episode.
        """
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def reset(self, **kwargs):
        """Resets the environment and the step counter."""
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Steps the environment and checks if the time limit has been reached."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        # Set truncated to True if the step limit is exceeded
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        return observation, reward, terminated, truncated, info

class FrameStack(gym.ObservationWrapper):
    """
    Stacks the most recent k frames (observations) into a single observation.

    This is commonly used in environments where temporal information is important,
    like Atari games or CarRacing, to allow the agent to infer dynamics like velocity.

    Assumes the input observation space is 2D (e.g., grayscale image HxW).
    The output observation space has shape (k, H, W).
    """
    def __init__(self, env, k: int):
        """
        Initializes the frame stacking wrapper.

        Args:
            env: The Gymnasium environment to wrap (must have 2D observation space).
            k: The number of frames to stack.
        """
        super().__init__(env)
        self.k = k
        # Use a deque to efficiently store the last k frames
        self.frames = deque([], maxlen=k)

        # Check input observation space shape (expects 2D, e.g., HxW after grayscale)
        assert len(env.observation_space.shape) == 2, \
               f"FrameStack expects 2D input shape (H, W), got {env.observation_space.shape}"

        # Define the new stacked observation space shape
        stacked_shape = (k,) + env.observation_space.shape # (k, H, W)
        # Define the new observation space bounds and dtype
        self.observation_space = Box(
            low=0, high=255, shape=stacked_shape, dtype=env.observation_space.dtype
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Adds the latest observation to the deque and returns the stacked frames."""
        self.frames.append(observation)
        return self._get_ob()

    def reset(self, **kwargs):
        """
        Resets the environment and the frame buffer.

        Fills the frame buffer with the first observation repeated k times.
        """
        obs, info = self.env.reset(**kwargs)
        # Clear the deque and fill it with the initial observation
        for _ in range(self.k):
            self.frames.append(obs)
        # Return the initial stacked observation and info dictionary
        return self._get_ob(), info

    def _get_ob(self) -> np.ndarray:
        """Retrieves the stacked frames from the deque as a NumPy array."""
        assert len(self.frames) == self.k, f"Frame buffer size mismatch: expected {self.k}, got {len(self.frames)}"
        # Stack the frames along the first axis (channel dimension)
        return np.stack(self.frames, axis=0) 
class ActionWrapper(gym.ActionWrapper):
    """
    Converts 2D action [steering, throttle] to 3D action [steering, gas, brake].
    throttle > 0 -> gas, throttle < 0 -> brake
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

    def action(self, action):
        steering = action[0]
        throttle = action[1]
        gas = max(0.0, float(throttle))
        brake = max(0.0, float(-throttle))
        return np.array([steering, gas, brake], dtype=np.float32)
