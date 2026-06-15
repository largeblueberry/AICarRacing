import gymnasium as gym
import torch
import numpy as np
import os
import time
import argparse
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import gymnasium.vector
from contextlib import nullcontext # Import nullcontext for mixed precision handling

# Import custom modules
from src.env_wrappers import GrayScaleObservation, FrameStack, TimeLimit, ActionWrapper
from src.ppo_agent import PPOAgent
from src.rollout_buffer import RolloutBuffer

# --- Configuration --- #
# Config for the training script. This config was used for both saved models.
# Config is optimized for my computer with a Ryzen 7 7800X3D (8 Cores), 32 GB RAM, and RTX 3080 GPU.
# num_envs, async_envs, batch_size, buffer_size, pin_memory, torch_num_threads, and mixed_precision are all optimized for my computer, they will likely need to be changed for different hardware.

config = {
    # Environment
    "env_id": "CarRacing-v3",           # ID for the Gymnasium environment
    "frame_stack": 4,                   # Number of consecutive frames to stack as input
    "num_envs": 16,                      # Number of parallel environments for vectorized training (Change based on your CPU/GPU)
    "max_episode_steps": 2000,          # Maximum steps allowed per episode
    "seed": 42,                         # Seed used for all evaluations and model training

    # PPO Core Parameters
    "total_timesteps": 6_000_000,       # Total number of training steps across all environments
    "learning_rate": 1e-4,              # Learning rate for the optimizers
    "buffer_size": 2048,                # Size of the rollout buffer per environment before updates
    "batch_size": 256,                  # Minibatch size for PPO updates
    "ppo_epochs": 6,                    # Number of optimization epochs per rollout
    "gamma": 0.99,                      # Discount factor for future rewards
    "gae_lambda": 0.95,                  # Factor for Generalized Advantage Estimation (GAE)
    "clip_epsilon": 0.15,               # Clipping parameter for the PPO policy loss
    "vf_coef": 0.5,                     # Coefficient for the value function loss in the total loss
    "ent_coef": 0.008,                  # Coefficient for the entropy bonus in the total loss
    "max_grad_norm": 0.75,              # Maximum norm for gradient clipping
    "target_kl": 0.2,                  # Target KL divergence threshold (for monitoring, not early stopping)
    "features_dim": 256,                # Dimensionality of features extracted by the CNN

    # Agent specific hyperparameters (previously defaults in PPOAgent)
    "initial_action_std": 0.75,          # Initial standard deviation for the action distribution
    "weight_decay": 1e-5,               # Weight decay (L2 regularization) for optimizers
    "fixed_std": False,                 # Whether to use a fixed or learned action standard deviation
    "lr_warmup_steps": 5000,            # Number of steps for learning rate warmup
    "min_learning_rate": 1e-8,          # Minimum learning rate allowed by the scheduler

    # Reward shaping
    "use_reward_shaping": True,         # Flag to enable custom reward shaping
    "velocity_reward_weight": 0.005,    # Weight for the velocity component of the reward
    "survival_reward": 0.05,            # Constant reward added at each step for surviving
    "track_penalty": 5.0,               # Penalty for going off-track
    "steering_smooth_weight": 0.3,      # Weight for the penalty encouraging smooth steering
    "acceleration_while_turning_penalty_weight": 0.8, # Weight for penalizing acceleration during sharp turns

    # Performance optimizations
    "torch_num_threads": 7,             # Number of threads for PyTorch CPU operations
    "mixed_precision": True,           # Flag to enable/disable mixed precision training (requires CUDA)
    "pin_memory": True,                 # Flag to use pinned memory for faster CPU-GPU data transfer
    "async_envs": True,                # Flag to use asynchronous vectorized environments

    # Logging and saving
    "log_interval": 1,                  # Number of rollouts between logging summary statistics
    "save_interval": 10,                # Number of rollouts between saving model checkpoints
    "save_dir": "./models/ppo_3action",  # Directory to save model checkpoints
    "log_dir": "./logs/ppo_3action",     # Directory to save TensorBoard logs

    # Checkpoint to load (set to None to start fresh training)
    "checkpoint_path": None, # Path to load a pre-trained model checkpoint

    # Hardware
    "device": "cuda" if torch.cuda.is_available() else "cpu", # Automatically select CUDA if available, else CPU
}

# --- Helper Functions --- #
def set_seeds(seed: int):
    """
    Sets random seeds for NumPy and PyTorch to ensure reproducibility.

    Args:
        seed: The random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Set performance-enhancing environment variables for multi-threading
os.environ['OMP_NUM_THREADS'] = str(config["torch_num_threads"])
os.environ['MKL_NUM_THREADS'] = str(config["torch_num_threads"])
torch.set_num_threads(config["torch_num_threads"])

# Configure GPU settings if using CUDA
if config["device"] == "cuda":
    torch.backends.cudnn.benchmark = True # Enable cuDNN auto-tuner for best performance
    if config["mixed_precision"]:
        # Enable TensorFloat-32 for faster matrix multiplications on compatible hardware
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def make_env(env_id: str, seed: int, frame_stack: int, max_episode_steps: int, idx: int = 0):
    """
    Creates and wraps a single Gymnasium environment instance.

    Args:
        env_id: The ID of the Gymnasium environment.
        seed: The base random seed.
        frame_stack: The number of frames to stack.
        max_episode_steps: The maximum number of steps per episode.
        idx: An index to create a unique seed for this environment instance.

    Returns:
        A callable function that initializes the environment.
    """
    def _init():
        # Create a unique seed for each parallel environment instance
        env_seed = seed + idx

        # Create the base environment
        env = gym.make(env_id, continuous=True, domain_randomize=False, render_mode=None)

        # Seed the environment and its action space
        env.reset(seed=env_seed)
        env.action_space.seed(env_seed)

        # Add custom reward shaping wrapper if enabled
        if config["use_reward_shaping"]:
            env = RewardShapingWrapper(env,
                                      velocity_weight=config["velocity_reward_weight"],
                                      survival_reward=config["survival_reward"],
                                      track_penalty=config["track_penalty"],
                                      steering_smooth_weight=config["steering_smooth_weight"],
                                      acceleration_while_turning_penalty_weight=config["acceleration_while_turning_penalty_weight"])

        # Apply ActionWrapper
        env = ActionWrapper(env)

        # Apply GrayScaleObservation wrapper (expects RGB input from RewardShapingWrapper or base env)
        env = GrayScaleObservation(env)

        # Apply TimeLimit wrapper
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

        # Apply FrameStack wrapper
        env = FrameStack(env, frame_stack)

        return env

    return _init

class RewardShapingWrapper(gym.Wrapper):
    """
    Applies custom reward shaping to the CarRacing environment.

    Adds rewards for velocity, survival, staying on track, and smooth driving.
    Adds penalties for going off-track, jerky steering, and accelerating while turning sharply.
    """
    def __init__(self, env, velocity_weight: float = 0.005, survival_reward: float = 0.05,
                 track_penalty: float = 2.0, steering_smooth_weight: float = 0.1,
                 acceleration_while_turning_penalty_weight: float = 0.5):
        """
        Initializes the RewardShapingWrapper.

        Args:
            env: The Gymnasium environment to wrap.
            velocity_weight: Weight for the speed-based reward component.
            survival_reward: Constant reward added at each step.
            track_penalty: Penalty applied for each step off-track.
            steering_smooth_weight: Weight for the steering change penalty.
            acceleration_while_turning_penalty_weight: Weight for the penalty for accelerating during sharp turns.
        """
        super().__init__(env)
        self.velocity_weight = velocity_weight
        self.survival_reward = survival_reward
        self.track_penalty = track_penalty
        self.steering_smooth_weight = steering_smooth_weight
        self.acceleration_while_turning_penalty_weight = acceleration_while_turning_penalty_weight

        # Track previous action for smoothness calculations
        self.last_steering = 0.0
        self.last_speed = 0.0

        # Track cumulative reward components per episode
        self.episode_velocity_rewards = 0.0
        self.episode_survival_rewards = 0.0
        self.episode_track_penalties = 0.0
        self.episode_steering_penalties = 0.0
        self.episode_acceleration_while_turning_penalties = 0.0
        self.steps_off_track = 0

        # Additional reward components
        self.centerline_reward_weight = 0.5 # Reward for staying near the track center
        self.track_return_weight = 0.3    # Reward for steering back towards the track when off-track
        self.speed_consistency_weight = 0.05 # Penalty for large speed changes

    def reset(self, **kwargs):
        """Resets the environment and internal state trackers."""
        self.last_steering = 0.0
        self.last_speed = 0.0

        # Reset episode trackers
        self.episode_velocity_rewards = 0.0
        self.episode_survival_rewards = 0.0
        self.episode_track_penalties = 0.0
        self.episode_steering_penalties = 0.0
        self.episode_acceleration_while_turning_penalties = 0.0
        self.steps_off_track = 0

        obs, info = self.env.reset(**kwargs)

        # Initialize reward components in the info dictionary
        info['velocity_rewards'] = 0.0
        info['survival_rewards'] = 0.0
        info['track_penalties'] = 0.0
        info['steering_penalties'] = 0.0
        info['acceleration_while_turning_penalties'] = 0.0
        info['steps_off_track'] = 0

        return obs, info

    def step(self, action):
        """
        Steps the environment and applies reward shaping.

        Args:
            action: The action taken by the agent.

        Returns:
            A tuple containing (observation, shaped_reward, terminated, truncated, info).
            The info dictionary includes detailed reward components.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Ensure info is a dictionary
        if info is None:
             info = {}

        # Extract action components
        steering = action[0]
        gas = action[1]
        # brake = action[2] # Brake is not used in current reward shaping

        # Initialize reward components for this step
        step_velocity_reward = 0.0
        step_survival_reward = 0.0
        step_track_penalty = 0.0
        step_steering_penalty = 0.0
        step_acceleration_while_turning_penalty = 0.0
        step_centerline_reward = 0.0
        step_speed_consistency_reward = 0.0
        step_track_return_reward = 0.0
        off_track = False

        # 1. Velocity rewards: Encourage higher speeds
        speed = info.get('speed', 0.0) # Get speed from info if available
        if speed > 0:
            step_velocity_reward = speed * self.velocity_weight
            reward += step_velocity_reward
            self.episode_velocity_rewards += step_velocity_reward

        # 2. Survival rewards: Encourage longer episodes
        step_survival_reward = self.survival_reward
        reward += step_survival_reward
        self.episode_survival_rewards += step_survival_reward

        # 3. Track adherence penalty: Penalize going off-track (onto grass)
        #    Requires RGB observation input to this wrapper.
        if len(obs.shape) == 3 and obs.shape[2] == 3: # Check if observation is RGB
            # Check bottom center pixels for green color (indicative of grass)
            car_area = obs[84:94, 42:54, :]
            green_channel = car_area[:, :, 1]
            red_channel = car_area[:, :, 0]

            # Heuristic: High green and low red likely means off-track
            off_track = np.mean(green_channel) > 150 and np.mean(red_channel) < 100

            if off_track:
                step_track_penalty = self.track_penalty
                reward -= step_track_penalty
                self.episode_track_penalties += step_track_penalty
                self.steps_off_track += 1

                # Add guidance reward to steer back towards the track center
                # Simplified: assumes track is generally ahead
                track_direction = np.array([1.0, 0.0])
                # Approximate car direction based on steering angle
                car_direction = np.array([np.cos(steering * np.pi / 2), np.sin(steering * np.pi / 2)])
                # Reward alignment with track direction
                step_track_return_reward = np.dot(track_direction, car_direction) * self.track_return_weight
                reward += step_track_return_reward
            else:
                # 5. Centerline reward: Reward staying near the center (higher red channel value)
                road_redness = np.mean(red_channel)
                step_centerline_reward = min(road_redness / 200, 1.0) * self.centerline_reward_weight
                reward += step_centerline_reward

        # 4. Steering smoothness penalty: Penalize large changes in steering, especially at high speed
        steering_change = abs(steering - self.last_steering)
        step_steering_penalty = steering_change * self.steering_smooth_weight * (1.0 + speed * 0.1)
        reward -= step_steering_penalty
        self.episode_steering_penalties += step_steering_penalty

        # 6. Speed consistency reward: Penalize large changes in speed
        speed_change = abs(speed - self.last_speed)
        step_speed_consistency_reward = -speed_change * self.speed_consistency_weight
        reward += step_speed_consistency_reward

        # 7. Acceleration while turning penalty: Penalize applying gas during sharp turns
        steering_threshold = 0.4 # Angle threshold for penalty
        gas_threshold = 0.1      # Gas threshold for penalty
        if abs(steering) > steering_threshold and gas > gas_threshold:
            step_acceleration_while_turning_penalty = (
                self.acceleration_while_turning_penalty_weight *
                (gas - gas_threshold) *
                (abs(steering) - steering_threshold)
            )
            reward -= step_acceleration_while_turning_penalty
            self.episode_acceleration_while_turning_penalties += step_acceleration_while_turning_penalty

        # Update state for next step
        self.last_steering = steering
        self.last_speed = speed

        # Add step-wise reward components to info
        info['velocity_rewards'] = step_velocity_reward
        info['survival_rewards'] = step_survival_reward
        info['track_penalties'] = step_track_penalty
        info['steering_penalties'] = step_steering_penalty
        info['acceleration_while_turning_penalties'] = step_acceleration_while_turning_penalty
        info['centerline_rewards'] = step_centerline_reward
        info['speed_consistency_rewards'] = step_speed_consistency_reward
        info['track_return_rewards'] = step_track_return_reward
        info['off_track'] = off_track

        # Add cumulative episode totals to info (useful for final info dict)
        info['episode_velocity_rewards'] = self.episode_velocity_rewards
        info['episode_survival_rewards'] = self.episode_survival_rewards
        info['episode_track_penalties'] = self.episode_track_penalties
        info['episode_steering_penalties'] = self.episode_steering_penalties
        info['episode_acceleration_while_turning_penalties'] = self.episode_acceleration_while_turning_penalties
        info['steps_off_track'] = self.steps_off_track

        return obs, reward, terminated, truncated, info

def load_checkpoint(agent: PPOAgent, checkpoint_path: str, config: dict, device: str):
    """
    Loads model weights and training state from a checkpoint file.

    Args:
        agent: The PPOAgent instance to load the weights into.
        checkpoint_path: Path to the checkpoint file (.pth).
        config: The configuration dictionary (used for reference, not loaded).
        device: The device ('cpu' or 'cuda') to load the checkpoint onto.

    Returns:
        A tuple (best_mean_reward, global_step) loaded from the checkpoint,
        or (-np.inf, 0) if loading fails or the checkpoint doesn't exist.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Starting fresh training.")
        return -np.inf, 0

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Loading checkpoint from {checkpoint_path}")

        # Load model weights
        agent.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        print("Model weights loaded successfully.")

        # Optionally load optimizer states (commented out for stability because this broke the model training)
        # if 'actor_optimizer_state_dict' in checkpoint and 'critic_optimizer_state_dict' in checkpoint:
        #     agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        #     agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        #     print("Loaded optimizer states.")
        # else:
        #     print("Optimizer states not found in checkpoint, initializing fresh optimizers.")
        print("Skipping optimizer state loading.")

        # Load global step and best mean reward
        global_step = checkpoint.get('global_step', 0)
        best_mean_reward = checkpoint.get('mean_reward', -np.inf)
        print(f"Resuming from global step {global_step}")
        print(f"Best mean reward from checkpoint: {best_mean_reward:.2f}")

        return best_mean_reward, global_step
    except KeyError as e:
        print(f"Error loading checkpoint: Missing key {e}. Checkpoint structure might be incompatible.")
        return -np.inf, 0
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return -np.inf, 0

# --- Main Training Loop --- #
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a PPO agent for CarRacing-v3")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint file to resume training from.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override the total number of training timesteps defined in the config.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the random seed defined in the config.")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Override the TensorBoard log directory defined in the config.")
    args = parser.parse_args()

    # --- Configuration Override ---
    # Update config dictionary with command-line arguments if provided
    if args.checkpoint:
        config["checkpoint_path"] = args.checkpoint
    if args.steps is not None:
        config["total_timesteps"] = args.steps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.log_dir is not None:
        config["log_dir"] = args.log_dir

    # --- Initialization ---
    print(f"--- Training Configuration ---")
    print(f"Device: {config['device']}")
    print(f"Number of Environments: {config['num_envs']} ({'Async' if config['async_envs'] else 'Sync'})")
    print(f"Total Timesteps: {config['total_timesteps']:,}")
    print(f"Seed: {config['seed']}")
    print(f"Log Directory: {config['log_dir']}")
    print(f"Save Directory: {config['save_dir']}")
    print(f"Mixed Precision: {'Enabled' if config['mixed_precision'] else 'Disabled'}")
    print(f"Torch Threads: {config['torch_num_threads']}")
    print(f"Resuming from Checkpoint: {config['checkpoint_path'] if config['checkpoint_path'] else 'None'}")
    print(f"-----------------------------")

    set_seeds(config["seed"])

    # Create necessary directories
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    # --- Environment Setup ---
    print(f"Creating {config['num_envs']} parallel environments...")
    env_fns = [make_env(config["env_id"], config["seed"], config["frame_stack"],
                        config["max_episode_steps"], i) for i in range(config["num_envs"])]

    # Choose between synchronous and asynchronous vectorized environments
    if config["async_envs"]:
        env = gymnasium.vector.AsyncVectorEnv(env_fns)
    else:
        env = gymnasium.vector.SyncVectorEnv(env_fns)

    print(f"Observation Space: {env.single_observation_space}")
    print(f"Action Space: {env.single_action_space}")

    # --- Agent Setup ---
    agent = PPOAgent(
        env.single_observation_space,
        env.single_action_space,
        config=config,  # Pass the entire config dictionary
        device=config["device"]
    )

    # --- Load Checkpoint ---
    best_mean_reward = -np.inf
    global_step = 0
    if config["checkpoint_path"]:
        loaded_reward, loaded_step = load_checkpoint(agent, config["checkpoint_path"], config, config["device"])
        if loaded_reward is not None: # Check if loading was successful
            best_mean_reward = loaded_reward
            global_step = loaded_step
            # Ensure the agent's internal step counter aligns for LR scheduling
            agent.steps_done = global_step
            print(f"Set agent's internal step counter to {agent.steps_done} for LR schedule.")

    # --- Rollout Buffer Setup ---
    # Calculate buffer size per environment
    buffer_size_per_env = config["buffer_size"] // config["num_envs"]
    if config["buffer_size"] % config["num_envs"] != 0:
         print(f"Warning: buffer_size ({config['buffer_size']}) not perfectly divisible by num_envs ({config['num_envs']}). Effective buffer size per env: {buffer_size_per_env}")

    buffer = RolloutBuffer(
        buffer_size_per_env, # Use size per env
        env.single_observation_space,
        env.single_action_space,
        num_envs=config["num_envs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        device=config["device"]
    )

    # --- Logging Setup ---
    writer = SummaryWriter(log_dir=config["log_dir"])
    # Track recent episode rewards and lengths for logging
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)

    # --- Training Start ---
    print(f"Starting training from step {global_step}/{config['total_timesteps']}")
    observations, infos = env.reset(seed=config["seed"]) # Initial reset with seed
    num_rollouts = 0
    # Track rewards and lengths for episodes currently in progress
    current_episode_rewards = np.zeros(config["num_envs"], dtype=np.float32)
    current_episode_lengths = np.zeros(config["num_envs"], dtype=np.int32)
    start_time = time.time()

    # --- Mixed Precision Setup ---
    # Use autocast context manager if CUDA and mixed precision are enabled
    autocast_context = torch.cuda.amp.autocast() if config["device"] == "cuda" and config["mixed_precision"] else nullcontext()
    # Initialize GradScaler if using mixed precision, otherwise None
    scaler = torch.cuda.amp.GradScaler() if config["device"] == "cuda" and config["mixed_precision"] else None

    # --- Main Training Loop ---
    try:
        while global_step < config["total_timesteps"]:
            rollout_episode_rewards = [] # Initialize list HERE
            buffer.reset() # Reset buffer position and full flag before each rollout
            rollout_start_time = time.time()
            steps_per_rollout = buffer.buffer_size # Steps to collect per environment in this rollout
            last_dones = np.zeros(config["num_envs"], dtype=bool) # Track dones from the final step

            # --- Rollout Phase ---
            for step in range(steps_per_rollout):
                # Ensure observations are tensors on the correct device
                obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=config["device"])

                # Agent selects actions based on observations
                with torch.no_grad():
                    actions, values, log_probs = agent.act(obs_tensor)

                # Environment steps forward with the selected actions
                next_observations, rewards, terminateds, truncateds, infos = env.step(actions)
                dones = terminateds | truncateds # Combine terminated and truncated flags

                # Update trackers for current episodes
                current_episode_rewards += rewards
                current_episode_lengths += 1

                # Store the transition in the rollout buffer
                buffer.add(observations, actions, rewards, terminateds, truncateds, values, log_probs)

                # Prepare for the next step
                observations = next_observations
                last_dones = dones # Store dones for GAE calculation

                # --- Handle Episode Completions ---
                # Check if any environments finished an episode using VecEnv's info dict
                if "_final_info" in infos:
                    # Identify which environments finished
                    finished_mask = infos["_final_info"]
                    if np.any(finished_mask):
                        # Extract final info for completed episodes
                        final_infos = infos["final_info"][finished_mask]
                        env_indices = np.where(finished_mask)[0] # Get original indices

                        for i, final_info in enumerate(final_infos):
                            if final_info is not None and "episode" in final_info:
                                ep_rew = final_info["episode"]["r"]
                                ep_len = final_info["episode"]["l"]
                                episode_rewards.append(ep_rew) # Add to logging queue (100 ep avg)
                                episode_lengths.append(ep_len)
                                rollout_episode_rewards.append(ep_rew) # Append reward HERE
                                print(f"Env {env_indices[i]} finished: Reward={ep_rew:.2f}, Length={ep_len}, Total Steps={global_step+step*config['num_envs']}")

                                # Reset trackers for the specific environment that finished
                                current_episode_rewards[env_indices[i]] = 0
                                current_episode_lengths[env_indices[i]] = 0

                # Fallback if _final_info is not present (e.g., older Gym versions)
                elif np.any(dones):
                    # Collect components for averaging across finished envs in this step
                    velocity_rews, survival_rews, track_pens, steering_pens = [], [], [], []
                    accel_turn_pens, steps_off, off_track_pcts = [], [], []

                    for i in range(config["num_envs"]):
                        if dones[i]:
                            ep_reward = current_episode_rewards[i]
                            ep_length = current_episode_lengths[i]
                            episode_rewards.append(ep_reward)
                            episode_lengths.append(ep_length)
                            rollout_episode_rewards.append(ep_reward) # Append reward HERE too
                            print(f"Env {i} finished (manual): Reward={ep_reward:.2f}, Length={ep_length}, Total Steps={global_step+step*config['num_envs']}")

                            # --- BEGIN MOVED LOGGING LOGIC ---
                            # Attempt to get detailed info from the info dict of the finished env
                            env_info = infos[i] if isinstance(infos, (list, tuple)) else infos.get(i) # Handle potential dict structure
                            if env_info:
                                if 'episode_velocity_rewards' in env_info: velocity_rews.append(env_info['episode_velocity_rewards'])
                                if 'episode_survival_rewards' in env_info: survival_rews.append(env_info['episode_survival_rewards'])
                                if 'episode_track_penalties' in env_info: track_pens.append(env_info['episode_track_penalties'])
                                if 'episode_steering_penalties' in env_info: steering_pens.append(env_info['episode_steering_penalties'])
                                if 'episode_acceleration_while_turning_penalties' in env_info: accel_turn_pens.append(env_info['episode_acceleration_while_turning_penalties'])
                                if 'steps_off_track' in env_info:
                                    steps_off.append(env_info['steps_off_track'])
                                    if ep_length > 0: # Use calculated ep_length
                                        off_track_pcts.append(100 * env_info['steps_off_track'] / ep_length)
                            # --- END MOVED LOGGING LOGIC ---

                            # Reset trackers
                            current_episode_rewards[i] = 0
                            current_episode_lengths[i] = 0

                    # --- BEGIN MOVED TENSORBOARD LOGGING ---
                    # Log averaged components if available (after checking all finished envs)
                    # Note: This logging now happens *inside* the rollout loop whenever an episode ends,
                    # rather than only at the end of the logging interval.
                    # This might lead to more frequent but potentially noisier component logs.
                    # Alternatively, accumulate these lists outside this loop and log them
                    # during the main logging phase (num_rollouts % log_interval == 0).
                    # For simplicity now, we log immediately.
                    if velocity_rews: writer.add_scalar("rewards/mean_velocity", np.mean(velocity_rews), global_step)
                    if survival_rews: writer.add_scalar("rewards/mean_survival", np.mean(survival_rews), global_step)
                    if track_pens: writer.add_scalar("penalties/mean_track", np.mean(track_pens), global_step)
                    if steering_pens: writer.add_scalar("penalties/mean_steering", np.mean(steering_pens), global_step)
                    if accel_turn_pens: writer.add_scalar("penalties/mean_accel_turn", np.mean(accel_turn_pens), global_step)
                    if steps_off: writer.add_scalar("driving/mean_steps_off_track", np.mean(steps_off), global_step)
                    if off_track_pcts: writer.add_scalar("driving/mean_percent_off_track", np.mean(off_track_pcts), global_step)
                    # --- END MOVED TENSORBOARD LOGGING ---

                # Update global step count (total steps across all envs)
                global_step += config["num_envs"]

                # Check if total timesteps limit is reached
                if global_step >= config["total_timesteps"]:
                    print(f"Reached total timesteps ({config['total_timesteps']}). Finishing rollout.")
                    break # Exit the inner rollout loop

            # --- Post-Rollout Phase ---
            # Compute advantages and returns after collecting the rollout data
            with torch.no_grad():
                # Get value estimate for the last observation in the rollout
                obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=config["device"])
                features = agent.feature_extractor(obs_tensor) # Pass raw obs, normalization happens inside
                last_values = agent.critic(features).cpu().numpy() # Get value estimates

            # Calculate GAE and returns using collected data and last value estimate
            buffer.compute_returns_and_advantages(last_values, last_dones)

            # --- Learning Phase ---
            # Update agent policy and value function using the collected rollout data
            if config["mixed_precision"] and config["device"] == "cuda":
                metrics = agent.learn_mixed_precision(buffer, scaler) # Use mixed precision update
            else:
                metrics = agent.learn(buffer) # Use standard precision update

            # Update learning rate based on schedule
            current_lr = agent.update_learning_rate(config['total_timesteps'])

            # Increment rollout counter
            num_rollouts += 1

            # --- Logging ---
            if num_rollouts % config["log_interval"] == 0 and len(episode_rewards) > 0:
                # --- Calculate performance metrics ---
                mean_reward_100 = np.mean(episode_rewards)
                mean_length_100 = np.mean(episode_lengths)
                rollout_duration = time.time() - rollout_start_time
                steps_in_rollout = buffer.size() # Get actual number of steps collected
                fps = int(steps_in_rollout / rollout_duration) if rollout_duration > 0 else 0

                # --- Calculate Mean Rollout Reward using the accumulated list ---
                mean_rollout_reward = np.mean(rollout_episode_rewards) if rollout_episode_rewards else -1 # Use -1 if no episodes finished in interval

                # --- Print summary to console ---
                print(f"====== Rollout {num_rollouts} | Step {global_step}/{config['total_timesteps']} ======")
                print(f"Mean Reward (Last 100): {mean_reward_100:.2f}")
                if mean_rollout_reward != -1:
                    print(f"Mean Reward (This Rollout): {mean_rollout_reward:.2f} ({len(rollout_episode_rewards)} episodes)")
                print(f"Mean Episode Length: {mean_length_100:.1f}")
                print(f"FPS: {fps}")
                print(f"Learning Rate: {current_lr:.2e}")
                print(f"Policy Loss: {metrics['policy_loss']:.4f}")
                print(f"Value Loss: {metrics['value_loss']:.4f}")
                print(f"Entropy: {metrics['entropy_loss']:.4f}")
                print(f"Approx KL: {metrics['approx_kl']:.4f}")
                print(f"Clip Fraction: {metrics['clip_fraction']:.4f}")

                # --- Log metrics to TensorBoard ---
                writer.add_scalar("charts/mean_reward_100", mean_reward_100, global_step)
                writer.add_scalar("charts/mean_length_100", mean_length_100, global_step)
                if mean_rollout_reward != -1:
                    writer.add_scalar("charts/mean_rollout_reward", mean_rollout_reward, global_step)
                writer.add_scalar("charts/fps", fps, global_step)
                writer.add_scalar("charts/learning_rate", current_lr, global_step)
                writer.add_scalar("losses/policy_loss", metrics["policy_loss"], global_step)
                writer.add_scalar("losses/value_loss", metrics["value_loss"], global_step)
                writer.add_scalar("losses/entropy", metrics["entropy_loss"], global_step)
                writer.add_scalar("losses/approx_kl", metrics["approx_kl"], global_step)
                writer.add_scalar("losses/clip_fraction", metrics["clip_fraction"], global_step)

                # --- Save Best Model ---
                if mean_reward_100 > best_mean_reward:
                    best_mean_reward = mean_reward_100
                    best_model_path = os.path.join(config["save_dir"], "best_model.pth")
                    print(f"New best mean reward: {best_mean_reward:.2f}. Saving model to {best_model_path}")
                    torch.save({
                        'feature_extractor_state_dict': agent.feature_extractor.state_dict(),
                        'actor_state_dict': agent.actor.state_dict(),
                        'critic_state_dict': agent.critic.state_dict(),
                        'global_step': global_step,
                        'mean_reward': mean_reward_100, # Save the reward that triggered the save
                        'config': config # Optionally save the config used
                    }, best_model_path)

            # --- Save Checkpoint Periodically ---
            if num_rollouts > 0 and num_rollouts % config["save_interval"] == 0:
                checkpoint_path = os.path.join(config["save_dir"], f"checkpoint_{global_step}.pth")
                print(f"Saving checkpoint at step {global_step} to {checkpoint_path}")
                torch.save({
                    'feature_extractor_state_dict': agent.feature_extractor.state_dict(),
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    # Save optimizer states if needed for exact resumption
                    # 'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    # 'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                    'global_step': global_step,
                    'config': config,
                    'mean_reward': best_mean_reward, # Save current best reward
                }, checkpoint_path)

            # Check again if total timesteps reached after learning phase
            if global_step >= config["total_timesteps"]:
                print("Total timesteps reached. Exiting training loop.")
                break

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        print(f"Closing environment and TensorBoard writer...")
        env.close()
        writer.close()
        print(f"Training finished after {global_step} steps.")
        print(f"Best mean reward achieved: {best_mean_reward:.2f}")
        print(f"Model checkpoints saved in: {config['save_dir']}")
        print(f"Logs saved in: {config['log_dir']}") 