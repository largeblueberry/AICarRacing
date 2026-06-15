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
config = {
    # Environment (A100 서버의 CPU-GPU 밸런스를 고려한 멀티환경 최적화)
    "env_id": "CarRacing-v3",           
    "frame_stack": 4,                    
    "num_envs": 64,                      # ★ 128 -> 64로 하향 조정 (원격 서버 CPU 병목을 해소하여 오히려 FPS 급상승)
    "max_episode_steps": 1000,           
    "seed": 42,                         

    # PPO Core Parameters (★ A100 80GB 맞춤형 스케일업 파라미터)
    "total_timesteps": 20_000_000,       
    "learning_rate": 8e-5,               # ★ 8e-5 고정 (Early Stopping 완화 및 정책 급발진 방지)
    "buffer_size": 16384,                # ★ 64 envs * 256 steps = 16384 (충분한 버퍼 크기 유지)
    "batch_size": 4096,                  # ★ 4096 유지 (A100의 Tensor Core를 완벽하게 활용하는 대형 배치)
    "ppo_epochs": 4,                     
    "gamma": 0.99,                      
    "gae_lambda": 0.95,                  
    "clip_epsilon": 0.2,                 
    "vf_coef": 0.5,                      
    "ent_coef": 0.01,                    
    "max_grad_norm": 0.5,               
    "target_kl": 0.015,                  
    "features_dim": 256,                 

    # Agent specific hyperparameters
    "initial_action_std": 0.4,           
    "weight_decay": 1e-6,               
    "fixed_std": False,                 
    "lr_warmup_steps": 0,                # ★ 0으로 변경 (LR이 역으로 상승하던 스케줄러 버그 원천 차단)
    "min_learning_rate": 1e-6,          

    # Reward shaping 
    "use_reward_shaping": True,         
    "velocity_reward_weight": 0.01,      
    "survival_reward": 0.01,            
    "track_penalty": 1.0,               
    "steering_smooth_weight": 0.01,      
    "acceleration_while_turning_penalty_weight": 0.05, 

    # Performance optimizations
    "torch_num_threads": 2,              # ★ 16 -> 2로 차단 (파이토치가 CPU를 독점하지 못하게 막아야 env 분산 처리가 빨라집니다)
    "mixed_precision": True,             
    "pin_memory": True,                 
    "async_envs": True,                 

    # Logging and saving
    "log_interval": 1,                  
    "save_interval": 10,                
    "save_dir": "./models/ppo_2action",  
    "log_dir": "./logs/ppo_2action",     

    "checkpoint_path": None, 
    "device": "cuda" if torch.cuda.is_available() else "cpu", 
}

# --- Helper Functions --- #
def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ★ 원격 멀티프로세싱 최적화를 위한 CPU 스레드 통제 (MKL/OMP 전부 제한)
os.environ['OMP_NUM_THREADS'] = str(config["torch_num_threads"])
os.environ['MKL_NUM_THREADS'] = str(config["torch_num_threads"])
torch.set_num_threads(config["torch_num_threads"])

if config["device"] == "cuda":
    torch.backends.cudnn.benchmark = True 
    if config["mixed_precision"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def make_env(env_id: str, seed: int, frame_stack: int, max_episode_steps: int, idx: int = 0):
    def _init():
        env_seed = seed + idx
        env = gym.make(env_id, continuous=True, domain_randomize=False, render_mode=None)
        env.reset(seed=env_seed)
        env.action_space.seed(env_seed)

        if config["use_reward_shaping"]:
            env = RewardShapingWrapper(env,
                                      velocity_weight=config["velocity_reward_weight"],
                                      survival_reward=config["survival_reward"],
                                      track_penalty=config["track_penalty"],
                                      steering_smooth_weight=config["steering_smooth_weight"],
                                      acceleration_while_turning_penalty_weight=config["acceleration_while_turning_penalty_weight"])

        env = ActionWrapper(env)
        env = GrayScaleObservation(env)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = FrameStack(env, frame_stack)
        return env
    return _init

class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, velocity_weight: float = 0.005, survival_reward: float = 0.05,
                 track_penalty: float = 2.0, steering_smooth_weight: float = 0.1,
                 acceleration_while_turning_penalty_weight: float = 0.5):
        super().__init__(env)
        self.velocity_weight = velocity_weight
        self.survival_reward = survival_reward
        self.track_penalty = track_penalty
        self.steering_smooth_weight = steering_smooth_weight
        self.acceleration_while_turning_penalty_weight = acceleration_while_turning_penalty_weight

        self.last_steering = 0.0
        self.last_speed = 0.0

        self.episode_velocity_rewards = 0.0
        self.episode_survival_rewards = 0.0
        self.episode_track_penalties = 0.0
        self.episode_steering_penalties = 0.0
        self.episode_acceleration_while_turning_penalties = 0.0
        self.steps_off_track = 0

        self.centerline_reward_weight = 0.5 
        self.track_return_weight = 0.3    
        self.speed_consistency_weight = 0.05 

    def reset(self, **kwargs):
        self.last_steering = 0.0
        self.last_speed = 0.0
        self.episode_velocity_rewards = 0.0
        self.episode_survival_rewards = 0.0
        self.episode_track_penalties = 0.0
        self.episode_steering_penalties = 0.0
        self.episode_acceleration_while_turning_penalties = 0.0
        self.steps_off_track = 0

        obs, info = self.env.reset(**kwargs)
        info['velocity_rewards'] = 0.0
        info['survival_rewards'] = 0.0
        info['track_penalties'] = 0.0
        info['steering_penalties'] = 0.0
        info['acceleration_while_turning_penalties'] = 0.0
        info['steps_off_track'] = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info is None: info = {}

        steering = action[0]
        gas = action[1]

        step_velocity_reward = 0.0
        step_survival_reward = 0.0
        step_track_penalty = 0.0
        step_steering_penalty = 0.0
        step_acceleration_while_turning_penalty = 0.0
        step_centerline_reward = 0.0
        step_speed_consistency_reward = 0.0
        step_track_return_reward = 0.0
        off_track = False

        speed = info.get('speed', 0.0)
        if speed > 0:
            step_velocity_reward = speed * self.velocity_weight
            reward += step_velocity_reward
            self.episode_velocity_rewards += step_velocity_reward

        step_survival_reward = self.survival_reward
        reward += step_survival_reward
        self.episode_survival_rewards += step_survival_reward

        if len(obs.shape) == 3 and obs.shape[2] == 3: 
            car_area = obs[84:94, 42:54, :]
            green_channel = car_area[:, :, 1]
            red_channel = car_area[:, :, 0]

            off_track = np.mean(green_channel) > 150 and np.mean(red_channel) < 100

            if off_track:
                step_track_penalty = self.track_penalty
                reward -= step_track_penalty
                self.episode_track_penalties += step_track_penalty
                self.steps_off_track += 1

                track_direction = np.array([1.0, 0.0])
                car_direction = np.array([np.cos(steering * np.pi / 2), np.sin(steering * np.pi / 2)])
                step_track_return_reward = np.dot(track_direction, car_direction) * self.track_return_weight
                reward += step_track_return_reward
            else:
                road_redness = np.mean(red_channel)
                step_centerline_reward = min(road_redness / 200, 1.0) * self.centerline_reward_weight
                reward += step_centerline_reward

        steering_change = abs(steering - self.last_steering)
        step_steering_penalty = steering_change * self.steering_smooth_weight * (1.0 + speed * 0.1)
        reward -= step_steering_penalty
        self.episode_steering_penalties += step_steering_penalty

        speed_change = abs(speed - self.last_speed)
        step_speed_consistency_reward = -speed_change * self.speed_consistency_weight
        reward += step_speed_consistency_reward

        steering_threshold = 0.4 
        gas_threshold = 0.1      
        if abs(steering) > steering_threshold and gas > gas_threshold:
            step_acceleration_while_turning_penalty = (
                self.acceleration_while_turning_penalty_weight *
                (gas - gas_threshold) *
                (abs(steering) - steering_threshold)
            )
            reward -= step_acceleration_while_turning_penalty
            self.episode_acceleration_while_turning_penalties += step_acceleration_while_turning_penalty

        self.last_steering = steering
        self.last_speed = speed

        info['velocity_rewards'] = step_velocity_reward
        info['survival_rewards'] = step_survival_reward
        info['track_penalties'] = step_track_penalty
        info['steering_penalties'] = step_steering_penalty
        info['acceleration_while_turning_penalties'] = step_acceleration_while_turning_penalty
        info['centerline_rewards'] = step_centerline_reward
        info['speed_consistency_rewards'] = step_speed_consistency_reward
        info['track_return_rewards'] = step_track_return_reward
        info['off_track'] = off_track

        info['episode_velocity_rewards'] = self.episode_velocity_rewards
        info['episode_survival_rewards'] = self.episode_survival_rewards
        info['episode_track_penalties'] = self.episode_track_penalties
        info['episode_steering_penalties'] = self.episode_steering_penalties
        info['episode_acceleration_while_turning_penalties'] = self.episode_acceleration_while_turning_penalties
        info['steps_off_track'] = self.steps_off_track

        return obs, reward, terminated, truncated, info

def load_checkpoint(agent: PPOAgent, checkpoint_path: str, config: dict, device: str):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Starting fresh training.")
        return -np.inf, 0

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Loading checkpoint from {checkpoint_path}")
        agent.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        print("Model weights loaded successfully. Skipping optimizer state loading for stability.")

        global_step = checkpoint.get('global_step', 0)
        best_mean_reward = checkpoint.get('mean_reward', -np.inf)
        print(f"Resuming from global step {global_step} | Best mean reward: {best_mean_reward:.2f}")
        return best_mean_reward, global_step
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return -np.inf, 0

# --- Main Training Loop --- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent for CarRacing-v3")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--steps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--log-dir", type=str, default=None, help="Override log directory")
    args = parser.parse_args()

    if args.checkpoint: config["checkpoint_path"] = args.checkpoint
    if args.steps is not None: config["total_timesteps"] = args.steps
    if args.seed is not None: config["seed"] = args.seed
    if args.log_dir is not None: config["log_dir"] = args.log_dir

    print(f"--- Optimized Training Configuration (A100 Headless Server) ---")
    print(f"Device: {config['device']} | Mixed Precision: {config['mixed_precision']}")
    print(f"Number of Environments: {config['num_envs']} | Batch Size: {config['batch_size']}")
    print(f"Total Timesteps: {config['total_timesteps']:,} | Torch Threads: {config['torch_num_threads']}")
    print(f"-----------------------------------------------------------------")

    set_seeds(config["seed"])
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    print(f"Creating {config['num_envs']} parallel environments...")
    env_fns = [make_env(config["env_id"], config["seed"], config["frame_stack"], config["max_episode_steps"], i) for i in range(config["num_envs"])]
    env = gymnasium.vector.AsyncVectorEnv(env_fns) if config["async_envs"] else gymnasium.vector.SyncVectorEnv(env_fns)

    agent = PPOAgent(env.single_observation_space, env.single_action_space, config=config, device=config["device"])

    best_mean_reward = -np.inf
    global_step = 0
    if config["checkpoint_path"]:
        loaded_reward, loaded_step = load_checkpoint(agent, config["checkpoint_path"], config, config["device"])
        if loaded_reward is not None:
            best_mean_reward = loaded_reward
            global_step = loaded_step
            agent.steps_done = global_step

    buffer_size_per_env = config["buffer_size"] // config["num_envs"]
    buffer = RolloutBuffer(buffer_size_per_env, env.single_observation_space, env.single_action_space, num_envs=config["num_envs"], gamma=config["gamma"], gae_lambda=config["gae_lambda"], device=config["device"])

    writer = SummaryWriter(log_dir=config["log_dir"])
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)

    print(f"Starting training from step {global_step}/{config['total_timesteps']}")
    observations, infos = env.reset(seed=config["seed"])
    num_rollouts = 0
    current_episode_rewards = np.zeros(config["num_envs"], dtype=np.float32)
    current_episode_lengths = np.zeros(config["num_envs"], dtype=np.int32)
    start_time = time.time()

    autocast_context = torch.cuda.amp.autocast() if config["device"] == "cuda" and config["mixed_precision"] else nullcontext()
    scaler = torch.cuda.amp.GradScaler() if config["device"] == "cuda" and config["mixed_precision"] else None

    try:
        while global_step < config["total_timesteps"]:
            rollout_episode_rewards = []
            buffer.reset()
            steps_per_rollout = buffer.buffer_size
            last_dones = np.zeros(config["num_envs"], dtype=bool)

            # --- Rollout Phase ---
            for step in range(steps_per_rollout):
                obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=config["device"])
                with torch.no_grad():
                    actions, values, log_probs = agent.act(obs_tensor)

                next_observations, rewards, terminateds, truncateds, infos = env.step(actions)
                dones = terminateds | truncateds

                current_episode_rewards += rewards
                current_episode_lengths += 1

                buffer.add(observations, actions, rewards, terminateds, truncateds, values, log_probs)
                observations = next_observations
                last_dones = dones

                # --- Handle Episode Completions ---
                if "_final_info" in infos:
                    finished_mask = infos["_final_info"]
                    if np.any(finished_mask):
                        final_infos = infos["final_info"][finished_mask]
                        env_indices = np.where(finished_mask)[0]

                        for i, final_info in enumerate(final_infos):
                            if final_info is not None and "episode" in final_info:
                                ep_rew = final_info["episode"]["r"]
                                ep_len = final_info["episode"]["l"]
                                episode_rewards.append(ep_rew)
                                episode_lengths.append(ep_len)
                                rollout_episode_rewards.append(ep_rew)
                                print(f"Env {env_indices[i]} finished: Reward={ep_rew:.2f}, Length={ep_len}, Total Steps={global_step}")

                                current_episode_rewards[env_indices[i]] = 0
                                current_episode_lengths[env_indices[i]] = 0

                elif np.any(dones):
                    velocity_rews, survival_rews, track_pens, steering_pens = [], [], [], []
                    accel_turn_pens, steps_off, off_track_pcts = [], [], []

                    for i in range(config["num_envs"]):
                        if dones[i]:
                            ep_reward = current_episode_rewards[i]
                            ep_length = current_episode_lengths[i]
                            episode_rewards.append(ep_reward)
                            episode_lengths.append(ep_length)
                            rollout_episode_rewards.append(ep_reward)
                            print(f"Env {i} finished (manual): Reward={ep_reward:.2f}, Length={ep_length}, Total Steps={global_step}")

                            env_info = infos[i] if isinstance(infos, (list, tuple)) else infos.get(i)
                            if env_info:
                                if 'episode_velocity_rewards' in env_info: velocity_rews.append(env_info['episode_velocity_rewards'])
                                if 'episode_survival_rewards' in env_info: survival_rews.append(env_info['episode_survival_rewards'])
                                if 'episode_track_penalties' in env_info: track_pens.append(env_info['episode_track_penalties'])
                                if 'episode_steering_penalties' in env_info: steering_pens.append(env_info['episode_steering_penalties'])
                                if 'episode_acceleration_while_turning_penalties' in env_info: accel_turn_pens.append(env_info['episode_acceleration_while_turning_penalties'])
                                if 'steps_off_track' in env_info:
                                    steps_off.append(env_info['steps_off_track'])
                                    if ep_length > 0: off_track_pcts.append(100 * env_info['steps_off_track'] / ep_length)

                            current_episode_rewards[i] = 0
                            current_episode_lengths[i] = 0

                    if velocity_rews: writer.add_scalar("rewards/mean_velocity", np.mean(velocity_rews), global_step)
                    if survival_rews: writer.add_scalar("rewards/mean_survival", np.mean(survival_rews), global_step)
                    if track_pens: writer.add_scalar("penalties/mean_track", np.mean(track_pens), global_step)
                    if steering_pens: writer.add_scalar("penalties/mean_steering", np.mean(steering_pens), global_step)
                    if accel_turn_pens: writer.add_scalar("penalties/mean_accel_turn", np.mean(accel_turn_pens), global_step)
                    if steps_off: writer.add_scalar("driving/mean_steps_off_track", np.mean(steps_off), global_step)
                    if off_track_pcts: writer.add_scalar("driving/mean_percent_off_track", np.mean(off_track_pcts), global_step)

                global_step += config["num_envs"]
                if global_step >= config["total_timesteps"]:
                    break

            # --- Post-Rollout Phase ---
            with torch.no_grad():
                obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=config["device"])
                features = agent.feature_extractor(obs_tensor)
                last_values = agent.critic(features).cpu().numpy()

            buffer.compute_returns_and_advantages(last_values, last_dones)

            # --- Learning Phase ---
            if config["mixed_precision"] and config["device"] == "cuda":
                metrics = agent.learn_mixed_precision(buffer, scaler)
            else:
                metrics = agent.learn(buffer)

            # Update learning rate based on schedule
            current_lr = agent.update_learning_rate(config['total_timesteps'])
            num_rollouts += 1

            # --- Logging Dashboard Terminal Output ---
            if num_rollouts % config["log_interval"] == 0:
                fps = int(global_step / (time.time() - start_time))
                mean_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else -np.inf
                mean_length = np.mean(episode_lengths) if len(episode_lengths) > 0 else 0
                
                print(f"====== Rollout {num_rollouts} | Step {global_step}/{config['total_timesteps']} ======")
                print(f"Mean Reward (Last 100): {mean_reward:.2f}")
                print(f"Mean Episode Length: {mean_length:.1f}")
                print(f"FPS (SPS): {fps}")
                print(f"Learning Rate: {current_lr:.2e}")
                for k, v in metrics.items():
                    print(f"{k}: {v:.4f}")
                    writer.add_scalar(f"losses/{k}", v, global_step)
                writer.add_scalar("charts/mean_reward", mean_reward, global_step)
                writer.add_scalar("charts/mean_length", mean_length, global_step)
                writer.add_scalar("charts/fps", fps, global_step)
                writer.add_scalar("charts/learning_rate", current_lr, global_step)

            # --- Save Checkpoint ---
            if num_rollouts % config["save_interval"] == 0:
                mean_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else -np.inf
                checkpoint_data = {
                    'global_step': global_step,
                    'mean_reward': mean_reward,
                    'feature_extractor_state_dict': agent.feature_extractor.state_dict(),
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                }
                torch.save(checkpoint_data, os.path.join(config["save_dir"], "latest_model.pth"))
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    torch.save(checkpoint_data, os.path.join(config["save_dir"], "best_model.pth"))
                    print(f"✨ New best model saved with mean reward: {best_mean_reward:.2f}")

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current state...")
    finally:
        env.close()
        writer.close()
        print("Training loop cleaned up and closed.")