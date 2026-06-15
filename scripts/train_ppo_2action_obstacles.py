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
from src.ppo_agent_2 import PPOAgent  # 2-action 전용 (per-minibatch KL early stop)
from src.rollout_buffer import RolloutBuffer
import src.car_racing_obstacles  # noqa: F401  CarRacingObstacles-v0 등록

# --- Configuration --- #
config = {
    # Environment (A100 서버의 CPU-GPU 밸런스를 고려한 멀티환경 최적화)
    "env_id": "CarRacingObstacles-v0",   # 장애물 변형 환경 (src/car_racing_obstacles.py)
    "n_obstacles": 10,                   # 트랙당 장애물 수
    "obstacle_penalty": 15.0,            # 충돌 1회(스텝)당 패널티
    "obstacle_size_min": 0.25,           # ★ 장애물 크기 랜덤화: 개별 장애물마다 [min,max]*TRACK_WIDTH에서 샘플
    "obstacle_size_max": 0.6,            #   (min==max로 두면 고정 크기. max 0.6 = 도로폭의 ~30%라 통과 틈 보장)
    "frame_stack": 4,
    "num_envs": 64,                      # ★ 128 -> 64로 하향 조정 (원격 서버 CPU 병목을 해소하여 오히려 FPS 급상승)
    "max_episode_steps": 1000,           
    "seed": 42,                         

    # PPO Core Parameters (★ A100 80GB 맞춤형 스케일업 파라미터)
    "total_timesteps": 6000000,          # round 2는 행동 교정(코너 감속)이라 6M이면 충분 (--steps 로 변경 가능)
    "learning_rate": 1e-4,               # ★ 3e-4 -> 1e-4로 하향 (3e-4는 KL 폭발/collapse 유발, 3action 안정값)
    "buffer_size": 32768,                # ★ 64 envs * 512 steps = 32768 (더 풍부한 데이터 수집)
    "batch_size": 2048,                  # ★ 2048로 상향 (미니배치 업데이트 횟수를 늘려 학습 효율 극대화)
    "ppo_epochs": 6,                     # ★ 10 -> 6으로 하향 (수집한 버퍼를 더 알차게 여러번 학습)                     
    "gamma": 0.99,                      
    "gae_lambda": 0.95,                  
    "clip_epsilon": 0.15,                # ★ 0.2 -> 0.15 (trust region 축소로 안정화)
    "vf_coef": 0.5,                      # ★ 0.75 -> 0.5로 하향 (표준 PPO 값으로 안정화)                      
    "ent_coef": 0.01,                    # ★ 0.02 -> 0.01로 하향 (탐색 강박 완화)                    
    "max_grad_norm": 0.5,               
    "target_kl": 0.03,                   # ★ 0.015 -> 0.03 (이제 미니배치마다 강제되므로 살짝 완화해 학습속도 확보)
    "features_dim": 256,                 

    # Agent specific hyperparameters
    "initial_action_std": 0.5,           # ★ 0.4 -> 0.5 (초기 탐색 안정화)
    "weight_decay": 1e-6,               
    "fixed_std": False,                 
    "lr_warmup_steps": 0,                # ★ 0으로 변경 (LR이 역으로 상승하던 스케줄러 버그 원천 차단)
    "min_learning_rate": 1e-5,          # ★ 최저 러닝레이트 하한선 보장          

    # Reward shaping 
    "use_reward_shaping": True,         
    "velocity_reward_weight": 0.003,     # ★ 0.003 (실속도≈20~60 사용; 0.03이면 step당 ~0.9로 기본보상 압도 → 0.003으로 낮춤)
    "survival_reward": 0.0,              # ★ 0.01 -> 0.0으로 제거 (가만히 서서 버티는 꼼수 차단)            
    "track_penalty": 1.0,               
    "steering_smooth_weight": 0.001,     # ★ 0.01 -> 0.001로 대폭 하향 (움직임을 너무 방해함)     
    "acceleration_while_turning_penalty_weight": 0.5, # ★ 0.5로 활성화: 급커브 풀스로틀 진입(트랙 이탈 원인) 억제. 직진 가속은 영향 없음

    # Performance optimizations
    "torch_num_threads": 2,              # ★ 16 -> 2로 차단 (파이토치가 CPU를 독점하지 못하게 막아야 env 분산 처리가 빨라집니다)
    "mixed_precision": True,             
    "pin_memory": True,                 
    "async_envs": True,                 

    # Logging and saving
    "log_interval": 1,                  
    "save_interval": 10,                
    "save_dir": "./models/ppo_2action_obstacles2",   # round 2 (코너 감속 학습) — round 1 결과 보존
    "log_dir": "./logs/ppo_2action_obstacles2",

    # round 2 기본: round 1 장애물 회피 best(475)에서 이어서 코너 감속만 추가 학습.
    # (--checkpoint 로 변경 가능, None 주면 처음부터)
    "checkpoint_path": "./models/ppo_2action_obstacles/best_model.pth",
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
        env = gym.make(env_id, continuous=True, domain_randomize=False, render_mode=None,
                       n_obstacles=config["n_obstacles"],
                       obstacle_penalty=config["obstacle_penalty"],
                       obstacle_size_min=config["obstacle_size_min"],
                       obstacle_size_max=config["obstacle_size_max"])
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

# --- RewardShapingWrapper 수정안 --- #
class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, velocity_weight: float = 0.2, survival_reward: float = 0.0,
                 track_penalty: float = 1.5, steering_smooth_weight: float = 0.0005,
                 acceleration_while_turning_penalty_weight: float = 0.0):
        super().__init__(env)
        self.velocity_weight = velocity_weight
        self.survival_reward = survival_reward
        self.track_penalty = track_penalty
        self.steering_smooth_weight = steering_smooth_weight
        self.accel_turn_weight = acceleration_while_turning_penalty_weight

        self.last_steering = 0.0
        self.episode_velocity_rewards = 0.0
        self.episode_track_penalties = 0.0
        self.episode_steering_penalties = 0.0
        self.episode_accel_turn_penalties = 0.0
        self.steps_off_track = 0
        self.episode_obstacle_hits = 0

    def reset(self, **kwargs):
        self.last_steering = 0.0
        self.episode_velocity_rewards = 0.0
        self.episode_track_penalties = 0.0
        self.episode_steering_penalties = 0.0
        self.episode_accel_turn_penalties = 0.0
        self.steps_off_track = 0
        self.episode_obstacle_hits = 0

        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info is None: info = {}

        # 장애물 충돌 누적 (패널티 자체는 CarRacingObstacles 환경이 이미 reward에 반영)
        if info.get('obstacle_hit'):
            self.episode_obstacle_hits += 1
        info['episode_obstacle_hits'] = self.episode_obstacle_hits

        steering = action[0]
        # CarRacing-v3는 info에 'speed'를 주지 않는다(항상 0이 되어 속도 보상이 죽음).
        # 차량 물리(linearVelocity)에서 실제 속도를 계산한다.
        car = getattr(self.env.unwrapped, "car", None)
        if car is not None:
            vx, vy = car.hull.linearVelocity
            speed = float((vx * vx + vy * vy) ** 0.5)
        else:
            speed = 0.0

        step_velocity_reward = 0.0
        step_track_penalty = 0.0
        step_steering_penalty = 0.0
        off_track = False

        # 1. 오프 트랙 감지 (기존의 초록색 채널 분석 활용)
        if len(obs.shape) == 3 and obs.shape[2] == 3: 
            car_area = obs[84:94, 42:54, :]
            green_channel = car_area[:, :, 1]
            red_channel = car_area[:, :, 0]
            off_track = np.mean(green_channel) > 150 and np.mean(red_channel) < 100

        # 2. 보상 설계 단순화 (꼼수 차단)
        if off_track:
            # 트랙을 벗어나면 속도 보상을 주지 않고 강한 패널티만 부여
            step_track_penalty = self.track_penalty
            reward -= step_track_penalty
            self.episode_track_penalties += step_track_penalty
            self.steps_off_track += 1
        else:
            # 트랙 위에 있을 때만 속도에 비례한 보상 부여 (트랙 위에서 질주하도록 유도)
            if speed > 0:
                step_velocity_reward = speed * self.velocity_weight
                reward += step_velocity_reward
                self.episode_velocity_rewards += step_velocity_reward

        # 3. 조향 급변 패널티 (지나친 좌우 흔들림 방지용 최소한의 장치)
        steering_change = abs(steering - self.last_steering)
        step_steering_penalty = steering_change * self.steering_smooth_weight
        reward -= step_steering_penalty
        self.episode_steering_penalties += step_steering_penalty

        self.last_steering = steering

        # 4. 코너 가속 패널티: 조향 중 가속(급커브 과속 진입 -> 트랙 이탈) 억제.
        #    이 wrapper는 ActionWrapper 안쪽이라 action은 3D [steer, gas, brake].
        #    직진 가속(|steer|~0)은 비용이 없고, 조향하며 밟을 때만 패널티.
        step_accel_turn_penalty = 0.0
        if self.accel_turn_weight > 0 and len(action) >= 2:
            gas = max(0.0, float(action[1]))
            step_accel_turn_penalty = self.accel_turn_weight * gas * abs(float(steering))
            reward -= step_accel_turn_penalty
            self.episode_accel_turn_penalties += step_accel_turn_penalty

        # 정보 저장
        info['velocity_rewards'] = step_velocity_reward
        info['track_penalties'] = step_track_penalty
        info['steering_penalties'] = step_steering_penalty
        info['off_track'] = off_track
        info['episode_velocity_rewards'] = self.episode_velocity_rewards
        info['episode_track_penalties'] = self.episode_track_penalties
        info['episode_steering_penalties'] = self.episode_steering_penalties
        info['episode_acceleration_while_turning_penalties'] = self.episode_accel_turn_penalties
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
    parser.add_argument("--save-dir", type=str, default=None, help="Override save directory")
    parser.add_argument("--n-obstacles", type=int, default=None, help="Override number of obstacles")
    parser.add_argument("--obstacle-size-min", type=float, default=None,
                        help="Min obstacle size (fraction of TRACK_WIDTH). size_frac=2.0 == full road width; >2.0 is bigger than the road.")
    parser.add_argument("--obstacle-size-max", type=float, default=None,
                        help="Max obstacle size (fraction of TRACK_WIDTH). Set min==max for fixed size.")
    parser.add_argument("--track-penalty", type=float, default=None,
                        help="Off-track penalty per step. Lower it for oversized obstacles where detouring off-road is the only way past.")
    parser.add_argument("--accel-turn-weight", type=float, default=None,
                        help="Penalty weight for accelerating while steering (gas*|steer|). 0 disables. 0.5 was too high (cost reward without helping).")
    parser.add_argument("--ent-coef", type=float, default=None,
                        help="Entropy coefficient. Lower (e.g. 0.005) reduces action noise -> fewer sampling-induced obstacle hits.")
    args = parser.parse_args()

    if args.checkpoint: config["checkpoint_path"] = args.checkpoint
    if args.steps is not None: config["total_timesteps"] = args.steps
    if args.seed is not None: config["seed"] = args.seed
    if args.log_dir is not None: config["log_dir"] = args.log_dir
    if args.save_dir is not None: config["save_dir"] = args.save_dir
    if args.n_obstacles is not None: config["n_obstacles"] = args.n_obstacles
    if args.obstacle_size_min is not None: config["obstacle_size_min"] = args.obstacle_size_min
    if args.obstacle_size_max is not None: config["obstacle_size_max"] = args.obstacle_size_max
    if args.track_penalty is not None: config["track_penalty"] = args.track_penalty
    if args.accel_turn_weight is not None: config["acceleration_while_turning_penalty_weight"] = args.accel_turn_weight
    if args.ent_coef is not None: config["ent_coef"] = args.ent_coef

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
        # Fine-tune: 가중치만 가져오고 step/best는 리셋한다.
        # - global_step을 체크포인트(9.8M)로 이어받으면 total 10M 중 0.2M만 돌고 끝나고
        #   LR 코사인도 끝물에서 시작해 학습이 안 된다.
        # - 장애물 환경의 shaped reward는 기존 트랙(838)과 비교 불가라 best도 -inf로 리셋
        #   (안 하면 best_model.pth가 영영 저장되지 않음).
        load_checkpoint(agent, config["checkpoint_path"], config, config["device"])
        best_mean_reward = -np.inf
        global_step = 0
        agent.steps_done = 0
        print("Fine-tuning: weights loaded; global_step/best_mean_reward reset for the new task.")

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
                    obstacle_hits_list = []

                    for i in range(config["num_envs"]):
                        if dones[i]:
                            ep_reward = current_episode_rewards[i]
                            ep_length = current_episode_lengths[i]
                            episode_rewards.append(ep_reward)
                            episode_lengths.append(ep_length)
                            rollout_episode_rewards.append(ep_reward)
                            print(f"Env {i} finished (manual): Reward={ep_reward:.2f}, Length={ep_length}, Total Steps={global_step}")

                            # gymnasium 벡터 env의 infos는 {키: (num_envs,) 배열} 형태다.
                            # (이전 코드의 infos.get(i)는 정수키라 항상 None → 셰이핑 지표가
                            #  전부 미기록됐었음.) done 시점의 infos[key][i]가 그 env의 누적값.
                            def _m(key):
                                arr = infos.get(key)
                                if arr is not None and np.ndim(arr) >= 1 and len(arr) > i:
                                    return arr[i]
                                return None
                            v = _m('episode_velocity_rewards');               velocity_rews.append(v) if v is not None else None
                            v = _m('episode_survival_rewards');               survival_rews.append(v) if v is not None else None
                            v = _m('episode_track_penalties');                track_pens.append(v) if v is not None else None
                            v = _m('episode_steering_penalties');             steering_pens.append(v) if v is not None else None
                            v = _m('episode_acceleration_while_turning_penalties'); accel_turn_pens.append(v) if v is not None else None
                            v = _m('episode_obstacle_hits');                  obstacle_hits_list.append(v) if v is not None else None
                            so = _m('steps_off_track')
                            if so is not None:
                                steps_off.append(so)
                                if ep_length > 0: off_track_pcts.append(100 * so / ep_length)

                            current_episode_rewards[i] = 0
                            current_episode_lengths[i] = 0

                    if velocity_rews: writer.add_scalar("rewards/mean_velocity", np.mean(velocity_rews), global_step)
                    if survival_rews: writer.add_scalar("rewards/mean_survival", np.mean(survival_rews), global_step)
                    if track_pens: writer.add_scalar("penalties/mean_track", np.mean(track_pens), global_step)
                    if steering_pens: writer.add_scalar("penalties/mean_steering", np.mean(steering_pens), global_step)
                    if accel_turn_pens: writer.add_scalar("penalties/mean_accel_turn", np.mean(accel_turn_pens), global_step)
                    if steps_off: writer.add_scalar("driving/mean_steps_off_track", np.mean(steps_off), global_step)
                    if off_track_pcts: writer.add_scalar("driving/mean_percent_off_track", np.mean(off_track_pcts), global_step)
                    if obstacle_hits_list: writer.add_scalar("driving/mean_obstacle_hits", np.mean(obstacle_hits_list), global_step)

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

            # 코사인 감쇠: env-step(global_step) 기준으로 LR이 1e-4 -> min_lr로 감쇠.
            # (롤아웃 카운터로 나누던 기존 방식은 progress가 항상 ~0이라 LR이 고정됐음)
            current_lr = agent.update_learning_rate(global_step, config["total_timesteps"])

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