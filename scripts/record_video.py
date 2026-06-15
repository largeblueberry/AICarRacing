"""Record a video of a trained PPO agent driving in CarRacing-v3.

This script reproduces the *exact* environment that ``scripts/evaluate_agent.py``
uses (same wrappers, same seeding scheme, same stochastic action sampling) but
renders with ``render_mode="rgb_array"`` so every frame can be captured and
written to an ``.mp4`` file. Use it to actually watch how the best model drove
during evaluation.

Key design choices for faithfulness to the evaluation:
  * Same wrapper stack: GrayScaleObservation -> TimeLimit -> FrameStack.
  * Same seeding: ``set_seeds(seed)`` once, ``env.reset(seed=seed+100)`` at
    creation, and per-episode ``env.reset(seed=seed+episode)``.
  * Same action selection: ``agent.act()`` samples from the policy (matches the
    evaluation graphs). Pass ``--deterministic`` to use the distribution mean
    instead for a cleaner, repeatable run.
  * Agent dimensions (``features_dim``, action dim) are read from the
    checkpoint's embedded ``config`` when present, so any downloaded model just
    works without manually matching flags.

Examples:
    # Record 3 episodes of the best saved agent (seed 42, like the eval graphs)
    python scripts/record_video.py --model BestSavedAgents/Evaluated679.pth

    # Record a single deterministic run of a freshly downloaded model
    python scripts/record_video.py --model models/ppo_simple/best_model.pth \
        --episodes 1 --deterministic
"""

import argparse
import os
import sys
import typing

import numpy as np
import torch

# Make ``src`` importable no matter how this script is launched.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gymnasium as gym  # noqa: E402

from src.env_wrappers import (  # noqa: E402
    GrayScaleObservation, FrameStack, TimeLimit, ActionWrapper)
from src.ppo_agent import PPOAgent  # noqa: E402
from src.random_agent import RandomAgent  # noqa: E402


# --- Defaults kept consistent with scripts/evaluate_agent.py --- #
DEFAULTS = {
    "env_id": "CarRacing-v3",
    "frame_stack": 4,
    "seed": 42,
    "max_episode_steps": 1000,
    "features_dim": 256,
    # Placeholder PPO hyperparameters (only used to build the network skeleton).
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.1,
    "ppo_epochs": 5,
    "batch_size": 64,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "max_grad_norm": 0.5,
    "target_kl": 0.02,
    "initial_action_std": 1.0,
    "weight_decay": 1e-5,
    "fixed_std": False,
    "lr_warmup_steps": 0,
    "min_learning_rate": 1e-7,
}


def set_seeds(seed: int) -> None:
    """Seed NumPy and PyTorch (mirrors evaluate_agent.set_seeds)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def make_env(env_id: str, seed: int, frame_stack: int,
             render_mode: typing.Union[str, None],
             max_episode_steps: int,
             two_action: bool = False,
             obstacles: bool = False,
             n_obstacles: int = 10,
             obstacle_size_min: float = 0.25,
             obstacle_size_max: float = 0.6) -> gym.Env:
    """Create the evaluation environment, identical to evaluate_agent.make_env.

    The only difference from evaluation is ``render_mode`` (we use
    ``"rgb_array"`` here). The render mode does not affect the physics, track
    generation, or RNG consumption, so the driven trajectory matches the
    ``render_mode=None`` evaluation run for the same seeds.

    ``two_action=True`` inserts the ActionWrapper (2D [steering, throttle] ->
    3D [steering, gas, brake]) right after env creation, matching
    evaluate_agent_2action.py, so 2-action checkpoints load and run correctly.
    """
    # CarRacing-v3 is registered with a built-in TimeLimit of 1000 steps, so the
    # effective limit is min(this inner limit, our outer TimeLimit). Pass
    # max_episode_steps here too so --max-steps can actually exceed 1000.
    # (At the default 1000 this is identical to evaluate_agent.py.)
    if obstacles:
        import src.car_racing_obstacles  # noqa: F401  registers the env id
        env = gym.make("CarRacingObstacles-v0", continuous=True,
                       domain_randomize=False, render_mode=render_mode,
                       max_episode_steps=max_episode_steps,
                       n_obstacles=n_obstacles,
                       obstacle_size_min=obstacle_size_min,
                       obstacle_size_max=obstacle_size_max)
    else:
        env = gym.make(env_id, continuous=True, domain_randomize=False,
                       render_mode=render_mode, max_episode_steps=max_episode_steps)
    env.reset(seed=seed + 100)  # same offset evaluate_agent.py uses
    env.action_space.seed(seed + 100)

    if two_action:
        env = ActionWrapper(env)
    env = GrayScaleObservation(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = FrameStack(env, frame_stack)
    return env


def detect_action_dim(checkpoint: typing.Optional[dict]) -> typing.Optional[int]:
    """Read the policy action dimension from a checkpoint (None if unknown)."""
    if not isinstance(checkpoint, dict):
        return None
    actor = checkpoint.get("actor_state_dict")
    if isinstance(actor, dict) and "fc_mean.weight" in actor:
        return int(actor["fc_mean.weight"].shape[0])
    return None


def pick_device(requested: str) -> str:
    """Resolve the compute device string."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_config(checkpoint: dict, args: argparse.Namespace) -> dict:
    """Build the agent config, preferring values embedded in the checkpoint.

    The training checkpoints save the exact ``config`` used. Reading
    ``features_dim`` (and friends) from there means a downloaded model with a
    different feature dimension still loads correctly without extra flags.
    """
    config = dict(DEFAULTS)
    embedded = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if isinstance(embedded, dict):
        for key in ("features_dim", "frame_stack", "initial_action_std",
                    "fixed_std", "max_episode_steps"):
            if key in embedded and embedded[key] is not None:
                config[key] = embedded[key]
    # Explicit CLI overrides win over everything.
    if args.features_dim is not None:
        config["features_dim"] = args.features_dim
    config["max_episode_steps"] = args.max_steps
    return config


def load_agent(model_path: str, env: gym.Env, config: dict, device: str) -> PPOAgent:
    """Instantiate a PPOAgent and load weights from a checkpoint."""
    agent = PPOAgent(env.observation_space, env.action_space,
                     config=config, device=device)
    # weights_only=False: these are our own training checkpoints (trusted) and
    # contain non-tensor objects (numpy scalars, optimizer state, config dict).
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    agent.feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    if "critic_state_dict" in checkpoint:
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])
    agent.feature_extractor.eval()
    agent.actor.eval()
    agent.critic.eval()
    if "mean_reward" in checkpoint:
        print(f"Checkpoint reports mean_reward = {checkpoint['mean_reward']:.2f}")
    return agent


def select_action(agent, observation: np.ndarray, deterministic: bool,
                  device: str) -> np.ndarray:
    """Pick an action for the current observation.

    ``deterministic=False`` reproduces evaluate_agent.py exactly (samples from
    the policy via ``agent.act``). ``deterministic=True`` uses the distribution
    mean for a cleaner, fully repeatable rollout.
    """
    if isinstance(agent, RandomAgent):
        actions, _, _ = agent.act(observation)
        return actions[0]

    if not deterministic:
        actions, _, _ = agent.act(observation)
        return actions[0]

    # Deterministic: use the mean of the action distribution.
    with torch.no_grad():
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        if obs_tensor.ndim == 3:
            obs_tensor = obs_tensor.unsqueeze(0)
        features = agent.feature_extractor(obs_tensor)
        dist = agent.actor.get_action_dist(features)
        action = dist.mean
    return action.detach().cpu().numpy()[0]


def annotate(frame: np.ndarray, lines: typing.List[str]) -> np.ndarray:
    """Draw informational text onto a copy of an RGB frame (best effort)."""
    try:
        import cv2
    except Exception:
        return frame
    frame = np.ascontiguousarray(frame)
    scale = max(frame.shape[0] / 400.0, 0.4)
    y = int(18 * scale)
    for line in lines:
        # Black outline then white text so it is readable on any background.
        cv2.putText(frame, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                    (0, 0, 0), max(2, int(3 * scale)), cv2.LINE_AA)
        cv2.putText(frame, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                    (255, 255, 255), max(1, int(1 * scale)), cv2.LINE_AA)
        y += int(20 * scale)
    return frame


def write_video(path: str, frames: typing.List[np.ndarray], fps: int) -> None:
    """Write frames to an mp4 using imageio (ffmpeg backend)."""
    import imageio.v2 as imageio
    # macro_block_size=1 avoids silent resizing of odd dimensions.
    with imageio.get_writer(path, fps=fps, macro_block_size=1,
                            codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a video of a trained PPO agent driving CarRacing-v3.")
    parser.add_argument("--model", type=str,
                        default="BestSavedAgents/Evaluated679.pth",
                        help="Path to the .pth checkpoint to load.")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to record (default: 3).")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                        help="Base seed; matches the evaluation graphs (default: 42).")
    parser.add_argument("--max-steps", type=int, default=DEFAULTS["max_episode_steps"],
                        help="Max steps per episode (default: 1000).")
    parser.add_argument("--features-dim", type=int, default=None,
                        help="Override CNN feature dim (otherwise read from checkpoint).")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use the policy mean instead of sampling.")
    parser.add_argument("--random", action="store_true",
                        help="Record a random-agent baseline instead of a model.")
    parser.add_argument("--out-dir", type=str, default="videos",
                        help="Directory to write mp4 files into (default: videos/).")
    parser.add_argument("--fps", type=int, default=None,
                        help="Video frame rate (default: env render_fps, ~50).")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Disable the reward/step text overlay.")
    parser.add_argument("--action-dim", type=str, default="auto",
                        choices=["auto", "2", "3"],
                        help="Policy action dim. 'auto' detects it from the "
                             "checkpoint; '2' adds the ActionWrapper "
                             "(steering/throttle), '3' is native CarRacing.")
    parser.add_argument("--obstacles", action="store_true",
                        help="Use the CarRacingObstacles-v0 env (random static "
                             "obstacles on the road).")
    parser.add_argument("--n-obstacles", type=int, default=10,
                        help="Number of obstacles when --obstacles is set (default: 10).")
    parser.add_argument("--obstacle-size-min", type=float, default=0.25,
                        help="Min obstacle size frac of TRACK_WIDTH (2.0==road width; >2.0 bigger than road).")
    parser.add_argument("--obstacle-size-max", type=float, default=0.6,
                        help="Max obstacle size frac of TRACK_WIDTH.")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps", "auto"],
                        help="Compute device (default: cpu for reproducibility).")
    args = parser.parse_args()

    device = pick_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.random and not os.path.exists(args.model):
        print(f"Error: model checkpoint not found: {args.model}")
        sys.exit(1)

    set_seeds(args.seed)

    # Load checkpoint first so we can read the embedded config before building env.
    checkpoint = None
    if not args.random:
        # Trusted local checkpoint (our own training output); see load_agent note.
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    config = build_config(checkpoint or {}, args)

    # Decide whether the policy uses the 2D action space (needs ActionWrapper).
    if args.action_dim == "auto":
        detected = detect_action_dim(checkpoint)
        two_action = (detected == 2)
        if detected is not None:
            print(f"Detected action_dim={detected} from checkpoint.")
    else:
        two_action = (args.action_dim == "2")

    env = make_env(DEFAULTS["env_id"], args.seed, config["frame_stack"],
                   render_mode="rgb_array",
                   max_episode_steps=config["max_episode_steps"],
                   two_action=two_action,
                   obstacles=args.obstacles,
                   n_obstacles=args.n_obstacles,
                   obstacle_size_min=args.obstacle_size_min,
                   obstacle_size_max=args.obstacle_size_max)

    fps = args.fps or int(env.metadata.get("render_fps", 50))

    if args.random:
        agent = RandomAgent(env.observation_space, env.action_space, device=device)
        model_name = "random_baseline"
        print("Recording random-agent baseline.")
    else:
        agent = load_agent(args.model, env, config, device)
        model_name = os.path.splitext(os.path.basename(args.model))[0]

    print("--- Recording Configuration ---")
    print(f"Device:        {device}")
    print(f"Model:         {args.model if not args.random else 'random'}")
    print(f"Features dim:  {config['features_dim']}")
    print(f"Action space:  {'2D (steering, throttle) + ActionWrapper' if two_action else '3D (steering, gas, brake)'}")
    print(f"Episodes:      {args.episodes}")
    print(f"Seed (base):   {args.seed}")
    print(f"Max steps:     {config['max_episode_steps']}")
    print(f"Action mode:   {'deterministic (mean)' if args.deterministic else 'sampled (matches eval)'}")
    print(f"Output dir:    {os.path.abspath(args.out_dir)}")
    print(f"Video FPS:     {fps}")
    print("-------------------------------")

    saved_paths = []
    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        frames = []

        # Capture the very first frame after reset.
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        while not (terminated or truncated):
            action = select_action(agent, observation, args.deterministic, device)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            frame = env.render()
            if frame is not None:
                if not args.no_overlay:
                    frame = annotate(frame, [
                        f"{model_name}",
                        f"ep {episode + 1}/{args.episodes}  seed {args.seed + episode}",
                        f"step {steps}  reward {total_reward:7.2f}",
                    ])
                frames.append(frame)

        reward_tag = f"r{int(round(total_reward))}"
        filename = f"{model_name}_ep{episode + 1}_seed{args.seed + episode}_{reward_tag}.mp4"
        out_path = os.path.join(args.out_dir, filename)
        write_video(out_path, frames, fps)
        saved_paths.append(out_path)
        print(f"Episode {episode + 1}/{args.episodes}: "
              f"reward={total_reward:.2f} steps={steps} -> {out_path}")

    env.close()

    print("\nSaved videos:")
    for path in saved_paths:
        print(f"  {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
