import gymnasium as gym
import torch
import numpy as np
import os
import time
import argparse
import sys
import matplotlib.pyplot as plt
import typing

# Make ``src`` importable when run as `python scripts/evaluate_agent_2action.py`.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.env_wrappers import GrayScaleObservation, FrameStack, TimeLimit, ActionWrapper
from src.ppo_agent import PPOAgent
from src.random_agent import RandomAgent

# --- Configuration --- #
# Default configuration for evaluation script
config = {
    # Environment settings
    "env_id": "CarRacing-v3",
    "frame_stack": 4,
    "seed": 42, # Seed used for all evaluation graphs
    "max_episode_steps": 1000, # Max steps per evaluation episode

    # Agent settings (Required for PPOAgent initialization, even if not used for eval logic)
    "features_dim": 256,          # Feature dimension (MUST match trained model or else this will break)
    "learning_rate": 1e-4,        # Placeholder LR
    "gamma": 0.99,                # Placeholder gamma
    "gae_lambda": 0.95,            # Placeholder lambda
    "clip_epsilon": 0.1,          # Placeholder clip epsilon
    "ppo_epochs": 5,              # Placeholder epochs
    "batch_size": 64,             # Placeholder batch size
    "vf_coef": 0.5,               # Placeholder vf coef
    "ent_coef": 0.01,             # Placeholder ent coef
    "max_grad_norm": 0.5,         # Placeholder grad norm
    "target_kl": 0.02,            # Placeholder target kl
    "initial_action_std": 1.0,    # Placeholder action std
    "weight_decay": 1e-5,         # Placeholder weight decay
    "fixed_std": False,           # Placeholder fixed std flag
    "lr_warmup_steps": 0,         # Placeholder warmup steps
    "min_learning_rate": 1e-7,    # Placeholder min LR

    # Evaluation settings
    "n_eval_episodes": 10,        # Number of episodes to run for evaluation (100 for all evaluation graphs)
    "render_mode": None,       # Set to "human" to watch the agent play

    # Hardware
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def set_seeds(seed: int):
    """
    Sets random seeds for NumPy and PyTorch for reproducible evaluation.

    Args:
        seed: The random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def make_env(env_id: str, seed: int, frame_stack: int, render_mode: typing.Union[str, None] = None, max_episode_steps: int = 1000):
    """
    Creates and wraps the (base, no-obstacle) evaluation environment.

    Applies necessary wrappers (GrayScaleObservation, TimeLimit, FrameStack)
    consistent with the training setup, but without reward shaping. For obstacle
    evaluation use scripts/evaluate_agent_2action_obstacles.py instead.

    Args:
        env_id: The ID of the Gymnasium environment.
        seed: The random seed for environment initialization.
        frame_stack: The number of frames to stack.
        render_mode: The render mode ('human' or None).
        max_episode_steps: Maximum steps per episode for the TimeLimit wrapper.

    Returns:
        The wrapped Gymnasium environment.
    """
    # Create the base environment
    env = gym.make(env_id, continuous=True, domain_randomize=False, render_mode=render_mode)
    # Seed the environment (use a different offset than training if desired)
    env.reset(seed=seed + 100) # Use a different seed offset for evaluation
    env.action_space.seed(seed + 100)

    # Apply standard wrappers (must match training configuration except for reward shaping)
    env = ActionWrapper(env)
    env = GrayScaleObservation(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = FrameStack(env, frame_stack)
    return env

# --- Main Evaluation Script --- #
if __name__ == "__main__":
    # --- Default Model Path (override with --model) --- #
    DEFAULT_MODEL_PATH = "./models/ppo_2action2/best_model.pth"
    # ---------------------------------- #

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent on CarRacing-v3")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Path to the .pth checkpoint to evaluate (default: {DEFAULT_MODEL_PATH}).")
    parser.add_argument("--episodes", type=int, default=config["n_eval_episodes"],
                        help=f"Number of episodes to run for evaluation (default: {config['n_eval_episodes']}).")
    parser.add_argument("--seed", type=int, default=config["seed"],
                        help=f"Base random seed for environment reset during evaluation (default: {config['seed']}).")
    parser.add_argument("--render", action='store_true',
                        help="Enable rendering to watch the agent play (default: disabled).")
    parser.add_argument("--features-dim", type=int, default=config["features_dim"],
                        help=f"Feature dimension of the loaded model's CNN (default: {config['features_dim']}). MUST match the trained model.")
    parser.add_argument("--max-steps", type=int, default=config["max_episode_steps"],
                        help=f"Maximum steps per evaluation episode (default: {config['max_episode_steps']}).")
    parser.add_argument("--random", action='store_true',
                        help="Use a random agent instead of loading a trained model (default: disabled).")
    args = parser.parse_args()

    # All downstream code references HARDCODED_MODEL_PATH; bind it to --model.
    HARDCODED_MODEL_PATH = args.model

    # --- Configuration Update ---
    # Override default config with command-line arguments
    config["features_dim"] = args.features_dim
    config["n_eval_episodes"] = args.episodes
    config["seed"] = args.seed
    config["max_episode_steps"] = args.max_steps
    render_mode = "human" if args.render else None

    print("--- Evaluation Configuration ---")
    print(f"Device: {config['device']}")
    if not args.random:
        print(f"Model Path: {HARDCODED_MODEL_PATH}") # Use hardcoded path
    else:
        print(f"Agent: Random (baseline comparison)")
    print(f"Evaluation Episodes: {config['n_eval_episodes']}")
    print(f"Environment Seed: {config['seed']}")
    print(f"Features Dimension: {config['features_dim']}")
    print(f"Max Steps per Episode: {config['max_episode_steps']}")
    print(f"Rendering: {'Enabled' if render_mode else 'Disabled'}")
    print("------------------------------")

    # Set random seeds for evaluation
    set_seeds(config["seed"])

    # --- Environment and Agent Setup ---
    # Create the evaluation environment
    env = make_env(config["env_id"], config["seed"], config["frame_stack"], render_mode, config["max_episode_steps"])

    if args.random:
        # Use RandomAgent for baseline comparison
        agent = RandomAgent(env.observation_space, env.action_space, device=config["device"])
        print("Using random agent as baseline")
    else:
        # Initialize the PPOAgent structure (weights will be loaded)
        # Crucially, features_dim must match the loaded checkpoint
        agent = PPOAgent(env.observation_space,
                         env.action_space,
                         config=config, # Pass the config dictionary
                         device=config["device"])

        # --- Load Model Weights ---
        if not os.path.exists(HARDCODED_MODEL_PATH): # Use hardcoded path
            print(f"Error: Model checkpoint not found at {HARDCODED_MODEL_PATH}")
            exit(1)

        print(f"Loading model weights from {HARDCODED_MODEL_PATH}...") # Use hardcoded path
        try:
            # Load the checkpoint onto the specified device
            # weights_only=False: our own training checkpoint contains numpy scalars
            # (mean_reward) etc.; torch>=2.6 defaults to True and would refuse to load.
            checkpoint = torch.load(HARDCODED_MODEL_PATH, map_location=config["device"], weights_only=False)

            # Load state dictionaries for the networks
            agent.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            # Critic state dict might not always be saved or needed for evaluation, but load if present
            if 'critic_state_dict' in checkpoint:
                agent.critic.load_state_dict(checkpoint['critic_state_dict'])
                print("Loaded Feature Extractor, Actor, and Critic weights.")
            else:
                print("Loaded Feature Extractor and Actor weights (Critic weights not found in checkpoint).")

        except KeyError as e:
            print(f"Error loading model: Missing key {e} in checkpoint. Ensure the checkpoint structure is correct and matches the agent definition (especially features_dim).")
            exit(1)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            exit(1)

        # Set agent networks to evaluation mode (disables dropout, etc.)
        agent.feature_extractor.eval()
        agent.actor.eval()
        agent.critic.eval() # Set critic to eval mode as well

    # --- Evaluation Loop ---
    episode_rewards = []
    episode_lengths = []

    print(f"\nStarting evaluation for {config['n_eval_episodes']} episodes...")
    for episode in range(config["n_eval_episodes"]):
        # Reset environment with a unique seed for each episode for variability
        observation, info = env.reset(seed=config["seed"] + episode)
        terminated = False
        truncated = False
        current_episode_reward = 0
        current_episode_length = 0

        # Run one episode
        while not (terminated or truncated):
            try:
                # Observation shape check (should be k, H, W)
                if len(observation.shape) != 3:
                    print(f"Warning: Unexpected observation shape {observation.shape} in episode {episode + 1}. Expected 3 dims.")
                    # Attempt to reshape if it looks like a batch dim issue
                    if len(observation.shape) == 4 and observation.shape[0] == 1:
                        observation = observation.squeeze(0)
                    else:
                         raise ValueError("Cannot proceed with incompatible observation shape.")

                # Add batch dimension for the agent's act method
                # obs_batch = np.expand_dims(observation, axis=0)
                # Convert to tensor on the correct device (agent.act now handles numpy input)
                # obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=config["device"])

                # Get action from agent (samples from the policy, matching the
                # evaluation graphs).
                with torch.no_grad():
                    # Pass observation directly to act (handles tensor conversion)
                    actions, _, _ = agent.act(observation)
                    # actions is shape (1, action_dim), take the first element
                    action = actions[0]

                # Step the environment
                observation, reward, terminated, truncated, info = env.step(action)

                current_episode_reward += reward
                current_episode_length += 1

                # Optional delay for smoother rendering
                if render_mode == "human":
                    time.sleep(0.01)

            except Exception as e:
                print(f"Error during episode {episode + 1} step {current_episode_length}: {e}")
                # Log details for debugging
                print(f"Observation shape: {observation.shape if 'observation' in locals() else 'unknown'}")
                print(f"Action attempted: {action if 'action' in locals() else 'unknown'}")
                import traceback
                traceback.print_exc()
                terminated = True # End the episode prematurely on error

        # Log episode results
        print(f"Episode {episode + 1}/{config['n_eval_episodes']}: Reward = {current_episode_reward:.2f}, Length = {current_episode_length}")
        episode_rewards.append(current_episode_reward)
        episode_lengths.append(current_episode_length)

    # --- Cleanup and Results --- 
    env.close()

    # Calculate summary statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    median_reward = np.median(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    lower_quartile = np.percentile(episode_rewards, 25)
    upper_quartile = np.percentile(episode_rewards, 75)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    print("\n--- Evaluation Summary ---")
    print(f"Number of Episodes: {config['n_eval_episodes']}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Median Reward: {median_reward:.2f}")
    print(f"Lower Quartile Reward: {lower_quartile:.2f}")
    print(f"Upper Quartile Reward: {upper_quartile:.2f}")
    print(f"Min Reward (Floor): {min_reward:.2f}")
    print(f"Max Reward (Ceiling): {max_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.1f} +/- {std_length:.1f}")

    # --- Performance Assessment ---
    # Provide a qualitative assessment based on typical CarRacing scores
    if mean_reward >= 900:
        performance = "Exceptional (>= 900)"
    elif mean_reward >= 800:
        performance = "Excellent (800-899)"
    elif mean_reward >= 700:
        performance = "Very Good (700-799)"
    elif mean_reward >= 500:
        performance = "Good (500-699)"
    elif mean_reward >= 300:
        performance = "Fair (300-499)"
    else:
        performance = "Needs Improvement (< 300)"
    print(f"Performance Rating: {performance}")

    # --- Plotting Results ---
    plt.figure(figsize=(12, 7))
    # Plot individual episode rewards
    plt.plot(episode_rewards, label='Episode Reward', marker='o', linestyle='-', markersize=4, alpha=0.7)
    # Add lines for mean, min, max
    plt.axhline(mean_reward, color='r', linestyle='--', label=f'Mean Reward ({mean_reward:.2f})')
    plt.axhline(median_reward, color='y', linestyle='--', label=f'Median Reward ({median_reward:.2f})')
    plt.axhline(min_reward, color='g', linestyle=':', label=f'Min Reward ({min_reward:.2f})')
    plt.axhline(max_reward, color='b', linestyle=':', label=f'Max Reward ({max_reward:.2f})')
    plt.axhline(lower_quartile, color='m', linestyle=':', label=f'Lower Quartile ({lower_quartile:.2f})')
    plt.axhline(upper_quartile, color='c', linestyle=':', label=f'Upper Quartile ({upper_quartile:.2f})')
    if args.random:
        plt.title(f'Random Agent Evaluation Results ({config["n_eval_episodes"]} Episodes)')
    else:
        plt.title(f'Agent Evaluation Results ({config["n_eval_episodes"]} Episodes)\nModel: {os.path.basename(HARDCODED_MODEL_PATH)}')
    plt.xlabel('Episode Number')
    plt.ylabel('Total Episodic Reward')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # --- Saving Plot ---
    # Construct a meaningful filename
    if args.random:
        model_name = "random_baseline"
    else:
        model_name = os.path.splitext(os.path.basename(HARDCODED_MODEL_PATH))[0] # Use hardcoded path
    plot_filename = f"evaluation_{model_name}_{config['n_eval_episodes']}ep_seed{config['seed']}.png"
    try:
        plt.savefig(plot_filename)
        print(f"\nEvaluation plot saved as: {plot_filename}")
    except Exception as e:
        print(f"\nError saving plot: {e}")

    # --- Display Plot ---
    # Optionally display the plot
    plt.show()
    # print("Plot display complete.") 