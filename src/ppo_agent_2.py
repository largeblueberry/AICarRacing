"""PPO agent variant dedicated to the 2-action experiment (train_ppo_2action2.py).

Forked from ``ppo_agent.py`` so the 2-action line can be tuned independently of
the working 3-action setup. The key behavioural difference is a *per-minibatch*
KL early stop in ``learn``/``learn_mixed_precision`` (the original checked KL only
once per epoch, which let the policy blow past target_kl with large buffers).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import itertools

from .cnn_model import CNNFeatureExtractor
from gymnasium import spaces
from typing import Generator, Tuple, Optional

# Constants for clamping the log standard deviation
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    """
    Actor Network (Policy) for PPO with continuous actions.

    Takes features extracted by a CNN and outputs the mean and log standard
    deviation of a Gaussian distribution representing the policy.
    """
    def __init__(self, features_dim: int, action_dim: int, initial_action_std: float = 1.0, fixed_std: bool = False):
        """
        Initializes the Actor network.

        Args:
            features_dim: Dimensionality of the input feature vector from the CNN.
            action_dim: Dimensionality of the continuous action space.
            initial_action_std: Initial value for the standard deviation of the action distribution.
            fixed_std: If True, use a fixed standard deviation; otherwise, learn it.
        """
        super().__init__()
        self.action_dim = action_dim
        self.initial_action_std = initial_action_std
        self.fixed_std = fixed_std

        hidden_dim = 256 # Dimension of hidden layers
        self.fc1 = nn.Linear(features_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer for the mean of the action distribution
        self.fc_mean = nn.Linear(hidden_dim, action_dim)

        # Output layer for the log standard deviation (learned or fixed)
        if not fixed_std:
            self.fc_logstd = nn.Linear(hidden_dim, action_dim)
            # Initialize log_std weights near zero and bias to initial_action_std
            self.fc_logstd.weight.data.fill_(0.0)
            self.fc_logstd.bias.data.fill_(np.log(self.initial_action_std))
        else:
            # Use a non-trainable parameter for fixed standard deviation
            self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(self.initial_action_std), requires_grad=False)

        # Initialize the mean layer weights orthogonally with small gain for stability
        nn.init.orthogonal_(self.fc_mean.weight, gain=0.01)
        nn.init.constant_(self.fc_mean.bias, 0.0)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass through the Actor network.

        Args:
            features: Feature tensor from the CNN (Batch, features_dim).

        Returns:
            A tuple containing:
                - mean: The mean of the action distribution (Batch, action_dim).
                - log_std: The log standard deviation of the action distribution (Batch, action_dim).
        """
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))

        # Apply tanh activation to constrain the mean to [-1, 1]
        mean = torch.tanh(self.fc_mean(x))

        if not self.fixed_std:
            log_std = self.fc_logstd(x)
            # Clamp log_std to prevent numerical instability
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        else:
            # Expand the fixed log_std to match the batch size
            batch_size = mean.size(0)
            log_std = self.log_std.expand(batch_size, -1)

        return mean, log_std

    def get_action_dist(self, features: torch.Tensor) -> Normal:
        """
        Creates the action distribution for the given features.

        Args:
            features: Feature tensor from the CNN (Batch, features_dim).

        Returns:
            A PyTorch Normal distribution object representing the policy.
        """
        mean, log_std = self.forward(features)
        std = log_std.exp() # Convert log_std to std
        return Normal(mean, std)

    def evaluate_actions(self, features: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the log probability and entropy of given actions under the current policy.

        Used during the PPO update step.

        Args:
            features: Features tensor (Batch, features_dim).
            actions: Actions tensor (Batch, action_dim).

        Returns:
            A tuple containing:
                - log_prob: Log probability of the actions (Batch,).
                - entropy: Entropy of the action distribution (Batch,).
        """
        action_dist = self.get_action_dist(features)
        log_prob = action_dist.log_prob(actions).sum(axis=-1) # Sum log probs across action dimensions
        entropy = action_dist.entropy().sum(axis=-1) # Sum entropy across action dimensions
        return log_prob, entropy


class Critic(nn.Module):
    """
    Critic Network (Value Function) for PPO.

    Takes features extracted by a CNN and outputs a scalar value representing
    the estimated value of the state.
    """
    def __init__(self, features_dim: int):
        """
        Initializes the Critic network.

        Args:
            features_dim: Dimensionality of the input feature vector from the CNN.
        """
        super().__init__()
        hidden_dim = 256 # Dimension of hidden layers
        self.fc1 = nn.Linear(features_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer for the state value (a single scalar)
        self.fc_value = nn.Linear(hidden_dim, 1)

        # Initialize the value layer weights/biases with small values
        nn.init.uniform_(self.fc_value.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_value.bias, -3e-3, 3e-3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the Critic network.

        Args:
            features: Feature tensor from the CNN (Batch, features_dim).

        Returns:
            The estimated state value (Batch,).
        """
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        value = self.fc_value(x)
        return value.squeeze(-1) # Remove the last dimension (size 1)


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent.

    Combines a CNN feature extractor, an Actor network (policy), and a Critic
    network (value function) to learn a policy for continuous control tasks.
    Implements features like GAE, PPO clipping, entropy bonus, gradient clipping,
    learning rate scheduling (warmup and cosine decay), and optional mixed precision.
    """
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Box,
                 config: dict, device: str = 'cpu'):
        """
        Initializes the PPO Agent using a configuration dictionary.

        Args:
            observation_space: The environment's observation space.
            action_space: The environment's action space.
            config: A dictionary containing agent hyperparameters.
                    Expected keys: learning_rate, gamma, gae_lambda, clip_epsilon,
                                   ppo_epochs, batch_size, vf_coef, ent_coef,
                                   max_grad_norm, features_dim, target_kl,
                                   initial_action_std, weight_decay, fixed_std,
                                   lr_warmup_steps, min_learning_rate.
            device: The computation device ('cpu' or 'cuda').
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_dim = action_space.shape[0]
        self.device = torch.device(device)

        # --- Extract Hyperparameters from Config (with defaults for safety otherwise things break) ---
        self.initial_lr = config.get("learning_rate", 1e-4)
        self.lr = self.initial_lr # Current learning rate starts at initial
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_epsilon = config.get("clip_epsilon", 0.1)
        self.epochs = config.get("ppo_epochs", 5)
        self.batch_size = config.get("batch_size", 64)
        self.vf_coef = config.get("vf_coef", 0.5)
        self.ent_coef = config.get("ent_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        features_dim = config.get("features_dim", 64) # Needed for network init
        self.target_kl = config.get("target_kl", 0.02)
        self.initial_action_std = config.get("initial_action_std", 1.0)
        self.weight_decay = config.get("weight_decay", 1e-5)
        self.fixed_std = config.get("fixed_std", False)
        self.lr_warmup_steps = config.get("lr_warmup_steps", 5000)
        self.min_lr = config.get("min_learning_rate", 1e-7) # Store min_lr

        self.steps_done = 0 # Counter for learning rate scheduling

        # --- Network Initialization ---
        self.feature_extractor = CNNFeatureExtractor(observation_space, features_dim).to(self.device)
        self.actor = Actor(features_dim, self.action_dim,
                           initial_action_std=self.initial_action_std,
                           fixed_std=self.fixed_std).to(self.device)
        self.critic = Critic(features_dim).to(self.device)

        # --- Optimizer Setup ---        
        # refactor — Integrated into a single optimizer
        self.optimizer = optim.Adam(
            itertools.chain(
                self.feature_extractor.parameters(),
                self.actor.parameters(),
                self.critic.parameters()
            ),
            lr=self.initial_lr,
            eps=1e-5,
            weight_decay=self.weight_decay
        )

        print(f"--- PPO Agent Initialized ---")
        print(f"Device: {self.device}")
        print(f"Feature Extractor Params: {sum(p.numel() for p in self.feature_extractor.parameters()):,}")
        print(f"Actor Params: {sum(p.numel() for p in self.actor.parameters()):,}")
        print(f"Critic Params: {sum(p.numel() for p in self.critic.parameters()):,}")
        print(f"Action Standard Deviation: {'Fixed' if self.fixed_std else 'Learned'} (Initial: {self.initial_action_std})")
        print(f"Learning Rate: Initial={self.initial_lr:.2e}, Warmup Steps={self.lr_warmup_steps:,}")
        print(f"---------------------------")

    def act(self, observation: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Selects an action based on the current observation using the actor network.

        Also returns the estimated value from the critic and the log probability
        of the selected action.

        Args:
            observation: Current environment observation(s) (Batch, k, H, W) as a tensor.

        Returns:
            A tuple containing:
                - action: Sampled action(s) from the policy distribution (Batch, action_dim) as NumPy array.
                - value: Estimated state value(s) from the critic (Batch,) as NumPy array.
                - log_prob: Log probability of the sampled action(s) (Batch,) as NumPy array.
        """
        # Set networks to evaluation mode
        self.feature_extractor.eval()
        self.actor.eval()
        self.critic.eval()

        # Ensure observation is a tensor on the correct device
        if not isinstance(observation, torch.Tensor):
            # Move to device only if it's not already there (minor optimization)
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        elif observation.device != self.device:
            observation_tensor = observation.to(self.device)
        else:
            observation_tensor = observation

        # --- Add batch dimension if missing ---
        if observation_tensor.ndim == 3:
            observation_tensor = observation_tensor.unsqueeze(0) # Add batch dim: (k, H, W) -> (1, k, H, W)
        # -------------------------------------

        # Perform inference without tracking gradients
        with torch.no_grad():
            # Note: Feature extractor handles normalization internally
            features = self.feature_extractor(observation_tensor)
            value = self.critic(features)
            action_dist = self.actor.get_action_dist(features)
            action = action_dist.sample() # Sample action from the distribution
            log_prob = action_dist.log_prob(action).sum(axis=-1) # Calculate log probability

        # Convert results to NumPy arrays for interaction with the environment
        action_np = action.detach().cpu().numpy()
        value_np = value.detach().cpu().numpy()
        log_prob_np = log_prob.detach().cpu().numpy()

        return action_np, value_np, log_prob_np

    def update_learning_rate(self, current_step: int, total_timesteps: int) -> float:
        """
        Updates the learning rate for both optimizers based on a schedule.

        Implements linear warmup followed by cosine decay, driven by the number
        of ENVIRONMENT steps elapsed (``current_step``). The original version
        divided an internal per-call counter (rollout count, only a few hundred)
        by ``total_timesteps`` (millions), so progress stayed ~0 and the LR never
        actually decayed. Driving the schedule by ``current_step`` makes progress
        sweep 0 -> 1 across the run as intended.

        Args:
            current_step: Environment steps elapsed so far (e.g. global_step).
            total_timesteps: The total number of environment steps planned.

        Returns:
            The newly calculated learning rate.
        """
        self.steps_done += 1  # kept for external bookkeeping/compatibility

        # Warmup Phase: Linearly increase LR from 30% to 100% of initial_lr
        if self.lr_warmup_steps > 0 and current_step < self.lr_warmup_steps:
            alpha = current_step / self.lr_warmup_steps
            current_lr = self.initial_lr * (0.3 + 0.7 * alpha)
        # Decay Phase: Cosine annealing from initial_lr to near zero
        else:
            denom = max(total_timesteps - self.lr_warmup_steps, 1)
            progress = min(max((current_step - self.lr_warmup_steps) / denom, 0.0), 1.0)
            current_lr = self.initial_lr * 0.5 * (1.0 + np.cos(np.pi * progress))

        # Ensure LR doesn't drop below the configured minimum threshold
        current_lr = max(current_lr, self.min_lr) # Use self.min_lr

        # Apply the updated learning rate to both optimizers
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

        self.lr = current_lr # Store the current LR
        return current_lr

    def learn_mixed_precision(self, rollout_buffer, scaler: torch.cuda.amp.GradScaler):
        """
        Performs the PPO learning update using mixed precision (FP16/FP32).

        Requires a CUDA device and a GradScaler.

        Args:
            rollout_buffer: Buffer containing collected experiences.
            scaler: PyTorch GradScaler for handling mixed precision gradients.

        Returns:
            A dictionary containing training metrics (losses, KL divergence, etc.).
        """
        # Set networks to training mode
        self.feature_extractor.train()
        self.actor.train()
        self.critic.train()

        # Accumulate metrics across all epochs and batches
        all_policy_losses, all_value_losses, all_entropy_losses = [], [], []
        all_kl_divs, clip_fractions = [], []

        # PPO Optimization Loop
        continue_training = True  # cleared when a minibatch exceeds target_kl
        for epoch in range(self.epochs):
            epoch_kl_divs = [] # Track KL divergence per epoch for potential early stopping

            # Iterate over minibatches from the rollout buffer
            for batch in rollout_buffer.get_batches(self.batch_size):
                obs_batch, actions_batch, old_log_probs_batch, advantages_batch, returns_batch = batch

                # Forward pass within autocast context for mixed precision
                with torch.cuda.amp.autocast():
                    # Feature extraction (normalization inside extractor)
                    features = self.feature_extractor(obs_batch)
                    # Get current value estimates and policy evaluation
                    values = self.critic(features)
                    log_probs, entropy = self.actor.evaluate_actions(features, actions_batch)


                    # --- PPO Loss Calculation ---
                    # Ratio of new policy probability to old policy probability
                    ratio = torch.exp(log_probs - old_log_probs_batch)

                    # Clipped Surrogate Objective (Policy Loss)
                    policy_loss_1 = advantages_batch * ratio
                    policy_loss_2 = advantages_batch * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                    value_loss = F.mse_loss(values, returns_batch)

                    # Entropy- Learning in the direction of increasing entropy
                    entropy_loss = torch.mean(entropy)
                    
                    # Total Loss
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

                # --- Backward Pass and Optimization --- 
                # Zero gradients before backward pass
                self.optimizer.zero_grad()            

                # Scale the loss and compute gradients using the scaler
                scaler.scale(loss).backward()

                # Unscale gradients before clipping
                scaler.unscale_(self.optimizer)
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(
                        self.feature_extractor.parameters(),
                        self.actor.parameters(),
                        self.critic.parameters()
                    ),
                    self.max_grad_norm
                )

                # Step the optimizers using the scaler
                scaler.step(self.optimizer)

                # Update the scaler for the next iteration
                scaler.update()

                # --- Logging Metrics (within epoch loop) ---
                with torch.no_grad():
                    # Approximate KL divergence between old and new policies
                    log_ratio = log_probs - old_log_probs_batch
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    clip_frac = torch.mean(
                        (torch.abs(torch.exp(log_ratio) - 1.0) > self.clip_epsilon).float()
                    ).item()
                    # Fraction of samples where the policy ratio was clipped
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > self.clip_epsilon).float()).item()

                # Store metrics for this batch
                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropy_losses.append(entropy_loss.item())
                all_kl_divs.append(approx_kl)
                epoch_kl_divs.append(approx_kl)
                clip_fractions.append(clip_frac)

                # --- Per-minibatch KL early stop (2-action stability fix) ---
                if self.target_kl is not None and approx_kl > self.target_kl * 1.5:
                    print(f"Early stop: epoch {epoch+1} minibatch KL {approx_kl:.4f} > {self.target_kl*1.5:.4f}")
                    continue_training = False
                    break

            if not continue_training:
                break

        # --- Return Averaged Metrics ---
        avg_metrics = {
            "policy_loss": np.mean(all_policy_losses),
            "value_loss": np.mean(all_value_losses),
            "entropy_loss": np.mean(all_entropy_losses),
            "approx_kl": np.mean(all_kl_divs),
            "clip_fraction": np.mean(clip_fractions),
        }
        return avg_metrics

    def learn(self, rollout_buffer):
        """
        Performs the PPO learning update using standard precision (FP32).

        Args:
            rollout_buffer: Buffer containing collected experiences.

        Returns:
            A dictionary containing training metrics (losses, KL divergence, etc.).
        """
        # Set networks to training mode
        self.feature_extractor.train()
        self.actor.train()
        self.critic.train()

        # Accumulate metrics across all epochs and batches
        all_policy_losses, all_value_losses, all_entropy_losses = [], [], []
        all_kl_divs, clip_fractions = [], []

        # PPO Optimization Loop
        continue_training = True  # cleared when a minibatch exceeds target_kl
        for epoch in range(self.epochs):
            epoch_kl_divs = [] # Track KL divergence per epoch

            # Iterate over minibatches from the rollout buffer
            for batch in rollout_buffer.get_batches(self.batch_size):
                obs_batch, actions_batch, old_log_probs_batch, advantages_batch, returns_batch = batch

                # --- Forward Pass ---
                # Feature extraction (normalization inside extractor)
                features = self.feature_extractor(obs_batch)
                # Get current value estimates and policy evaluation
                values = self.critic(features)
                log_probs, entropy = self.actor.evaluate_actions(features, actions_batch)

                # --- PPO Loss Calculation ---
                # Ratio of new policy probability to old policy probability
                ratio = torch.exp(log_probs - old_log_probs_batch)
                # ratio = torch.clamp(ratio, 0.1, 10.0) # Optional clamping

                # Clipped Surrogate Objective (Policy Loss)
                policy_loss_1 = advantages_batch * ratio
                policy_loss_2 = advantages_batch * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value Function Loss (Mean Squared Error)
                value_loss = F.mse_loss(values, returns_batch)

                # Entropy Bonus
                entropy_loss = -torch.mean(entropy)

                # Total Loss
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # --- Backward Pass and Optimization ---
                # Zero gradients
                self.optimizer.zero_grad()
                # Compute gradients
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(
                        self.feature_extractor.parameters(),
                        self.actor.parameters(),
                        self.critic.parameters()
                    ),
                    self.max_grad_norm
                )
                # Update weights
                self.optimizer.step()

                # --- Logging Metrics ---
                with torch.no_grad():
                    # Approximate KL divergence
                    approx_kl = 0.5 * torch.mean((log_probs - old_log_probs_batch)**2).item()
                    # Clip fraction
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > self.clip_epsilon).float()).item()

                # Store metrics for this batch
                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropy_losses.append(entropy_loss.item())
                all_kl_divs.append(approx_kl)
                epoch_kl_divs.append(approx_kl)
                clip_fractions.append(clip_frac)

                # --- Per-minibatch KL early stop (2-action stability fix) ---
                # Check after EVERY update, not once per epoch. With a large
                # buffer there are many updates per epoch; a per-epoch check lets
                # the policy blow far past target_kl before stopping (KL spikes).
                if self.target_kl is not None and approx_kl > self.target_kl * 1.5:
                    print(f"Early stop: epoch {epoch+1} minibatch KL {approx_kl:.4f} > {self.target_kl*1.5:.4f}")
                    continue_training = False
                    break

            if not continue_training:
                break

        # --- Return Averaged Metrics ---
        avg_metrics = {
            "policy_loss": np.mean(all_policy_losses),
            "value_loss": np.mean(all_value_losses),
            "entropy_loss": np.mean(all_entropy_losses),
            "approx_kl": np.mean(all_kl_divs),
            "clip_fraction": np.mean(clip_fractions),
        }
        return avg_metrics

    def save(self, path: str):
        """
        Saves the state dictionaries of the feature extractor, actor, and critic networks.

        Note: Does not save optimizer states or other training parameters.
              Use checkpoint saving in the training script for full state saving.

        Args:
            path: The file path to save the model weights.
        """
        print(f"Saving model components to {path}")
        torch.save({
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, path)
        print("Model components saved.")

    def load(self, path: str):
        """
        Loads the state dictionaries for the feature extractor, actor, and critic.

        Args:
            path: The file path from which to load the model weights.
        """
        try:
            print(f"Loading model components from {path}")
            checkpoint = torch.load(path, map_location=self.device)
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            print("Model components loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model file not found at {path}")
        except KeyError as e:
            print(f"Error: Missing key {e} in checkpoint file {path}. Structure mismatch?")
        except Exception as e:
            print(f"Error loading model components from {path}: {e}") 