# controllers/rl/agent.py
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import os
import gymnasium as gym
import numpy as np

from controllers.rl.fuzzy_rewards import FuzzyRewardSystem
from config import (RL_LEARNING_RATE, RL_BUFFER_SIZE, RL_BATCH_SIZE,
                    RL_TAU, RL_GAMMA, RL_TRAIN_FREQ, RL_GRADIENT_STEPS)

class FuzzyRewardCallback(BaseCallback):
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.fuzzy_reward_system = FuzzyRewardSystem() # Initialize the fuzzy system

    def _on_step(self) -> bool:
        
        # Retrieve reward components from the info dictionary
        # For VecEnv, info is a list, one item per environment.
        assert "reward_components" in self.locals["infos"][0], \
            "Info dictionary missing 'reward_components' for fuzzy reward calculation."

        new_rewards = []
        for info in self.locals["infos"]:
            reward_components = info["reward_components"]
            # Calculate the fuzzy reward for this step/env
            fuzzy_reward = self.fuzzy_reward_system.calculate_reward(reward_components)
            new_rewards.append(fuzzy_reward)

            # Log fuzzy reward components for monitoring (optional)
            # for key, value in reward_components.items():
            #     self.logger.record_mean(f'fuzzy_reward_inputs/{key}', value)
            # self.logger.record_mean('fuzzy_reward/output', fuzzy_reward)

        # Overwrite the rewards array used by the RL algorithm
        self.locals["rewards"] = np.array(new_rewards, dtype=np.float32)

        return True # Continue training


def setup_rl_agent(env_id, log_dir="./rl_logs/", model_path=None, train=True):
    # ---Sets up the RL agent (SAC) and environment.# ---

    os.makedirs(log_dir, exist_ok=True)

    # Wrap the environment with Monitor for logging stats
    # Use make_vec_env for potential parallel training (n_envs=1 for simplicity here)
    vec_env = make_vec_env(lambda: Monitor(gym.make(env_id), log_dir), n_envs=1)

    if train:
        # Define the SAC agent
        model = SAC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=RL_LEARNING_RATE,
            buffer_size=RL_BUFFER_SIZE,
            batch_size=RL_BATCH_SIZE,
            tau=RL_TAU,
            gamma=RL_GAMMA,
            train_freq=RL_TRAIN_FREQ,
            gradient_steps=RL_GRADIENT_STEPS,
            tensorboard_log=log_dir,
            # policy_kwargs=dict(net_arch=[256, 256]) # Example architecture
        )
        print("New SAC model created for training.")
    else:
        if model_path and os.path.exists(model_path):
             print(f"Loading pre-trained SAC model from: {model_path}")
             model = SAC.load(model_path, env=vec_env)
        else:
            raise FileNotFoundError(f"Model path not found: {model_path}. Cannot load model for evaluation.")

    return model, vec_env

def train_rl_agent(model, total_timesteps=100000, save_path="./pwr_sac_model"):
    # ---Trains the RL agent.# ---
    print(f"Starting RL training for {total_timesteps} timesteps...")
    callback = FuzzyRewardCallback() # Use the fuzzy reward callback
    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=10)
    model.save(save_path)
    print(f"Training finished. Model saved to {save_path}.zip")

def evaluate_rl_agent(model, env, num_episodes=10):
    # ---Evaluates the trained RL agent.# ---
    print(f"Evaluating RL agent for {num_episodes} episodes...")
    all_rewards = []
    all_lengths = []
    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # IMPORTANT: Use the reward processed by the callback mechanism
            # For evaluation, we need to simulate the callback's effect or get reward from info.
            # Let's assume the 'reward' returned by step is placeholder, get fuzzy from info.
            # Re-calculate reward here for eval consistency if not using callback during eval.
            eval_fuzzy_reward_sys = FuzzyRewardSystem() # Need independent instance for eval
            fuzzy_reward = eval_fuzzy_reward_sys.calculate_reward(info["reward_components"])
            episode_reward += fuzzy_reward # Accumulate fuzzy reward
            episode_length += 1
        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        print(f"Episode {i+1}: Length={episode_length}, Reward={episode_reward:.2f}")

    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    print(f"Evaluation Complete: Mean Reward={mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward
