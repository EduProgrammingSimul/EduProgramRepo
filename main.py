# main.py
import gymnasium as gym
import pandas as pd
import numpy as np
import time as timer # Avoid conflict with time variable in loops
import os
import argparse # Import argparse
import sys # For exiting cleanly

# Register the custom environment
from gymnasium.envs.registration import register
register(
    id='PWREnv-v0',
    entry_point='envs.pwr_env:PWREnv',
)

# Import necessary components AFTER registration
from config import * # Import configurations like DEFAULT_SIMULATION_TIME
from controllers.pid import PIDController
from controllers.flc import FLCController
from controllers.mpc import MPCController # Placeholder
from controllers.rl.agent import setup_rl_agent, train_rl_agent, evaluate_rl_agent
# Import only needed function if scenario list isn't required globally
from scenarios.definitions import get_scenario_profile
from utils.evaluation import calculate_performance_metrics
from utils.plotting import plot_simulation_results
# Removed plot_comparison_metrics as we run single simulation

# --- Helper Function (remains the same) ---
def run_simulation(env_id, controller_type, scenario_name, sim_time, rl_model=None):
    """Runs a single simulation episode."""
    print(f"\n--- Running Scenario: {scenario_name} with Controller: {controller_type} ---")

    # Create environment instance for this run
    env = gym.make(env_id, scenario_name=scenario_name, simulation_time=sim_time, render_mode=None) # No render during batch runs

    # Initialize controller for this run
    controller = None
    try:
        initial_power_fraction = env.unwrapped.initial_power_fraction
        initial_valve = initial_power_fraction
        print(f"Debug: Accessed initial_power_fraction: {initial_power_fraction:.2f}")
    except AttributeError:
         print("Warning: Could not get initial_power_fraction from unwrapped env. Using default.")
         initial_valve = 0.9 # Default fallback

    if controller_type == 'PID':
        controller = PIDController(setpoint=NOMINAL_SPEED_RPM)
        controller.reset()
    elif controller_type == 'FLC':
        controller = FLCController(setpoint=NOMINAL_SPEED_RPM)
        controller.reset(initial_valve_pos=initial_valve)
    elif controller_type == 'MPC':
        controller = MPCController() # Placeholder
        controller.reset(initial_valve_pos=initial_valve)
    elif controller_type == 'RL':
        if rl_model is None:
            raise ValueError("RL model must be provided for RL controller type evaluation.")
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    # Simulation Loop
    obs, info = env.reset()
    done = False
    history = []
    start_time_exec = timer.time()

    while not done:
        current_measurement = info['speed_rpm']
        current_valve_pos = info['valve_position']
        current_step = env.unwrapped.current_step

        # Sensor Fault Injection placeholder (can be refined)
        if scenario_name == 'sensor_fail':
             fault_start_time = 20
             fault_duration = 30
             if fault_start_time <= env.unwrapped.current_time < fault_start_time + fault_duration:
                  current_measurement = NOMINAL_SPEED_RPM

        # Get action from controller or RL agent
        if controller_type == 'RL':
            action, _states = rl_model.predict(obs, deterministic=True)
        elif controller_type == 'PID':
            action = controller.calculate_action(current_measurement)
        elif controller_type == 'FLC':
            action = controller.calculate_action(current_measurement)
        elif controller_type == 'MPC':
            action = controller.calculate_action(current_measurement, current_step, current_valve_pos)
        else: # Should not happen due to argparse choices
             action = env.action_space.sample()

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        history.append(info)

    end_time_exec = timer.time()
    exec_time = end_time_exec - start_time_exec
    print(f"--- Simulation Complete ({exec_time:.2f}s execution time) ---")
    env.close()

    history_df = pd.DataFrame(history)
    return history_df


if __name__ == "__main__":

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run PWR Governor Control Simulation")
    parser.add_argument(
        "-c", "--controller",
        required=True,
        choices=['PID', 'FLC', 'MPC', 'RL'],
        help="Type of controller to use."
    )
    parser.add_argument(
        "-s", "--scenario",
        default='stable',
        # TODO: Dynamically list available scenarios if possible, or list manually
        help="Name of the scenario to run (e.g., 'stable', 'gradual_increase', 'sudden_load_increase', etc.). Default: 'stable'."
    )
    parser.add_argument(
        "-t", "--time",
        type=int,
        default=DEFAULT_SIMULATION_TIME,
        help=f"Simulation duration in seconds. Default: {DEFAULT_SIMULATION_TIME}s."
    )
    # RL specific arguments
    parser.add_argument(
        "--train",
        action='store_true',
        help="Train a new RL model (only applies if --controller RL)."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=150000, # Default training timesteps
        help="Total timesteps for RL training (only applies with --train)."
    )
    parser.add_argument(
        "--model_path",
        default=None, # Default to constructing path later
        help="Path to load a pre-trained RL model (only applies if --controller RL and not --train)."
    )

    args = parser.parse_args()

    # --- Configuration from args ---
    ENV_ID = 'PWREnv-v0'
    SELECTED_CONTROLLER = args.controller
    SELECTED_SCENARIO = args.scenario
    SIMULATION_TIME = args.time
    RUN_RL_TRAINING = args.train
    RL_TOTAL_TIMESTEPS = args.timesteps
    PROVIDED_RL_MODEL_PATH = args.model_path

    # Construct default RL paths if not provided
    RL_MODEL_DIR = "./rl_models/"
    RL_MODEL_NAME = "pwr_sac_model_final" # Default name
    RL_LOG_DIR = "./rl_logs/"
    DEFAULT_RL_MODEL_PATH = os.path.join(RL_MODEL_DIR, f"{RL_MODEL_NAME}.zip")
    RL_MODEL_PATH_TO_USE = PROVIDED_RL_MODEL_PATH if PROVIDED_RL_MODEL_PATH else DEFAULT_RL_MODEL_PATH

    # --- RL Agent Training / Loading ---
    rl_agent = None
    if SELECTED_CONTROLLER == 'RL':
        os.makedirs(RL_MODEL_DIR, exist_ok=True)
        os.makedirs(RL_LOG_DIR, exist_ok=True)

        if RUN_RL_TRAINING:
            print(f"--- Preparing for RL Training ({RL_TOTAL_TIMESTEPS} timesteps) ---")
            try:
                # Setup agent for training
                rl_agent_train, vec_env_for_train = setup_rl_agent(
                    ENV_ID, log_dir=RL_LOG_DIR, train=True
                )
                # Train the agent
                train_rl_agent(
                    rl_agent_train,
                    total_timesteps=RL_TOTAL_TIMESTEPS,
                    save_path=os.path.join(RL_MODEL_DIR, RL_MODEL_NAME) # Save with default name
                )
                print("--- Training complete. Exiting. ---")
                sys.exit(0) # Exit after training
            except Exception as e:
                print(f"!!! ERROR during RL training setup or execution !!!")
                print(f"Error details: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1) # Exit with error code

        else: # Load for evaluation
            print(f"--- Loading RL Agent for Evaluation ---")
            try:
                rl_agent, _ = setup_rl_agent( # We don't need the vec_env here
                    ENV_ID, train=False, model_path=RL_MODEL_PATH_TO_USE
                )
                print(f"RL model loaded successfully from {RL_MODEL_PATH_TO_USE}")
            except FileNotFoundError as e:
                print(f"!!! ERROR: RL model file not found at {RL_MODEL_PATH_TO_USE} !!!")
                print("Cannot run RL controller evaluation.")
                sys.exit(1) # Exit if model not found for eval
            except Exception as e:
                print(f"!!! ERROR loading RL model !!!")
                print(f"Error details: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1) # Exit with error code

    elif RUN_RL_TRAINING:
        print("Warning: --train flag provided but controller is not RL. Training flag ignored.")


    # --- Run the Single Simulation ---
    print(f"\n=============================================")
    print(f" Starting Simulation:")
    print(f" Controller: {SELECTED_CONTROLLER}")
    print(f" Scenario:   {SELECTED_SCENARIO}")
    print(f" Duration:   {SIMULATION_TIME}s")
    print(f"=============================================")

    try:
        history_df = run_simulation(
            env_id=ENV_ID,
            controller_type=SELECTED_CONTROLLER,
            scenario_name=SELECTED_SCENARIO,
            sim_time=SIMULATION_TIME,
            rl_model=rl_agent # Pass agent if loaded (will be None otherwise)
        )

        # --- Process and Plot Results ---
        metrics = calculate_performance_metrics(history_df)
        print("\n--- Performance Metrics ---")
        # Pretty print metrics
        max_key_len = max(len(k) for k in metrics.keys())
        for key, value in metrics.items():
             print(f"{key:<{max_key_len}} : {value:.4f}")

        # Save metrics to a file specific to this run
        metrics_filename = f"metrics_{SELECTED_SCENARIO}_{SELECTED_CONTROLLER}.csv"
        metrics_series = pd.Series(metrics)
        metrics_series.to_csv(metrics_filename)
        print(f"\nMetrics saved to {metrics_filename}")

        # Save history to a file specific to this run
        history_filename = f"history_{SELECTED_SCENARIO}_{SELECTED_CONTROLLER}.csv"
        history_df.to_csv(history_filename, index=False)
        print(f"Full history saved to {history_filename}")


        plot_simulation_results(
            history_df,
            title=f"Scenario: {SELECTED_SCENARIO} - Controller: {SELECTED_CONTROLLER}"
        )

    except Exception as e:
        print(f"\n!!! ERROR during simulation run for {SELECTED_SCENARIO} with {SELECTED_CONTROLLER} !!!")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n--- Simulation Finished Successfully ---")
    sys.exit(0)
