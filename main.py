# main.py
import gymnasium as gym
import pandas as pd
import numpy as np
import time as timer # Avoid conflict with time variable in loops
import os

# Register the custom environment
from gymnasium.envs.registration import register
register(
    id='PWREnv-v0',
    entry_point='envs.pwr_env:PWREnv'
)

from config import *
from controllers.pid import PIDController
from controllers.flc import FLCController
from controllers.mpc import MPCController # Placeholder
from controllers.rl.agent import setup_rl_agent, train_rl_agent, evaluate_rl_agent
from scenarios.definitions import get_scenario_profile # Used to get initial conditions
from utils.evaluation import calculate_performance_metrics
from utils.plotting import plot_simulation_results, plot_comparison_metrics

def run_simulation(env_id, controller_type, scenario_name, sim_time, rl_model=None):
    
    print(f"\n--- Running Scenario: {scenario_name} with Controller: {controller_type} ---")

    # Create environment instance for this run
    env = gym.make(env_id, scenario_name=scenario_name, simulation_time=sim_time, render_mode=None) # No render during batch runs

    # Initialize controller for this run
    controller = None
    initial_valve = env.unwrapped.initial_power_fraction # Get initial valve state from env's init
    if controller_type == 'PID':
        controller = PIDController(setpoint=NOMINAL_SPEED_RPM)
        controller.reset() # Ensure PID state is reset
    elif controller_type == 'FLC':
        controller = FLCController(setpoint=NOMINAL_SPEED_RPM)
        controller.reset(initial_valve_pos=initial_valve)
    elif controller_type == 'MPC':
        controller = MPCController() # Placeholder
        controller.reset(initial_valve_pos=initial_valve)
    elif controller_type == 'RL':
        if rl_model is None:
            raise ValueError("RL model must be provided for RL controller type.")
        # RL agent action is determined externally
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    # Simulation Loop
    obs, info = env.reset()
    done = False
    history = []
    start_time_exec = timer.time()

    while not done:
        current_measurement = info['speed_rpm'] # Use actual speed for controllers
        current_valve_pos = info['valve_position']
        current_step = env.unwrapped.current_step # Get current step for MPC

        # --- Sensor Fault Injection (Example for Scenario 3) ---
        if scenario_name == 'sensor_fail':
             fault_start_time = 20
             fault_duration = 30
             if fault_start_time <= env.unwrapped.current_time < fault_start_time + fault_duration:
                  # Example: Sensor stuck at nominal
                  current_measurement = NOMINAL_SPEED_RPM
                  # print(f"Time {env.current_time:.2f}: Sensor fault active, reporting {current_measurement} RPM")


        # Get action from controller or RL agent
        if controller_type == 'RL':
            action, _states = rl_model.predict(obs, deterministic=True)
        elif controller_type == 'PID':
            action = controller.calculate_action(current_measurement)
        elif controller_type == 'FLC':
            action = controller.calculate_action(current_measurement)
        elif controller_type == 'MPC':
            # MPC needs more state info potentially
            action = controller.calculate_action(current_measurement, current_step, current_valve_pos)
        else:
            action = env.action_space.sample() # Should not happen

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        history.append(info)

    end_time_exec = timer.time()
    exec_time = end_time_exec - start_time_exec
    print(f"--- Simulation Complete ({exec_time:.2f}s execution time) ---")
    env.close()

    # Convert history to DataFrame
    history_df = pd.DataFrame(history)
    return history_df


if __name__ == "__main__":

    # --- Configuration ---
    ENV_ID = 'PWREnv-v0'
    CONTROLLERS_TO_TEST = ['PID', 'FLC', 'RL'] # Add 'MPC' when implemented
    SCENARIOS_TO_TEST = [
        'stable',
        'gradual_increase',
        'sudden_load_increase',
        # 'sensor_fail', # Uncomment when fault logic is robust
        'emergency_shutdown',
        # 'steam_pressure_drop', # Needs model change
        'oscillating_load',
        # 'efficiency_decline', # Needs model change
        'fluctuating_demand',
    ]
    SIMULATION_TIME = 120 # Seconds per scenario run
    RUN_RL_TRAINING = False # Set to True to train a new RL model first
    RL_TOTAL_TIMESTEPS = 150000 # Timesteps for training
    RL_MODEL_DIR = "./rl_models/"
    RL_MODEL_NAME = "pwr_sac_model_final"
    RL_LOG_DIR = "./rl_logs/"
    RL_MODEL_PATH = os.path.join(RL_MODEL_DIR, f"{RL_MODEL_NAME}.zip")

    # --- RL Agent Training (Optional) ---
    rl_agent = None
    if RUN_RL_TRAINING or 'RL' in CONTROLLERS_TO_TEST:
        os.makedirs(RL_MODEL_DIR, exist_ok=True)
        try:
            # Setup agent for training or loading
            rl_agent, vec_env_for_train = setup_rl_agent(ENV_ID, log_dir=RL_LOG_DIR, train=RUN_RL_TRAINING, model_path=RL_MODEL_PATH if not RUN_RL_TRAINING else None)

            if RUN_RL_TRAINING:
                 train_rl_agent(rl_agent, total_timesteps=RL_TOTAL_TIMESTEPS, save_path=os.path.join(RL_MODEL_DIR, RL_MODEL_NAME))
                 print("Training complete. Exiting. Re-run with RUN_RL_TRAINING=False to evaluate.")
                 exit()
            elif 'RL' not in CONTROLLERS_TO_TEST:
                 print("RL model loaded but 'RL' not in CONTROLLERS_TO_TEST. Exiting.")
                 exit()
            else:
                 print("RL model loaded successfully for evaluation.")

        except FileNotFoundError as e:
             print(f"Error setting up RL Agent: {e}")
             print("Cannot run RL controller evaluations.")
             if 'RL' in CONTROLLERS_TO_TEST:
                 CONTROLLERS_TO_TEST.remove('RL') # Remove RL if model not found
        except Exception as e:
             print(f"An unexpected error occurred during RL setup: {e}")
             if 'RL' in CONTROLLERS_TO_TEST:
                 CONTROLLERS_TO_TEST.remove('RL')

    # --- Run Simulations & Collect Results ---
    all_results_history = {}
    all_metrics = []

    if not CONTROLLERS_TO_TEST:
         print("No controllers available or configured to test. Exiting.")
         exit()

    for scenario in SCENARIOS_TO_TEST:
        for controller in CONTROLLERS_TO_TEST:
            try:
                print("CURRENT CONTROLLER TO TEST - " + controller);
                history_df = run_simulation(
                    env_id=ENV_ID,
                    controller_type=controller,
                    scenario_name=scenario,
                    sim_time=SIMULATION_TIME,
                    rl_model=rl_agent if controller == 'RL' else None
                )
                all_results_history[(scenario, controller)] = history_df

                # Calculate metrics for this run
                metrics = calculate_performance_metrics(history_df)
                metrics['Scenario'] = scenario
                metrics['Controller'] = controller
                all_metrics.append(metrics)

                # Optional: Plot results for each individual run
                # plot_simulation_results(history_df, title=f"Scenario: {scenario} - Controller: {controller}")

            except Exception as e:
                print(f"!!! ERROR running simulation for Scenario: {scenario}, Controller: {controller} !!!")
                print(f"Error details: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for debugging

    # --- Aggregate Metrics & Plot Comparisons ---
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        print("\n--- Aggregate Performance Metrics ---")
        print(metrics_df.to_string())
        metrics_df.to_csv("simulation_metrics.csv", index=False)

        # Plot comparison charts
        plot_comparison_metrics(metrics_df, SCENARIOS_TO_TEST, CONTROLLERS_TO_TEST)
    else:
        print("No simulation metrics were collected.")

    print("\n--- All Simulations Finished ---")
