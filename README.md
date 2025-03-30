

This project simulates an adaptive governor control system for Pressurized Water Reactor (PWR) nuclear power plants, leveraging Reinforcement Learning (RL) with Fuzzy Logic-based rewards. The goal is to enhance the plant's load-following capabilities and contribute to grid stability, particularly in the context of increasing variable renewable energy sources.

## Project Description

Traditional PID controllers used in PWR turbine governors excel at baseload operation but struggle with the non-linear dynamics encountered during load-following maneuvers (adjusting power output to match grid demand). This simulation framework explores advanced control strategies to overcome these limitations.

This research proposes an RL-based governor control system (specifically using Soft Actor-Critic - SAC) guided by a novel Fuzzy Reward function. This approach allows the controller to:

1.  **Adapt:** Learn optimal control policies through interaction with a simulated environment, handling non-linearities without needing a perfect analytical model.
2.  **Balance Objectives:** Utilize fuzzy logic to intelligently weigh competing objectives like operational safety (fuel temperature, turbine speed), grid stability (frequency deviation), and efficiency (power tracking, control effort).
3.  **Enhance Safety:** Integrate operational limits directly into the learning process and reward structure.

The project provides a high-fidelity simulation environment and compares the proposed RL-Fuzzy controller against traditional PID, Fuzzy Logic Control (FLC), and a placeholder for Model Predictive Control (MPC).

## Simulation Environment

The simulation environment, built using Python and Gymnasium (the maintained fork of OpenAI Gym), models the key dynamics of a PWR system relevant to governor control:

*   **Reactor Model (`models/reactor.py`):** Implements point kinetics equations with 6 delayed neutron groups and a 2-node thermal-hydraulic feedback model (fuel and coolant temperatures affecting reactivity).
*   **Turbine Model (`models/turbine.py`):** Represents the steam turbine-generator dynamics using a torque-balance approach, incorporating inertia (H), damping (D), steam chest lag, and valve rate limits.
*   **Grid Interface Model (`models/grid.py`):** Uses the standard swing equation to model the interaction of the generator with an infinite bus, simulating grid frequency response to power imbalances.
*   **Integration (`envs/pwr_env.py`):** Couples these models within a Gymnasium environment (`PWREnv-v0`). The coupled Ordinary Differential Equations (ODEs) are solved simultaneously using `scipy.integrate.solve_ivp` for accuracy.
*   **Scenarios (`scenarios/definitions.py`):** Defines various operational scenarios including stable operation, power ramps, sudden load changes, oscillating demand, and placeholders for fault conditions (e.g., sensor failure).

## Control Strategies Implemented

*   **PID Controller (`controllers/pid.py`):** A standard Proportional-Integral-Derivative controller tuned for stable operation (based on Ziegler-Nichols or similar methods, requires tuning).
*   **Fuzzy Logic Controller (FLC) (`controllers/flc.py`):** Uses fuzzy rules based on speed error and its rate of change to determine valve adjustments. (Requires significant rule base tuning for optimal performance).
*   **Model Predictive Control (MPC) (`controllers/mpc.py`):** Placeholder structure. Requires implementation of a predictive model and an optimization solver (e.g., using `scipy.optimize` or dedicated libraries like CasADi, GEKKO, do-mpc).
*   **Reinforcement Learning + Fuzzy Rewards (`controllers/rl/`):**
    *   **Agent (`agent.py`):** Uses the Soft Actor-Critic (SAC) algorithm from the `stable-baselines3` library.
    *   **Fuzzy Rewards (`fuzzy_rewards.py`):** Calculates a scalar reward signal based on safety margins, stability metrics (speed/frequency deviations), and efficiency metrics (power tracking, control effort) using fuzzy logic rules. This guides the RL agent's learning process.
    *   **Callback (`agent.py`):** A `stable-baselines3` callback injects the calculated fuzzy reward into the RL training loop.

## Features

*   Modular and extensible object-oriented code structure.
*   High-fidelity (though simplified) physics models for core components.
*   Standard Gymnasium interface for easy integration with RL libraries.
*   Implementation of PID, FLC, and RL (SAC) controllers.
*   Novel Fuzzy Reward system for multi-objective RL training.
*   Scenario definition framework for testing controllers under various conditions.
*   Command-line interface for running specific controller/scenario combinations.
*   Built-in plotting and metric calculation utilities.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <EduProgramRepo>
    cd pwr_rl_control
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you encounter issues with PyTorch (a dependency of stable-baselines3), follow the installation instructions on the official PyTorch website: https://pytorch.org/)*

## Usage

The main script `main.py` is used to run simulations via the command line.

**Command-Line Arguments:**

*   `-c` or `--controller`: **Required**. Specifies the controller type. Choices: `PID`, `FLC`, `MPC`, `RL`.
*   `-s` or `--scenario`: *Optional*. Specifies the scenario name. Default: `stable`. (See `scenarios/definitions.py` for available names like `gradual_increase`, `sudden_load_increase`, `oscillating_load`, etc.).
*   `-t` or `--time`: *Optional*. Sets the simulation duration in seconds. Default is defined in `config.py`.
*   `--train`: *Optional*. If specified with `--controller RL`, trains a new SAC model instead of evaluating.
*   `--timesteps`: *Optional*. Number of timesteps for RL training (used with `--train`). Default: 150000.
*   `--model_path`: *Optional*. Path to a pre-trained RL model (`.zip` file) to load for evaluation (used with `--controller RL` and without `--train`). Defaults to `./rl_models/pwr_sac_model_final.zip`.

**Examples:**

1.  **Run PID controller on the 'gradual_increase' scenario for 150 seconds:**
    ```bash
    python main.py -c PID -s gradual_increase -t 150
    ```

2.  **Run FLC controller on the 'stable' scenario (default time):**
    ```bash
    python main.py --controller FLC --scenario stable
    ```

3.  **Train a new RL (SAC) model for 200,000 timesteps:**
    ```bash
    python main.py --controller RL --train --timesteps 200000
    ```
    *(This will save the trained model to `./rl_models/pwr_sac_model_final.zip` by default. The script will exit after training.)*

4.  **Evaluate the default pre-trained RL model on the 'sudden_load_increase' scenario:**
    ```bash
    python main.py -c RL -s sudden_load_increase
    ```

5.  **Evaluate a specific RL model on the 'oscillating_load' scenario:**
    ```bash
    python main.py -c RL --model_path ./rl_models/my_trained_agent.zip -s oscillating_load
    ```

**Output:**

*   Console output showing simulation progress and final metrics.
*   A plot displaying the time-series results for the simulation run.
*   CSV files saved in the root directory containing the performance metrics (`metrics_<scenario>_<controller>.csv`) and the full simulation history (`history_<scenario>_<controller>.csv`).
*   During RL training, logs will be saved in the `./rl_logs/` directory (viewable with TensorBoard: `tensorboard --logdir ./rl_logs/`).

## Future Work & Extensions

*   Implement a detailed steam generator model for more accurate reactor-turbine coupling.
*   Develop and integrate the Model Predictive Controller (MPC).
*   Refine and extensively tune the Fuzzy Logic Controller (FLC) rules.
*   Refine and extensively tune the Fuzzy Reward system rules for optimal RL performance.
*   Implement Constrained SAC or other safe RL algorithms.
*   Explore adaptive fuzzy weights using context (e.g., BERT embeddings on operational state descriptions).
*   Expand the scenario library, including more complex faults and multi-machine grid interactions (requires grid model changes).
*   Conduct hyperparameter optimization for the RL agent.
*   Perform quantitative comparisons across a wider range of metrics and scenarios.

## License

(Choose a license, e.g., MIT, Apache 2.0, or specify if it's proprietary)
Example:
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgements

*   (Optional: Acknowledge libraries used, inspirations, funding sources, etc.)
*   Built using Gymnasium, Stable Baselines3, NumPy, SciPy, Matplotlib, and scikit-fuzzy.
