# envs/pwr_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import copy

from config import *
from models.reactor import ReactorModel
from models.turbine import TurbineModel
from models.grid import GridInterfaceModel
from scenarios.definitions import get_scenario_profile # Import scenario function

class PWREnv(gym.Env):
    # ---Gym environment for PWR Turbine Governor Control.# ---
    metadata = {'render_modes': ['human'], 'render_fps': 10}

    def __init__(self, scenario_name='stable', simulation_time=DEFAULT_SIMULATION_TIME, render_mode=None):
        super().__init__()

        self.scenario_name = scenario_name
        self.simulation_time = simulation_time
        self.max_steps = int(self.simulation_time / DT)
        self.render_mode = render_mode

        # --- Initialize Models ---
        # Use scenario definition to set initial power fraction if needed
        initial_load_mw = get_scenario_profile(scenario_name, 0.0) # Get load at t=0
        self.initial_power_fraction = (initial_load_mw / BASE_POWER_MVA) / EFFICIENCY_THERMAL # Estimate initial reactor power frac
        self.initial_power_fraction = np.clip(self.initial_power_fraction, 0.1, 1.0) # Bound it reasonably

        self.reactor = ReactorModel(initial_power_fraction=self.initial_power_fraction)
        self.turbine = TurbineModel(initial_speed_rpm=NOMINAL_SPEED_RPM,
                                    initial_valve_pos=self.initial_power_fraction) # Assume valve matches power initially
        self.grid = GridInterfaceModel(initial_freq_hz=F_NOMINAL)

        # --- Define Gym Spaces ---
        # Action: Target valve position command (normalized 0-1)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: Key normalized state variables
        # [speed_dev_norm, freq_dev_norm, valve_pos, fuel_temp_norm, coolant_temp_norm, P_load_norm]
        obs_low = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0, 1.5, 1.5, 1.5], dtype=np.float32) # Allow some overshoot in norm
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.current_step = 0
        self.current_time = 0.0
        self.last_action = np.array([self.initial_power_fraction], dtype=np.float32) # Initialize last action

        # For rendering or logging
        self.state_history = []

        print(f"PWREnv initialized. Scenario: {scenario_name}, Sim Time: {simulation_time}s, Initial Power Fraction: {self.initial_power_fraction:.2f}")

    def _get_observation(self):
        # ---Constructs the observation vector from model states.# ---
        # Speed deviation normalized by control band
        speed_dev = self.turbine.get_speed_rpm() - NOMINAL_SPEED_RPM
        speed_range = MAX_SPEED_RPM - MIN_SPEED_RPM
        speed_dev_norm = np.clip(2 * speed_dev / speed_range, -1.0, 1.0) if speed_range > 0 else 0.0

        # Frequency deviation normalized by control band
        freq_dev = self.grid.get_frequency_hz() - F_NOMINAL
        freq_range = CONTROL_FREQ_DEV_HZ * 2
        freq_dev_norm = np.clip(freq_dev / CONTROL_FREQ_DEV_HZ, -1.0, 1.0) if CONTROL_FREQ_DEV_HZ > 0 else 0.0

        valve_pos = self.turbine.get_valve_position()

        # Temperatures normalized by nominal values (approximation)
        T_fuel, T_coolant = self.reactor.get_temperatures()
        fuel_temp_norm = T_fuel / NOMINAL_FUEL_TEMP if NOMINAL_FUEL_TEMP > 0 else 0.0
        coolant_temp_norm = T_coolant / NOMINAL_AVG_COOLANT_TEMP if NOMINAL_AVG_COOLANT_TEMP > 0 else 0.0

        # Load normalized by nominal electrical power
        load_norm = self.current_electrical_load_mw / NOMINAL_POWER_ELEC if NOMINAL_POWER_ELEC > 0 else 0.0

        return np.array([speed_dev_norm, freq_dev_norm, valve_pos,
                         fuel_temp_norm, coolant_temp_norm, load_norm], dtype=np.float32)

    def _get_info(self):
        # ---Returns auxiliary information about the state.# ---
        T_fuel, T_coolant = self.reactor.get_temperatures()
        P_mech_pu = self.turbine.get_mechanical_power_pu()
        P_elec_pu = self.current_electrical_load_mw / BASE_POWER_MVA

        # Calculate margins for fuzzy reward system (0-100%)
        speed_margin = max(0, 100 * (1 - abs(self.turbine.get_speed_rpm() - NOMINAL_SPEED_RPM) / (MAX_SPEED_RPM - NOMINAL_SPEED_RPM))) if (MAX_SPEED_RPM - NOMINAL_SPEED_RPM) > 0 else 100
        freq_margin = max(0, 100 * (1 - abs(self.grid.get_frequency_hz() - F_NOMINAL) / CONTROL_FREQ_DEV_HZ)) if CONTROL_FREQ_DEV_HZ > 0 else 100
        temp_margin = max(0, 100 * (1 - max(0, T_fuel) / MAX_FUEL_TEMP)) if MAX_FUEL_TEMP > 0 else 100

        # Calculate stability (closeness to nominal, 0-100%)
        speed_stability = max(0, 100 * (1 - abs(self.turbine.get_speed_rpm() - NOMINAL_SPEED_RPM) / (0.01 * NOMINAL_SPEED_RPM))) # 1% band
        freq_stability = max(0, 100 * (1 - abs(self.grid.get_frequency_hz() - F_NOMINAL) / (0.001 * F_NOMINAL))) # 0.1% band

        # Calculate efficiency metrics (0-100%)
        power_match = max(0, 100 * (1 - abs(P_mech_pu - P_elec_pu) / max(0.1, P_elec_pu))) # Match mech power to elec load demand
        control_action_diff = abs(self.turbine.valve_position_target - self.last_action[0]) # How much did the target change?
        control_effort_inv = max(0, 100 * (1 - control_action_diff / (VALVE_RATE_LIMIT * DT))) if (VALVE_RATE_LIMIT*DT > 0) else 100 # Inverse effort

        return {
            'time': self.current_time,
            'speed_rpm': self.turbine.get_speed_rpm(),
            'frequency_hz': self.grid.get_frequency_hz(),
            'valve_position': self.turbine.get_valve_position(),
            'fuel_temp_c': T_fuel,
            'coolant_temp_c': T_coolant,
            'reactor_power_mwth': self.reactor.get_thermal_power(),
            'mech_power_mw': self.turbine.get_mechanical_power_mw(),
            'elec_load_mw': self.current_electrical_load_mw,
            'reward_components': {
                 'safety_margin_speed': speed_margin,
                 'safety_margin_freq': freq_margin,
                 'safety_margin_temp': temp_margin,
                 'stability_speed': speed_stability,
                 'stability_freq': freq_stability,
                 'efficiency_power_match': power_match,
                 'efficiency_control_effort_inv': control_effort_inv,
            },
            'action_taken': self.turbine.valve_position_target # Store the target action
        }

    def _system_dynamics(self, t, y, target_valve_pos, electrical_load_pu):
        # ---Combined ODE function for all models.# ---
        # Unpack state vector y based on model sizes
        reactor_states = self.reactor.num_states
        turbine_states = self.turbine.num_states
        grid_states = self.grid.num_states

        y_reactor = y[0 : reactor_states]
        y_turbine = y[reactor_states : reactor_states + turbine_states]
        y_grid = y[reactor_states + turbine_states : reactor_states + turbine_states + grid_states]

        # --- Internal Calculations & Couplings ---
        # 1. Set Turbine Valve Target (already done before calling dynamics)
        #    self.turbine.valve_position_target = target_valve_pos # Stored in turbine instance

        # 2. Turbine Dynamics -> produces Pmech_pu
        #    Requires electrical load feedback Pelec_pu
        turbine_derivs = self.turbine.get_ode_derivatives(y_turbine, t, electrical_load_pu)
        #    Get intermediate Pmech_pu calculated *during* derivative evaluation if needed,
        #    otherwise it's based on the state *within* the derivative func. Here based on steam pressure state.
        Pmech_pu_current = y_turbine[self.turbine.state_vector_indices['steam_pressure_pu']]

        # 3. Grid Dynamics -> updates frequency based on Pmech/Pelec mismatch
        grid_derivs = self.grid.get_ode_derivatives(y_grid, t, Pmech_pu_current, electrical_load_pu,
                                                    self.turbine.inertia_constant_2H, TURBINE_DAMPING_D)

        # 4. Reactor Dynamics -> updates power/temps
        #    Coupling: Power removed from coolant depends on Turbine's mech power demand
        #    Estimate thermal power required = Pmech / efficiency
        reactor_power_removed_mw = (Pmech_pu_current * BASE_POWER_MVA) / EFFICIENCY_THERMAL if EFFICIENCY_THERMAL > 0 else 0
        #    Control Rods: Assume a simplified underlying controller tries to match reactor power
        #    to this demand (e.g., proportional control on power error). This is a MAJOR simplification.
        #    Or, keep external_reactivity=0 for pure governor response study. Let's assume 0 for now.
        external_reactivity = 0.0
        reactor_derivs = self.reactor.get_ode_derivatives(y_reactor, t, external_reactivity, reactor_power_removed_mw)

        # Combine derivatives into a single flat array
        return np.concatenate([reactor_derivs, turbine_derivs, grid_derivs])

    def step(self, action):
        # ---Run one timestep of the environment's dynamics.# ---
        target_valve_pos = float(action[0])
        self.turbine.set_valve_target(target_valve_pos) # Set target for ODE

        # Get current electrical load from scenario
        self.current_electrical_load_mw = get_scenario_profile(self.scenario_name, self.current_time)
        electrical_load_pu = self.current_electrical_load_mw / BASE_POWER_MVA

        # Get current combined state vector
        y0 = np.concatenate([
            self.reactor.get_initial_state(), # Use current internal states
            self.turbine.get_initial_state(),
            self.grid.get_initial_state()
        ])
        y0 = np.nan_to_num(y0) # Ensure no NaNs before solver

        # --- Solve Coupled ODEs ---
        sol = solve_ivp(
            fun=lambda t, y: self._system_dynamics(t, y, target_valve_pos, electrical_load_pu),
            t_span=[self.current_time, self.current_time + DT],
            y0=y0,
            method=SOLVER_METHOD,
            t_eval=[self.current_time + DT]
        )

        if not sol.success:
            print(f"Warning: ODE Solver failed at time {self.current_time:.2f}s! Status: {sol.status}, Message: {sol.message}")
            # Handle failure: Terminate? Use last valid state? Use y0?
            # For stability, let's terminate the episode.
            observation = self._get_observation() # Get obs from last valid state
            reward = -500 # Heavy penalty for solver failure
            terminated = True
            truncated = False
            info = self._get_info()
            info['solver_failed'] = True
            return observation, reward, terminated, truncated, info

        # Update model states from solution
        y_final = sol.y[:, -1]
        reactor_states = self.reactor.num_states
        turbine_states = self.turbine.num_states
        self.reactor.update_state_from_solution(y_final[0 : reactor_states])
        self.turbine.update_state_from_solution(y_final[reactor_states : reactor_states + turbine_states])
        self.grid.update_state_from_solution(y_final[reactor_states + turbine_states :])

        self.current_time += DT
        self.current_step += 1

        # --- Check Termination Conditions ---
        terminated = False
        safety_violations = {**self.reactor.check_safety(), **self.turbine.check_safety(), **self.grid.check_safety()}

        if safety_violations:
            print(f"Termination due to safety violation at t={self.current_time:.2f}s: {safety_violations}")
            terminated = True
            # Reward calculation happens via callback, but we can add penalty here if needed.
            # reward = -100 # Example penalty

        # Check for truncation (max steps reached)
        truncated = self.current_step >= self.max_steps

        # --- Get Results ---
        observation = self._get_observation()
        info = self._get_info() # Includes components for fuzzy reward

        # Reward is calculated by the RL agent's callback/wrapper using info['reward_components']
        # We return a placeholder reward (or 0) from the bare environment step.
        reward = 0.0

        # Store state for rendering/logging
        if self.render_mode == 'human':
            self.state_history.append(copy.deepcopy(info)) # Store full info dict
            self._render_frame()

        self.last_action = action # Store the action taken for next step's info

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # ---Resets the environment to an initial state.# ---
        super().reset(seed=seed)

        # Re-initialize models based on scenario's t=0 state
        initial_load_mw = get_scenario_profile(self.scenario_name, 0.0)
        self.initial_power_fraction = (initial_load_mw / BASE_POWER_MVA) / EFFICIENCY_THERMAL
        self.initial_power_fraction = np.clip(self.initial_power_fraction, 0.1, 1.0)

        self.reactor = ReactorModel(initial_power_fraction=self.initial_power_fraction)
        self.turbine = TurbineModel(initial_speed_rpm=NOMINAL_SPEED_RPM,
                                    initial_valve_pos=self.initial_power_fraction)
        self.grid = GridInterfaceModel(initial_freq_hz=F_NOMINAL)

        self.current_step = 0
        self.current_time = 0.0
        self.current_electrical_load_mw = initial_load_mw
        self.last_action = np.array([self.initial_power_fraction], dtype=np.float32)
        self.state_history = []

        observation = self._get_observation()
        info = self._get_info()

        # Reset rendering if applicable
        if self.render_mode == 'human':
            self._reset_render()

        return observation, info

    def _reset_render(self):
        # ---Resets rendering resources.# ---
        self.fig = None
        self.axs = None

    def _render_frame(self):
        # ---Renders the current state using matplotlib.# ---
        if self.render_mode != 'human':
            return

        import matplotlib.pyplot as plt

        if self.fig is None:
            plt.ion() # Turn on interactive mode
            self.fig, self.axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            self.lines = {
                'speed': self.axs[0].plot([], [], label='Speed (RPM)')[0],
                'freq': self.axs[1].plot([], [], label='Frequency (Hz)')[0],
                'valve': self.axs[2].plot([], [], label='Valve Pos')[0],
                'power': self.axs[3].plot([], [], label='P_mech (MW)', linestyle='-')[0],
                'load': self.axs[3].plot([], [], label='P_load (MW)', linestyle='--')[0],
                'temp': self.axs[3].plot([], [], label='T_fuel (C)', linestyle=':', color='red')[0] # Secondary axis?
            }
            self.axs[0].set_ylabel("Speed (RPM)")
            self.axs[0].axhline(NOMINAL_SPEED_RPM, color='grey', linestyle='--')
            self.axs[0].axhline(MAX_SPEED_RPM, color='orange', linestyle=':')
            self.axs[0].axhline(MIN_SPEED_RPM, color='orange', linestyle=':')
            self.axs[0].legend(loc='upper right')
            self.axs[0].grid(True)

            self.axs[1].set_ylabel("Frequency (Hz)")
            self.axs[1].axhline(F_NOMINAL, color='grey', linestyle='--')
            self.axs[1].axhline(F_NOMINAL + CONTROL_FREQ_DEV_HZ, color='orange', linestyle=':')
            self.axs[1].axhline(F_NOMINAL - CONTROL_FREQ_DEV_HZ, color='orange', linestyle=':')
            self.axs[1].legend(loc='upper right')
            self.axs[1].grid(True)

            self.axs[2].set_ylabel("Valve Position")
            self.axs[2].set_ylim(-0.05, 1.05)
            self.axs[2].legend(loc='upper right')
            self.axs[2].grid(True)

            self.axs[3].set_ylabel("Power (MW) / Temp (C)")
            self.axs[3].set_xlabel("Time (s)")
            self.axs[3].legend(loc='upper right')
            self.axs[3].grid(True)

            self.fig.tight_layout()

        # Update data
        times = [s['time'] for s in self.state_history]
        self.lines['speed'].set_data(times, [s['speed_rpm'] for s in self.state_history])
        self.lines['freq'].set_data(times, [s['frequency_hz'] for s in self.state_history])
        self.lines['valve'].set_data(times, [s['valve_position'] for s in self.state_history])
        self.lines['power'].set_data(times, [s['mech_power_mw'] for s in self.state_history])
        self.lines['load'].set_data(times, [s['elec_load_mw'] for s in self.state_history])
        self.lines['temp'].set_data(times, [s['fuel_temp_c'] for s in self.state_history])


        # Adjust plot limits
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view(True, True, True)

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001) # Small pause to allow plot to update


    def close(self):
        # ---Clean up resources.# ---
        if self.render_mode == 'human' and self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
            self.axs = None
            plt.ioff() # Turn off interactive mode

