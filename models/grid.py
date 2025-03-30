# models/grid.py
import numpy as np
from config import F_NOMINAL, BASE_POWER_MVA, MAX_FREQ_HZ, MIN_FREQ_HZ

class GridInterfaceModel:
    # ---Simplified Grid Model using Swing Equation for a single machine connected to infinite bus.# ---
    def __init__(self, initial_freq_hz=F_NOMINAL):
        self.frequency_hz = initial_freq_hz
        self.rotor_angle_rad = 0.0 # Relative angle to infinite bus

        # Note: The swing equation is typically solved together with turbine dynamics,
        # using shared H and D parameters. Here we separate it for modularity,
        # assuming the turbine provides Pmech_pu and Pelec_pu.
        # The H and D in turbine model should represent combined turbine-generator.

        self.state_vector_indices = {
            'delta_omega_pu': 0, # Frequency deviation in pu
            'rotor_angle_rad': 1
        }
        self.num_states = 2

    def get_initial_state(self):
        # ---Returns the initial state vector for the ODE solver.# ---
        initial_delta_omega_pu = (self.frequency_hz - F_NOMINAL) / F_NOMINAL if F_NOMINAL > 0 else 0
        return np.array([initial_delta_omega_pu, self.rotor_angle_rad])

    def get_ode_derivatives(self, y, t, Pmech_pu, Pelec_pu, inertia_2H, damping_D):
        idx = self.state_vector_indices
        delta_omega_pu = y[idx['delta_omega_pu']]
        # delta_rad = y[idx['rotor_angle_rad']] # Angle state if needed

        # Swing Equation: d(delta_omega_pu)/dt = (1 / 2H) * (Pmech_pu - Pelec_pu - D * delta_omega_pu)
        d_delta_omega_pu_dt = (Pmech_pu - Pelec_pu - damping_D * delta_omega_pu) / inertia_2H if inertia_2H > 0 else 0

        # Angle dynamics: d(delta_rad)/dt = omega_nominal * delta_omega_pu
        omega_nominal_rad_s = 2 * np.pi * F_NOMINAL
        d_delta_rad_dt = omega_nominal_rad_s * delta_omega_pu

        return np.array([d_delta_omega_pu_dt, d_delta_rad_dt])

    def update_state_from_solution(self, y_solution):
        # ---Updates internal state variables from the ODE solution vector.# ---
        idx = self.state_vector_indices
        delta_omega_pu = y_solution[idx['delta_omega_pu']]
        self.rotor_angle_rad = y_solution[idx['rotor_angle_rad']]
        self.frequency_hz = F_NOMINAL * (1.0 + delta_omega_pu)

    def get_frequency_hz(self):
        return self.frequency_hz

    def get_rotor_angle_deg(self):
        return np.degrees(self.rotor_angle_rad)

    def check_safety(self):
        # ---Checks if safety limits are violated.# ---
        violations = {}
        if self.frequency_hz > MAX_FREQ_HZ:
            violations['freq_high'] = self.frequency_hz
        elif self.frequency_hz < MIN_FREQ_HZ:
             violations['freq_low'] = self.frequency_hz
        return violations
