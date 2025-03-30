# models/reactor.py
import numpy as np
from config import (BETA, BETA_I, LAMBDA, NEUTRON_LIFETIME, ALPHA_F, ALPHA_C,
                    FUEL_HEAT_CAPACITY, COOLANT_HEAT_CAPACITY, HEAT_TRANSFER_FC,
                    NOMINAL_POWER_THERMAL, NOMINAL_AVG_COOLANT_TEMP, NOMINAL_FUEL_TEMP,
                    MAX_FUEL_TEMP)

class ReactorModel:
    # ---PWR Point Kinetics Reactor Model with 2-Node Thermal Feedback.# ---
    def __init__(self, initial_power_fraction):
        self.n_groups = 6
        self.P0 = initial_power_fraction # Initial power fraction
        self.neutron_power = self.P0     # Normalized neutron power (n = P/P_nominal)
        # Initial precursor concentrations at equilibrium
        self.precursors = (BETA_I / (LAMBDA * NEUTRON_LIFETIME)) * self.neutron_power

        # Initial temperatures (assuming steady state at initial power)
        # Need a better way to initialize temps based on power - simplified here
        self.T_fuel = NOMINAL_FUEL_TEMP * initial_power_fraction
        self.T_coolant = NOMINAL_AVG_COOLANT_TEMP # Assume avg coolant temp initially

        # Reference temps for reactivity feedback calculation
        self.T_fuel_ref = NOMINAL_FUEL_TEMP
        self.T_coolant_ref = NOMINAL_AVG_COOLANT_TEMP

        self.state_vector_indices = {
            'neutron_power': 0,
            'precursors': slice(1, 1 + self.n_groups),
            'T_fuel': 1 + self.n_groups,
            'T_coolant': 2 + self.n_groups
        }
        self.num_states = 3 + self.n_groups

    def get_initial_state(self):
        # ---Returns the initial state vector for the ODE solver.# ---
        return np.concatenate(([self.neutron_power], self.precursors, [self.T_fuel], [self.T_coolant]))

    def _calculate_reactivity(self, T_fuel, T_coolant, external_reactivity):
        # ---Calculates total reactivity including feedback.# ---
        rho_thermal = ALPHA_F * (T_fuel - self.T_fuel_ref) + \
                      ALPHA_C * (T_coolant - self.T_coolant_ref)
        return external_reactivity + rho_thermal

    def get_ode_derivatives(self, y, t, external_reactivity, power_removed_mw):
        # ---Calculates the derivatives for the reactor ODE system.# ---
        idx = self.state_vector_indices
        n = y[idx['neutron_power']]
        C = y[idx['precursors']]
        Tf = y[idx['T_fuel']]
        Tc = y[idx['T_coolant']]

        # Clamp neutron power to avoid negative values during transients
        n = max(n, 0.0)

        rho_total = self._calculate_reactivity(Tf, Tc, external_reactivity)

        # Point Kinetics Equations
        dn_dt = (rho_total - BETA) / NEUTRON_LIFETIME * n + np.sum(LAMBDA * C)
        dC_dt = (BETA_I / NEUTRON_LIFETIME) * n - LAMBDA * C

        # Thermal Hydraulics Equations (2-Node)
        power_deposited_mw = n * NOMINAL_POWER_THERMAL # MWth
        dTf_dt = (power_deposited_mw - HEAT_TRANSFER_FC * (Tf - Tc)) / FUEL_HEAT_CAPACITY
        dTc_dt = (HEAT_TRANSFER_FC * (Tf - Tc) - power_removed_mw) / COOLANT_HEAT_CAPACITY

        # Ensure derivatives are returned as a flat numpy array
        return np.concatenate(([dn_dt], dC_dt, [dTf_dt], [dTc_dt]))

    def update_state_from_solution(self, y_solution):
        # ---Updates internal state variables from the ODE solution vector.# ---
        idx = self.state_vector_indices
        self.neutron_power = y_solution[idx['neutron_power']]
        self.precursors = y_solution[idx['precursors']]
        self.T_fuel = y_solution[idx['T_fuel']]
        self.T_coolant = y_solution[idx['T_coolant']]

    def get_thermal_power(self):
        # ---Returns the current thermal power in MWth.# ---
        return max(0.0, self.neutron_power * NOMINAL_POWER_THERMAL) # Ensure non-negative

    def get_temperatures(self):
        # ---Returns the current fuel and coolant temperatures.# ---
        return self.T_fuel, self.T_coolant

    def check_safety(self):
        # ---Checks if safety limits are violated.# ---
        violations = {}
        if self.T_fuel > MAX_FUEL_TEMP:
            violations['fuel_temp'] = self.T_fuel
        return violations
