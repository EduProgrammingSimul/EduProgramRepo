import numpy as np
from config import (NOMINAL_POWER_ELEC, BASE_POWER_MVA, TURBINE_INERTIA_H, TURBINE_DAMPING_D,
                    NOMINAL_SPEED_RPM, NOMINAL_SPEED_RAD_S, VALVE_RATE_LIMIT,
                    EFFICIENCY_THERMAL, STEAM_CHEST_TAU, DT,
                    MAX_SPEED_RPM, TRIP_SPEED_RPM, MIN_SPEED_RPM)

class TurbineModel:
    # ---Steam Turbine Governor Model with Steam Chest and Torque Balance.# ---
    def __init__(self, initial_speed_rpm=NOMINAL_SPEED_RPM, initial_valve_pos=0.9):
        self.speed_rpm = initial_speed_rpm
        self.speed_rad_s = initial_speed_rpm * 2 * np.pi / 60
        self.valve_position_target = initial_valve_pos # Target set by controller
        self.valve_position_actual = initial_valve_pos # Actual position after rate limit
        self.mechanical_power_pu = initial_valve_pos # Approximation: Pmech = Valve Pos initially in pu
        self.steam_pressure_pu = initial_valve_pos  # Simplified steam chest pressure (pu) relative to valve pos

        # Calculate effective inertia J on system base power (MVA)
        # J = 2 * H * BaseMVA / (omega_nominal_rad_s^2) -> gives units MW*s^2/rad or MJ*s/rad
        # We need J such that J * d(omega_rad_s)/dt = T_mech_MW - T_elec_MW
        # Alternatively, work in pu: 2H d(omega_pu)/dt = T_mech_pu - T_elec_pu
        self.inertia_constant_2H = 2 * TURBINE_INERTIA_H # In seconds

        self.state_vector_indices = {
            'speed_rad_s': 0,
            'steam_pressure_pu': 1,
            'valve_position_actual': 2,
        }
        self.num_states = 3

    def get_initial_state(self):
        # ---Returns the initial state vector for the ODE solver.# ---
        return np.array([self.speed_rad_s, self.steam_pressure_pu, self.valve_position_actual])

    def set_valve_target(self, target_pos):
        # ---Sets the target valve position from the controller.# ---
        self.valve_position_target = np.clip(target_pos, 0.0, 1.0)

    def get_ode_derivatives(self, y, t, electrical_power_pu):
        # ---Calculates the derivatives for the turbine ODE system.# ---
        idx = self.state_vector_indices
        omega_rad_s = y[idx['speed_rad_s']]
        pressure_pu = y[idx['steam_pressure_pu']]
        valve_actual = y[idx['valve_position_actual']]

        omega_pu = omega_rad_s / NOMINAL_SPEED_RAD_S if NOMINAL_SPEED_RAD_S > 0 else 0

        # Apply valve rate limit dynamics
        # d(valve_actual)/dt = rate_limited_change
        max_change = VALVE_RATE_LIMIT * DT
        target_diff = self.valve_position_target - valve_actual
        actual_change = np.clip(target_diff, -max_change/DT, max_change/DT) # Rate per second
        dvalve_actual_dt = actual_change

        # Steam chest dynamics (first-order lag from valve to pressure/flow)
        dpressure_pu_dt = (valve_actual - pressure_pu) / STEAM_CHEST_TAU

        # Mechanical Power (proportional to steam pressure/flow)
        # Simplified: Pmech_pu proportional to pressure_pu
        # Could add dependency on speed (efficiency)
        Pmech_pu = pressure_pu

        # Torque balance using pu values: 2H d(omega_pu)/dt = Pmech_pu - Pelec_pu - D * omega_pu_deviation
        # Or using physical values: J dw/dt = Tm - Te - D_physical * w
        # Let's use pu version (more common in power systems)
        Pelec_pu = electrical_power_pu
        omega_dev_pu = omega_pu - 1.0 # Deviation from nominal speed in pu
        damping_term = TURBINE_DAMPING_D * omega_dev_pu

        # Avoid division by zero if inertia is zero
        domega_pu_dt = (Pmech_pu - Pelec_pu - damping_term) / self.inertia_constant_2H if self.inertia_constant_2H > 0 else 0
        # Convert back to rad/s derivative
        domega_rad_s_dt = domega_pu_dt * NOMINAL_SPEED_RAD_S

        return np.array([domega_rad_s_dt, dpressure_pu_dt, dvalve_actual_dt])

    def update_state_from_solution(self, y_solution):
        # ---Updates internal state variables from the ODE solution vector.# ---
        idx = self.state_vector_indices
        self.speed_rad_s = y_solution[idx['speed_rad_s']]
        self.speed_rpm = self.speed_rad_s * 60 / (2 * np.pi)
        self.steam_pressure_pu = y_solution[idx['steam_pressure_pu']]
        self.valve_position_actual = y_solution[idx['valve_position_actual']]

        # Recalculate mechanical power based on final state for reporting
        self.mechanical_power_pu = self.steam_pressure_pu # Based on final steam chest state

    def get_speed_rpm(self):
        return self.speed_rpm

    def get_valve_position(self):
        return self.valve_position_actual

    def get_mechanical_power_mw(self):
        # ---Returns mechanical power in MW.# ---
        # Convert pu power on Base MVA to MW
        return self.mechanical_power_pu * BASE_POWER_MVA

    def get_mechanical_power_pu(self):
        # ---Returns mechanical power in pu on Base MVA.# ---
        return self.mechanical_power_pu

    def check_safety(self):
        # ---Checks if safety limits are violated.# ---
        violations = {}
        if self.speed_rpm > TRIP_SPEED_RPM:
            violations['speed_trip'] = self.speed_rpm
        elif self.speed_rpm > MAX_SPEED_RPM:
            violations['speed_high'] = self.speed_rpm
        elif self.speed_rpm < MIN_SPEED_RPM:
             violations['speed_low'] = self.speed_rpm
        return violations
