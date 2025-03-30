# controllers/flc.py
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from config import (NOMINAL_SPEED_RPM, DT, FLC_ERROR_RANGE, FLC_ERROR_DOT_RANGE,
                    FLC_VALVE_CHANGE_RANGE, VALVE_RATE_LIMIT)

class FLCController:
    # ---Fuzzy Logic Controller for Turbine Speed.# ---
    def __init__(self, setpoint=NOMINAL_SPEED_RPM, dt=DT):
        self.setpoint = setpoint
        self.dt = dt
        self._previous_speed = setpoint
        self.current_valve_position = 0.9 # Initial guess, should align with initial power

        # --- Define Fuzzy Variables ---
        # Inputs
        speed_error = ctrl.Antecedent(np.arange(FLC_ERROR_RANGE[0], FLC_ERROR_RANGE[1] + 1, 1), 'speed_error')
        speed_error_dot = ctrl.Antecedent(np.arange(FLC_ERROR_DOT_RANGE[0], FLC_ERROR_DOT_RANGE[1] + 1, 1), 'speed_error_dot')
        # Output (Change in valve position per step)
        valve_change = ctrl.Consequent(np.arange(FLC_VALVE_CHANGE_RANGE[0], FLC_VALVE_CHANGE_RANGE[1] + 0.001, 0.001), 'valve_change', defuzzify_method='centroid')

        # --- Define Membership Functions (Example - NEEDS SIGNIFICANT TUNING) ---
        # Speed Error MFs
        neg_large_err = FLC_ERROR_RANGE[0] * 0.6
        neg_small_err = FLC_ERROR_RANGE[0] * 0.2
        pos_small_err = FLC_ERROR_RANGE[1] * 0.2
        pos_large_err = FLC_ERROR_RANGE[1] * 0.6
        speed_error['NL'] = fuzz.trimf(speed_error.universe, [FLC_ERROR_RANGE[0], FLC_ERROR_RANGE[0], neg_large_err])
        speed_error['NS'] = fuzz.trimf(speed_error.universe, [neg_large_err, neg_small_err, 0])
        speed_error['ZE'] = fuzz.trimf(speed_error.universe, [neg_small_err, 0, pos_small_err])
        speed_error['PS'] = fuzz.trimf(speed_error.universe, [0, pos_small_err, pos_large_err])
        speed_error['PL'] = fuzz.trimf(speed_error.universe, [pos_large_err, FLC_ERROR_RANGE[1], FLC_ERROR_RANGE[1]])

        # Speed Error Dot MFs
        neg_fast_dot = FLC_ERROR_DOT_RANGE[0] * 0.6
        neg_slow_dot = FLC_ERROR_DOT_RANGE[0] * 0.2
        pos_slow_dot = FLC_ERROR_DOT_RANGE[1] * 0.2
        pos_fast_dot = FLC_ERROR_DOT_RANGE[1] * 0.6
        speed_error_dot['NF'] = fuzz.trimf(speed_error_dot.universe, [FLC_ERROR_DOT_RANGE[0], FLC_ERROR_DOT_RANGE[0], neg_fast_dot])
        speed_error_dot['NS'] = fuzz.trimf(speed_error_dot.universe, [neg_fast_dot, neg_slow_dot, 0])
        speed_error_dot['ZE'] = fuzz.trimf(speed_error_dot.universe, [neg_slow_dot, 0, pos_slow_dot])
        speed_error_dot['PS'] = fuzz.trimf(speed_error_dot.universe, [0, pos_slow_dot, pos_fast_dot])
        speed_error_dot['PF'] = fuzz.trimf(speed_error_dot.universe, [pos_fast_dot, FLC_ERROR_DOT_RANGE[1], FLC_ERROR_DOT_RANGE[1]])

        # Valve Change MFs
        close_fast = FLC_VALVE_CHANGE_RANGE[0] * 0.6
        close_slow = FLC_VALVE_CHANGE_RANGE[0] * 0.2
        open_slow = FLC_VALVE_CHANGE_RANGE[1] * 0.2
        open_fast = FLC_VALVE_CHANGE_RANGE[1] * 0.6
        valve_change['CF'] = fuzz.trimf(valve_change.universe, [FLC_VALVE_CHANGE_RANGE[0], FLC_VALVE_CHANGE_RANGE[0], close_fast])
        valve_change['CS'] = fuzz.trimf(valve_change.universe, [close_fast, close_slow, 0])
        valve_change['NC'] = fuzz.trimf(valve_change.universe, [close_slow * 0.5, 0, open_slow * 0.5]) # Narrower zero change band
        valve_change['OS'] = fuzz.trimf(valve_change.universe, [0, open_slow, open_fast])
        valve_change['OF'] = fuzz.trimf(valve_change.universe, [open_fast, FLC_VALVE_CHANGE_RANGE[1], FLC_VALVE_CHANGE_RANGE[1]])

        # --- Define Fuzzy Rules (CRITICAL - Example subset, needs expert tuning!) ---
        # If speed is too low (negative error) and falling (negative dot), open fast.
        rule1 = ctrl.Rule(speed_error['NL'] & speed_error_dot['NF'], valve_change['OF'])
        rule2 = ctrl.Rule(speed_error['NL'] & speed_error_dot['NS'], valve_change['OF'])
        rule3 = ctrl.Rule(speed_error['NS'] & speed_error_dot['NF'], valve_change['OS'])
        # If speed is too high (positive error) and rising (positive dot), close fast.
        rule4 = ctrl.Rule(speed_error['PL'] & speed_error_dot['PF'], valve_change['CF'])
        rule5 = ctrl.Rule(speed_error['PL'] & speed_error_dot['PS'], valve_change['CF'])
        rule6 = ctrl.Rule(speed_error['PS'] & speed_error_dot['PF'], valve_change['CS'])
        # If speed is near zero and stable, no change.
        rule7 = ctrl.Rule(speed_error['ZE'] & speed_error_dot['ZE'], valve_change['NC'])
        # If speed is slightly low but rising slowly, open slowly.
        rule8 = ctrl.Rule(speed_error['NS'] & speed_error_dot['PS'], valve_change['OS'])
        # If speed is slightly high but falling slowly, close slowly.
        rule9 = ctrl.Rule(speed_error['PS'] & speed_error_dot['NS'], valve_change['CS'])
        # Add rules for all combinations (potentially 25 rules for 5x5 MFs)

        # Create control system
        # TODO: Add ALL defined rules here!
        self.control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

        print("FLC Controller Initialized (RULES NEED THOROUGH REVIEW AND TUNING!).")

    def calculate_action(self, current_measurement):
        # ---Calculates the control action (target valve position).# ---
        error_val = self.setpoint - current_measurement
        error_dot_val = (current_measurement - self._previous_speed) / self.dt if self.dt > 0 else 0
        self._previous_speed = current_measurement

        # Clamp inputs to universe ranges to avoid skfuzzy errors
        error_val_clamped = np.clip(error_val, FLC_ERROR_RANGE[0], FLC_ERROR_RANGE[1])
        error_dot_val_clamped = np.clip(error_dot_val, FLC_ERROR_DOT_RANGE[0], FLC_ERROR_DOT_RANGE[1])

        try:
            self.simulation.input['speed_error'] = error_val_clamped
            self.simulation.input['speed_error_dot'] = error_dot_val_clamped
            self.simulation.compute()
            delta_valve = self.simulation.output['valve_change']

            # Integrate the change to get the new target position
            self.current_valve_position += delta_valve
            self.current_valve_position = np.clip(self.current_valve_position, 0.0, 1.0)

        except Exception as e:
            print(f"FLC computation error: {e}. Inputs: err={error_val:.2f}, err_dot={error_dot_val:.2f}. Holding valve position.")
            # Keep previous valve position on error

        return np.array([self.current_valve_position], dtype=np.float32)

    def reset(self, initial_valve_pos=0.9):
        # ---Resets the internal state of the FLC controller.# ---
        self._previous_speed = self.setpoint
        self.current_valve_position = initial_valve_pos
        # self.simulation.reset() # Check if skfuzzy simulation needs explicit reset
