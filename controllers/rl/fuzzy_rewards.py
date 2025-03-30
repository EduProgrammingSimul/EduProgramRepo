# controllers/rl/fuzzy_rewards.py
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from config import (FR_MARGIN_RANGE, FR_STABILITY_RANGE, FR_EFFICIENCY_RANGE,
                    FR_REWARD_RANGE, FR_WEIGHTS)

class FuzzyRewardSystem:
    # ---Calculates a scalar reward using fuzzy logic based on performance components.# ---
    def __init__(self, weights=FR_WEIGHTS):
        self.weights = weights

        # --- Define Fuzzy Variables (using standardized 0-100 range) ---
        # Inputs (from env info['reward_components'])
        margin_speed = ctrl.Antecedent(np.arange(0, 101, 1), 'safety_margin_speed')
        margin_freq = ctrl.Antecedent(np.arange(0, 101, 1), 'safety_margin_freq')
        margin_temp = ctrl.Antecedent(np.arange(0, 101, 1), 'safety_margin_temp')
        stability_speed = ctrl.Antecedent(np.arange(0, 101, 1), 'stability_speed')
        stability_freq = ctrl.Antecedent(np.arange(0, 101, 1), 'stability_freq')
        eff_power_match = ctrl.Antecedent(np.arange(0, 101, 1), 'efficiency_power_match')
        eff_control_effort_inv = ctrl.Antecedent(np.arange(0, 101, 1), 'efficiency_control_effort_inv')

        # Output Reward
        reward = ctrl.Consequent(np.arange(FR_REWARD_RANGE[0], FR_REWARD_RANGE[1] + 0.1, 0.1), 'reward', defuzzify_method='centroid')

        # --- Define Membership Functions (Example - Needs Tuning!) ---
        # Generic Poor/Okay/Good for margins/stability/efficiency (0=Bad, 100=Good)
        levels = ['poor', 'okay', 'good']
        mf_ranges = [[0, 0, 40], [20, 50, 80], [60, 100, 100]]
        for var in [margin_speed, margin_freq, margin_temp, stability_speed, stability_freq, eff_power_match, eff_control_effort_inv]:
            for level, mf_range in zip(levels, mf_ranges):
                 var[level] = fuzz.trimf(var.universe, mf_range)

        # Reward Membership Functions
        reward['very_low'] = fuzz.trimf(reward.universe, [FR_REWARD_RANGE[0], FR_REWARD_RANGE[0], -4])
        reward['low'] = fuzz.trimf(reward.universe, [-6, -3, 0])
        reward['medium'] = fuzz.trimf(reward.universe, [-1, 0, 1])
        reward['high'] = fuzz.trimf(reward.universe, [0, 3, 6])
        reward['very_high'] = fuzz.trimf(reward.universe, [4, FR_REWARD_RANGE[1], FR_REWARD_RANGE[1]])

        # --- Define Fuzzy Rules (CRITICAL - Example subset!) ---
        # Weighting can be implicitly handled by how rules combine factors.
        # High importance on Safety: If any margin is poor -> very low reward
        rule_s1 = ctrl.Rule(margin_speed['poor'] | margin_freq['poor'] | margin_temp['poor'], reward['very_low'])

        # High importance on Stability: If stability is poor -> low reward (unless safety is also poor)
        rule_st1 = ctrl.Rule(stability_speed['poor'] | stability_freq['poor'], reward['low'])

        # Baseline good performance: If all margins/stability are good -> high reward
        rule_g1 = ctrl.Rule(margin_speed['good'] & margin_freq['good'] & margin_temp['good'] &
                          stability_speed['good'] & stability_freq['good'], reward['high'])

        # Excellent performance: If all margins/stability/efficiency are good -> very high reward
        rule_ex1 = ctrl.Rule(margin_speed['good'] & margin_freq['good'] & margin_temp['good'] &
                           stability_speed['good'] & stability_freq['good'] &
                           eff_power_match['good'] & eff_control_effort_inv['good'], reward['very_high'])

        # Trade-offs: Good stability but poor efficiency -> medium reward
        rule_tr1 = ctrl.Rule(stability_speed['good'] & stability_freq['good'] &
                           (eff_power_match['poor'] | eff_control_effort_inv['poor']), reward['medium'])

        # TODO: Define a comprehensive rule base covering various combinations.

        # Create control system
        self.control_system = ctrl.ControlSystem([rule_s1, rule_st1, rule_g1, rule_ex1, rule_tr1]) # Add all rules
        self.simulation = ctrl.ControlSystemSimulation(self.control_system, clip_consequents=True)

        print("Fuzzy Reward System Initialized (RULES NEED THOROUGH REVIEW AND TUNING!).")


    def calculate_reward(self, reward_components):
        # ---Computes the scalar fuzzy reward based on input components.# ---
        # Input values into the fuzzy simulation
        try:
            # Clip inputs to ensure they are within the defined universe ranges (0-100)
            self.simulation.input['safety_margin_speed'] = np.clip(reward_components['safety_margin_speed'], 0, 100)
            self.simulation.input['safety_margin_freq'] = np.clip(reward_components['safety_margin_freq'], 0, 100)
            self.simulation.input['safety_margin_temp'] = np.clip(reward_components['safety_margin_temp'], 0, 100)
            self.simulation.input['stability_speed'] = np.clip(reward_components['stability_speed'], 0, 100)
            self.simulation.input['stability_freq'] = np.clip(reward_components['stability_freq'], 0, 100)
            self.simulation.input['efficiency_power_match'] = np.clip(reward_components['efficiency_power_match'], 0, 100)
            self.simulation.input['efficiency_control_effort_inv'] = np.clip(reward_components['efficiency_control_effort_inv'], 0, 100)

            # Compute the fuzzy reward
            self.simulation.compute()
            scalar_reward = self.simulation.output['reward']

            # Handle potential NaN output from skfuzzy if no rules fire
            if np.isnan(scalar_reward):
                # print("Warning: Fuzzy reward computation resulted in NaN (likely no rules fired). Returning 0.")
                scalar_reward = 0.0 # Default reward if no rules match


        except Exception as e:
            print(f"Fuzzy reward computation error: {e}. Components: {reward_components}. Returning 0.")
            scalar_reward = 0.0

        # Optional: Apply weights here if not handled implicitly by rules (less common)
        # scalar_reward *= self.weights['safety'] * self.weights['stability'] * ... (needs careful design)

        return scalar_reward

    def set_context_weights(self, context):
         # ---Placeholder for adapting weights (e.g., via BERT analysis). Not directly used in skfuzzy rules easily.# ---
         # Example adaptation logic
         if context == "emergency":
             self.weights = {'safety': 1.5, 'stability': 0.8, 'efficiency': 0.5}
             print("Fuzzy Reward Context: EMERGENCY (Weights adjusted - Note: affect external scaling, not internal rules directly)")
         elif context == "load_follow":
             self.weights = {'safety': 1.0, 'stability': 1.2, 'efficiency': 1.0}
             print("Fuzzy Reward Context: LOAD FOLLOW")
         else: # Normal
             self.weights = FR_WEIGHTS
             print("Fuzzy Reward Context: NORMAL")
         # How these weights affect the reward requires further design.
         # Maybe scale the final reward output, or try to modify rule consequent strengths.
