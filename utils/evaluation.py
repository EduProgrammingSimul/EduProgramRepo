# utils/evaluation.py
import numpy as np
import pandas as pd
from config import (NOMINAL_SPEED_RPM, F_NOMINAL, CONTROL_FREQ_DEV_HZ,
                    MAX_SPEED_RPM, MIN_SPEED_RPM, MAX_FREQ_HZ, MIN_FREQ_HZ, DT)

def calculate_performance_metrics(history_df):
    # ---Calculates key performance metrics from simulation history.# ---
    metrics = {}

    time = history_df['time'].values
    speed = history_df['speed_rpm'].values
    freq = history_df['frequency_hz'].values
    valve = history_df['valve_position'].values
    action = history_df['action_taken'].values # Controller target action
    p_mech = history_df['mech_power_mw'].values
    p_load = history_df['elec_load_mw'].values

    # --- Dynamic Response Metrics (Focus on speed/freq) ---
    # Settling Time (Time to enter and stay within X% of final value/band)
    # Requires identifying the end of transients, complex for varying loads. Simplified: Time to enter control band.
    speed_control_band = (MIN_SPEED_RPM, MAX_SPEED_RPM) # Use operational limits
    freq_control_band = (F_NOMINAL - CONTROL_FREQ_DEV_HZ, F_NOMINAL + CONTROL_FREQ_DEV_HZ)

    try:
        first_time_in_speed_band = time[np.where((speed >= speed_control_band[0]) & (speed <= speed_control_band[1]))[0][0]]
        metrics['time_enter_speed_band_s'] = first_time_in_speed_band
    except IndexError:
        metrics['time_enter_speed_band_s'] = np.inf

    try:
        first_time_in_freq_band = time[np.where((freq >= freq_control_band[0]) & (freq <= freq_control_band[1]))[0][0]]
        metrics['time_enter_freq_band_s'] = first_time_in_freq_band
    except IndexError:
        metrics['time_enter_freq_band_s'] = np.inf

    # Overshoot/Undershoot (relative to nominal or final setpoint if changed)
    max_speed_overshoot = (np.max(speed) - NOMINAL_SPEED_RPM) / NOMINAL_SPEED_RPM * 100 if np.max(speed) > NOMINAL_SPEED_RPM else 0
    min_speed_undershoot = (NOMINAL_SPEED_RPM - np.min(speed)) / NOMINAL_SPEED_RPM * 100 if np.min(speed) < NOMINAL_SPEED_RPM else 0
    metrics['max_speed_overshoot_pct'] = max_speed_overshoot
    metrics['min_speed_undershoot_pct'] = min_speed_undershoot

    max_freq_overshoot = (np.max(freq) - F_NOMINAL) / F_NOMINAL * 100 if np.max(freq) > F_NOMINAL else 0
    min_freq_undershoot = (F_NOMINAL - np.min(freq)) / F_NOMINAL * 100 if np.min(freq) < F_NOMINAL else 0
    metrics['max_freq_overshoot_pct'] = max_freq_overshoot
    metrics['min_freq_undershoot_pct'] = min_freq_undershoot

    # Integral Absolute Error (IAE) - Speed and Frequency
    metrics['speed_iae'] = np.sum(np.abs(speed - NOMINAL_SPEED_RPM)) * DT
    metrics['freq_iae'] = np.sum(np.abs(freq - F_NOMINAL)) * DT

    # --- Safety Adherence Metrics ---
    time_outside_speed_limits = np.sum((speed < MIN_SPEED_RPM) | (speed > MAX_SPEED_RPM)) * DT
    time_outside_freq_limits = np.sum((freq < MIN_FREQ_HZ) | (freq > MAX_FREQ_HZ)) * DT
    metrics['time_outside_speed_limits_s'] = time_outside_speed_limits
    metrics['time_outside_freq_limits_s'] = time_outside_freq_limits
    metrics['max_fuel_temp_c'] = history_df['fuel_temp_c'].max()

    # --- Efficiency Metrics ---
    # Control Effort (Sum of squared changes in controller *action*)
    control_effort = np.sum(np.diff(action)**2) if len(action) > 1 else 0
    metrics['control_effort_sum_sq_diff'] = control_effort

    # Power Tracking Error (IAE between mechanical power and load demand)
    power_tracking_iae = np.sum(np.abs(p_mech - p_load)) * DT
    metrics['power_tracking_iae'] = power_tracking_iae

    # --- Consistency (Not directly calculable from single run) ---
    # Requires running multiple scenarios and analyzing variation in metrics.

    return metrics
