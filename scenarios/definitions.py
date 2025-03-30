# scenarios/definitions.py
import numpy as np
from config import NOMINAL_POWER_ELEC, F_NOMINAL, DT

# Base load in MW (assuming nominal electrical power)
BASE_LOAD = NOMINAL_POWER_ELEC * 0.9 # Start at 90% load typically

def get_scenario_profile(scenario_name, time_s):
    
    load = BASE_LOAD # Default

    # 1. Gradual power increase from 90% (BASE_LOAD) to 100% over 5 minutes
    if scenario_name == 'gradual_increase':
        start_time = 10
        duration = 300 # 5 minutes
        end_time = start_time + duration
        target_load = NOMINAL_POWER_ELEC * 1.0
        if start_time <= time_s < end_time:
            load = BASE_LOAD + (target_load - BASE_LOAD) * (time_s - start_time) / duration
        elif time_s >= end_time:
            load = target_load
        else:
            load = BASE_LOAD

    # 2. Sudden grid frequency drop (Simulated by sudden load increase)
    elif scenario_name == 'sudden_load_increase':
        start_time = 20
        increase_factor = 1.10 # 10% load increase
        if time_s >= start_time:
             load = BASE_LOAD * increase_factor
        else:
             load = BASE_LOAD

    # 3. Sensor failure (Handled within environment/controller logic, load is stable)
    elif scenario_name == 'sensor_fail':
        load = BASE_LOAD # Load profile is stable

    # 4. Emergency shutdown (Simulated by load dropping to near zero rapidly)
    elif scenario_name == 'emergency_shutdown':
        start_time = 15
        duration = 2 # seconds
        end_time = start_time + duration
        if start_time <= time_s < end_time:
            load = BASE_LOAD * (1 - (time_s - start_time) / duration)
        elif time_s >= end_time:
            load = 0.01 * NOMINAL_POWER_ELEC # Minimal load
        else:
            load = BASE_LOAD
        load = max(0.0, load) # Ensure non-negative

    # 5. Steam pressure drop (Simulated by reduced Pmech - Modify Turbine Model or add fault)
    elif scenario_name == 'steam_pressure_drop':
         # This is harder to simulate via load alone. Needs modification in Turbine dynamics.
         # For now, assume stable load. Fault injection needed in Env/Turbine.
        load = BASE_LOAD

    # 6. Grid frequency oscillating (Simulated by oscillating load demand)
    elif scenario_name == 'oscillating_load':
        start_time = 10
        oscillation_freq = 0.5 # Hz (Frequency of load oscillation)
        oscillation_amplitude = 0.05 * BASE_LOAD # 5% amplitude
        if time_s >= start_time:
             load = BASE_LOAD + oscillation_amplitude * np.sin(2 * np.pi * oscillation_freq * (time_s - start_time))
        else:
             load = BASE_LOAD

    # 7. Gradual efficiency decline (Simulated by needing higher valve pos for same Pmech - Modify Turbine)
    elif scenario_name == 'efficiency_decline':
        # Needs modification in Turbine efficiency representation. Assume stable load profile.
        load = BASE_LOAD

    # 8. Demand fluctuating +/- 50 MW every minute
    elif scenario_name == 'fluctuating_demand':
        fluctuation = 50 # MW
        period = 60 # seconds
        phase = (time_s % period) / period
        if phase < 0.5: # Ramp up first half
             load = BASE_LOAD + fluctuation * (phase * 2)
        else: # Ramp down second half
             load = BASE_LOAD + fluctuation * (1 - (phase - 0.5) * 2)

    # 9. Two turbines on one grid (Requires multi-machine model - Out of scope for this structure)
    elif scenario_name == 'multi_turbine':
        print("Warning: 'multi_turbine' scenario requires a different grid model structure.")
        load = BASE_LOAD # Default to single machine load

    # Default case / Stable operation
    elif scenario_name == 'stable':
        load = BASE_LOAD

    else:
        print(f"Warning: Unknown scenario name '{scenario_name}'. Using stable base load.")
        load = BASE_LOAD

    return load
