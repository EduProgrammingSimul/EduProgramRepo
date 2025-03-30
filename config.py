# config.py
# ---Configuration file for PWR Simulation Parameters and Constants.# ---

import numpy as np

# --- Simulation Settings ---
DT = 0.01                     # Simulation time step (s)
DEFAULT_SIMULATION_TIME = 300 # Default simulation duration (s)
SOLVER_METHOD = 'RK45'        # ODE solver method ('RK45', 'LSODA', etc.)

# --- Reactor Parameters (Illustrative - USE PLANT SPECIFIC DATA) ---
# Core Physics
NOMINAL_POWER_THERMAL = 3000  # MWth (Example: 1000 MWe plant ~ 3000 MWth)
INITIAL_POWER_FRACTION = 0.9  # Initial power as fraction of nominal
BETA = 0.0065                 # Total delayed neutron fraction
LAMBDA = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]) # Decay constants (1/s)
BETA_I = np.array([0.00021, 0.00142, 0.00127, 0.00257, 0.00075, 0.00027]) # Fraction per group
NEUTRON_LIFETIME = 2e-5       # Prompt neutron lifetime (s)
# Thermal Feedback Coefficients (dk/k / C) - Highly dependent on design
ALPHA_F = -2.5e-5             # Fuel temperature reactivity coefficient
ALPHA_C = -1.0e-4             # Coolant temperature reactivity coefficient
# Thermal Hydraulics (Simplified 2-Node Model - Needs Calibration)
FUEL_HEAT_CAPACITY = 300      # MJ/C (Effective capacity of fuel)
COOLANT_HEAT_CAPACITY = 1000    # MJ/C (Effective capacity of coolant in core)
HEAT_TRANSFER_FC = 500        # MW/C (Effective Fuel-to-Coolant Heat Transfer Coeff)
NOMINAL_COOLANT_TEMP_IN = 290 # Celsius (Core Inlet)
NOMINAL_COOLANT_TEMP_OUT = 330 # Celsius (Core Outlet)
NOMINAL_AVG_COOLANT_TEMP = (NOMINAL_COOLANT_TEMP_IN + NOMINAL_COOLANT_TEMP_OUT) / 2
NOMINAL_FUEL_TEMP = 650      # Celsius (Average fuel temp at nominal power)

# --- Turbine Parameters (Illustrative) ---
NOMINAL_POWER_ELEC = 1000     # MWe
BASE_POWER_MVA = 1100         # MVA (Generator Base for pu calculations)
EFFICIENCY_THERMAL = NOMINAL_POWER_ELEC / NOMINAL_POWER_THERMAL # Approx thermal efficiency
TURBINE_INERTIA_H = 5.0       # Generator + Turbine Inertia Constant (s) on Base MVA
TURBINE_DAMPING_D = 0.8       # Damping coefficient (pu) - Combined mech/elec damping
NOMINAL_SPEED_RPM = 1800      # RPM (for 60Hz, 4-pole generator)
NOMINAL_SPEED_RAD_S = NOMINAL_SPEED_RPM * 2 * np.pi / 60
VALVE_RATE_LIMIT = 0.1        # Max valve position change per second (%/s)
STEAM_CHEST_TAU = 0.5         # Steam chest time constant (s) - simplified steam dynamics

# --- Grid Parameters ---
F_NOMINAL = 60.0              # Nominal grid frequency (Hz)
INFINITE_BUS_VOLTAGE = 1.0    # pu

# --- Safety & Operational Limits ---
MAX_FUEL_TEMP = 2800          # degrees C (Safety Limit)
MIN_SPEED_RPM = 1750 #1790 #1750          # RPM (Operational lower bound for control)
MAX_SPEED_RPM = 1850 #1810 #1850          # RPM (Operational upper bound for control)
TRIP_SPEED_RPM = 1980         # RPM (Turbine trip speed - 110% of nominal)
MIN_FREQ_HZ = F_NOMINAL - 0.5   # Hz (Safety Limit)
MAX_FREQ_HZ = F_NOMINAL + 0.5   # Hz (Safety Limit)
CONTROL_FREQ_DEV_HZ = 0.05    # Hz (Target operational band)

# --- Controller Settings ---
# PID Gains (Needs Tuning)
PID_KP = 2.5
PID_KI = 1.0
PID_KD = 0.1
PID_OUTPUT_LIMITS = (0.0, 1.0) # Valve position limits

# FLC Settings (Placeholder - Ranges need tuning)
FLC_ERROR_RANGE = (-100, 100) # RPM
FLC_ERROR_DOT_RANGE = (-50, 50) # RPM/s
FLC_VALVE_CHANGE_RANGE = (-VALVE_RATE_LIMIT * DT, VALVE_RATE_LIMIT * DT) # Max change per step

# MPC Settings (Placeholder)
MPC_PREDICTION_HORIZON = 10 # seconds
MPC_CONTROL_INTERVAL = 1   # seconds

# RL Settings
RL_LEARNING_RATE = 3e-4
RL_BUFFER_SIZE = int(1e6)
RL_BATCH_SIZE = 256
RL_TAU = 0.005
RL_GAMMA = 0.99
RL_TRAIN_FREQ = 1
RL_GRADIENT_STEPS = 1
# For Constrained SAC (if implemented), add constraint parameters

# --- Fuzzy Reward Settings ---
# Define ranges for fuzzy inputs (margins/deviations are typically 0-100%)
FR_MARGIN_RANGE = (0, 100)
FR_STABILITY_RANGE = (0, 100)
FR_EFFICIENCY_RANGE = (0, 100)
FR_REWARD_RANGE = (-10, 10)
# Weights (can be adapted contextually)
FR_WEIGHTS = {'safety': 1.2, 'stability': 1.0, 'efficiency': 0.8}
