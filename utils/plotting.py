# utils/plotting.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from config import * # Import limits and nominal values

def plot_simulation_results(history_df, title="Simulation Results"):
    # ---Plots key variables from the simulation history.# ---
    time = history_df['time'].values

    fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(title, fontsize=16)

    # Speed Plot
    axs[0].plot(time, history_df['speed_rpm'], label='Turbine Speed')
    axs[0].axhline(NOMINAL_SPEED_RPM, color='grey', linestyle='--', label='Nominal')
    axs[0].axhline(MAX_SPEED_RPM, color='orange', linestyle=':', label='Control Limit High')
    axs[0].axhline(MIN_SPEED_RPM, color='orange', linestyle=':', label='Control Limit Low')
    axs[0].axhline(TRIP_SPEED_RPM, color='red', linestyle=':', label='Trip Limit')
    axs[0].set_ylabel('Speed (RPM)')
    axs[0].legend(loc='best')
    axs[0].grid(True)

    # Frequency Plot
    axs[1].plot(time, history_df['frequency_hz'], label='Grid Frequency')
    axs[1].axhline(F_NOMINAL, color='grey', linestyle='--', label='Nominal')
    axs[1].axhline(F_NOMINAL + CONTROL_FREQ_DEV_HZ, color='orange', linestyle=':', label='Control Limit High')
    axs[1].axhline(F_NOMINAL - CONTROL_FREQ_DEV_HZ, color='orange', linestyle=':', label='Control Limit Low')
    axs[1].axhline(MAX_FREQ_HZ, color='red', linestyle=':', label='Safety Limit High')
    axs[1].axhline(MIN_FREQ_HZ, color='red', linestyle=':', label='Safety Limit Low')
    axs[1].set_ylabel('Frequency (Hz)')
    axs[1].legend(loc='best')
    axs[1].grid(True)

    # Valve Position Plot
    axs[2].plot(time, history_df['valve_position'], label='Actual Valve Pos', drawstyle='steps-post')
    axs[2].plot(time, history_df['action_taken'], label='Target Valve Pos', linestyle='--', alpha=0.7)
    axs[2].set_ylabel('Valve Position (%)')
    axs[2].set_ylim(-0.05, 1.05)
    axs[2].legend(loc='best')
    axs[2].grid(True)

    # Power Plot
    axs[3].plot(time, history_df['mech_power_mw'], label='P Mechanical (MW)')
    axs[3].plot(time, history_df['elec_load_mw'], label='P Electrical Load (MW)', linestyle='--')
    axs[3].plot(time, history_df['reactor_power_mwth'] * EFFICIENCY_THERMAL, label='P Reactor (MW_e equiv)', linestyle=':', alpha=0.7)
    axs[3].set_ylabel('Power (MW_e)')
    axs[3].legend(loc='best')
    axs[3].grid(True)

    # Temperature Plot
    axs[4].plot(time, history_df['fuel_temp_c'], label='Fuel Temp (C)', color='red')
    axs[4].plot(time, history_df['coolant_temp_c'], label='Coolant Temp (C)', color='blue')
    axs[4].axhline(MAX_FUEL_TEMP, color='darkred', linestyle=':', label='Fuel Temp Limit')
    axs[4].set_ylabel('Temperature (C)')
    axs[4].set_xlabel('Time (s)')
    axs[4].legend(loc='best')
    axs[4].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to prevent title overlap
    plt.show()


def plot_comparison_metrics(all_metrics_df, scenarios, controllers):
     # ---Plots comparative metrics across scenarios and controllers.# ---
     # Example: Plot Speed Overshoot using seaborn barplot
     metric_to_plot = 'max_speed_overshoot_pct'
     if metric_to_plot in all_metrics_df.columns:
         plt.figure(figsize=(12, 6))
         sns.barplot(data=all_metrics_df, x='Scenario', y=metric_to_plot, hue='Controller')
         plt.title(f'Comparison: {metric_to_plot}')
         plt.ylabel('Overshoot (%)')
         plt.xticks(rotation=45, ha='right')
         plt.tight_layout()
         plt.grid(axis='y')
         plt.show()
     else:
         print(f"Metric '{metric_to_plot}' not found in results for comparison plot.")

     # Add more plots for other key metrics (IAE, safety violations, etc.)
     metric_to_plot = 'freq_iae'
     if metric_to_plot in all_metrics_df.columns:
         plt.figure(figsize=(12, 6))
         sns.barplot(data=all_metrics_df, x='Scenario', y=metric_to_plot, hue='Controller')
         plt.title(f'Comparison: {metric_to_plot}')
         plt.ylabel('Frequency IAE (Hz*s)')
         plt.xticks(rotation=45, ha='right')
         plt.tight_layout()
         plt.grid(axis='y')
         plt.show()
     else:
         print(f"Metric '{metric_to_plot}' not found in results for comparison plot.")

