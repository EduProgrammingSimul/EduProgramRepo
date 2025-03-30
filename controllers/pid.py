# controllers/pid.py
import numpy as np
from config import DT, PID_KP, PID_KI, PID_KD, NOMINAL_SPEED_RPM, PID_OUTPUT_LIMITS

class PIDController:
    # ---Standard PID Controller for Turbine Speed.# ---
    def __init__(self, Kp=PID_KP, Ki=PID_KI, Kd=PID_KD, setpoint=NOMINAL_SPEED_RPM,
                 output_limits=PID_OUTPUT_LIMITS, dt=DT):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.dt = dt
        self._integral = 0
        self._previous_error = 0
        self._previous_measurement = None # For derivative on measurement

        print(f"PID Controller Initialized: Kp={Kp}, Ki={Ki}, Kd={Kd}, Setpoint={setpoint}")

    def calculate_action(self, current_measurement):
        # ---Calculates the control action (target valve position).# ---
        error = self.setpoint - current_measurement

        # Integral term with anti-windup (optional, basic clamping here)
        self._integral += error * self.dt
        self._integral = np.clip(self._integral, -10.0, 10.0) # Basic anti-windup limits, tune needed

        # Derivative term (on measurement to reduce kick)
        if self._previous_measurement is not None:
            derivative = (current_measurement - self._previous_measurement) / self.dt
        else:
            derivative = 0
        self._previous_measurement = current_measurement

        # PID formula (Note: Derivative term is negative of error derivative)
        output = self.Kp * error + self.Ki * self._integral - self.Kd * derivative

        # Clamp output to valve limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        self._previous_error = error
        return np.array([output], dtype=np.float32)

    def reset(self):
        # ---Resets the internal state of the PID controller.# ---
        self._integral = 0
        self._previous_error = 0
        self._previous_measurement = None
