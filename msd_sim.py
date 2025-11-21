#!/usr/bin/env python3

"""
Mass–Spring–Damper System Simulation with PID Control (using RK4 integration)

This script simulates the response of a mass–spring–damper (MSD) system to various force inputs, with optional closed-loop PID control.

Features:
    - Simulates the ODE: m * x'' + c * x' + k * x = F(t)
    - Uses a RK4 integrator
    - Supports open-loop (preset force input) and closed-loop (PID-controlled) simulations
    - Includes anti-windup and noise filtering in the PID controller
    - Provides example force/reference input functions (step, impulse, sinusoid, chirp)
    - Plots system response, control input, and energy over time

Usage:
    - Run the script as-is for a step input or closed-loop PID example
    - Edit the main block to select different force/reference inputs or PID parameters
    - Use the plotting functions to visualize system behavior
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.base import clone

try:
    from bayes_opt import BayesianOptimization
    from bayes_opt.acquisition import UpperConfidenceBound, ExpectedImprovement
    BAYES_OPT_AVAILABLE = True
except ImportError as e:
    BAYES_OPT_AVAILABLE = False
    print(f"Warning: bayesian-optimization import failed: {e}")
    print("Run: pip install bayesian-optimization")

@dataclass
class MSDParams:
    mass: float = 1.0     # mass [kg]
    damping: float = 0.5     # damping [N·s/m]
    stiffness: float = 20.0    # stiffness [N/m]

@dataclass
class SimConfig:
    t0: float = 0.0
    tf: float = 10.0
    dt: float = 0.05
    x0: float = 0.0     # initial displacement [m]
    v0: float = 0.0     # initial velocity [m/s]

def rk4_step(f, t, y, h):
    """
    4th-order Runge–Kutta step for y' = f(t, y).
    y is a 1D numpy array.
    """
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h+k2)
    k4 = f(t + h,     y + h*k3)
    return y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

class MassSpringDamper:
    def __init__(self, params: MSDParams, force_fn):
        """
        force_fn: callable F(t) returning scalar force at time t.
        """
        self.p = params
        self.F = force_fn

    def f(self, t, y):
        """
        systems's derivate function: state y = [x, v] & y' = [v, a]
        """
        x, v = y                                            # unpack 2-element state array
        m, c, k = self.p.mass, self.p.damping, self.p.stiffness
        a = (self.F(t) - c*v - k*x) / m                     # computes acceleration
        return np.array([v, a])

    def simulate(self, cfg: SimConfig):
        n = int(np.floor((cfg.tf - cfg.t0) / cfg.dt)) + 1   # compute simulation steps
        t = np.linspace(cfg.t0, cfg.tf, n)                  # create time vector
        y = np.zeros((n, 2))                                # 2D array for y
        y[0, 0] = cfg.x0
        y[0, 1] = cfg.v0

        # For plotting the input (and energy later)
        u = np.zeros(n)                                     # input force array

        for i in range(n - 1):
            u[i] = self.F(t[i])                             # compute input force with function self.F
            y[i+1] = rk4_step(self.f, t[i], y[i], cfg.dt)   # Advance system's state one step using the RK4
        u[-1] = self.F(t[-1])                               # set force at last time step

        x = y[:, 0]
        v = y[:, 1]

        # Energies
        m, c, k = self.p.mass, self.p.damping, self.p.stiffness
        KE = 0.5 * m * v**2
        PE = 0.5 * k * x**2
        E = KE + PE

        return t, x, v, u, E, KE, PE

    def simulate_closed_loop(self, cfg: SimConfig, controller, ref_fn, mode="position", use_feedforward=True):
        """
        Closed-loop sim with PID (holds the force (u) constant during each RK4 step.
        If use_feedforward is True, self.F(t) is added as an external/disturbance feedforward.
        """
        n = int(np.floor((cfg.tf - cfg.t0) / cfg.dt)) + 1   # compute simulation steps
        t = np.linspace(cfg.t0, cfg.tf, n)                  # create time vector
        y = np.zeros((n, 2))                                # 2D array for y
        y[0, 0] = cfg.x0
        y[0, 1] = cfg.v0

         # For plotting the input (and energy later)
        u = np.zeros(n)                                     # input force array
        r_hist = np.zeros(n)                                # reference signal (desired value) history

        m, c, k = self.p.mass, self.p.damping, self.p.stiffness
        controller.reset()                                  # call reset method of the controller

        def f_with_u(_, y_local, u_k):
            """
            Computes  derivative with control input
            """
            x, v = y_local
            a = (u_k - c*v - k*x) / m
            return np.array([v, a])

        for i in range(n - 1):
            x_i, v_i = y[i]                                 # extract current state
            y_meas = x_i if mode == "position" else v_i     # choose what to control (measured output)
            r = ref_fn(t[i])                                # computue reference force (r) at time t
            r_hist[i] = r

            u_pid = controller.update(r=r, y=y_meas, Ts=cfg.dt)
            u_ff  = self.F(t[i]) if use_feedforward else 0.0
            u[i]  = u_pid + u_ff

            f = lambda tt, yy: f_with_u(tt, yy, u[i])       # function for derivative with current control input
            y[i+1] = rk4_step(f, t[i], y[i], cfg.dt)

        r_hist[-1] = ref_fn(t[-1]); u[-1] = u[-2]
        x, v = y[:,0], y[:,1]
        KE = 0.5*m*v**2
        PE = 0.5*k*x**2
        E = KE + PE

        return t, x, v, u, E, KE, PE, r_hist

    def simulate_cascade_loop(self, cfg: SimConfig, cascade_controller, ref_fn, mode="position", use_feedforward=True):
        """
        Closed-loop simulation with cascade PID controller.
        
        Args:
            cfg: Simulation configuration
            cascade_controller: CascadePID instance
            ref_fn: Reference function (position or velocity depending on mode)
            mode: "position" for position control, "velocity" for direct velocity control
            use_feedforward: Whether to add external force as feedforward
            
        Returns:
            t, x, v, u, E, KE, PE, x_ref_hist, v_ref_hist
        """
        n = int(np.floor((cfg.tf - cfg.t0) / cfg.dt)) + 1
        t = np.linspace(cfg.t0, cfg.tf, n)
        y = np.zeros((n, 2))  # [position, velocity]
        y[0, 0] = cfg.x0
        y[0, 1] = cfg.v0
        
        # Arrays for logging
        u = np.zeros(n)                    # control force
        x_ref_hist = np.zeros(n)           # position reference history
        v_ref_hist = np.zeros(n)           # velocity setpoint/reference history
        
        m, c, k = self.p.mass, self.p.damping, self.p.stiffness
        cascade_controller.reset()
        
        def f_with_u(_, y_local, u_k):
            """Dynamics with control input"""
            x, v = y_local
            a = (u_k - c*v - k*x) / m
            return np.array([v, a])
        
        for i in range(n - 1):
            x_i, v_i = y[i]
            
            if mode == "velocity":
                # Direct velocity control mode
                v_ref = ref_fn(t[i])
                x_ref_hist[i] = float('nan')  # No position reference in this mode
                
                # Update cascade controller with direct velocity reference
                u_pid, v_setpoint = cascade_controller.update(
                    v_ref=v_ref, v_meas=v_i, Ts=cfg.dt
                )
                v_ref_hist[i] = v_setpoint  # Same as v_ref in this mode
                
            else:  # mode == "position" (default)
                # Position control mode with cascade structure
                x_ref = ref_fn(t[i])
                x_ref_hist[i] = x_ref
                
                # Update cascade controller (returns force and intermediate velocity setpoint)
                u_pid, v_setpoint = cascade_controller.update(
                    x_ref=x_ref, x_meas=x_i, v_meas=v_i, Ts=cfg.dt
                )
                v_ref_hist[i] = v_setpoint
            
            # Add feedforward if specified
            u_ff = self.F(t[i]) if use_feedforward else 0.0
            u[i] = u_pid + u_ff
            
            # Integrate system dynamics
            f = lambda tt, yy: f_with_u(tt, yy, u[i])
            y[i+1] = rk4_step(f, t[i], y[i], cfg.dt)
        
        # Final values
        if mode == "velocity":
            x_ref_hist[-1] = float('nan')
            v_ref_hist[-1] = ref_fn(t[-1])
        else:
            x_ref_hist[-1] = ref_fn(t[-1])
            v_ref_hist[-1] = v_ref_hist[-2]  # Use previous velocity setpoint
        u[-1] = u[-2]
        
        # Extract states and compute energies
        x, v = y[:, 0], y[:, 1]
        KE = 0.5 * m * v**2
        PE = 0.5 * k * x**2
        E = KE + PE
        
        return t, x, v, u, E, KE, PE, x_ref_hist, v_ref_hist

class PID:
    def __init__(self, Kp=10.0, Ki=0.0, Kd=0.0, u_min=-50.0, u_max=50.0, deriv_tau=0.0):
        """
        PID including:
          - output saturation limits (force bounds)
          - anti-windup (integrator clamping on output saturation)
          - derivative on measurement (better noise behavior)
          - optional 1st-order LPF on Kd to filter out noise: tau / (tau + Ts)

        Args:
            Kp, Ki, Kd: PID gains (Ki = Kp/Ti, Kd = Kp*Td if you prefer Ti/Td language)
            u_min, u_max: output saturation limits (force bounds)
            deriv_tau: LPF on Kd (0 => no filter)
        """

        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.u_min, self.u_max = u_min, u_max
        self.deriv_tau = deriv_tau

        self.I = 0.0        # integrator state
        self.y_prev = None  # last measurement
        self.d_filt = 0.0   # filtered derivative

    def reset(self):
        self.I = 0.0
        self.y_prev = None
        self.d_filt = 0.0

    def update(self, r, y, Ts):
        e = r - y

        # Proportional
        P = self.Kp * e

        # Derivative
        if self.y_prev is None:
            dy = 0.0
        else:
            dy = (y - self.y_prev) / Ts
        self.y_prev = y

        d_raw = -self.Kd * dy

        # LPF on derivative
        if self.deriv_tau > 0.0:
            alpha = self.deriv_tau / (self.deriv_tau + Ts) 
            self.d_filt = alpha * self.d_filt + (1 - alpha) * d_raw
            D = self.d_filt
        else:
            D = d_raw

        # prevent saturation
        u_unsat = P + self.I + D
        u = np.clip(u_unsat, self.u_min, self.u_max)

        # Intergral

        # prevent intergrator windup
        anti_windup_block = not ((u == self.u_max and e > 0) or (u == self.u_min and e < 0))
        if anti_windup_block:
            self.I += self.Ki * Ts * e

        # Recompute with updated integrator
        u_unsat = P + self.I + D
        u = np.clip(u_unsat, self.u_min, self.u_max)

        return u

class CascadePID:
    """
    Cascade PID controller with position outer loop and velocity inner loop.
    
    Two operating modes:
    1. Position Reference -> [Outer PID] -> Velocity Setpoint -> [Inner PID] -> Force Output
    2. Direct Velocity Reference -> [Inner PID] -> Force Output (position loop disabled)
    
    This provides the possibility to tune position and velocity loops separately,
    or use direct velocity control when needed.
    """
    
    def __init__(self, 
                 outer_Kp=10.0, outer_Ki=0.0, outer_Kd=0.0,
                 inner_Kp=50.0, inner_Ki=10.0, inner_Kd=1.0,
                 u_min=-50.0, u_max=50.0,
                 outer_deriv_tau=0.0, inner_deriv_tau=0.0,
                 velocity_limit=None):
        """
        Initialize cascade PID controller.
        
        Args:
            outer_Kp, outer_Ki, outer_Kd: Outer loop (position) PID gains
            inner_Kp, inner_Ki, inner_Kd: Inner loop (velocity) PID gains
            u_min, u_max: Output saturation limits for force
            outer_deriv_tau, inner_deriv_tau: Derivative filtering time constants
            velocity_limit: velocity saturation limit (tuple: (v_min, v_max))
        """
        # Create outer loop PID (position controller)
        # Note: outer loop output is velocity setpoint -> so no saturation on force
        self.outer_pid = PID(Kp=outer_Kp, Ki=outer_Ki, Kd=outer_Kd, 
                            u_min=-1e6, u_max=1e6,
                            deriv_tau=outer_deriv_tau)
        
        # Create inner loop PID (velocity controller) 
        self.inner_pid = PID(Kp=inner_Kp, Ki=inner_Ki, Kd=inner_Kd,
                            u_min=u_min, u_max=u_max,
                            deriv_tau=inner_deriv_tau)
        
        # Velocity saturation limits
        self.velocity_limit = velocity_limit
        
        # Store original outer loop gains for enabling/disabling position loop
        self.original_outer_gains = (outer_Kp, outer_Ki, outer_Kd)
        
    def reset(self):
        """Reset both PID controllers"""
        self.outer_pid.reset()
        self.inner_pid.reset()
        
    def disable_position_loop(self):
        """Disable position loop by setting all gains to zero"""
        self.outer_pid.Kp = 0.0
        self.outer_pid.Ki = 0.0 
        self.outer_pid.Kd = 0.0
        self.outer_pid.reset()  # Clear integrator and derivative states
        
    def enable_position_loop(self):
        """Re-enable position loop with original gains"""
        self.outer_pid.Kp, self.outer_pid.Ki, self.outer_pid.Kd = self.original_outer_gains
        self.outer_pid.reset()  # Reset states when re-enabling
    
    def set_outer_gains(self, Kp_x, Ki_x):
        """Set outer loop (position) PI gains for Bayesian Optimization"""
        self.outer_pid.Kp, self.outer_pid.Ki = Kp_x, Ki_x
    
    def set_inner_gains(self, Kp_v, Ki_v):
        """Set inner loop (velocity) PI gains for Bayesian Optimization"""
        self.inner_pid.Kp, self.inner_pid.Ki = Kp_v, Ki_v
        
    def update(self, x_ref=None, x_meas=None, v_ref=None, v_meas=None, Ts=None):
        """
        Update cascade controller with flexible input modes.
        
        Two modes:
        1. Position control: provide x_ref, x_meas, v_meas, Ts
        2. Direct velocity control: provide v_ref, v_meas, Ts (x_ref, x_meas ignored)
        
        Args:
            x_ref: Position reference/setpoint (for position control mode)
            x_meas: Measured position (for position control mode)
            v_ref: Direct velocity reference (for velocity control mode)
            v_meas: Measured velocity (required for both modes)
            Ts: Sample time (required for both modes)
            
        Returns:
            u: Control force output
            v_setpoint: Velocity setpoint used (from outer loop or direct reference)
        """
        if v_ref is not None:
            # Direct velocity control mode - bypass position loop
            v_setpoint = v_ref
        elif x_ref is not None and x_meas is not None:
            # Position control mode - use cascade structure
            v_setpoint = self.outer_pid.update(r=x_ref, y=x_meas, Ts=Ts)
        else:
            raise ValueError("Must provide either (x_ref, x_meas) for position control or (v_ref) for velocity control")
        
        # Apply velocity limits if specified
        if self.velocity_limit is not None:
            v_min, v_max = self.velocity_limit
            v_setpoint = np.clip(v_setpoint, v_min, v_max)
        
        # Inner loop: velocity error -> force output
        u = self.inner_pid.update(r=v_setpoint, y=v_meas, Ts=Ts)
        
        return u, v_setpoint

# Some forces to use as inputs of feedforward

def step_force(magnitude=1.0, t_delay=0.0):
    return lambda t: magnitude * (t >= t_delay)

def impulse_force(area=1.0, t0=0.0, width=1e-3):
    # Gaussian approximating a Dirac impulse with integral ≈ area
    sigma = width / 3.0
    A = area / (np.sqrt(2*np.pi) * sigma)
    return lambda t: A * np.exp(-0.5 * ((t - t0)/sigma)**2)

def sinusoid_force(A=1.0, freq_hz=1.0, phase=0.0):
    w = 2*np.pi*freq_hz
    return lambda t: A * np.sin(w*t + phase)

def chirp_force(A=1.0, f0=0.1, f1=5.0, tf=10.0):
    # Linear chirp from f0 at t=0 to f1 at t=tf
    k = (f1 - f0) / tf
    return lambda t: A * np.sin(2*np.pi*(f0*t + 0.5*k*t*t))

def step_ref(magnitude=0.1, t_delay=0.5):
    return lambda t: magnitude * (t >= t_delay)

class Scenario:
    """
    Generates reference signals and disturbances for controller testing.
    
    Provides different test scenarios:
    1. Steps and ramps (position and velocity)
    2. Station-keeping (zero velocity reference) under disturbances
    3. Straight-line cruise (constant velocity)
    4. Path following: Combined maneuvers
    """
    
    @staticmethod
    def step_position(magnitude=0.1, t_start=0.5):
        """Position step reference."""
        return lambda t: magnitude * (t >= t_start)
    
    @staticmethod
    def step_velocity(magnitude=0.05, t_start=0.5):
        """Velocity step reference."""
        return lambda t: magnitude * (t >= t_start)
    
    @staticmethod
    def ramp_position(slope=0.02, t_start=0.5, t_end=None):
        """Position ramp reference."""
        def ramp(t):
            if t < t_start:
                return 0.0
            elif t_end is not None and t >= t_end:
                return slope * (t_end - t_start)
            else:
                return slope * (t - t_start)
        return ramp
    
    @staticmethod
    def ramp_velocity(slope=0.01, t_start=0.5, t_end=None):
        """Velocity ramp reference."""
        def ramp(t):
            if t < t_start:
                return 0.0
            elif t_end is not None and t >= t_end:
                return slope * (t_end - t_start)
            else:
                return slope * (t - t_start)
        return ramp
    
    @staticmethod
    def station_keeping(position=0.0):
        """Station-keeping: maintain fixed position (zero velocity)."""
        return lambda t: position
    
    @staticmethod
    def cruise(velocity=0.05):
        """Constant velocity cruise."""
        return lambda t: velocity
    
    @staticmethod
    def multi_step_position(steps, times):
        """
        Multiple position steps at different times.
        
        Args:
            steps: List of step magnitudes
            times: List of step times (must be same length as steps)
        """
        def multi_step(t):
            value = 0.0
            for magnitude, t_step in zip(steps, times):
                if t >= t_step:
                    value = magnitude
            return value
        return multi_step
    
    @staticmethod
    def multi_step_velocity(steps, times):
        """
        Multiple velocity steps at different times.
        
        Args:
            steps: List of step magnitudes
            times: List of step times (must be same length as steps)
        """
        def multi_step(t):
            value = 0.0
            for magnitude, t_step in zip(steps, times):
                if t >= t_step:
                    value = magnitude
            return value
        return multi_step
    
    @staticmethod
    def sinusoidal_position(amplitude=0.05, frequency=0.5, phase=0.0, offset=0.0):
        """Sinusoidal position reference."""
        omega = 2 * np.pi * frequency
        return lambda t: offset + amplitude * np.sin(omega * t + phase)
    
    @staticmethod
    def sinusoidal_velocity(amplitude=0.05, frequency=0.5, phase=0.0, offset=0.0):
        """Sinusoidal velocity reference."""
        omega = 2 * np.pi * frequency
        return lambda t: offset + amplitude * np.sin(omega * t + phase)
    
    @staticmethod
    def combined_maneuver(position_amplitude=0.1, velocity_offset=0.02, 
                         frequency=0.3, t_transition=2.0):
        """
        Combined position and velocity maneuver.
        First phase: position tracking with sinusoid
        Second phase: velocity cruise
        
        Args:
            position_amplitude: Amplitude of position sinusoid
            velocity_offset: Constant velocity after transition
            frequency: Frequency of position sinusoid
            t_transition: Time to switch from position to velocity mode
        """
        omega = 2 * np.pi * frequency
        def maneuver(t):
            if t < t_transition:
                return position_amplitude * np.sin(omega * t)
            else:
                # After transition, maintain the position reached
                return position_amplitude * np.sin(omega * t_transition)
        return maneuver
    
    @staticmethod
    def disturbance_step(magnitude=1.0, t_start=2.0):
        """Step disturbance force."""
        return lambda t: magnitude * (t >= t_start)
    
    @staticmethod
    def disturbance_impulse(area=1.0, t_impulse=2.0, width=0.01):
        """Impulse disturbance (approximated as Gaussian)."""
        sigma = width / 3.0
        A = area / (np.sqrt(2 * np.pi) * sigma)
        return lambda t: A * np.exp(-0.5 * ((t - t_impulse) / sigma)**2)
    
    @staticmethod
    def disturbance_sinusoidal(amplitude=0.5, frequency=1.0):
        """Sinusoidal disturbance force."""
        omega = 2 * np.pi * frequency
        return lambda t: amplitude * np.sin(omega * t)
    
    @staticmethod
    def disturbance_random_walk(magnitude=0.1, seed=42):
        """
        Random walk disturbance (requires time array).
        Returns a function that generates random values with memory.
        """
        rng = np.random.RandomState(seed)
        state = {'last_t': -np.inf, 'value': 0.0}
        
        def random_disturbance(t):
            # Update only if time has advanced
            if t > state['last_t']:
                state['value'] += rng.normal(0, magnitude)
                state['last_t'] = t
            return state['value']
        
        return random_disturbance

class ControllerMetrics:
    """
    Calculates and displays performance metrics for controller evaluation.
    Provides methods to assess
    - tracking performance (IAE, ITAE, RMSE)
    - dynamic response (Rise time, Settling time, Overshoot, Steady-state error)
    - constraint handling (saturation, control rate)
    - plots to visualize tracking performance and error
    """
    def __init__(self):
        pass

    def evaluate(self, t, y, r, u, u_min, u_max, eps=0.02, final_pct=10):

        """
        Evaluates controller performance based on time series data.

        Args:
        t: Array of time points
            y: Array of measured outputs (position or velocity)
            r: Array of reference signals (setpoints)
            u: Array of control inputs
            u_min, u_max: Minimum and maximum control input limits
            eps: Settling band tolerance (default: 2%)
            final_pct: Percentage of final points to use for steady-state calculation (default: 10%)
        """

        # Calculate error and time step
        e = r - y
        dt = t[1] - t[0] if len(t) > 1 else 0.001
        N = len(t) - 1

        # Initialize results dictionary
        metrics = {}
        
        # ===== Tracking Performance =====
        
        # Integral Absolute Error (IAE)
        iae = np.sum(np.abs(e)) * dt
        metrics['IAE'] = iae
        
        # Integral Time Absolute Error (ITAE)
        itae = np.sum(t * np.abs(e)) * dt
        metrics['ITAE'] = itae
        
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.mean(e**2))
        metrics['RMSE'] = rmse
        
        # ===== Dynamic Response =====
        
        # Detect reference type: constant vs step
        r_unique = np.unique(r)
        is_constant = len(r_unique) == 1  # Only one unique value
        is_step = len(r_unique) == 2 and np.sum(np.abs(np.diff(r)) > 1e-6) == 1  # Two values, one transition
        
        if is_step:
            # STEP RESPONSE ANALYSIS
            r_initial = r[0]
            r_final = r[-1]
            r_step = r_final - r_initial
            
            # Find step transition point
            step_idx = np.where(np.abs(np.diff(r)) > 1e-6)[0][0] + 1
            t_step = t[step_idx]
            
            # Rise time calculation (10% to 90% of step)
            y_norm = (y - r_initial) / r_step
            try:
                t_10 = t[np.where(y_norm >= 0.1)[0][0]]
                t_90 = t[np.where(y_norm >= 0.9)[0][0]]
                metrics['Rise time (10-90%)'] = t_90 - t_10
            except IndexError:
                metrics['Rise time (10-90%)'] = float('nan')

            # Overshoot
            if r_step > 0:
                overshoot = (np.max(y) - r_final) / abs(r_final) * 100 if abs(r_final) > 1e-6 else 0
            else:
                overshoot = (r_final - np.min(y)) / abs(r_final) * 100 if abs(r_final) > 1e-6 else 0
            metrics['Overshoot (%)'] = max(0, overshoot)

            # Settling time (2% band around final value)
            settling_threshold = eps * abs(r_final) if abs(r_final) > 1e-6 else eps * abs(r_step)
            settled = np.where(np.abs(e) <= settling_threshold)[0]
            if len(settled) > 0:
                # Find first point that stays settled
                for idx in settled:
                    if np.all(np.abs(e[idx:]) <= settling_threshold):
                        metrics['Settling time (2%)'] = t[idx] - t_step
                        break
                else:
                    metrics['Settling time (2%)'] = float('nan')
            else:
                metrics['Settling time (2%)'] = float('nan')
        
        elif is_constant:
            # CONSTANT REFERENCE (STATION-KEEPING / DISTURBANCE REJECTION)
            r_setpoint = r[0]
            abs_error = np.abs(e)
            max_error = np.max(abs_error)
            
            # Check if there's significant disturbance (deviation from setpoint)
            # Lowered threshold to 0.001 to detect even small disturbances that are well-controlled
            if max_error > 0.001:  # Threshold: 0.001 m or 0.001 m/s
                max_error_idx = np.argmax(abs_error)
                t_disturbance = t[max_error_idx]
                
                # Normalize error relative to peak disturbance (like y_norm in step response)
                # error_norm = 1.0 at peak, 0.0 when fully recovered
                error_norm = abs_error[max_error_idx:] / max_error
                t_post = t[max_error_idx:]
                
                # Rise time: 90% recovery (error drops to 10% of peak)
                try:
                    t_90_recovered = t_post[np.where(error_norm <= 0.1)[0][0]]
                    metrics['Rise time (10-90%)'] = t_90_recovered - t_disturbance
                except IndexError:
                    metrics['Rise time (10-90%)'] = float('nan')
                
                # Settling time: 98% recovery (error drops to 2% of peak and stays there)
                settling_threshold = 0.02  # 2% of peak error
                settled = np.where(error_norm <= settling_threshold)[0]
                if len(settled) > 0:
                    # Find first point that stays settled (same logic as step response)
                    for idx in settled:
                        if np.all(error_norm[idx:] <= settling_threshold):
                            metrics['Settling time (2%)'] = t_post[idx] - t_disturbance
                            break
                    else:
                        metrics['Settling time (2%)'] = float('nan')
                else:
                    metrics['Settling time (2%)'] = float('nan')
                
                # Overshoot = peak deviation as percentage of reference
                # If setpoint is zero, normalize by a reasonable value (0.1 m or 0.1 m/s)
                norm_value = abs(r_setpoint) if abs(r_setpoint) > 1e-6 else 0.1
                metrics['Overshoot (%)'] = (max_error / norm_value) * 100
            else:
                # No significant disturbance - no meaningful rise/settling time
                metrics['Rise time (10-90%)'] = float('nan')
                metrics['Settling time (2%)'] = float('nan')
                metrics['Overshoot (%)'] = 0.0
        
        else:
            # Complex reference (ramp, multi-step, etc.) - skip step metrics
            metrics['Rise time (10-90%)'] = float('nan')
            metrics['Settling time (2%)'] = float('nan')
            metrics['Overshoot (%)'] = 0.0
        
        # Steady-state error (average of last X% of data points)
        n_final = max(1, int(N * final_pct / 100))
        metrics['Steady-state error'] = np.mean(np.abs(e[-n_final:]))
        
        # ===== Constraint Handling =====
        
        # Control saturation analysis
        saturation_min = np.sum(np.isclose(u, u_min, atol=1e-6))
        saturation_max = np.sum(np.isclose(u, u_max, atol=1e-6))
        metrics['Saturation min (%)'] = saturation_min / len(u) * 100
        metrics['Saturation max (%)'] = saturation_max / len(u) * 100
        metrics['Total saturation (%)'] = (saturation_min + saturation_max) / len(u) * 100
        
        # Control rate analysis
        u_rate = np.diff(u) / dt
        metrics['Max control rate'] = np.max(np.abs(u_rate))
        metrics['Mean control rate'] = np.mean(np.abs(u_rate))
        
        # Return all metrics
        return metrics

    def print_metrics(self, metrics):
        """
        Prints the calculated metrics in a formatted way.
        """
        print("\n" + "="*50)
        print("CONTROLLER PERFORMANCE METRICS")
        print("="*50)
        
        # Tracking performance
        print("\n--- Tracking Performance ---")
        print(f"IAE:   {metrics.get('IAE', 'N/A'):.4g}")
        print(f"ITAE:  {metrics.get('ITAE', 'N/A'):.4g}")
        print(f"RMSE:  {metrics.get('RMSE', 'N/A'):.4g}")
        
        # Dynamic response
        print("\n--- Dynamic Response ---")
        rise_time = metrics.get('Rise time (10-90%)', 'N/A')
        settling_time = metrics.get('Settling time (2%)', 'N/A')
        overshoot = metrics.get('Overshoot (%)', 'N/A')
        sse = metrics.get('Steady-state error', 'N/A')
        
        print(f"Rise time (10-90%): {rise_time:.4g} s" if isinstance(rise_time, (int, float)) else f"Rise time (10-90%): {rise_time}")
        print(f"Settling time (2%): {settling_time:.4g} s" if isinstance(settling_time, (int, float)) else f"Settling time (2%): {settling_time}")
        print(f"Overshoot:          {overshoot:.2f} %" if isinstance(overshoot, (int, float)) else f"Overshoot:          {overshoot}")
        print(f"Steady-state error: {sse:.6g}" if isinstance(sse, (int, float)) else f"Steady-state error: {sse}")
        
        # Constraint handling
        print("\n--- Constraint Handling ---")
        sat = metrics.get('Total saturation (%)', 'N/A')
        max_rate = metrics.get('Max control rate', 'N/A')
        
        print(f"Control saturation: {sat:.2f} %" if isinstance(sat, (int, float)) else f"Control saturation: {sat}")
        print(f"Max control rate:   {max_rate:.4g}" if isinstance(max_rate, (int, float)) else f"Max control rate:   {max_rate}")
        
        print("\n" + "="*50 + "\n")
        
    def plot_error_analysis(self, t, y, r, e):
        """
        Creates plots to visualize tracking performance and error.
        """
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot 1: Reference vs Output
        axs[0].plot(t, r, 'r--', label='Reference')
        axs[0].plot(t, y, 'b-', label='Output')
        axs[0].set_ylabel('Value')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        axs[0].set_title('Reference Tracking')
        
        # Plot 2: Error
        axs[1].plot(t, e, 'k-')
        axs[1].set_ylabel('Error')
        axs[1].set_xlabel('Time [s]')
        axs[1].grid(True, alpha=0.3)
        axs[1].set_title('Control Error')
        
        plt.tight_layout()
        plt.show()
    
    def cost_from_metrics(self, metrics_list, weights, normalization_refs, use_worst_case=False):
        """
        Aggregates per-scenario metrics into a scalar cost J for optimization.
        
        Pipeline: timeseries → metrics → normalized metrics → weighted sum → J
        
        This method is purely about metric aggregation and does NOT perform constraint
        checking. Constraint validation should be done upstream (e.g., in BayesianOptimizer).
        
        Args:
            metrics_list: List of per-scenario metrics dictionaries from evaluate()
            weights: Dict with keys {'rise_time', 'settling_time', 'overshoot', 'sse'} 
                    (REQUIRED - no defaults, caller must provide)
            normalization_refs: Dict with reference values for normalization
                               (REQUIRED - no defaults, caller must provide)
            use_worst_case: If True, use max aggregator (robust); if False, use mean aggregator
                        
        Returns:
            float: Scalar cost J (lower is better)
        """
        
        scenario_costs = []
        
        for i, metrics in enumerate(metrics_list):
            # ===== Metric Extraction and Normalization =====
            
            # Extract and normalize rise time (t_r)
            t_r = metrics.get('Rise time (10-90%)', float('nan'))
            if np.isnan(t_r):
                t_r = 10.0  # Pessimistic value if not computed
            t_r_norm = t_r / normalization_refs['rise_time']
            
            # Extract and normalize settling time (t_s)
            t_s = metrics.get('Settling time (2%)', float('nan'))
            if np.isnan(t_s):
                t_s = 20.0  # Pessimistic value if not computed
            t_s_norm = t_s / normalization_refs['settling_time']
            
            # Extract and normalize overshoot (M_p)
            M_p = metrics.get('Overshoot (%)', 0.0)
            M_p_norm = M_p / normalization_refs['overshoot']
            
            # Extract and normalize steady-state error (SSE)
            sse = metrics.get('Steady-state error', float('nan'))
            if np.isnan(sse):
                sse = 1.0  # Pessimistic value if not computed
            sse_norm = sse / normalization_refs['sse']
            
            # ===== Weighted Cost Calculation =====
            # J_s = w_r * t_r_norm + w_s * t_s_norm + w_o * M_p_norm + w_e * sse_norm
            
            cost_s = (
                weights['rise_time'] * t_r_norm +
                weights['settling_time'] * t_s_norm +
                weights['overshoot'] * M_p_norm +
                weights['sse'] * sse_norm
            )
            
            scenario_costs.append(cost_s)
        
        # ===== Aggregation Across Scenarios =====
        
        if use_worst_case:
            # Robust aggregator: J_wc = max over scenarios
            J = np.max(scenario_costs)
        else:
            # Mean aggregator: J = mean over scenarios
            J = np.mean(scenario_costs)
        
        return J

class BayesianOptimizer:
    """
    Bayesian Optimization Evaluator for cascade PID tuning.
    
    Evaluates controller candidates for multiple scenarios,
    implements constraint checking, and provides a callable interface for
    Bayesian optimization libraries.
    
    Supports three tuning stages:
    - 'inner': Tune velocity loop (inner) PI gains only
    - 'outer': Tune position loop (outer) PI gains only (with fixed inner gains)
    - 'joint': Tune all four PI gains simultaneously
    """
    
    def __init__(self, plant, controller, evaluator, scenarios, bounds_log10, 
                 weights_cfg, safety_cfg, sim_cfg=None, rng=None):
        """
        Initialize Bayesian Optimization Evaluator.
        
        Args:
            plant: MassSpringDamper instance (the system to control)
            controller: CascadePID instance (the controller to tune)
            evaluator: ControllerMetrics instance (for computing metrics)
            scenarios: List of scenario tuples: [(ref_fn, disturbance_fn, mode), ...]
                      Each tuple contains (reference_function, disturbance_function, control_mode)
            bounds_log10: Dict of log10 parameter bounds per stage:
                         {
                             'inner': {'log10_Kp_v': [a, b], 'log10_Ki_v': [c, d]},
                             'outer': {'log10_Kp_x': [e, f], 'log10_Ki_x': [g, h]},
                             'joint': {'log10_Kp_v': [...], 'log10_Ki_v': [...],
                                      'log10_Kp_x': [...], 'log10_Ki_x': [...]}
                         }
            weights_cfg: Dict with cost function configuration (REQUIRED):
                        {
                            'weights': {'rise_time': 0.3, 'settling_time': 0.3, ...},
                            'normalization_refs': {'rise_time': 1.0, ...},
                            'use_worst_case': False
                        }
            safety_cfg: Dict with constraint thresholds (REQUIRED):
                       {
                           'max_saturation': 95.0,
                           'max_energy': 1e6,
                           'max_position': 100.0
                       }
            sim_cfg: SimConfig instance for simulation settings (optional)
            rng: Random number generator (for reproducibility, optional)
        """
        self.plant = plant
        self.controller = controller
        self.evaluator = evaluator
        self.scenarios = scenarios
        self.bounds_log10 = bounds_log10
        self.weights_cfg = weights_cfg
        self.safety_cfg = safety_cfg
        self.sim_cfg = sim_cfg if sim_cfg is not None else SimConfig(tf=10.0, dt=0.05)
        self.rng = rng if rng is not None else np.random.RandomState(42)
        
        # Store best results per stage
        self.best = {
            'inner': {'J': float('inf'), 'gains': None, 'metrics': None},
            'outer': {'J': float('inf'), 'gains': None, 'metrics': None},
            'joint': {'J': float('inf'), 'gains': None, 'metrics': None}
        }
        
        # Current tuning stage
        self.current_stage = None
        
        # Evaluation counter
        self.eval_count = 0
    
    def set_stage(self, stage):
        """
        Set the current tuning stage.
        
        Args:
            stage: 'inner', 'outer', or 'joint'
        """
        if stage not in ['inner', 'outer', 'joint']:
            raise ValueError(f"Invalid stage '{stage}'. Must be 'inner', 'outer', or 'joint'.")
        self.current_stage = stage
    
    def _log10_to_gains(self, **log10_params):
        """
        Convert log10 parameters to actual gains.
        
        Args:
            **log10_params: Keyword arguments with log10 parameter values
                          (e.g., log10_Kp_v=1.5, log10_Ki_v=0.8)
        
        Returns:
            dict: Actual gain values {'Kp_v': ..., 'Ki_v': ..., 'Kp_x': ..., 'Ki_x': ...}
        """
        gains = {}
        
        # Map log10 parameters to actual gains
        for param_name, log10_value in log10_params.items():
            if param_name.startswith('log10_'):
                gain_name = param_name[6:]  # Remove 'log10_' prefix
                gains[gain_name] = 10 ** log10_value
        
        return gains
    
    def _apply_gains(self, gains):
        """
        Apply gains to the controller based on current stage.
        
        Args:
            gains: Dict with gain values {'Kp_v': ..., 'Ki_v': ..., 'Kp_x': ..., 'Ki_x': ...}
        """
        # Apply inner loop gains if available
        if 'Kp_v' in gains and 'Ki_v' in gains:
            self.controller.set_inner_gains(gains['Kp_v'], gains['Ki_v'])
        
        # Apply outer loop gains if available
        if 'Kp_x' in gains and 'Ki_x' in gains:
            self.controller.set_outer_gains(gains['Kp_x'], gains['Ki_x'])
    
    def _run_scenarios(self):
        """
        Run all scenarios and collect metrics.
        
        Returns:
            tuple: (metrics_list, all_feasible)
                  - metrics_list: List of metric dicts for each scenario
                  - all_feasible: Boolean indicating if all scenarios were feasible
        """
        metrics_list = []
        all_feasible = True
        
        for i, (ref_fn, disturbance_fn, mode) in enumerate(self.scenarios):
            # Update plant with disturbance
            self.plant.F = disturbance_fn
            
            # Run simulation
            try:
                t, x, v, u, E, KE, PE, x_ref_hist, v_ref_hist = self.plant.simulate_cascade_loop(
                    cfg=self.sim_cfg,
                    cascade_controller=self.controller,
                    ref_fn=ref_fn,
                    mode=mode,
                    use_feedforward=True  # Enable disturbances
                )
            except Exception as e:
                print(f"Scenario {i}: SIMULATION FAILED - {str(e)}")
                all_feasible = False
                return None, False
            
            # Choose appropriate reference and output based on mode
            if mode == "velocity":
                y = v
                r = v_ref_hist
            else:  # position mode
                y = x
                r = x_ref_hist
            
            # Compute metrics
            metrics = self.evaluator.evaluate(
                t=t, y=y, r=r, u=u,
                u_min=self.controller.inner_pid.u_min,
                u_max=self.controller.inner_pid.u_max
            )
            
            # Add energy and position envelope metrics
            metrics['max_energy'] = np.max(E)
            metrics['max_position'] = np.max(np.abs(x))
            
            # Early exit checks (per-scenario safety)
            if metrics['max_energy'] > self.safety_cfg['max_energy']:
                print(f"Scenario {i}: EARLY EXIT - Energy divergence ({metrics['max_energy']:.2e})")
                all_feasible = False
                return None, False
            
            if metrics['Total saturation (%)'] > self.safety_cfg['max_saturation']:
                print(f"Scenario {i}: EARLY EXIT - Excessive saturation ({metrics['Total saturation (%)']:.1f}%)")
                all_feasible = False
                return None, False
            
            if metrics['max_position'] > self.safety_cfg['max_position']:
                print(f"Scenario {i}: EARLY EXIT - Position envelope violation ({metrics['max_position']:.2f})")
                all_feasible = False
                return None, False
            
            # Check for NaN/Inf in critical metrics
            for key in ['Rise time (10-90%)', 'Settling time (2%)', 'Steady-state error']:
                if key in metrics:
                    if np.isnan(metrics[key]) or np.isinf(metrics[key]):
                        print(f"Scenario {i}: EARLY EXIT - {key} is {metrics[key]}")
                        all_feasible = False
                        return None, False
            
            metrics_list.append(metrics)
        
        return metrics_list, all_feasible
    
    def __call__(self, **log10_params):
        """
        Evaluate a candidate controller configuration.
        
        This method is called by the Bayesian Optimization library.
        
        Args:
            **log10_params: Log10-scaled parameters (e.g., log10_Kp_v=1.5, log10_Ki_v=0.8)
        
        Returns:
            float: Negative cost J (for maximization by BO library)
                  Returns large negative value for infeasible candidates
        """
        self.eval_count += 1
        
        # Convert log10 parameters to actual gains
        gains = self._log10_to_gains(**log10_params)
        
        # Apply gains to controller
        self._apply_gains(gains)
        
        # Reset controller state
        self.controller.reset()
        
        # Run all scenarios
        metrics_list, feasible = self._run_scenarios()
        
        # Handle infeasible cases
        if not feasible or metrics_list is None:
            J = 1e6  # Large penalty
            print(f"Eval {self.eval_count}: INFEASIBLE - J={J:.2e}")
            return -J  # Return negative for maximization
        
        # Compute aggregate cost (no constraint checking here - done upstream in _run_scenarios)
        J = self.evaluator.cost_from_metrics(
            metrics_list,
            weights=self.weights_cfg['weights'],
            normalization_refs=self.weights_cfg['normalization_refs'],
            use_worst_case=self.weights_cfg.get('use_worst_case', False)
        )
        
        # Update best if improved
        if self.current_stage and J < self.best[self.current_stage]['J']:
            self.best[self.current_stage]['J'] = J
            self.best[self.current_stage]['gains'] = gains.copy()
            self.best[self.current_stage]['metrics'] = metrics_list
            print(f"Eval {self.eval_count}: NEW BEST for '{self.current_stage}' - J={J:.4f}, gains={gains}")
        else:
            print(f"Eval {self.eval_count}: J={J:.4f}, gains={gains}")
        
        # Return negative cost for maximization (BO maximizes the objective)
        return -J
    
    def evaluate_with_breakdown(self, **log10_params):
        """
        Evaluate a candidate and return detailed breakdown.
        
        Args:
            **log10_params: Log10-scaled parameters
        
        Returns:
            tuple: (J, breakdown, feasible)
                  - J: Scalar cost
                  - breakdown: Dict with per-scenario costs and metrics
                  - feasible: Boolean indicating feasibility
        """
        # Convert log10 parameters to actual gains
        gains = self._log10_to_gains(**log10_params)
        
        # Apply gains to controller
        self._apply_gains(gains)
        
        # Reset controller state
        self.controller.reset()
        
        # Run all scenarios
        metrics_list, feasible = self._run_scenarios()
        
        # Handle infeasible cases
        if not feasible or metrics_list is None:
            J = 1e6
            breakdown = {'feasible': False, 'reason': 'Early exit due to safety constraint'}
            return J, breakdown, False
        
        # Compute aggregate cost (no constraint checking here - done upstream in _run_scenarios)
        J = self.evaluator.cost_from_metrics(
            metrics_list,
            weights=self.weights_cfg['weights'],
            normalization_refs=self.weights_cfg['normalization_refs'],
            use_worst_case=self.weights_cfg.get('use_worst_case', False)
        )
        
        # Build breakdown
        breakdown = {
            'feasible': True,
            'J': J,
            'gains': gains,
            'metrics_list': metrics_list,
            'num_scenarios': len(metrics_list)
        }
        
        return J, breakdown, True
    
    def get_bounds(self, stage=None):
        """
        Get parameter bounds for the specified stage.
        
        Args:
            stage: 'inner', 'outer', or 'joint'. If None, uses current_stage.
        
        Returns:
            dict: Parameter bounds for BayesianOptimization library
        """
        if stage is None:
            stage = self.current_stage
        
        if stage not in self.bounds_log10:
            raise ValueError(f"No bounds defined for stage '{stage}'")
        
        return self.bounds_log10[stage]
    
    def print_best(self, stage=None):
        """
        Print the best result for a given stage.
        
        Args:
            stage: 'inner', 'outer', or 'joint'. If None, prints all stages.
        """
        if stage is None:
            stages = ['inner', 'outer', 'joint']
        else:
            stages = [stage]
        
        print("\n" + "="*60)
        print("BEST RESULTS FROM BAYESIAN OPTIMIZATION")
        print("="*60)
        
        for s in stages:
            best = self.best[s]
            if best['gains'] is not None:
                print(f"\n--- Stage: {s.upper()} ---")
                print(f"Best Cost J: {best['J']:.6f}")
                print(f"Best Gains: {best['gains']}")
            else:
                print(f"\n--- Stage: {s.upper()} ---")
                print("No evaluations completed yet.")
        
        print("\n" + "="*60 + "\n")

def plot_results(t, x, v, u, E, KE, PE):
    fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

    axs[0].plot(t, u)
    axs[0].set_ylabel("Force F(t) [N]")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t, x)
    axs[1].set_ylabel("Displacement x [m]")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(t, v)
    axs[2].set_ylabel("Velocity v [m/s]")
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(t, E, label="Total")
    axs[3].plot(t, KE, label="Kinetic")
    axs[3].plot(t, PE, label="Potential")
    axs[3].set_ylabel("Energy [J]")
    axs[3].set_xlabel("Time [s]")
    axs[3].legend()
    axs[3].grid(True, alpha=0.3)

    zeta = c_to_zeta(params.damping, params.mass, params.stiffness)
    wn = np.sqrt(params.stiffness / params.mass)
    fig.suptitle(f"Mass–Spring–Damper Response\n"
                 f"m={params.mass} kg, c={params.damping} N·s/m, k={params.stiffness} N/m, "
                 f"ζ={zeta:.3f}, ωₙ={wn:.3f} rad/s")
    plt.tight_layout()
    plt.show()

def plot_closed_loop(t, x, v, u, E, KE, PE, r_hist, mode, params):
    fig, axs = plt.subplots(5, 1, figsize=(9, 12), sharex=True)

    # Reference vs measured
    if mode == "position":
        axs[0].plot(t, r_hist, label="x* (ref)")
        axs[0].plot(t, x, label="x (meas)")
        axs[0].set_ylabel("Position [m]")
    else:
        axs[0].plot(t, r_hist, label="v* (ref)")
        axs[0].plot(t, v, label="v (meas)")
        axs[0].set_ylabel("Velocity [m/s]")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # State plots
    axs[1].plot(t, x); axs[1].set_ylabel("x [m]"); axs[1].grid(True, alpha=0.3)
    axs[2].plot(t, v); axs[2].set_ylabel("v [m/s]"); axs[2].grid(True, alpha=0.3)

    # Control
    axs[3].plot(t, u)
    axs[3].set_ylabel("Force u [N]")
    axs[3].grid(True, alpha=0.3)

    # Energy
    axs[4].plot(t, E, label="Total")
    axs[4].plot(t, KE, label="Kinetic")
    axs[4].plot(t, PE, label="Potential")
    axs[4].set_ylabel("Energy [J]")
    axs[4].set_xlabel("Time [s]")
    axs[4].legend()
    axs[4].grid(True, alpha=0.3)

    zeta = c_to_zeta(params.damping, params.mass, params.stiffness)
    wn = np.sqrt(params.stiffness / params.mass)
    fig.suptitle(f"Closed-loop MSD ({mode} control)\n"
                 f"m={params.mass} kg, c={params.damping} N·s/m, k={params.stiffness} N/m, "
                 f"ζ={zeta:.3f}, ωₙ={wn:.3f} rad/s")
    plt.tight_layout()
    plt.show()
 
def plot_cascade(t, x, v, u, E, KE, PE, x_ref_hist, v_ref_hist, params, mode="position"):
    """Plot results for cascade PID control showing both position and velocity loops"""
    fig, axs = plt.subplots(6, 1, figsize=(10, 14), sharex=True)
    
    # Position tracking (outer loop) - skip if in velocity mode
    if mode == "position" and not np.all(np.isnan(x_ref_hist)):
        axs[0].plot(t, x_ref_hist, 'r--', label="x* (position ref)")
        axs[0].plot(t, x, 'b-', label="x (measured)")
        axs[0].set_ylabel("Position [m]")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        axs[0].set_title("Outer Loop: Position Control")
    else:
        axs[0].plot(t, x, 'b-', label="x (position - not controlled)")
        axs[0].set_ylabel("Position [m]")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        axs[0].set_title("Position (Free Response)")
    
    # Velocity tracking (inner loop)
    axs[1].plot(t, v_ref_hist, 'g--', label="v* (velocity reference)")
    axs[1].plot(t, v, 'b-', label="v (measured)")
    axs[1].set_ylabel("Velocity [m/s]")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    if mode == "velocity":
        axs[1].set_title("Direct Velocity Control")
    else:
        axs[1].set_title("Inner Loop: Velocity Control")
    
    # State plots
    axs[2].plot(t, x)
    axs[2].set_ylabel("Position [m]")
    axs[2].grid(True, alpha=0.3)
    
    axs[3].plot(t, v)
    axs[3].set_ylabel("Velocity [m/s]")
    axs[3].grid(True, alpha=0.3)
    
    # Control force
    axs[4].plot(t, u)
    axs[4].set_ylabel("Force [N]")
    axs[4].grid(True, alpha=0.3)
    
    # Energy
    axs[5].plot(t, E, label="Total")
    axs[5].plot(t, KE, label="Kinetic")
    axs[5].plot(t, PE, label="Potential")
    axs[5].set_ylabel("Energy [J]")
    axs[5].set_xlabel("Time [s]")
    axs[5].legend()
    axs[5].grid(True, alpha=0.3)
    
    # Title with system parameters
    zeta = c_to_zeta(params.damping, params.mass, params.stiffness)
    wn = np.sqrt(params.stiffness / params.mass) if params.mass != 0 else float('nan')
    fig.suptitle(f"Cascade PID Control\n"
                 f"m={params.mass} kg, c={params.damping} N·s/m, k={params.stiffness} N/m, "
                 f"ζ={zeta:.3f}, ωₙ={wn:.3f} rad/s")
    plt.tight_layout()
    plt.show()

def c_to_zeta(c, m, k):
    # Damping ratio ζ = c / (2*sqrt(k*m))
    if k <= 0:
        return float('inf')  # Pure damping system (no spring)
    return c / (2.0*np.sqrt(k*m))


def plot_gp_and_acquisition_2d(optimizer, stage_name='inner', param_names=None, acquisition_func=None):
    """
    Visualize GP mean/uncertainty and acquisition function for 2D optimization.
    
    Similar to the example at:
    https://bayesian-optimization.github.io/BayesianOptimization/3.1.0/acquisition_functions.html
    
    Args:
        optimizer: BayesianOptimization instance with fitted GP
        stage_name: Name for saving plots ('inner' or 'outer')
        param_names: List of 2 parameter names [param1, param2]
        acquisition_func: The acquisition function object (ExpectedImprovement, etc.)
    """
    if len(optimizer.space) < 2:
        print(f"  Skipping {stage_name} visualization: need at least 2 evaluations")
        return
    
    # Get parameter bounds
    if param_names is None:
        param_names = list(optimizer.space.keys)
    
    bounds = optimizer.space.bounds
    x1_min, x1_max = bounds[0]
    x2_min, x2_max = bounds[1]
    
    # Create grid for evaluation
    n_grid = 100
    x1 = np.linspace(x1_min, x1_max, n_grid)
    x2 = np.linspace(x2_min, x2_max, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    grid_points = np.column_stack([X1.ravel(), X2.ravel()])
    
    # Get GP predictions
    mu, sigma = optimizer._gp.predict(grid_points, return_std=True)
    mu = mu.reshape(n_grid, n_grid)
    sigma = sigma.reshape(n_grid, n_grid)
    
    # Get acquisition function values
    if acquisition_func is not None:
        acquisition_func._fit_gp(optimizer._gp, optimizer._space)
        acq_func = acquisition_func._get_acq(gp=optimizer._gp)
        utility = -1 * acq_func(grid_points)  # Negate because library minimizes internally
        utility = utility.reshape(n_grid, n_grid)
        
        # Find next best point
        next_idx = np.argmax(utility)
        next_x1 = grid_points[next_idx, 0]
        next_x2 = grid_points[next_idx, 1]
    else:
        utility = None
        next_x1, next_x2 = None, None
    
    # Get observed points
    x_obs = np.array([[res["params"][param_names[0]], res["params"][param_names[1]]] 
                      for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 5))
    steps = len(optimizer.space)
    fig.suptitle(f'{stage_name.upper()} Loop: GP and Utility Function After {steps} Steps', 
                 fontsize=16, fontweight='bold')
    
    # 1. GP Mean (Prediction)
    ax1 = plt.subplot(131)
    contour1 = ax1.contourf(X1, X2, mu, levels=20, cmap='viridis')
    ax1.plot(x_obs[:, 0], x_obs[:, 1], 'r.', markersize=10, label='Observations')
    ax1.plot(x_obs[-1, 0], x_obs[-1, 1], 'r*', markersize=15, label='Last Point')
    cbar1 = plt.colorbar(contour1, ax=ax1)
    cbar1.set_label('GP Mean (predicted target)')
    ax1.set_xlabel(param_names[0])
    ax1.set_ylabel(param_names[1])
    ax1.set_title('GP Mean Prediction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. GP Uncertainty (Std Dev)
    ax2 = plt.subplot(132)
    contour2 = ax2.contourf(X1, X2, sigma, levels=20, cmap='coolwarm')
    ax2.plot(x_obs[:, 0], x_obs[:, 1], 'k.', markersize=10, label='Observations')
    ax2.plot(x_obs[-1, 0], x_obs[-1, 1], 'k*', markersize=15, label='Last Point')
    cbar2 = plt.colorbar(contour2, ax=ax2)
    cbar2.set_label('GP Uncertainty (std)')
    ax2.set_xlabel(param_names[0])
    ax2.set_ylabel(param_names[1])
    ax2.set_title('GP Uncertainty (Standard Deviation)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Acquisition Function (Utility)
    if utility is not None:
        ax3 = plt.subplot(133)
        contour3 = ax3.contourf(X1, X2, utility, levels=20, cmap='plasma')
        ax3.plot(x_obs[:, 0], x_obs[:, 1], 'w.', markersize=10, label='Observations')
        if next_x1 is not None:
            ax3.plot(next_x1, next_x2, 'y*', markersize=20, 
                    markeredgecolor='k', markeredgewidth=1.5,
                    label='Next Best Guess')
        cbar3 = plt.colorbar(contour3, ax=ax3)
        cbar3.set_label('Acquisition Function (EI)')
        ax3.set_xlabel(param_names[0])
        ax3.set_ylabel(param_names[1])
        ax3.set_title('Utility Function (Expected Improvement)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'gp_acquisition_{stage_name}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()  # Close to avoid showing during optimization


def plot_bo_progression_2d(optimizer, stage_name='inner', param_names=None, 
                           iterations_to_show=[5, 15, 30, 50], 
                           acquisition_func=None):
    """
    Create 2D contour progression plots showing GP evolution through iterations.
    
    Shows snapshots of GP mean, uncertainty, and acquisition function at different
    iteration milestones. Much more informative than just the final converged state.
    
    Args:
        optimizer: BayesianOptimization instance
        stage_name: 'inner' or 'outer'
        param_names: List of 2 parameter names [param1, param2]
        iterations_to_show: List of iteration milestones to visualize
        acquisition_func: The acquisition function object (for EI calculation)
    """
    if param_names is None:
        param_names = list(optimizer.space.keys)
    
    # Get parameter bounds
    bounds = optimizer.space.bounds
    x1_min, x1_max = bounds[0]
    x2_min, x2_max = bounds[1]
    
    # Create grid for evaluation
    n_grid = 80  # Slightly lower resolution for faster computation
    x1 = np.linspace(x1_min, x1_max, n_grid)
    x2 = np.linspace(x2_min, x2_max, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    grid_points = np.column_stack([X1.ravel(), X2.ravel()])
    
    # Filter iterations to those that actually exist
    max_iter = len(optimizer.res)
    iterations_to_show = [i for i in iterations_to_show if i <= max_iter]
    
    if not iterations_to_show:
        print(f"  Warning: No valid iterations to show for {stage_name}")
        return None
    
    # Create figure with subplots (3 columns x N rows)
    n_rows = len(iterations_to_show)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    print(f"  Creating {stage_name} progression with {n_rows} snapshots...")
    
    for row_idx, n_iter in enumerate(iterations_to_show):
        # Get data up to this iteration
        if n_iter == 0:
            # Just show prior (no data)
            X_obs = np.array([]).reshape(0, 2)
            y_obs = np.array([])
        else:
            res_subset = optimizer.res[:n_iter]
            X_obs = np.array([[res["params"][pname] for pname in param_names] 
                             for res in res_subset])
            y_obs = np.array([res["target"] for res in res_subset])
        
        # Debug: Show what's really being plotted
        if len(y_obs) > 0:
            print(f"    Iter {n_iter}: {len(X_obs)} obs, y_max={y_obs.max():.6f}, xi={acquisition_func.xi if acquisition_func else 'N/A'}")
        
        # Fit GP with data up to this iteration
        if len(X_obs) > 0:
            temp_gp = clone(optimizer._gp)
            temp_gp.fit(X_obs, y_obs)
            
            # Get GP predictions
            mu, sigma = temp_gp.predict(grid_points, return_std=True)
            mu = mu.reshape(n_grid, n_grid)
            sigma = sigma.reshape(n_grid, n_grid)
            
            # Calculate acquisition function
            if acquisition_func is not None:
                y_max = y_obs.max()
                mu_flat = mu.ravel()
                sigma_flat = sigma.ravel()
                
                # Manually calculate EI
                with np.errstate(divide='warn', invalid='ignore'):
                    imp = mu_flat - y_max - acquisition_func.xi
                    Z = imp / sigma_flat
                    ei = imp * norm.cdf(Z) + sigma_flat * norm.pdf(Z)
                    ei[sigma_flat == 0.0] = 0.0
                
                utility = ei.reshape(n_grid, n_grid)
                
                # Debug: Show EI statistics
                print(f"      EI: min={ei.min():.2e}, max={ei.max():.2e}, mean={ei.mean():.2e}, sigma: min={sigma_flat.min():.2e}, max={sigma_flat.max():.2e}")
                
                # Find next best point
                next_idx = np.argmax(ei)
                next_x1 = grid_points[next_idx, 0]
                next_x2 = grid_points[next_idx, 1]
            else:
                utility = None
                next_x1, next_x2 = None, None
        else:
            # Prior prediction
            mu = np.zeros((n_grid, n_grid))
            sigma = np.ones((n_grid, n_grid))
            utility = None
            next_x1, next_x2 = None, None
        
        # COLUMN 1: GP Mean
        ax1 = axes[row_idx, 0]
        contour1 = ax1.contourf(X1, X2, mu, levels=20, cmap='viridis')
        if len(X_obs) > 0:
            ax1.plot(X_obs[:, 0], X_obs[:, 1], 'r.', markersize=8, label='Observations')
            # Highlight best point
            best_idx = np.argmax(y_obs)
            ax1.plot(X_obs[best_idx, 0], X_obs[best_idx, 1], 'g*', markersize=15,
                    markeredgecolor='black', markeredgewidth=1, label='Best So Far')
        cbar1 = plt.colorbar(contour1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('GP Mean', fontsize=10)
        ax1.set_xlabel(param_names[0], fontsize=11)
        ax1.set_ylabel(param_names[1], fontsize=11)
        ax1.set_title(f'Iteration {n_iter}: GP Mean', fontsize=12, fontweight='bold')
        if len(X_obs) > 0:
            ax1.legend(fontsize=9, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # COLUMN 2: GP Uncertainty
        ax2 = axes[row_idx, 1]
        contour2 = ax2.contourf(X1, X2, sigma, levels=20, cmap='coolwarm')
        if len(X_obs) > 0:
            ax2.plot(X_obs[:, 0], X_obs[:, 1], 'k.', markersize=8, label='Observations')
        cbar2 = plt.colorbar(contour2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Std Dev', fontsize=10)
        ax2.set_xlabel(param_names[0], fontsize=11)
        ax2.set_ylabel(param_names[1], fontsize=11)
        ax2.set_title(f'Iteration {n_iter}: GP Uncertainty', fontsize=12, fontweight='bold')
        if len(X_obs) > 0:
            ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # COLUMN 3: Acquisition Function
        ax3 = axes[row_idx, 2]
        if utility is not None:
            contour3 = ax3.contourf(X1, X2, utility, levels=20, cmap='plasma')
            ax3.plot(X_obs[:, 0], X_obs[:, 1], 'w.', markersize=8, alpha=0.7, label='Observations')
            if next_x1 is not None:
                ax3.plot(next_x1, next_x2, 'y*', markersize=18,
                        markeredgecolor='k', markeredgewidth=1.5,
                        label='Next Sample')
            cbar3 = plt.colorbar(contour3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.set_label('EI', fontsize=10)
            ax3.legend(fontsize=9, loc='upper right')
        else:
            ax3.text(0.5, 0.5, 'No acquisition\n(prior only)', 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=14, color='gray')
        
        ax3.set_xlabel(param_names[0], fontsize=11)
        ax3.set_ylabel(param_names[1], fontsize=11)
        ax3.set_title(f'Iteration {n_iter}: Acquisition (EI)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'{stage_name.upper()} Loop: Bayesian Optimization Progression (2D)', 
                 fontsize=16, fontweight='bold', y=0.998)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'bo_progression_2d_{stage_name}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()
    
    return fig


def plot_bo_progression_1d(optimizer, stage_name='inner', param_to_vary='log10_Kp_v', 
                           iterations_to_show=[0, 5, 15, 30, 50], 
                           acquisition_func=None, best_gains=None):
    """
    Create a progression plot showing GP evolution through iterations (1D slices).
    
    Similar to: https://github.com/bayesian-optimization/BayesianOptimization/blob/master/docsrc/static/bo_example.png
    
    Shows multiple snapshots of the optimization process with GP mean, uncertainty,
    and acquisition function at different iteration milestones. This is much more
    informative than the final converged state where EI ≈ 0.
    
    Args:
        optimizer: BayesianOptimization instance
        stage_name: 'inner' or 'outer'
        param_to_vary: Which parameter to vary (the other is fixed at optimal)
        iterations_to_show: List of iteration milestones to visualize
        acquisition_func: The acquisition function object (for EI calculation)
        best_gains: Dict with best gain values to fix the other parameter
    """
    if param_to_vary not in optimizer._space.keys:
        print(f"  Warning: {param_to_vary} not in parameter space")
        return None
    
    # Get parameter bounds
    param_names = list(optimizer._space.keys)
    bounds = optimizer._space.bounds
    
    param_idx = param_names.index(param_to_vary)
    param_bounds = bounds[param_idx]
    
    # Create 1D grid for the parameter we're varying
    x_plot = np.linspace(param_bounds[0], param_bounds[1], 300).reshape(-1, 1)
    
    # Fix other parameter at its best value
    other_param_idx = 1 - param_idx  # 0 -> 1, 1 -> 0
    
    # Determine the fixed value for the other parameter
    # NOTE: We use the OPTIMAL value from best_gains
    # This can result in FLAT slices if cost is insensitive to param_to_vary when other is optimal
    # Alternative: Use np.mean(bounds[other_param_idx]) for more variation (but less relevant slice)
    if best_gains:
        if param_to_vary == 'log10_Kp_v':
            other_param_value = np.log10(best_gains.get('Ki_v', 1700))
        elif param_to_vary == 'log10_Ki_v':
            other_param_value = np.log10(best_gains.get('Kp_v', 600))
        elif param_to_vary == 'log10_Kp_x':
            other_param_value = np.log10(best_gains.get('Ki_x', 20))
        elif param_to_vary == 'log10_Ki_x':
            other_param_value = np.log10(best_gains.get('Kp_x', 0.1))
        else:
            other_param_value = np.mean(bounds[other_param_idx])
    else:
        other_param_value = np.mean(bounds[other_param_idx])
    
    # Create 2D grid for GP prediction (1D slice through 2D space)
    X_plot = np.zeros((len(x_plot), 2))
    X_plot[:, param_idx] = x_plot.flatten()
    X_plot[:, other_param_idx] = other_param_value
    
    # Filter iterations to those that actually exist
    max_iter = len(optimizer.res)
    iterations_to_show = [i for i in iterations_to_show if i <= max_iter]
    
    if not iterations_to_show:
        print(f"  Warning: No valid iterations to show for {stage_name}")
        return None
    
    # Create figure with subplots (2 columns: GP + Acquisition)
    n_rows = len(iterations_to_show)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, n_iter in enumerate(iterations_to_show):
        # Get data up to this iteration
        if n_iter == 0:
            # Just show prior (no data)
            X_obs = np.array([]).reshape(0, 2)
            y_obs = np.array([])
        else:
            res_subset = optimizer.res[:n_iter]
            X_obs = np.array([[res["params"][pname] for pname in param_names] 
                             for res in res_subset])
            y_obs = np.array([res["target"] for res in res_subset])
        
        # Fit GP with data up to this iteration
        if len(X_obs) > 0:
            temp_gp = clone(optimizer._gp)
            temp_gp.fit(X_obs, y_obs)
        else:
            temp_gp = clone(optimizer._gp)
        
        # Predict on 1D slice
        if len(X_obs) > 0:
            mu, sigma = temp_gp.predict(X_plot, return_std=True)
        else:
            # Prior prediction
            mu = np.zeros(len(X_plot))
            sigma = np.ones(len(X_plot))
        
        # Check if posterior is relatively flat (low variation)
        if len(X_obs) > 0 and row_idx == len(iterations_to_show) - 1:  # Check on last iteration
            mu_range = mu.max() - mu.min()
            if mu_range < 0.01:  # Very flat
                print(f"    Note: {param_to_vary} slice is relatively flat (range={mu_range:.6f})")
                print(f"          This suggests low sensitivity when {param_names[other_param_idx]} is fixed at optimal")
        
        # Convert negative target back to cost (remember we maximize -J)
        mu_cost = -mu
        
        # Get observed points on this 1D slice (only show points NEAR the slice)
        # Calculate distance from slice in the other dimension
        if len(X_obs) > 0:
            # Calculate how far each observation is from the slice
            other_param_range = param_bounds[1] - param_bounds[0] if param_idx == 0 else bounds[0][1] - bounds[0][0]
            threshold = 0.15 * other_param_range  # Only show points within 15% of range
            
            distances_from_slice = np.abs(X_obs[:, other_param_idx] - other_param_value)
            near_slice_mask = distances_from_slice < threshold
            
            obs_on_slice = X_obs[near_slice_mask, param_idx]
            obs_costs = -y_obs[near_slice_mask]  # Convert back to cost
            
            # Also mark ALL observations faintly to show the 2D distribution
            all_obs_on_axis = X_obs[:, param_idx]
            all_obs_costs = -y_obs
        else:
            obs_on_slice = np.array([])
            obs_costs = np.array([])
            all_obs_on_axis = np.array([])
            all_obs_costs = np.array([])
        
        # LEFT PANEL: GP Mean + Uncertainty
        ax_gp = axes[row_idx, 0]
        
        # Plot GP mean
        ax_gp.plot(x_plot, mu_cost, 'b-', lw=2, label='GP Mean')
        
        # Plot uncertainty bands (±2 sigma = 95% confidence)
        ax_gp.fill_between(
            x_plot.flatten(),
            mu_cost - 2*sigma,
            mu_cost + 2*sigma,
            alpha=0.3,
            color='blue',
            label='95% Confidence'
        )
        
        # Plot ALL observed points (faintly, to show 2D distribution projected to 1D)
        if len(all_obs_on_axis) > 0:
            ax_gp.plot(all_obs_on_axis, all_obs_costs, 'o', 
                      color='lightcoral', markersize=4, alpha=0.3,
                      label='All Obs (projected)', zorder=3)
        
        # Plot observed points NEAR this slice (these are relevant!)
        if len(obs_on_slice) > 0:
            ax_gp.plot(obs_on_slice, obs_costs, 'ro', markersize=8, 
                      label='Obs Near Slice', zorder=5)
            
            # Highlight best point so far (among near-slice observations)
            best_idx = np.argmin(obs_costs)
            ax_gp.plot(obs_on_slice[best_idx], obs_costs[best_idx], 
                      'g*', markersize=15, markeredgecolor='black', markeredgewidth=1,
                      label='Best Near Slice', zorder=6)
        
        ax_gp.set_xlabel(param_to_vary, fontsize=12)
        ax_gp.set_ylabel('Cost J', fontsize=12)
        
        # Create informative title showing the fixed parameter value
        other_param_name = param_names[other_param_idx].replace('log10_', '')
        fixed_param_value = 10 ** other_param_value
        title_text = f'Iteration {n_iter}: GP Posterior (fixing {other_param_name}={fixed_param_value:.1f})'
        ax_gp.set_title(title_text, fontsize=14, fontweight='bold')
        
        ax_gp.legend(loc='upper right', fontsize=9)
        ax_gp.grid(True, alpha=0.3)
        ax_gp.set_xlim(param_bounds)
        
        # RIGHT PANEL: Acquisition Function
        ax_acq = axes[row_idx, 1]
        
        if acquisition_func and len(X_obs) > 0:
            # Calculate acquisition function (Expected Improvement)
            y_max = y_obs.max()
            
            # Manually calculate EI
            with np.errstate(divide='warn', invalid='ignore'):
                imp = mu - y_max - acquisition_func.xi
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
            
            ax_acq.plot(x_plot, ei, 'purple', lw=2, label='Expected Improvement')
            ax_acq.fill_between(x_plot.flatten(), 0, ei, alpha=0.3, color='purple')
            
            # Mark next sampling point (argmax of EI)
            next_idx = np.argmax(ei)
            ax_acq.plot(x_plot[next_idx], ei[next_idx], 'r*', markersize=15, 
                       markeredgecolor='black', markeredgewidth=1,
                       label='Next Sample', zorder=5)
            
            ax_acq.set_ylabel('EI Value', fontsize=12)
            ax_acq.legend(loc='upper right', fontsize=10)
        else:
            ax_acq.text(0.5, 0.5, 'No acquisition\n(prior only)', 
                       ha='center', va='center', transform=ax_acq.transAxes,
                       fontsize=14, color='gray')
            ax_acq.set_ylabel('Acquisition', fontsize=12)
        
        ax_acq.set_xlabel(param_to_vary, fontsize=12)
        ax_acq.set_title(f'Iteration {n_iter}: Acquisition Function', fontsize=14, fontweight='bold')
        ax_acq.grid(True, alpha=0.3)
        ax_acq.set_xlim(param_bounds)
    
    # Overall title
    param_label = param_to_vary.replace('log10_', '')
    other_param_label = param_names[other_param_idx].replace('log10_', '')
    fixed_val = 10 ** other_param_value
    
    title_main = f'{stage_name.upper()} Loop: BO Progression for {param_label}'
    title_sub = f'(1D slice: fixing {other_param_label}={fixed_val:.1f})'
    
    # Add explanation if slice appears flat
    if len(iterations_to_show) > 0 and hasattr(fig, '_flatness_note'):
        title_sub += '\nNote: Flat posterior indicates low sensitivity to this parameter when other is at optimal'
    
    fig.suptitle(f'{title_main}\n{title_sub}', 
                 fontsize=16, fontweight='bold', y=0.998)
    
    plt.tight_layout()
    
    # Save figure
    param_short = param_to_vary.replace('log10_', '')
    filename = f'bo_progression_{stage_name}_{param_short}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()
    
    return fig


if __name__ == "__main__":
    # System parameters
    params = MSDParams(mass=20.0, damping=20.0, stiffness=0.0)

    # Simulation config for RK4
    cfg = SimConfig(t0=0.0, tf=10.0, dt=0.05, x0=0.0, v0=0.0)

    # Only necessary for manually single PID controller
    # Standard tune guide for PID:
    # Increase Kp until response with minor overshoot
    # Add Ki to remove steady-state error
    # Add Kd to dampen overshoot (set deriv_tau ~ 1–5*dt for smoothing)
    pid_p = PID(Kp=600.0, Ki=60.0, Kd=10.0, u_min=-50.0, u_max=50.0, deriv_tau=0.002)   # position PID
    pid_v = PID(Kp=600.0, Ki=2000.0, Kd=0.0, u_min=-50.0, u_max=50.0, deriv_tau=0.005)    # velocity PID
    
    # Cascade PID controller (outer loop: position, inner loop: velocity)
    cascade_pid = CascadePID(
        outer_Kp=0.8, outer_Ki=20, outer_Kd=0.0,    # Position loop gains
        inner_Kp=565.0, inner_Ki=2500.0, inner_Kd=0.0,   # Velocity loop gains
        u_min=-50.0, u_max=50.0,
        outer_deriv_tau=0.002, inner_deriv_tau=0.005,
        velocity_limit=(-2.0, 2.0)  # Optional velocity saturation
    )

    # Initialize controller evaluator for performance metrics
    evaluator = ControllerMetrics()

    # Choose what to run
    # Set to "open-loop", "position", "velocity", "cascade", "velocity-setpoint", "comparison", or "bayesian_optimization"
    run_mode = "velocity-setpoint"

    # Initialize MSD
    msd = MassSpringDamper(params, force_fn=lambda t: 0.0)  # force is provided by the PID in closed-loop
    
    if run_mode == "open-loop":
        # Open-loop simulation with step force

        # Choose one force input
        # F = step_force(magnitude=1.0, t_delay=0.5)
        # F = impulse_force(area=1.0, t0=0.2, width=0.01)
        F = sinusoid_force(A=1.0, freq_hz=1.0, phase=0.0)
        # F = chirp_force(A=1.0, f0=0.2, f1=5.0, tf=cfg.tf)

        msd = MassSpringDamper(params, F)
        t, x, v, u, E, KE, PE = msd.simulate(cfg)
        plot_results(t, x, v, u, E, KE, PE)
    
    elif run_mode == "position":
        # Position control example
        ref_p = step_ref(magnitude=0.1, t_delay=0.5)
        t, x, v, u, E, KE, PE, r_hist = msd.simulate_closed_loop(cfg, pid_p, ref_fn=ref_p, mode="position", use_feedforward=False)
        plot_closed_loop(t, x, v, u, E, KE, PE, r_hist, mode="position", params=params)
        
        # Evaluate position controller performance
        metrics = evaluator.evaluate(
            t=t,
            y=x,  # position is measured output
            r=r_hist, 
            u=u, 
            u_min=pid_p.u_min, 
            u_max=pid_p.u_max
        )
        evaluator.print_metrics(metrics)
        
        # Plot error analysis
        error = r_hist - x
        evaluator.plot_error_analysis(t, x, r_hist, error)
    
    elif run_mode == "velocity":
        # Velocity control example
        ref_v = step_ref(magnitude=0.2, t_delay=0.5)  # 0.5 m/s
        t, x, v, u, E, KE, PE, r_hist = msd.simulate_closed_loop(cfg, pid_v, ref_fn=ref_v, mode="velocity")
        plot_closed_loop(t, x, v, u, E, KE, PE, r_hist, mode="velocity", params=params)
        
        # Evaluate velocity controller performance
        metrics = evaluator.evaluate(
            t=t, 
            y=v,
            r=r_hist, 
            u=u, 
            u_min=pid_v.u_min, 
            u_max=pid_v.u_max
        )
        evaluator.print_metrics(metrics)
        
        # Plot error analysis
        error = r_hist - v
        evaluator.plot_error_analysis(t, v, r_hist, error)
    
    elif run_mode == "cascade":
        # Change this to switch between velocity and position control:
        TEST_MODE = "position"  # "velocity" or "position"
        
        # Configure test based on mode
        if TEST_MODE == "velocity":
            # VELOCITY MODE: Test inner loop directly (position loop disabled)
            # Choose one of these reference scenarios:
            
            reference = Scenario.cruise(velocity=0.0)
            disturbance = Scenario.disturbance_impulse(area=1.0, t_impulse=2.0, width=0.01)
            test_description = "Cruise at 0 m/s with impulse disturbance"
            
            # reference = Scenario.step_velocity(magnitude=0.2, t_start=0.5)
            # disturbance = lambda t: 0.0
            # test_description = "Velocity step 0.2 m/s"
            
            control_mode = "velocity"
            active_loop = "Inner (velocity)"
            
        else:
            # POSITION MODE: Test full cascade (both loops active)
            # Choose one of these reference scenarios:
            
            reference = Scenario.step_position(magnitude=0.1, t_start=0.5)
            disturbance = lambda t: 0.0
            test_description = "Position step 0.1 m"
            
            # reference = Scenario.step_position(magnitude=0.1, t_start=0.5)
            # disturbance = Scenario.disturbance_step(magnitude=5.0, t_start=5.0)
            # test_description = "Position step 0.1 m with disturbance at t=5s"
            
            control_mode = "position"
            active_loop = "Both (cascade)"
        
        # Update MSD with disturbance
        msd.F = disturbance
        
        # Run simulation
        test_cfg = SimConfig(t0=0.0, tf=15.0, dt=0.05, x0=0.0, v0=0.0)
        
        t, x, v, u, E, KE, PE, x_ref_hist, v_ref_hist = msd.simulate_cascade_loop(
            test_cfg, cascade_pid, ref_fn=reference, mode=control_mode, use_feedforward=True
        )
        
        # Select controlled variable and reference based on mode
        if TEST_MODE == "velocity":
            controlled_variable = v
            reference_signal = v_ref_hist
        else:  # position
            controlled_variable = x
            reference_signal = x_ref_hist
        
        # Plot results
        plot_cascade(t, x, v, u, E, KE, PE, x_ref_hist, v_ref_hist, params, mode=control_mode)
        
        # Evaluate controller performance
        metrics = evaluator.evaluate(
            t=t,
            y=controlled_variable,
            r=reference_signal,
            u=u,
            u_min=cascade_pid.inner_pid.u_min,
            u_max=cascade_pid.inner_pid.u_max
        )
        evaluator.print_metrics(metrics)
        
        # Calculate cost J
        weights_cfg = {
            'rise_time': 0.25,
            'settling_time': 0.35,
            'overshoot': 0.20,
            'sse': 0.20
        }
        normalization_refs = {
            'rise_time': 1.0,
            'settling_time': 5.0,
            'overshoot': 20.0,
            'sse': 0.01
        }
        J = evaluator.cost_from_metrics(
            [metrics],
            weights=weights_cfg,
            normalization_refs=normalization_refs,
            use_worst_case=False
        )
        print(f"\n=== COST FUNCTION ===")
        print(f"Cost J: {J:.6f} (lower is better)")
        
        # Plot error analysis
        error = reference_signal - controlled_variable
        evaluator.plot_error_analysis(t, controlled_variable, reference_signal, error)
        
        # Print controller info
        print(f"\n=== CASCADE CONTROLLER INFO ({TEST_MODE.upper()} MODE) ===")
        print(f"Test: {test_description}")
        print(f"Active loop: {active_loop}")
        print(f"Outer loop gains (position): Kp={cascade_pid.outer_pid.Kp}, Ki={cascade_pid.outer_pid.Ki}, Kd={cascade_pid.outer_pid.Kd}")
        print(f"Inner loop gains (velocity): Kp={cascade_pid.inner_pid.Kp}, Ki={cascade_pid.inner_pid.Ki}, Kd={cascade_pid.inner_pid.Kd}")
        if cascade_pid.velocity_limit:
            print(f"Velocity limits: {cascade_pid.velocity_limit} m/s")
    
    elif run_mode == "velocity-setpoint":
        # Direct velocity setpoint control (bypass position loop)
        ref_v = step_ref(magnitude=0.2, t_delay=.5) 
        t, x, v, u, E, KE, PE, x_ref_hist, v_ref_hist = msd.simulate_cascade_loop(
            cfg, cascade_pid, ref_fn=ref_v, mode="velocity", use_feedforward=False
        )
        plot_cascade(t, x, v, u, E, KE, PE, x_ref_hist, v_ref_hist, params, mode="velocity")
        
        # Evaluate velocity controller performance (based on velocity tracking)
        metrics = evaluator.evaluate(
            t=t,
            y=v,  # velocity is the controlled variable
            r=v_ref_hist,  # velocity reference
            u=u, 
            u_min=cascade_pid.inner_pid.u_min, 
            u_max=cascade_pid.inner_pid.u_max
        )
        evaluator.print_metrics(metrics)
        
        # Compute cost using the same weights as Bayesian Optimization
        weights_cfg = {
            'weights': {
                'rise_time': 0.25,
                'settling_time': 0.35,
                'overshoot': 0.20,
                'sse': 0.20
            },
            'normalization_refs': {
                'rise_time': 1.0,
                'settling_time': 5.0,
                'overshoot': 20.0,
                'sse': 0.01
            },
            'use_worst_case': False
        }
        
        J = evaluator.cost_from_metrics(
            [metrics],  # Single scenario, wrap in list
            weights=weights_cfg['weights'],
            normalization_refs=weights_cfg['normalization_refs'],
            use_worst_case=weights_cfg['use_worst_case']
        )
        
        # Plot error analysis for velocity tracking
        error = v_ref_hist - v
        evaluator.plot_error_analysis(t, v, v_ref_hist, error)
        
        print("\n=== DIRECT VELOCITY SETPOINT CONTROL ===")
        print(f"Inner loop gains (velocity): Kp={cascade_pid.inner_pid.Kp}, Ki={cascade_pid.inner_pid.Ki}, Kd={cascade_pid.inner_pid.Kd}")
        print("Position loop: DISABLED (direct velocity control)")
        if cascade_pid.velocity_limit:
            print(f"Velocity limits: {cascade_pid.velocity_limit} m/s")
        print(f"\n**Cost J = {J:.6f}**")
    
    elif run_mode == "comparison":  # compare controllers
        # Run position control
        print("\n=== POSITION CONTROL TEST ===")
        ref_p = step_ref(magnitude=0.1, t_delay=0.5)
        t_p, x_p, v_p, u_p, E_p, KE_p, PE_p, r_p = msd.simulate_closed_loop(
            cfg, pid_p, ref_fn=ref_p, mode="position", use_feedforward=False
        )
        
        # Evaluate position controller
        metrics_p = evaluator.evaluate(
            t=t_p, y=x_p, r=r_p, u=u_p, 
            u_min=pid_p.u_min, u_max=pid_p.u_max
        )
        evaluator.print_metrics(metrics_p)
        
        # Run velocity control
        print("\n=== VELOCITY CONTROL TEST ===")
        ref_v = step_ref(magnitude=0.5, t_delay=2.0)
        t_v, x_v, v_v, u_v, E_v, KE_v, PE_v, r_v = msd.simulate_closed_loop(
            cfg, pid_v, ref_fn=ref_v, mode="velocity"
        )
        
        # Evaluate velocity controller
        metrics_v = evaluator.evaluate(
            t=t_v, y=v_v, r=r_v, u=u_v,
            u_min=pid_v.u_min, u_max=pid_v.u_max
        )
        evaluator.print_metrics(metrics_v)
        
        # Plot results for both controllers
        plot_closed_loop(t_p, x_p, v_p, u_p, E_p, KE_p, PE_p, r_p, mode="position", params=params)
        plot_closed_loop(t_v, x_v, v_v, u_v, E_v, KE_v, PE_v, r_v, mode="velocity", params=params)
        
        # Plot error analysis for both controllers
        evaluator.plot_error_analysis(t_p, x_p, r_p, r_p - x_p)
        evaluator.plot_error_analysis(t_v, v_v, r_v, r_v - v_v)
    
    elif run_mode == "bayesian_optimization":
        # BAYESIAN OPTIMIZATION FOR CASCADE PID TUNING
        
        if not BAYES_OPT_AVAILABLE:
            print("ERROR: bayesian-optimization package not installed!")
            print("Install with: pip install bayesian-optimization")
            exit(1)
        
        print("\n" + "="*80)
        print("BAYESIAN OPTIMIZATION FOR CASCADE PID TUNING")
        print("="*80 + "\n")
        
        # 1. DEFINE TEST SCENARIOS
        print("Setting up test scenarios...")
        
        # Stage 1: Inner loop (velocity) test scenarios
        inner_scenarios = [
            # Velocity step response
            (Scenario.step_velocity(magnitude=0.2, t_start=0.5),
             lambda t: 0.0,
             "velocity"),
            
            # Cruise with step disturbance
            # (Scenario.cruise(velocity=0.0),
            #  Scenario.disturbance_impulse(area=1.0, t_impulse=2.0, width=0.01),
            #  "velocity"),
        ]
        
        # Stage 2: Outer loop (position) test scenarios
        outer_scenarios = [
            # Position step response
            (Scenario.step_position(magnitude=0.1, t_start=0.5),
             lambda t: 0.0,
             "position"),
        ]
        
        # Start with inner loop scenarios
        test_scenarios = inner_scenarios
        
        print(f"  ✓ Created {len(inner_scenarios)} inner loop scenarios")
        print(f"  ✓ Created {len(outer_scenarios)} outer loop scenarios\n")
        
        # 2. CONFIGURE BAYESIAN OPTIMIZATION
        
        bounds_log10 = {
            # Inner loop (velocity PI): tune first
            # VALIDATION TEST: Bounds include known optimal (Kp=600, Ki=1700)
            'inner': {
            'log10_Kp_v': [2.0, 3.0],   # log10(Kp_v) ∈ [2.0, 3.0] => Kp_v ∈ [100, 1000]
            'log10_Ki_v': [3.0, 3.4],   # log10(Ki_v) ∈ [3.0, 3.4] => Ki_v ∈ [1000, 2512]
            },
            # Outer loop (position PI): tune second with fixed inner loop
            # VALIDATION TEST: Bounds include known optimal (Kp=0.1, Ki=20)
            'outer': {
            'log10_Kp_x': [-1.3, -0.5],  # log10(Kp_x) ∈ [-1.3, -0.5] => Kp_x ∈ [0.05, 0.32]
            'log10_Ki_x': [1.0, 1.6],    # log10(Ki_x) ∈ [1.0, 1.6] => Ki_x ∈ [10, 40]
            },
            # Joint optimization: fine-tune all four gains together
            'joint': {
                'log10_Kp_v': [0.5, 3.0],
                'log10_Ki_v': [-1.0, 2.0],
                'log10_Kp_x': [0.0, 2.5],
                'log10_Ki_x': [-1.0, 2.0],
            }
        }
    
        # Cost function weights (prioritize fast settling and accuracy)
        weights_cfg = {
            'weights': {
                'rise_time': 0.25,
                'settling_time': 0.35,  # Prioritize stability
                'overshoot': 0.20,
                'sse': 0.20
            },
            'normalization_refs': {
                'rise_time': 1.0,
                'settling_time': 5.0,
                'overshoot': 20.0,
                'sse': 0.01
            },
            'use_worst_case': False  # Use mean aggregation
        }
        
        # Safety constraints
        safety_cfg = {
            'max_saturation': 90.0,   # Allow up to 90% saturation
            'max_energy': 1e5,        # Energy divergence threshold
            'max_position': 20.0      # Position envelope [m]
        }
        
        print("Configuration:")
        print(f"  Weights: {weights_cfg['weights']}")
        print(f"  Safety: max_sat={safety_cfg['max_saturation']}%, max_E={safety_cfg['max_energy']:.1e}, max_x={safety_cfg['max_position']}m")
        print()
        
        # 3. CREATE BayesianOptimizer
        
        # initial gains - will be overwritten by BO
        opt_cascade_pid = CascadePID(
            outer_Kp=10.0, outer_Ki=1.0, outer_Kd=0.0,
            inner_Kp=10.0, inner_Ki=1.0, inner_Kd=0.0,
            u_min=-50.0, u_max=50.0,
            velocity_limit=(-2.0, 2.0)
        )
        
        # Simulation config for optimization (long time to ensure even slow controllers can settle)
        opt_sim_cfg = SimConfig(t0=0.0, tf=10.0, dt=0.05, x0=0.0, v0=0.0)
        
        # Create BayesianOptimizer
        optimizer = BayesianOptimizer(
            plant=msd,
            controller=opt_cascade_pid,
            evaluator=evaluator,
            scenarios=test_scenarios,
            bounds_log10=bounds_log10,
            weights_cfg=weights_cfg,
            safety_cfg=safety_cfg,
            sim_cfg=opt_sim_cfg,
            rng=np.random.RandomState(42)  # Reproducibility
        )
        
        print("BayesianOptimizer created successfully\n")
        
        # 4. STAGE 1: OPTIMIZE INNER LOOP (VELOCITY)
        #
        # Acquisition Strategy:
        # - Using Expected Improvement (EI) with xi=0.01
        # - Manual optimization loop using suggest() and probe()
        # - EI provides faster convergence and focuses on exploitation

        print("="*80)
        print("STAGE 1: INNER LOOP OPTIMIZATION (Velocity PI Gains)")
        print("="*80 + "\n")
        
        optimizer.set_stage('inner')
        
        # Manual seed points (good starting guesses in log10 space)
        # VALIDATION TEST: Seeds distributed around the space (NOT at optimal 600/1700)
        inner_seeds = [
            {'log10_Kp_v': np.log10(200), 'log10_Ki_v': np.log10(1000)},   # Lower-left region
            {'log10_Kp_v': np.log10(800), 'log10_Ki_v': np.log10(2500)},   # Upper-right region
            # {'log10_Kp_v': np.log10(450), 'log10_Ki_v': np.log10(1800)},   # Middle region
            # {'log10_Kp_v': np.log10(700), 'log10_Ki_v': np.log10(1200)},   # Asymmetric 1
        ]
        
        # Create Bayesian Optimization object for inner loop
        # Note: We maximize -J (negative cost) since BO maximizes
        inner_optimizer = BayesianOptimization(
            f=optimizer,  # Callable that returns -J
            pbounds=bounds_log10['inner'],
            random_state=42,
            allow_duplicate_points=False
        )
        
        # Configure Gaussian Process (Matérn kernel with ν=2.5)
        from sklearn.gaussian_process.kernels import Matern
        inner_optimizer.set_gp_params(
            kernel=Matern(nu=2.5, length_scale=[1.0, 1.0], length_scale_bounds=[(0.1, 10.0), (0.1, 10.0)]),
            alpha=1e-6,  # VERY low noise = GP is VERY confident = follows observations EXACTLY
                        # Higher alpha (e.g., 1e-5 or 1e-4) = less confident GP = maintains higher EI longer
            n_restarts_optimizer=25  # More restarts = better GP hyperparameter optimization
        )
        
        # Add manual seed points
        print("Probing manual seed points...")
        for i, seed in enumerate(inner_seeds):
            try:
                inner_optimizer.probe(params=seed, lazy=False)
                print(f"  Seed {i+1}/{len(inner_seeds)} evaluated")
            except Exception as e:
                print(f"  Seed {i+1} failed: {e}")
        
        print()
        
        # Run Bayesian Optimization with Expected Improvement
        print("Running Bayesian Optimization...")
        print(f"  Initial random points: 2 (reduced for better EI visualization)")
        print(f"  Optimization iterations: 50")
        print(f"  Acquisition function: Expected Improvement (EI, xi=0.1)")
        print()
        
        # Create Expected Improvement acquisition function
        # Per docs: xi=0.0 is pure exploitation, xi=0.1 is high exploration
        # 
        # IMPORTANT: xi controls exploration vs exploitation tradeoff
        # - xi=0.01 (current): FAST convergence, minimal exploration
        #   Good for: tight bounds, quick optimization, limited budget
        # - xi=0.05: More exploration, slower convergence
        # - xi=0.1: High exploration, much slower convergence
        #
        # NOTE: With tight bounds ([100,1000], [1000,2512]) and 57 evaluations,
        # EI converges to ~0 around iteration 15-20 because:
        # 1. Small search space (narrow bounds)
        # 2. Low xi = exploitation-heavy
        # 3. Many evaluations relative to space size
        # 4. Low GP noise (alpha=1e-6) = high confidence
        #
        # To maintain higher EI for longer:
        # - Increase xi to 0.05 or 0.1 (more exploration)
        # - Widen bounds (larger search space)
        # - Reduce iterations (stop earlier)
        # - Increase alpha to 1e-5 or 1e-4 (less confident GP)
        ei_acquisition = ExpectedImprovement(xi=0.1)
        
        # Manual optimization loop with EI acquisition
        # Random initialization for diverse starting points
        # REDUCED from 5 to 2 to better visualize EI convergence
        for _ in range(2):
            next_point_dict = inner_optimizer._space.random_sample()
            inner_optimizer.probe(next_point_dict, lazy=False)
        
        # Then do BO iterations with EI
        for i in range(50):
            # Get next point using EI acquisition
            next_point_array = ei_acquisition.suggest(
                gp=inner_optimizer._gp,
                target_space=inner_optimizer._space,
                n_random=5000,  # More samples for better acquisition optimization
                n_smart=10      # More L-BFGS-B refinement starts
            )
            # Convert array to dict using pbounds keys to preserve order
            param_names = list(bounds_log10['inner'].keys())
            next_point_dict = {param_names[j]: next_point_array[j] for j in range(len(param_names))}
            inner_optimizer.probe(next_point_dict, lazy=False)
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i + 1}/50 complete")
        
        print()
        
        # Extract best inner loop gains
        best_inner = inner_optimizer.max
        best_inner_gains = {
            'Kp_v': 10 ** best_inner['params']['log10_Kp_v'],
            'Ki_v': 10 ** best_inner['params']['log10_Ki_v']
        }
        
        print("\n" + "-"*80)
        print(f"STAGE 1 COMPLETE")
        print(f"Best Inner Loop Gains: Kp_v={best_inner_gains['Kp_v']:.2f}, Ki_v={best_inner_gains['Ki_v']:.2f}")
        print(f"Best Cost J: {-best_inner['target']:.6f}")
        print("-"*80 + "\n")
        
        # Fix inner loop gains for next stage
        opt_cascade_pid.set_inner_gains(best_inner_gains['Kp_v'], best_inner_gains['Ki_v'])
        
        # 5. STAGE 2: OPTIMIZE OUTER LOOP (POSITION)

        print("="*80)
        print("STAGE 2: OUTER LOOP OPTIMIZATION (Position PI Gains)")
        print("="*80 + "\n")
        
        # Switch to outer loop scenarios
        optimizer.scenarios = outer_scenarios
        optimizer.set_stage('outer')
        
        # Manual seed points for outer loop
        # VALIDATION TEST: Seeds distributed around the space (NOT at optimal 0.1/20)
        outer_seeds = [
            {'log10_Kp_x': np.log10(0.05), 'log10_Ki_x': np.log10(10)},   # Lower region
            {'log10_Kp_x': np.log10(0.3), 'log10_Ki_x': np.log10(40)},    # Upper region  
            {'log10_Kp_x': np.log10(0.15), 'log10_Ki_x': np.log10(25)},   # Middle region
            {'log10_Kp_x': np.log10(0.2), 'log10_Ki_x': np.log10(15)},    # Asymmetric
        ]
        
        # Create optimizer for outer loop
        outer_optimizer = BayesianOptimization(
            f=optimizer,
            pbounds=bounds_log10['outer'],
            random_state=43,
            allow_duplicate_points=False
        )
        
        # Configure GP
        outer_optimizer.set_gp_params(
            kernel=Matern(nu=2.5, length_scale=[1.0, 1.0], length_scale_bounds=[(0.1, 10.0), (0.1, 10.0)]),
            alpha=1e-3,  # Lower noise = GP more confident = follows observations closely
            n_restarts_optimizer=25  # More restarts = better GP hyperparameter optimization
        )
        
        # Probe seed points
        print("Probing manual seed points...")
        for i, seed in enumerate(outer_seeds):
            try:
                outer_optimizer.probe(params=seed, lazy=False)
                print(f"  Seed {i+1}/{len(outer_seeds)} evaluated")
            except Exception as e:
                print(f"  Seed {i+1} failed: {e}")
        
        print()
        
        # Run optimization with Expected Improvement
        print("Running Bayesian Optimization...")
        print(f"  Initial random points: 5")
        print(f"  Optimization iterations: 40")
        print(f"  Acquisition function: Expected Improvement (EI, xi=0.01)")
        print()
        
        # Create Expected Improvement acquisition function
        # Per docs: xi=0.01 provides minimal exploration with gradual convergence
        ei_acquisition_outer = ExpectedImprovement(xi=0.1)
        
        # Manual optimization loop with EI acquisition
        # Random initialization for diverse starting points
        # REDUCED from 5 to 2 to better visualize EI convergence
        for _ in range(2):
            next_point_dict = outer_optimizer._space.random_sample()
            outer_optimizer.probe(next_point_dict, lazy=False)
        
        # Then do BO iterations with EI
        for i in range(40):
            # Get next point using EI acquisition
            next_point_array = ei_acquisition_outer.suggest(
                gp=outer_optimizer._gp,
                target_space=outer_optimizer._space,
                n_random=5000,  # More samples for better optimization
                n_smart=10      # More L-BFGS-B starts
            )
            # Convert array to dict using pbounds keys to preserve order
            param_names = list(bounds_log10['outer'].keys())
            next_point_dict = {param_names[j]: next_point_array[j] for j in range(len(param_names))}
            outer_optimizer.probe(next_point_dict, lazy=False)
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i + 1}/40 complete")
        
        print()
        
        # Extract best outer loop gains
        best_outer = outer_optimizer.max
        best_outer_gains = {
            'Kp_x': 10 ** best_outer['params']['log10_Kp_x'],
            'Ki_x': 10 ** best_outer['params']['log10_Ki_x']
        }
        
        print("\n" + "-"*80)
        print(f"STAGE 2 COMPLETE")
        print(f"Best Outer Loop Gains: Kp_x={best_outer_gains['Kp_x']:.2f}, Ki_x={best_outer_gains['Ki_x']:.2f}")
        print(f"Best Cost J: {-best_outer['target']:.6f}")
        print("-"*80 + "\n")
        
        # Fix outer loop gains for next stage
        opt_cascade_pid.set_outer_gains(best_outer_gains['Kp_x'], best_outer_gains['Ki_x'])
        
        # # 6. STAGE 3: JOINT OPTIMIZATION (FINE-TUNING)
        # # COMMENTED OUT - Using sequential optimization only
        #
        # print("="*80)
        # print("STAGE 3: JOINT OPTIMIZATION (All Four PI Gains)")
        # print("="*80 + "\n")
        #
        # optimizer.set_stage('joint')
        #
        # # Seed with best from previous stages
        # joint_seeds = [
        #     {
        #         'log10_Kp_v': best_inner['params']['log10_Kp_v'],
        #         'log10_Ki_v': best_inner['params']['log10_Ki_v'],
        #         'log10_Kp_x': best_outer['params']['log10_Kp_x'],
        #         'log10_Ki_x': best_outer['params']['log10_Ki_x']
        #     },
        # ]
        #
        # # Create optimizer for joint optimization
        # joint_optimizer = BayesianOptimization(
        #     f=optimizer,
        #     pbounds=bounds_log10['joint'],
        #     random_state=44,
        #     allow_duplicate_points=False
        # )
        #
        # # Configure GP (4D now)
        # joint_optimizer.set_gp_params(
        #     kernel=Matern(nu=2.5, length_scale=[1.0]*4, length_scale_bounds=[(0.1, 10.0)]*4),
        #     alpha=1e-4,
        #     n_restarts_optimizer=5
        # )
        #
        # # Probe seed point
        # print("Probing best from previous stages...")
        # for i, seed in enumerate(joint_seeds):
        #     try:
        #         joint_optimizer.probe(params=seed, lazy=False)
        #         print(f"  Seed {i+1}/{len(joint_seeds)} evaluated")
        #     except Exception as e:
        #         print(f"  Seed {i+1} failed: {e}")
        #
        # print()
        #
        # # Run joint optimization (fewer iterations since 4D is more expensive)
        # print("Running Joint Bayesian Optimization...")
        # print(f"  Initial random points: 8")
        # print(f"  Optimization iterations: 30")
        # print()
        #
        # joint_optimizer.maximize(
        #     init_points=8,
        #     n_iter=30
        # )
        #
        # # Extract best joint gains
        # best_joint = joint_optimizer.max
        # best_joint_gains = {
        #     'Kp_v': 10 ** best_joint['params']['log10_Kp_v'],
        #     'Ki_v': 10 ** best_joint['params']['log10_Ki_v'],
        #     'Kp_x': 10 ** best_joint['params']['log10_Kp_x'],
        #     'Ki_x': 10 ** best_joint['params']['log10_Ki_x']
        # }
        #
        # print("\n" + "-"*80)
        # print(f"STAGE 3 COMPLETE")
        # print(f"Best Joint Gains:")
        # print(f"  Inner: Kp_v={best_joint_gains['Kp_v']:.2f}, Ki_v={best_joint_gains['Ki_v']:.2f}")
        # print(f"  Outer: Kp_x={best_joint_gains['Kp_x']:.2f}, Ki_x={best_joint_gains['Ki_x']:.2f}")
        # print(f"Best Cost J: {-best_joint['target']:.6f}")
        # print("-"*80 + "\n")
        
        # Use best gains from sequential optimization
        best_final_gains = {
            'Kp_v': best_inner_gains['Kp_v'],
            'Ki_v': best_inner_gains['Ki_v'],
            'Kp_x': best_outer_gains['Kp_x'],
            'Ki_x': best_outer_gains['Ki_x']
        }
        
        # 7. SUMMARY AND CONVERGENCE PLOTS
    
        print("="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80 + "\n")
        
        optimizer.print_best()
        
        # Visualize GP and Acquisition Function
        print("\nGenerating GP and Utility Function visualizations...")
        plot_gp_and_acquisition_2d(inner_optimizer, stage_name='inner', 
                                    param_names=['log10_Kp_v', 'log10_Ki_v'],
                                    acquisition_func=ei_acquisition)
        plot_gp_and_acquisition_2d(outer_optimizer, stage_name='outer',
                                    param_names=['log10_Kp_x', 'log10_Ki_x'],
                                    acquisition_func=ei_acquisition_outer)
        
        # Generate BO progression plots (2D contours showing evolution)
        print("\nGenerating BO progression visualizations (2D contours)...")
        plot_bo_progression_2d(inner_optimizer, 'inner', 
                              param_names=['log10_Kp_v', 'log10_Ki_v'],
                              iterations_to_show=[2, 4, 7, 15, 30, 50], 
                              acquisition_func=ei_acquisition)
        plot_bo_progression_2d(outer_optimizer, 'outer',
                              param_names=['log10_Kp_x', 'log10_Ki_x'],
                              iterations_to_show=[2, 4, 7, 15, 30, 40], 
                              acquisition_func=ei_acquisition_outer)
        
        # Generate 1D progression plots (slices showing EI convergence clearly)
        print("\nGenerating 1D progression visualizations (showing EI convergence)...")
        plot_bo_progression_1d(inner_optimizer, 'inner', 'log10_Kp_v', 
                              [2, 4, 7, 15, 30, 50], ei_acquisition, best_inner_gains)
        plot_bo_progression_1d(inner_optimizer, 'inner', 'log10_Ki_v', 
                              [2, 4, 7, 15, 30, 50], ei_acquisition, best_inner_gains)
        plot_bo_progression_1d(outer_optimizer, 'outer', 'log10_Kp_x', 
                              [2, 4, 7, 15, 30, 40], ei_acquisition_outer, best_outer_gains)
        plot_bo_progression_1d(outer_optimizer, 'outer', 'log10_Ki_x', 
                              [2, 4, 7, 15, 30, 40], ei_acquisition_outer, best_outer_gains)
        
        # Plot convergence for each stage
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # Inner loop convergence
        inner_targets = [-res['target'] for res in inner_optimizer.res]  # Convert back to J (cost)
        
        # Filter out extreme values for better visualization
        # Keep values below 10x the minimum (or use a fixed threshold)
        min_J_inner = min(inner_targets)
        
        axs[0].plot(inner_targets, 'b-o', markersize=4)
        axs[0].axhline(y=min_J_inner, color='r', linestyle='--', label=f'Best J={min_J_inner:.4f}')
        axs[0].set_ylim([0, 0.7])  # Limit y-axis to filter extreme values
        axs[0].set_ylabel('Cost J')
        axs[0].set_title('Stage 1: Inner Loop (Velocity) Convergence')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
        
        # Outer loop convergence
        outer_targets = [-res['target'] for res in outer_optimizer.res]
        
        # Filter out extreme values for better visualization
        min_J_outer = min(outer_targets)
        
        axs[1].plot(outer_targets, 'g-o', markersize=4)
        axs[1].axhline(y=min_J_outer, color='r', linestyle='--', label=f'Best J={min_J_outer:.4f}')
        axs[1].set_ylim([0, 0.7])  # Limit y-axis
        axs[1].set_ylabel('Cost J')
        axs[1].set_xlabel('Iteration')
        axs[1].set_title('Stage 2: Outer Loop (Position) Convergence')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()
        
        # # Joint optimization convergence (COMMENTED OUT)
        # joint_targets = [-res['target'] for res in joint_optimizer.res]
        # axs[2].plot(joint_targets, 'm-o', markersize=4)
        # axs[2].axhline(y=min(joint_targets), color='r', linestyle='--', label=f'Best J={min(joint_targets):.4f}')
        # axs[2].set_ylabel('Cost J')
        # axs[2].set_xlabel('Iteration')
        # axs[2].set_title('Stage 3: Joint Optimization Convergence')
        # axs[2].grid(True, alpha=0.3)
        # axs[2].legend()
        
        plt.tight_layout()
        plt.savefig('bo_convergence.png', dpi=150, bbox_inches='tight')
        print("Convergence plot saved to: bo_convergence.png\n")
        plt.show()
        
        # 8. FINAL VALIDATION
        
        # Apply best gains to a fresh controller
        final_cascade_pid = CascadePID(
            outer_Kp=best_final_gains['Kp_x'],
            outer_Ki=best_final_gains['Ki_x'],
            outer_Kd=0.0,
            inner_Kp=best_final_gains['Kp_v'],
            inner_Ki=best_final_gains['Ki_v'],
            inner_Kd=0.0,
            u_min=-50.0, u_max=50.0,
            velocity_limit=(-2.0, 2.0)
        )
        
        # Run a longer validation simulation
        val_cfg = SimConfig(t0=0.0, tf=15.0, dt=0.05, x0=0.0, v0=0.0)
        ref_val = Scenario.step_position(magnitude=0.15, t_start=1.0)  # Simple step for clear metrics
        
        t, x, v, u, E, KE, PE, x_ref_hist, v_ref_hist = msd.simulate_cascade_loop(
            cfg=val_cfg,
            cascade_controller=final_cascade_pid,
            ref_fn=ref_val,
            mode="position",
            use_feedforward=False
        )
        
        # Evaluate performance
        val_metrics = evaluator.evaluate(
            t=t, y=x, r=x_ref_hist, u=u,
            u_min=final_cascade_pid.inner_pid.u_min,
            u_max=final_cascade_pid.inner_pid.u_max
        )
        
        evaluator.print_metrics(val_metrics)
        
        # Plot validation results
        plot_cascade(t, x, v, u, E, KE, PE, x_ref_hist, v_ref_hist, params, mode="position")
