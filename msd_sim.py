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

try:
    from bayes_opt import BayesianOptimization
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

class ControlEvaluator:
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
        checking. Constraint validation should be done upstream (e.g., in BOEvaluator).
        
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

class BOEvaluator:
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
            evaluator: ControlEvaluator instance (for computing metrics)
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
        self.sim_cfg = sim_cfg if sim_cfg is not None else SimConfig(tf=10.0, dt=0.001)
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

if __name__ == "__main__":
    # System parameters
    params = MSDParams(mass=20.0, damping=20.0, stiffness=0.0)

    # Simulation config for RK4
    cfg = SimConfig(t0=0.0, tf=10.0, dt=0.05, x0=0.0, v0=0.0)

    # Standard tune guide for PID:
    # Increase Kp until response with minor overshoot
    # Add Ki to remove steady-state error
    # Add Kd to dampen overshoot (set deriv_tau ~ 1–5*dt for smoothing)
    pid_p = PID(Kp=600.0, Ki=60.0, Kd=10.0, u_min=-50.0, u_max=50.0, deriv_tau=0.002)   # position PID
    pid_v = PID(Kp=600.0, Ki=9400.0, Kd=0.0, u_min=-50.0, u_max=50.0, deriv_tau=0.005)    # velocity PID
    
    # Cascade PID controller (outer loop: position, inner loop: velocity)
    cascade_pid = CascadePID(
        outer_Kp=160.0, outer_Ki=50.0, outer_Kd=0.0,    # Position loop gains
        inner_Kp=700.0, inner_Ki=5000.0, inner_Kd=0.0,   # Velocity loop gains
        u_min=-50.0, u_max=50.0,
        outer_deriv_tau=0.002, inner_deriv_tau=0.005,
        velocity_limit=(-2.0, 2.0)  # Optional velocity saturation
    )

    # Initialize controller evaluator for performance metrics
    evaluator = ControlEvaluator()

    # Choose what to run
    # Set to "open-loop", "position", "velocity", "cascade", "velocity-setpoint", "comparison", or "bayesian_optimization"
    run_mode = "bayesian_optimization"

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
        ref_v = step_ref(magnitude=0.3, t_delay=0.5)  # 0.5 m/s
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
        test_cfg = SimConfig(t0=0.0, tf=15.0, dt=0.001, x0=0.0, v0=0.0)
        
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
        ref_v = step_ref(magnitude=0.3, t_delay=1.0)  # 0.3 m/s step at t=1.0s
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
        
        # Plot error analysis for velocity tracking
        error = v_ref_hist - v
        evaluator.plot_error_analysis(t, v, v_ref_hist, error)
        
        print("\n=== DIRECT VELOCITY SETPOINT CONTROL ===")
        print(f"Inner loop gains (velocity): Kp={cascade_pid.inner_pid.Kp}, Ki={cascade_pid.inner_pid.Ki}, Kd={cascade_pid.inner_pid.Kd}")
        print("Position loop: DISABLED (direct velocity control)")
        if cascade_pid.velocity_limit:
            print(f"Velocity limits: {cascade_pid.velocity_limit} m/s")
    
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
            'inner': {
            'log10_Kp_v': [-1.0, 4.0],   # log10(Kp_v) ∈ [-1.0, 4.0] => Kp_v ∈ [0.1, 10000]
            'log10_Ki_v': [-1.0, 4.0],   # log10(Ki_v) ∈ [-1.0, 4.0] => Ki_v ∈ [0.1, 10000]
            },
            # Outer loop (position PI): tune second with fixed inner loop
            'outer': {
            'log10_Kp_x': [-1.0, 3.0],   # log10(Kp_x) ∈ [-1.0, 3.0] => Kp_x ∈ [0.1, 1000]
            'log10_Ki_x': [-10.0, 2.0],  # log10(Ki_x) ∈ [-10.0, 2.0] => Ki_x ∈ [1e-10, 100]
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
            'max_position': 20.0       # Position envelope [m]
        }
        
        print("Configuration:")
        print(f"  Weights: {weights_cfg['weights']}")
        print(f"  Safety: max_sat={safety_cfg['max_saturation']}%, max_E={safety_cfg['max_energy']:.1e}, max_x={safety_cfg['max_position']}m")
        print()
        
        # 3. CREATE BOEvaluator
        
        # initial gains - will be overwritten by BO
        opt_cascade_pid = CascadePID(
            outer_Kp=10.0, outer_Ki=1.0, outer_Kd=0.0,
            inner_Kp=10.0, inner_Ki=1.0, inner_Kd=0.0,
            u_min=-50.0, u_max=50.0,
            velocity_limit=(-2.0, 2.0)
        )
        
        # Simulation config for optimization (long time to ensure even slow controllers can settle)
        opt_sim_cfg = SimConfig(t0=0.0, tf=10.0, dt=0.05, x0=0.0, v0=0.0)
        
        # Create BOEvaluator
        bo_evaluator = BOEvaluator(
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
        
        print("BOEvaluator created successfully\n")
        
        # 4. STAGE 1: OPTIMIZE INNER LOOP (VELOCITY)

        print("="*80)
        print("STAGE 1: INNER LOOP OPTIMIZATION (Velocity PI Gains)")
        print("="*80 + "\n")
        
        bo_evaluator.set_stage('inner')
        
        # Manual seed points (good starting guesses in log10 space)
        # Centered around Kp=100.0, Ki=50.0
        inner_seeds = [
            {'log10_Kp_v': 3.0, 'log10_Ki_v': 4.0},   # Kp=1000, Ki=10000 (baseline)
            {'log10_Kp_v': 2.8, 'log10_Ki_v': 3.7},   # Kp=630, Ki=5000 (lower)
            {'log10_Kp_v': 3.2, 'log10_Ki_v': 4.3},   # Kp=1600, Ki=20000 (higher)
        ]
        
        # Create Bayesian Optimization object for inner loop
        # Note: We maximize -J (negative cost) since BO maximizes
        inner_optimizer = BayesianOptimization(
            f=bo_evaluator,  # Callable that returns -J
            pbounds=bounds_log10['inner'],
            random_state=42,
            allow_duplicate_points=False
        )
        
        # Configure Gaussian Process (Matérn kernel with ν=2.5)
        from sklearn.gaussian_process.kernels import Matern
        inner_optimizer.set_gp_params(
            kernel=Matern(nu=2.5, length_scale=[1.0, 1.0], length_scale_bounds=[(0.1, 10.0), (0.1, 10.0)]),
            alpha=1e-4,  # Noise level (small for deterministic sims)
            n_restarts_optimizer=5
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
        
        # Run Bayesian Optimization
        print("Running Bayesian Optimization...")
        print(f"  Initial random points: 5")
        print(f"  Optimization iterations: 25")
        print()
        
        inner_optimizer.maximize(
            init_points=5,      # Additional random initialization
            n_iter=25           # Number of BO iterations
        )
        
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
        bo_evaluator.scenarios = outer_scenarios
        bo_evaluator.set_stage('outer')
        
        # Manual seed points for outer loop
        # Centered around Kp=50.0, Ki=10.0
        outer_seeds = [
            {'log10_Kp_x': 2.0, 'log10_Ki_x': 2.0},   # Kp=100, Ki=100 (baseline)
            {'log10_Kp_x': 1.8, 'log10_Ki_x': 1.8},   # Kp=63, Ki=63 (lower)
            {'log10_Kp_x': 2.2, 'log10_Ki_x': 2.2},   # Kp=158, Ki=158 (higher)
        ]
        
        # Create optimizer for outer loop
        outer_optimizer = BayesianOptimization(
            f=bo_evaluator,
            pbounds=bounds_log10['outer'],
            random_state=43,
            allow_duplicate_points=False
        )
        
        # Configure GP
        outer_optimizer.set_gp_params(
            kernel=Matern(nu=2.5, length_scale=[1.0, 1.0], length_scale_bounds=[(0.1, 10.0), (0.1, 10.0)]),
            alpha=1e-4,
            n_restarts_optimizer=5
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
        
        # Run optimization
        print("Running Bayesian Optimization...")
        print(f"  Initial random points: 5")
        print(f"  Optimization iterations: 25")
        print()
        
        outer_optimizer.maximize(
            init_points=5,
            n_iter=25
        )
        
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
        # bo_evaluator.set_stage('joint')
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
        #     f=bo_evaluator,
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
        
        bo_evaluator.print_best()
        
        # Plot convergence for each stage
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # Inner loop convergence
        inner_targets = [-res['target'] for res in inner_optimizer.res]  # Convert back to J (cost)
        
        # Filter out extreme values for better visualization
        # Keep values below 10x the minimum (or use a fixed threshold)
        min_J_inner = min(inner_targets)
        
        axs[0].plot(inner_targets, 'b-o', markersize=4)
        axs[0].axhline(y=min_J_inner, color='r', linestyle='--', label=f'Best J={min_J_inner:.4f}')
        axs[0].set_ylim([0, 5])  # Limit y-axis to filter extreme values
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
        axs[1].set_ylim([0, 5])  # Limit y-axis
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
        val_cfg = SimConfig(t0=0.0, tf=15.0, dt=0.001, x0=0.0, v0=0.0)
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
