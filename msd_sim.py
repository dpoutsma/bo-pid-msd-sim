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

@dataclass
class MSDParams:
    mass: float = 1.0     # mass [kg]
    damping: float = 0.5     # damping [N·s/m]
    stiffness: float = 20.0    # stiffness [N/m]

@dataclass
class SimConfig:
    t0: float = 0.0
    tf: float = 10.0
    dt: float = 0.001
    x0: float = 0.0     # initial displacement [m]
    v0: float = 0.0     # initial velocity [m/s]

def rk4_step(f, t, y, h):
    """
    4th-order Runge–Kutta step for y' = f(t, y).
    y is a 1D numpy array.
    """
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
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
        
        # Only for step references
        if np.all(r == r[0]) or np.all(r[r>0] == r[np.where(r>0)[0][0]]):
            # Get step characteristics
            r_final = r[-1]  # Final reference value
            r_step = r_final - r[0]  # Step size
            
            if abs(r_step) > 1e-6:  # Only if we have a significant step
                # Rise time calculation (10% to 90%)
                y_norm = (y - y[0]) / r_step
                try:
                    t_10 = t[np.where(y_norm >= 0.1)[0][0]]
                    t_90 = t[np.where(y_norm >= 0.9)[0][0]]
                    metrics['Rise time (10-90%)'] = t_90 - t_10
                except IndexError:
                    metrics['Rise time (10-90%)'] = float('nan')

                # Overshoot
                if r_step > 0:
                    overshoot = (np.max(y) - r_final) / abs(r_final) * 100
                else:
                    overshoot = (r_final - np.min(y)) / abs(r_final) * 100
                metrics['Overshoot (%)'] = max(0, overshoot)

                # Settling time (2% band)
                settled = np.where(np.abs(e) <= eps * abs(r_final))[0]
                if len(settled) > 0:
                    # Check if it's settled for all future points
                    for idx in settled:
                        if np.all(np.abs(e[idx:]) <= eps * abs(r_final)):
                            metrics['Settling time (2%)'] = t[idx] - t[0]
                            break
                    else:
                        metrics['Settling time (2%)'] = float('nan')
                else:
                    metrics['Settling time (2%)'] = float('nan')
        
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
        print(f"Rise time (10-90%): {metrics.get('Rise time (10-90%)', 'N/A'):.4g} s")
        print(f"Settling time (2%): {metrics.get('Settling time (2%)', 'N/A'):.4g} s")
        print(f"Overshoot:          {metrics.get('Overshoot (%)', 'N/A'):.2f} %")
        print(f"Steady-state error: {metrics.get('Steady-state error', 'N/A'):.6g}")
        
        # Constraint handling
        print("\n--- Constraint Handling ---")
        print(f"Control saturation: {metrics.get('Total saturation (%)', 'N/A'):.2f} %")
        print(f"Max control rate:   {metrics.get('Max control rate', 'N/A'):.4g}")
        
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
    cfg = SimConfig(t0=0.0, tf=10.0, dt=0.001, x0=0.0, v0=0.0)

    # Standard tune guide for PID:
    # Increase Kp until response with minor overshoot
    # Add Ki to remove steady-state error
    # Add Kd to dampen overshoot (set deriv_tau ~ 1–5*dt for smoothing)
    pid_p = PID(Kp=200.0, Ki=60.0, Kd=10.0, u_min=-50.0, u_max=50.0, deriv_tau=0.002)   # position PID
    pid_v = PID(Kp=500.0, Ki=20.0, Kd=2.0, u_min=-50.0, u_max=50.0, deriv_tau=0.005)    # velocity PID
    
    # Cascade PID controller (outer loop: position, inner loop: velocity)
    cascade_pid = CascadePID(
        outer_Kp=50.0, outer_Ki=10.0, outer_Kd=5.0,    # Position loop gains
        inner_Kp=100.0, inner_Ki=20.0, inner_Kd=2.0,   # Velocity loop gains
        u_min=-50.0, u_max=50.0,
        outer_deriv_tau=0.002, inner_deriv_tau=0.005,
        velocity_limit=(-2.0, 2.0)  # Optional velocity saturation
    )

    # Initialize controller evaluator for performance metrics
    evaluator = ControlEvaluator()

    # Choose what to run
    # Set to "open-loop", "position", "velocity", "cascade", "velocity-setpoint", or "comparison"
    run_mode = "velocity-setpoint"  # Test cascade PID controller
    
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
        ref_v = step_ref(magnitude=0.5, t_delay=2.0)  # 0.5 m/s
        t, x, v, u, E, KE, PE, r_hist = msd.simulate_closed_loop(cfg, pid_v, ref_fn=ref_v, mode="velocity")
        plot_closed_loop(t, x, v, u, E, KE, PE, r_hist, mode="velocity", params=params)
        
        # Evaluate velocity controller performance
        metrics = evaluator.evaluate(
            t=t, 
            y=v,  # velocity is measured output
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
        # Cascade PID control example
        ref_p = step_ref(magnitude=0.1, t_delay=0.5)
        t, x, v, u, E, KE, PE, x_ref_hist, v_ref_hist = msd.simulate_cascade_loop(
            cfg, cascade_pid, ref_fn=ref_p, mode="position", use_feedforward=False
        )
        plot_cascade(t, x, v, u, E, KE, PE, x_ref_hist, v_ref_hist, params, mode="position")
        
        # Evaluate cascade controller performance (based on position tracking)
        metrics = evaluator.evaluate(
            t=t,
            y=x,  # position is the primary controlled variable
            r=x_ref_hist,  # position reference
            u=u, 
            u_min=cascade_pid.inner_pid.u_min, 
            u_max=cascade_pid.inner_pid.u_max
        )
        evaluator.print_metrics(metrics)
        
        # Plot error analysis for position tracking
        error = x_ref_hist - x
        evaluator.plot_error_analysis(t, x, x_ref_hist, error)
        
        print("\n=== CASCADE CONTROLLER INFO ===")
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
    
    else:  # "comparison" or any other value - compare controllers
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