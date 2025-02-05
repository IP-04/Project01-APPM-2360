import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from utils import plot_vector_field, plot_phase_plane, find_nullclines

def task_1_classification():
    """
    Task B.1: Classify the Lotka-Volterra system (4).
    Provide the order, linearity, and autonomy of the system.
    """
    # Classification:
    # - Order: First-order system of differential equations (dx1/dt, dx2/dt)
    # - Linear/Nonlinear: Nonlinear due to cross terms βx1x2 and -δx1x2
    # - Autonomous: Yes, because it only depends on x1 and x2, not explicitly on t

    print("Task B.1 completed: First-order, nonlinear, autonomous system.")

def task_2_nullclines_and_equilibrium():
    """
    Task B.2: Analytically find the v and h nullclines and all equilibrium solutions of (4).
    No specific parameter values should be used here.
    """
    # **Nullclines are found by setting dx1/dt = 0 and dx2/dt = 0:**
    # v-nullclines: Set dx1/dt = -αx1 + βx1x2 = 0
    # h-nullclines: Set dx2/dt = γx2 - δx1x2 = 0
    v_nullcline, h_nullcline = find_nullclines()
    print("Task B.2 completed: Nullclines identified and equilibrium solutions derived.")

def task_3_vector_field_and_phase_plane():
    """
    Task B.3: Assign parameter values and compute the vector field and phase plane.
    - Parameters: α = 1.5, β = 1.1, γ = 2.5, δ = 1.4
    - Vector field region: 0 ≤ x1 ≤ 5, 0 ≤ x2 ≤ 5
    """
    # Parameters
    params = {
        'alpha': 1.5,
        'beta': 1.1,
        'gamma': 2.5,
        'delta': 1.4
    }

    # **Compute and plot the vector field using utils function**
    plot_vector_field(params, x_range=(0, 5), y_range=(0, 5))

    # **Generate the phase plane plot (including nullclines and equilibrium points)**
    plot_phase_plane(params)

def task_3b_solve_ode_system():
    """
    Task B.3(b): Use a numerical ODE solver to simulate the Lotka-Volterra system.
    - Initial condition: x1(0) = 0.5, x2(0) = 1.0
    - Time interval: t ∈ [0, 20], step size h = 0.01
    """
    # Define the Lotka-Volterra system as a function
    def lotka_volterra(t, z, alpha, beta, gamma, delta):
        x1, x2 = z
        dx1_dt = -alpha * x1 + beta * x1 * x2
        dx2_dt = gamma * x2 - delta * x1 * x2
        return [dx1_dt, dx2_dt]

    # Initial conditions and time span
    t_span = (0, 20)
    t_eval = np.linspace(0, 20, 2000)  # Stepsize of approximately 0.01
    z0 = [0.5, 1.0]  # Initial conditions

    # Solve the system using solve_ivp
    sol = solve_ivp(lotka_volterra, t_span, z0, t_eval=t_eval, args=(1.5, 1.1, 2.5, 1.4))

    # **Plot the trajectory in the phase plane**
    plt.plot(sol.y[0], sol.y[1], label="Trajectory: (x1(0), x2(0)) = (0.5, 1.0)")
    plt.xlabel("Predator population (x1)")
    plt.ylabel("Prey population (x2)")
    plt.title("Phase Plane Trajectory")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/task_b_phase_plane_trajectory.png")
    plt.show()

def task_4_component_curves():
    """
    Task B.4: Plot the component curves x1(t) and x2(t) against t.
    Discuss whether they are in phase or out of phase and what this means.
    """
    # **Plot x1(t) and x2(t) from the ODE solver solution**
    t, x1, x2 = sol.t, sol.y[0], sol.y[1]

    plt.figure(figsize=(8, 6))
    plt.plot(t, x1, label="Predator population x1(t)", color='r')
    plt.plot(t, x2, label="Prey population x2(t)", color='b')
    plt.xlabel("Time (t)")
    plt.ylabel("Population size")
    plt.title("Component Curves: Predator and Prey Populations Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/task_b_component_curves.png")
    plt.show()

    # **Discuss phase relationship**
    print("Task B.4 discussion:")
    print("- If the curves are in phase, peaks in predator and prey populations occur at the same time.")
    print("- If they are out of phase, predator peaks follow prey peaks, which is typically the case in predator-prey systems.")

def main():
    # Execute each task sequentially
    task_1_classification()
    task_2_nullclines_and_equilibrium()
    task_3_vector_field_and_phase_plane()
    task_3b_solve_ode_system()
    task_4_component_curves()

if __name__ == "__main__":
    main()
