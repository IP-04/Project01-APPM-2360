import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from utils import plot_vector_field, vectorfield, dx1_dt, dx2_dt, plot_phase_plane2

def task_1_nullclines_and_equilibrium():
    """
    Task C.1: Analytically find the v and h nullclines and all equilibrium solutions of (5).
    DO NOT use specific parameter values for this task.
    """
    # **v-nullclines: Set dx1/dt = -αx1 + βx1x2 = 0**
    # Solve analytically for x2 in terms of x1 (symbolic algebra can be used if needed).
    # **h-nullclines: Set dx2/dt = γ(1 - κx2)x2 - δx1x2 = 0**
    # Solve analytically for x2 in terms of x1.

    print("Task C.1 completed: v and h nullclines analytically derived and equilibrium solutions found.")

def task_2a_vector_field():
    """
    Task C.2(a): Compute the vector field of the logistic predator-prey system (5).
    Parameters: α = 1.5, β = 1.1, γ = 2.5, δ = 1.4, κ = 0.5
    Region: 0 ≤ x1 ≤ 5, 0 ≤ x2 ≤ 5
    """
    params = (1.5, 1.1, 2.5, 1.4, 0.5)
    x1 = np.arange(0, 5, 0.17)
    x2 = np.arange(0, 5, 0.17)
    plt.figure(figsize=(8, 6))
    vectorfield(dx1_dt, dx2_dt, x1, x2, params)
    plt.xlabel("Predator population (x1)")
    plt.ylabel("Prey population (x2)")
    plt.title("Vector Field of Logistic Predator-Prey System")
    plt.savefig("plots/task_c_vector_field.png")
    print("Task C.2(a) completed: Vector field plotted.")
    return vectorfield(dx1_dt, dx2_dt, x1, x2, params)
    

def task_2b_solve_ode_system():
    """
    Task C.2(b): Solve the system numerically using solve_ivp.
    Initial conditions: (x1(0), x2(0)) = (5, 1) and (x1(0), x2(0)) = (1, 5)
    Time interval: t ∈ [0, 20] with a stepsize of h = 0.01
    """
    def logistic_predator_prey(t, f, alpha, beta, gamma, delta, kappa):
        x1, x2 = f
        return [dx1_dt(x1, x2, alpha, beta, gamma, delta, kappa ), dx2_dt(x1, x2, alpha, beta, gamma, delta, kappa)]

    # Initial conditions
    initial_conditions = [[5, 1], [1, 5]]
    t_span = (0, 20)
    t_eval = np.linspace(0, 20, 2000)
    params = (1.5, 1.1, 2.5, 1.4, 0.5)
    
    solutions = []
    for i in range(len(initial_conditions)):
        sol = solve_ivp(logistic_predator_prey, t_span, initial_conditions[i], t_eval=t_eval, args=params)
        solutions.append(sol)
    return solutions

def task_2c_phase_plane_and_trajectories(solutions):
    """
    Task C.2(c): Plot the following in the phase plane:
    i. Vector field
    ii. v-nullclines and h-nullclines
    iii. Equilibrium solutions
    iv. Solution trajectories
    """
    # **Plot phase plane setup using utils function (nullclines, vector field, equilibrium points)**
    params = (1.5, 1.1, 2.5, 1.4, 0.5)
    alpha, beta, gamma, delta, kappa = params
    x1 = np.arange(-0.01, 5, 0.2)
    vnulls = [alpha/beta, 0]
    hnulls = [(1/kappa)*(1-(x1*(delta/gamma))), 0]
    eq_sols = [[0,0,0], [0,2,0], [(gamma/delta)*(1-(kappa*alpha)/beta), vnulls[0], 1]]

    plt.figure(figsize=(8, 6))
    plot_phase_plane2(vnulls, hnulls, eq_sols, x1, task_2a_vector_field)
    solutions = task_2b_solve_ode_system()
    labels = ["[5, 1]", "[1, 5]"]
    colors = ["purple", "orange"]
    for i in range(len(solutions)):
        x1 = solutions[i].y[0]
        x2 = solutions[i].y[1]
        plt.plot(x1, x2, label=labels[i], color=colors[i])
    plt.legend(loc="upper right")
    plt.savefig("plots/task_c_phase_plane_and_traj.png")
    plt.show()
    print("Task C.2(c) completed: Phase plane with trajectories and equilibrium points plotted.")

def task_3_component_curves(solutions):
    """
    Task C.3: Plot component curves x1(t) and x2(t) together against t.
    Discuss whether the solutions are periodic or show asymptotic behavior.
    """
    plt.figure(figsize=(10, 6))
    solutions = task_2b_solve_ode_system()

    # Plot component curves for each initial condition
    label1 = ["x1, IC [5,1]", "x1, IC [1,5]"]
    label2 = ["x2, IC [5,1]", "x2, IC [1,5]"]
    for i in range(len(solutions)):
        t = solutions[i].t
        x1 = solutions[i].y[0]
        x2 = solutions[i].y[1]
        plt.plot(t, x1, label=label1[i])
        plt.plot(t, x2, label=label2[i])

    plt.xlabel("Time (t)")
    plt.ylabel("Population size")
    plt.title("Component Curves: Predator and Prey Populations Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/task_c_component_curves.png")
    plt.show()

    # **Discuss whether the system is periodic or asymptotic**
    print("Task C.3 discussion:")
    print("- If periodic, the predator and prey populations will oscillate indefinitely.")
    print("- If asymptotic, the populations will stabilize or diverge.")

def main():
    # Execute each task sequentially
    #task_1_nullclines_and_equilibrium()
    task_2a_vector_field()
    solutions = task_2b_solve_ode_system()
    task_2c_phase_plane_and_trajectories(solutions)
    task_3_component_curves(solutions)

if __name__ == "__main__":
    main()