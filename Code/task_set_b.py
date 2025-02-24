import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from utils import vectorfield1, LV_dx1, LV_dx2, LV_system

params = (1.5, 1.1, 2.5, 1.4)
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

def task_2_nullclines_and_equilibrium(params):
    #hnulls: x2 = 0 and x1 = gamma/delta
    #vnulls: x1 = 0 and x2 = alpha/beta
    a, b, g, d = params
    #h-nulls:
    plt.axhline(y=0, color="blue", )
    plt.axvline(x=g/d, color="blue", label="h-nullcline")
    #v-nulls:
    plt.axhline(y=a/b, color="red")
    plt.axvline(x=0, color="red", label="v-nullcline")
    plt.xlim(-0.01, 5)
    plt.ylim(-0.01, 5)

# Plotting functions for vector field and phase plane
def plot_vector_field2():
    x1 = np.arange(0, 5, 0.2)
    x2 = np.arange(0, 5, 0.2)
    params = (1.5, 1.1, 2.5, 1.4)
    return vectorfield1(LV_dx1, LV_dx2, x1, x2, params)

def task_3b_solve_ode_system():
    t_span = [0, 20]
    t_eval = np.arange(0, 20, 0.01)
    f0 = [0.5, 1]
    sol = solve_ivp(LV_system, t_span, f0, t_eval=t_eval, args=params)
    t = sol.t
    x1_sol = sol.y[0]
    x2_sol = sol.y[1]
    return (x1_sol, x2_sol, t)

def plot_phase_plane(params):
    plt.figure(figsize=(8, 6))
    a, b, g, d = params
    task_2_nullclines_and_equilibrium(params)
    plot_vector_field2()
    x1, x2, t = task_3b_solve_ode_system()
    eq_sols = [[g/d, a/b], [0, 0]]
    plt.plot(x1, x2, label="(x1(t), x2(t)) Trajectory", color="purple")
    for i in range(len(eq_sols)):
        x_points = eq_sols[i][0]
        y_points = eq_sols[i][1]
        if i == 0:
            plt.plot(x_points, y_points, marker="o", markerfacecolor='none', markeredgecolor='#39FF14', markeredgewidth=2, label="Unstable Eq. Pt.")
        plt.plot(x_points, y_points, marker="o", markerfacecolor='none', markeredgecolor='#39FF14', markeredgewidth=2)
    plt.title("Lotka-Volterra System With I.C. (0.5, 1.0)")
    plt.xlabel("x1(t) (Dozens of Predators)")
    plt.ylabel("x2(t) (Dozens of Prey)")
    plt.legend(loc="upper right", fontsize="7.5")
    plt.grid()
    plt.savefig("plots/task_b_phase_plane_trajectory.png")
    plt.show()

def task_4_component_curves():
    plt.figure(figsize=(8, 6))
    x1, x2, t = task_3b_solve_ode_system()
    plt.plot(t, x1, label="x1(t)", color="purple")
    plt.plot(t, x2, label="x2(t)", color="orange")
    plt.title("Lotka-Volterra Component Curves")
    plt.xlabel("Time (yrs)")
    plt.ylabel("Population (Dozens of Animals)")
    plt.legend(loc="upper right", fontsize="7.5")
    plt.ylim(0, 5.3)
    plt.grid()
    plt.savefig("plots/task_b_component_curves.png")
    plt.show()

def main():
    # Execute each task sequentially
    task_1_classification()
    plot_phase_plane(params)
    task_4_component_curves()


if __name__ == "__main__":
    main()
