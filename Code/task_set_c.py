import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from utils import vectorfield, LPP_dx1, LPP_dx2, LPP_system

params = (1.5, 1.1, 2.5, 1.4, 0.5)

def task_1_nullclines_and_equilibrium(params):
    a, b, g, d, k = params
    x1 = np.arange(0, 5, 0.2)
    vnulls = [a/b, 0]
    hnulls = [(1/k)*(1-(x1*(d/g))), 0]
    
    plt.plot(x1, hnulls[0], label='h-nullcline', color='blue')
    plt.axhline(y=vnulls[0], label='v-nullcline', color='red')
    plt.axhline(y=hnulls[1], color="blue")
    plt.axvline(x=vnulls[1], color='red')
    plt.xlim(-0.01, 5)
    plt.ylim(-0.01, 5)

def task_2a_vector_field():
    x1 = np.arange(-0.01, 5, 0.2)
    x2 = np.arange(-0.01, 5, 0.2)
    params = (1.5, 1.1, 2.5, 1.4, 0.5)
    return vectorfield(LPP_dx1, LPP_dx2, x1, x2, params)
    
def task_2b_solve_ode_system():
    initial_conditions = [[5, 1], [1, 5]]
    t_span = (0, 20)
    t_eval = np.linspace(0, 20, 2000)
    
    solutions = []
    for i in range(len(initial_conditions)):
        sol = solve_ivp(LPP_system, t_span, initial_conditions[i], t_eval=t_eval, args=params)
        solutions.append(sol)
    return solutions

def task_2c_phase_plane_and_trajectories(params):
    plt.figure(figsize=(8, 6))
    a, b, g, d, k = params
    task_1_nullclines_and_equilibrium(params)
    task_2a_vector_field()
    stable_eq_sols = [(g/d)*(1-(k*a)/b), a/b, "stable"]
    unstable_eq_sols = [[0,0,"unstable"], [0,2,"unstable"]]

    solutions = task_2b_solve_ode_system()
    labels = ["I.C. [5, 1] Trajectory", "I.C. [1, 5] Trajectory"]
    colors = ["purple", "orange"]
    for i in range(len(solutions)):
        x1 = solutions[i].y[0]
        x2 = solutions[i].y[1]
        plt.plot(x1, x2, label=labels[i], color=colors[i])

    for i in range(len(unstable_eq_sols)):
        if i == 0:
            x_points = unstable_eq_sols[i][0]
            y_points = unstable_eq_sols[i][1]
            plt.plot(x_points, y_points, marker="o", markerfacecolor='none', markeredgecolor='#39FF14', markeredgewidth=2, label="Unstable Eq. Pt.")
        x_points = unstable_eq_sols[i][0]
        y_points = unstable_eq_sols[i][1]
        plt.plot(x_points, y_points, marker="o", markerfacecolor='none', markeredgecolor='#39FF14', markeredgewidth=2)
    x_points = stable_eq_sols[0]
    y_points = stable_eq_sols[1]
    plt.plot(x_points, y_points, marker="o", label="Stable Eq. Pt.", color='#39FF14')
        
    plt.title("Logistic Predator-Prey System With Differing I.C.'s")
    plt.xlabel("x1(t) (Dozens of Predators)")
    plt.ylabel("x2(t) (Dozens of Prey)")
    plt.legend(loc="upper right")
    plt.grid(True)
    #Padding
    plt.xlim(-0.3, 5.3)
    plt.ylim(-0.3, 5.3)
    
    plt.savefig("plots/task_c_phase_plane_trajectory.png")
    plt.show()

def task_3_component_curves():
    plt.figure(figsize=(8, 6))
    solutions = task_2b_solve_ode_system()
    label1 = ["x1(t), I.C. [5,1]", "x1(t), I.C. [1,5]"]
    label2 = ["x2(t), I.C. [5,1]", "x2(t), I.C. [1,5]"]
    linestyles = ["solid", "dashed"]
    for i in range(len(solutions)):
        t = solutions[i].t
        x1 = solutions[i].y[0]
        x2 = solutions[i].y[1]
        plt.plot(t, x1, label=label1[i], color="purple", linestyle=linestyles[i])
        plt.plot(t, x2, label=label2[i], color="orange", linestyle=linestyles[i])

    plt.xlabel("Time (yrs)")
    plt.ylabel("Population (Dozens of Animals)")
    plt.title("Logistic Predator-Prey Component Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/task_c_component_curves.png")
    plt.show()

def main():
    task_2c_phase_plane_and_trajectories(params)
    task_3_component_curves()

if __name__ == "__main__":
    main()