import numpy as np
import matplotlib.pyplot as plt

#Uitlity functions for the tasks:

#euler_method function
def euler_method(r, L, x0, t_end, h):
    t = np.arange(0, t_end + h, h)
    x = np.zeros_like(t)
    x[0] = x0

    for i in range(1, len(t)):
        x[i] = x[i-1] + h * r * (1 - x[i-1] / L) * x[i-1]

    return t, x

#exact_solution function
def exact_solution(r, L, x0, t_end):
    t = np.linspace(0, t_end, 1000)
    C = L * x0 / (x0 * (L - 1) + L)  # Derived constant from separation of variables solution
    x_exact = L / (1 + (L / x0 - 1) * np.exp(-r * t))
    return t, x_exact

#For set C:
def plot_vector_field(params, x_range, y_range):
    x1, x2 = np.meshgrid(np.linspace(*x_range, 20), np.linspace(*y_range, 20))
    dx1_dt = -params['alpha'] * x1 + params['beta'] * x1 * x2
    dx2_dt = params['gamma'] * (1 - params['kappa'] * x2) * x2 - params['delta'] * x1 * x2
    plt.quiver(x1, x2, dx1_dt, dx2_dt)
    plt.xlabel("Predator population (x1)")
    plt.ylabel("Prey population (x2)")
    plt.title("Vector Field of Logistic Predator-Prey System")


#plot_solutions function
def plot_solutions(t_exact, x_exact, solutions, h_values):
    plt.figure(figsize=(8, 6))
    plt.plot(t_exact, x_exact, label="Exact Solution", color='black', linestyle='--')
    
    for (t, x), h in zip(solutions, h_values):
        plt.plot(t, x, label=f"Euler (h = {h})")

    plt.xlabel("Time (years)")
    plt.ylabel("Population (dozens)")
    plt.title("Mountain Lion Population vs. Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/task_a_plot.png")
    plt.show()
