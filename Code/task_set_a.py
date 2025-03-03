import numpy as np
import matplotlib.pyplot as plt
from utils import euler_method, exact_solution, plot_solutions, harvesting_function, dirfield
from scipy.optimize import fsolve

def task_1_units():
    """
    Task A.1: Determine the units of the parameters r and L.
    Provide explanation or comments within this function.
    """
    # units:
    # - r (intrinsic growth rate) has units of 1/time
    # - L (carrying capacity) has units of population size

    print("Task A.1 completed: Units of r = 1/time, Units of L = population size")

def task_2_equilibrium_and_exact_solution():
    """
    Task A.2: Find equilibrium solutions and derive the exact solution to the logistic equation.
    Return the symbolic or numeric expression of x(t) in explicit form.
    """
    # Equilibrium solutions are found by setting dx/dt = 0 in the logistic equation:
    # dx/dt = r(1 - x/L)x => set dx/dt = 0

    # Solve using separation of variables
    # Manually derive explicit form of x(t) = ...
    x_t_expression = "L / (1 + ((L - x0) / x0) * exp(-r * t))"
    print("Task A.2 completed: Explicit form of x(t) =", x_t_expression)
    return x_t_expression

def task_3_euler_method_and_plots():
    """
    Task A.3: Use Euler’s method and plot numerical vs. exact solutions for different step sizes.
    Also calculate and plot the absolute error.
    """
    # Parameters for mountain lion population
    r = 0.65
    L = 5.4
    x0 = .5  # Initial population (in dozens of mountain lions)
    t_end = 20
    h_values = [0.5, 0.1, 0.01]

    # Compute exact solution
    t_exact, x_exact = exact_solution(r, L, x0, t_end)

    # Compute numerical solutions using Euler's method
    solutions = [euler_method(r, L, x0, t_end, h) for h in h_values]

    # Plot the solutions
    plot_solutions(t_exact, x_exact, solutions, h_values)

    # Calculate and plot absolute errors
    plot_absolute_error(t_exact, x_exact, solutions, h_values)

def plot_absolute_error(t_exact, x_exact, solutions, h_values):
    """
    Plot the absolute error of numerical solutions against the exact solution.
    """
    plt.figure(figsize=(10, 6))

    for (t_num, x_num), h in zip(solutions, h_values):
        error = np.abs(np.interp(t_num, t_exact, x_exact) - x_num)
        print(error[-1])
        plt.semilogy(t_num, error, label=f"h = {h}")

    plt.xlabel("Time (years)")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error of Euler’s  with Different Step Sizes", fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/task_a_error_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

def task_4_classify_differential_eq():
    """
    Task A.4: Classify the differential equation (2) with harvesting.
    Provide explanations and classification in comments.
    """
    # Classification (manually):
    # - Order: First order
    # - Linear/Nonlinear: Nonlinear due to the product and harvesting terms
    # - Autonomous: Yes, if the differential equation only depends on the variable x and not explicitly on time t

    print("Task A.4 completed: Classified as nonlinear, first order, autonomous system")

def task_5_behavior_of_harvesting_function():
    """
    Task A.5: Explore the behavior of the harvesting function H(x) and plot the curves.
    """
    x_values = np.linspace(0, 10, 500)
    parameters = [(1, 1), (1, 3), (1, 5), (3, 1), (3, 3), (3, 5), (5, 1), (5, 3), (5, 5)]

    plt.figure(figsize=(10, 6))
    for p, q in parameters:
        y = harvesting_function(x_values, p, q)
        plt.plot(x_values, y, label=f"p = {p}, q = {q}")

    plt.xlabel("x(t), Deer Population (dozens)")
    plt.ylabel("H(x), Deer Harvested (dozens)")
    plt.title("Behavior of Harvesting Function H(x)")
    plt.legend(loc='upper left', fontsize="7.5")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.savefig("plots/task_a_harvesting_function.png", dpi=300, bbox_inches='tight')
    plt.show()

    

def task_6_equilibrium_solutions_and_euler():
    """
    Task A.6: Analyze the logistic equation with harvesting and solve using Euler’s method.
    Plot solutions for different initial conditions and the direction field.
    """
    # Parameters for deer population
    rd = 0.65 # intrinsic growth rate
    Ld = 8.1 # carrying capacity
    p = 1.2 # harvesting parameter
    q = 1 # harvesting parameter
    h = 0.1 # step size for Euler's method
    
    # Initial populations (in dozens of deer)
    pop = [7, 2, 2.5, 0.5]
    
    # Set up the time grid and solution array
    t_grid = np.arange(0, 30, h)
    x_sol = np.zeros_like(t_grid)
    x_grid = np.linspace(0, 8, 500)
    
    # Set up the function to find roots
    def fxt(x):
        y = rd*x*(1-(x/Ld))-harvesting_function(x, p, q)
        return y
    
    # Finding roots
    guess = [0, 0.7, 5]
    roots = np.array([])
    for i in range(len(guess)):
        root = fsolve(fxt, guess[i])
        roots = np.append(roots, root)
    print("Roots:", roots)

    # Plotting the direction field
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(pop)):
        x_sol[0] = pop[i]
    
        for j in range(len(t_grid)-1):
            H = harvesting_function(x_sol[j], p, q)
            x_sol[j+1] = x_sol[j] + h*(fxt(x_sol[j]))
        
        ax.plot(t_grid, x_sol, label="%g Dozen Deer" % pop[i])
        ax.set_title("Deer Population Over Time (Differing Initial Values of Deer)")
        ax.set_xlabel("Time (yrs)")
        ax.set_ylabel("Deer Population (dozens)")
        ax.legend(fontsize="7.5")

    for i in range(len(roots)):
        ax.axhline(y=roots[i], color='#39FF14', linestyle="dashed")

    x = np.arange(0, 30, 2.5)
    y = np.arange(0, 7, .25)
    dirfield(fxt, x, y)
    
    plt.savefig("plots/task_a_harvesting_with_dirfield.png")
    plt.show()
    print("Task A.6 completed: Equilibrium solutions and Euler's method applied with plots.")


def main():
    # Execute each task sequentially
    task_1_units()
    task_2_equilibrium_and_exact_solution()
    task_3_euler_method_and_plots()
    task_4_classify_differential_eq()
    task_5_behavior_of_harvesting_function()
    task_6_equilibrium_solutions_and_euler()

if __name__ == "__main__":
    main()