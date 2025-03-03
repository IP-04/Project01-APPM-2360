import numpy as np
import matplotlib.pyplot as plt

#Uitlity functions for the tasks:

# Task A.2: Exact solution of the logistic equation
def exact_solution(r, L, x0, t_end):
    t = np.linspace(0, t_end, 1000)
    C = L * x0 / (x0 * (L - 1) + L)  # Derived constant from separation of variables solution
    x_exact = L / (1 + (L / x0 - 1) * np.exp(-r * t))
    return t, x_exact
# Task A.3: Euler's method for solving differential equations
def euler_method(r, L, x0, t_end, h):
    t = np.arange(0, t_end + h, h)
    x = np.zeros_like(t)
    x[0] = x0

    for i in range(1, len(t)):
        x[i] = x[i-1] + h * r * (1 - x[i-1] / L) * x[i-1]

    return t, x
# Task A.3: Plot solutions from Euler's method and exact solution
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
# Task A.6: Direction field for differential equations
def dirfield(f, X, Y):
    #vector must have step size pre-specified
    x, y = np.meshgrid(X, Y)
    dy = f(y)
    dx = np.ones_like(dy)
    
    norm = np.sqrt(dx**2 + dy**2)
    dyu = dy/norm
    dxu = dx/norm

    plt.quiver(x,y,dxu,dyu, width=0.002)
    plt.grid(True)
    plt.show()
    
# Task B.1: Differential equation for x1 in the Lotka-Volterra system
def LV_dx1(x1, x2, a, b, g, d):
    y = (-a*x1) + (b*x1*x2)
    return y
# Task B.1: Differential equation for x2 in the Lotka-Volterra system
def LV_dx2(x1, x2, a, b, g, d):
    y = (g*x2) - (d*x1*x2)
    return y
# Task B.1: Lotka-Volterra system
def LV_system(t, f, a, b, g, d):
    #f is the vector containing x1 and x2
    x1, x2 = f
    return [LV_dx1(x1, x2, a, b, g, d), LV_dx2(x1, x2, a, b, g, d)]
# Task B.3: Vector field for the Lotka-Volterra system
def vectorfield1(f1, f2, X, Y, params):
    a, b, g, d = params
    x, y = np.meshgrid(X, Y)
    dy = f2(x, y, a, b, g, d)
    dx = f1(x, y, a, b, g, d)
    
    norm = np.sqrt(x**2 + y**2)
    dyu = dy/norm
    dxu = dx/norm
    
    plt.quiver(x,y,dxu,dyu, width=0.002)
# Task B.5: Harvesting function for the logistic equation
def harvesting_function(x, p, q):
        return (p * x**2) / (q + x**2)

# Task C.1: Differential equation for x1 in the Lotka-Volterra system with harvesting
def LPP_dx1(x1, x2, a, b, g, d, k):
    y = (-a*x1) + (b*x1*x2)
    return y
# Task C.1: Differential equation for x2 in the Lotka-Volterra system with harvesting
def LPP_dx2(x1, x2, a, b, g, d, k):
    y = x2*g*(1-k*x2) - d*x1*x2
    return y
# Task C.1: Lotka-Volterra system with harvesting
def LPP_system(t, f, a, b, g, d, k):
    x1, x2 = f
    return [LPP_dx1(x1, x2, a, b, g, d, k), LPP_dx2(x1, x2, a, b, g, d, k)]
# Task C.2: Vector field for the Lotka-Volterra system with harvesting
def vectorfield(f1, f2, X, Y, params):
    a, b, g, d, k = params
    x, y = np.meshgrid(X, Y)
    dy = f2(x, y, a, b, g, d, k)
    dx = f1(x, y, a, b, g, d, k)
    norm = np.sqrt(x**2 + y**2)
    dyu = dy/norm
    dxu = dx/norm
    plt.quiver(x,y,dxu,dyu, width=0.002)

    

