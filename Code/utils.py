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

def harvesting_function(x, p, q):
        return (p * x**2) / (q + x**2)

def dirfield(f, X, Y):
    #vector must have step size pre-specified
    x, y = np.meshgrid(X, Y)
    dy = f(y)
    dx = np.ones_like(dy)
    
    norm = np.sqrt(dx**2 + dy**2)
    dyu = dy/norm
    dxu = dx/norm

    plt.quiver(x,y,dxu,dyu, width=0.002)
    plt.show()

def dx1_dt(x1, x2, alpha, beta, gamma, delta, kappa):
        y = (-alpha*x1) + (beta*x1*x2)
        return y
def dx2_dt(x1, x2, alpha, beta, gamma, delta, kappa):
        y = x2*gamma*(1-kappa*x2) - delta*x1*x2
        return y
def vectorfield(f1, f2, X, Y, params):
    a, b, g, d, k = params
    x, y = np.meshgrid(X, Y)
    dy = f2(x, y, a, b, g, d, k)
    dx = f1(x, y, a, b, g, d, k)
    
    norm = np.sqrt(x**2 + y**2)
    dyu = dy/norm
    dxu = dx/norm
    
    plt.quiver(x,y,dxu,dyu, width=0.002, scale=120)
    

def plot_phase_plane(params):
    v_nullcline, h_nullcline, equilibrium_solutions = task_2_nullclines_and_equilibrium()
    alpha, beta, gamma, delta = params['alpha'], params['beta'], params['gamma'], params['delta']
    
    x1 = np.linspace(0, 5, 400)
    x2_v_nullcline = [v_nullcline[0].subs({'alpha': alpha, 'beta': beta, 'x1': x}) for x in x1]
    x2_h_nullcline = [h_nullcline[0].subs({'gamma': gamma, 'delta': delta, 'x1': x}) for x in x1]

    plt.plot(x1, x2_v_nullcline, label='v-nullcline')
    plt.plot(x1, x2_h_nullcline, label='h-nullcline')

    for sol in equilibrium_solutions:
        plt.plot(sol[0], sol[1], 'ro')  # Equilibrium points

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Phase Plane')
    plt.legend()
    plt.grid()
    plt.show()

def plot_phase_plane2(vnulls, hnulls, eq_sols, x1, task_2a_vector_field):
    task_2a_vector_field()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Phase Plane')
    plt.grid()
    plt.plot(x1, hnulls[0], label='h-nullcline', color='blue')
    plt.xlim(-0.01,5)
    plt.ylim(-0.01, 5)
    plt.axhline(y=vnulls[0], label='v-nullcline', color='red')
    plt.axhline(y=hnulls[1], color="blue")
    plt.axvline(x=vnulls[1], color='red')
    
    for i in range(len(eq_sols)):
        if eq_sols[i][2] == 1:
            x_points = eq_sols[i][0]
            y_points = eq_sols[i][1]
            plt.plot(x_points, y_points, color='#39FF14', marker="o")
        x_points = eq_sols[i][0]
        y_points = eq_sols[i][1]
        plt.plot(x_points, y_points, marker="o", markerfacecolor='none', markeredgecolor='#39FF14', markeredgewidth=2)

