using Plots
using Main.Attractors3Channel
using DifferentialEquations: ODEProblem, solve

# Simulation parameters
tspan = (0.0, 200.0)
dt = 0.1
num_timesteps = Int((tspan[2] - tspan[1]) / dt) + 1

# Initial conditions
u0 = [1.0, 1.0, 1.0]

# Define a function to simulate and plot the specified system
function simulate_and_plot_system(system_function, u0, tspan, dt)
    prob = ODEProblem(system_function, u0, tspan)
    sol = solve(prob, saveat=dt)
    
    p = plot(sol, vars=(1, 2, 3), title=string(nameof(system_function)), xlabel="x", ylabel="y", zlabel="z")
    return p
end

# Simulate and plot each system
p1 = simulate_and_plot_system(Attractors3Channel.genesio_tesi!, [0.1, 0.2, 0.3], tspan, dt)
p2 = simulate_and_plot_system(Attractors3Channel.rossler!, u0, tspan, dt)
p3 = simulate_and_plot_system(Attractors3Channel.lorenz!, u0, tspan.*0.3, dt*0.3)
p4 = simulate_and_plot_system(Attractors3Channel.rossler_variant!, u0, tspan, dt)
p5 = simulate_and_plot_system(Attractors3Channel.chua!, [0.001, 0., 0.0], tspan.*0.2, dt*0.2)
p6 = simulate_and_plot_system(Attractors3Channel.three_scroll!, [0.5, 0.1, 0.0], tspan, dt)

# Combine all plots into a grid
plot(p1, p2, p3, p4, p5, p6, layout=(2, 3), legend=false)