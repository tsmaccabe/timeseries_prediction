using DifferentialEquations
using Plots

# Define the Lorenz system
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    x, y, z = u
    
    du[1] = σ * (y - x)        # dx/dt
    du[2] = x * (ρ - z) - y     # dy/dt
    du[3] = x * y - β * z       # dz/dt
end

# Set parameters
σ = 10.0
ρ = 28.0
β = 8/3
p = [σ, ρ, β]

# Initial conditions
u0 = [1.0, 1.0, 1.0]

# Time span
tspan = (0.0, 10.0)

# Solve the ODE
prob = ODEProblem(lorenz!, u0, tspan, p)
sol = solve(prob, saveat=0.01)

# Extract x, y, z coordinates
x = [u[1] for u in sol.u]
y = [u[2] for u in sol.u]
z = [u[3] for u in sol.u]

# Create 3D plot
plot3d(x, y, z, 
    xlabel="x", 
    ylabel="y", 
    zlabel="z",
    title="Lorenz Attractor",
    legend=false,
    lw=0.5,
    color=:viridis,
    size=(800, 600))
