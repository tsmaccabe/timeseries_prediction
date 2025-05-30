using DifferentialEquations

# Define the Lorenz system
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# Define the Rossler system
function rossler!(du, u, p, t)
    a, b, c = p
    du[1] = -u[2] - u[3]
    du[2] = u[1] + a * u[2]
    du[3] = b + u[3] * (u[1] - c)
end

# Define the coupling function
function coupling!(du, u, p, t)
    x, y = u
    α, β, γ = p
    du[1] = -α * x + β * y
    du[2] = γ * x - β * y
end

# Define the initial conditions and parameters for each system
u0_l = [0.1, 0.0, 0.0]
p_l = [10.0, 28.0, 8/3]

u0_r = [0.1, 0.0, 0.0]
p_r = [0.2, 0.2, 5.7]

u0_c = [0.1, 0.1]
p_c = [1.0, 0.8, 0.5]

# Define the differential equation for the coupled system
function coupled!(du, u, p, t)
    x, y, z, a, b, c, α, β, γ = p
    du[1:3] .= lorenz!(similar(u[1:3]), u[1:3], [x, y, z], t)
    du[4:6] .= rossler!(similar(u[4:6]), u[4:6], [a, b, c], t)
    du[7:8] .= coupling!(similar(u[7:8]), u[7:8], [α, β, γ], t)
end

# Define the initial conditions and parameters for the coupled system
u0 = [u0_l..., u0_r..., u0_c...]
p = [p_l..., p_r..., p_c..., 1.0, 1.0, 1.0]

# Define the time span and solve the differential equation
tspan = (0.0, 100.0)
prob = ODEProblem(coupled!, u0, tspan, p)
sol = solve(prob, Vern7(), reltol=1e-10, abstol=1e-10)

# Plot the solutions for the Lorenz and Rossler systems
using Plots
plot(sol, vars=(1,4), xlabel="x", ylabel="y", zlabel="z")

using LyapunovExponents

# Define the Lorenz system
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# Define the Rossler system
function rossler!(du, u, p, t)
    a, b, c = p
    du[1] = -u[2] - u[3]
    du[2] = u[1] + a * u[2]
    du[3] = b + u[3] * (u[1] - c)
end

# Define the coupling function
function coupling!(du, u, p, t)
    x, y = u
    α, β, γ = p
    du[1] = -α * x + β * y
    du[2] = γ * x - β * y
end

# Define the initial conditions and parameters for each system
u0_l = [0.1, 0.0, 0.0]
p_l = [10.0, 28.0, 8/3]

u0_r = [0.1, 0.0, 0.0]
p_r = [0.2, 0.2, 5.7]

u0_c = [0.1, 0.1]
p_c = [1.0, 0.8, 0.5]

# Calculate the Lyapunov exponents for the Lorenz system
lyap_lorenz = lyapunov(lorenz!, u0_l, p_l, 0.0:0.01:100.0)

# Calculate the Lyapunov exponents for the Rossler system
lyap_rossler = lyapunov(rossler!, u0_r, p_r, 0.0:0.01:100.0)

# Calculate the Lyapunov exponents for the coupled system
function coupled!(du, u, p, t)
    x, y, z, a, b, c, α, β, γ = p
    du[1:3] .= lorenz!(similar(u[1:3]), u[1:3], [x, y, z], t)
    du[4:6] .= rossler!(similar(u[4:6]), u[4:6], [a, b, c], t)
    du[7:8] .= coupling!(similar(u[7:8]), u[7:8], [α, β, γ], t)
end

u0 = [u0_l..., u0_r..., u0_c...]
p = [p_l..., p_r..., p_c..., 1.0, 1.0, 1.0]

lyap_coupled = lyapunov(coupled!, u0, p, 0.0:0.01:100.0)

println("Lyapunov exponents for the Lorenz system: ", lyap_lorenz)
println("Lyapunov exponents for the Rossler system: ", lyap_rossler)
println("Lyapunov exponents for the coupled system: ", lyap_coupled)