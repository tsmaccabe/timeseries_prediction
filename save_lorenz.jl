using DifferentialEquations, JLD2, FileIO

# Lorenz attractor parameters
const σ = 10.0
const ρ = 28.0
const β = 8/3

# Define the Lorenz attractor differential equation
function lorenz!(du, u, p, t)
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

SAVE = true

# Initial condition
u0 = Float32[1.0, 1.0, 1.0]

# Time span for the simulation
tspan = (0.0, 10.0)

# Time step for saving data
dt = 0.001

# Chunk size
chunk_size = 1000

# Number of chunks
num_chunks = ceil(Int, (tspan[2] - tspan[1]) / (chunk_size * dt))

# Save function for storing data in JLD2 format
function save_chunk(data, chunk_index, save = true)
    if save
        filename = "timeseries_prediction/data/lorenz/lorenz_data_chunk_$(chunk_index).jld2"
        @save filename data
    end
end

# Integration and save chunks
for i in 1:num_chunks
    global u0, prob, sol

    # Calculate the time span for the current chunk
    tstart = (i - 1) * chunk_size * dt
    tend = min(tstart + chunk_size * dt, tspan[2])
    tspan_chunk = (tstart, tend)

    # Solve the Lorenz attractor for the current chunk
    prob = ODEProblem(lorenz!, u0, tspan_chunk)
    sol = solve(prob, saveat=dt)

    # Save the current chunk to a JLD2 file
    save_chunk(sol, i, SAVE)

    # Update the initial condition for the next chunk
    u0 = sol[end]
end