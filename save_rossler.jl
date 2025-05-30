using DifferentialEquations, JLD2, FileIO

# Rossler attractor parameters
const a = 0.1
const b = 0.1
const c = 14.0

# Define the Rossler attractor differential equation
function rossler!(du, u, p, t)
    du[1] = -u[2] - u[3]
    du[2] = u[1] + a*u[2]
    du[3] = b + u[3]*(u[1] - c)
end

SAVE = true

# Initial condition
u0 = Float32[1.0, 1.0, 1.0]

# Time span for the simulation
tspan = (0.0, 100.0)

# Time step for saving data
dt = 0.001

# Chunk size
chunk_size = 1000

# Number of chunks
num_chunks = ceil(Int, (tspan[2] - tspan[1]) / (chunk_size * dt))

# Save function for storing data in JLD2 format
function save_chunk(data, chunk_index, save = true)
    if save
        filename = "timeseries_prediction/data/rossler/rossler_data_chunk_$(chunk_index).jld2"
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

    # Solve the Rossler attractor for the current chunk
    prob = ODEProblem(rossler!, u0, tspan_chunk)
    sol = solve(prob, saveat=dt)

    # Save the current chunk to a JLD2 file
    save_chunk(sol, i, SAVE)

    # Update the initial condition for the next chunk
    u0 = sol[end]
end