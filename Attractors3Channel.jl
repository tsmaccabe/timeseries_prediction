module Attractors3Channel

# Genesio-Tesi parameters
const GT_ALPHA = 1.0
const GT_BETA = 2.0
const GT_GAMMA = 1.0

# Define the Genesio-Tesi differential equation
function genesio_tesi!(du, u, p, t)
    du[1] = u[2]
    du[2] = u[3]
    du[3] = -GT_ALPHA * u[3] - GT_BETA * u[2] - GT_GAMMA * sin(u[1])
end

# Rossler attractor parameters
const RA_A = 0.1
const RA_B = 0.1
const RA_C = 14.0

# Define the Rossler attractor differential equation
function rossler!(du, u, p, t)
    du[1] = -u[2] - u[3]
    du[2] = u[1] + RA_A * u[2]
    du[3] = RA_B + u[3] * (u[1] - RA_C)
end

# Lorenz attractor parameters
const LA_SIGMA = 10.0
const LA_RHO = 28.0
const LA_BETA = 8 / 3

# Define the Lorenz attractor differential equation
function lorenz!(du, u, p, t)
    du[1] = LA_SIGMA * (u[2] - u[1])
    du[2] = u[1] * (LA_RHO - u[3]) - u[2]
    du[3] = u[1] * u[2] - LA_BETA * u[3]
end

# Rössler variant parameters
const RV_A = 0.2
const RV_B = 0.2
const RV_C = 5.7

# Define the Rossler variant differential equation
function rossler_variant!(du, u, p, t)
    du[1] = -u[2] - u[3]
    du[2] = u[1] + RV_A * u[2]
    du[3] = RV_B + u[3] * (u[1] - RV_C)
end

# Chua system parameters
const CHUA_ALPHA = 9.0
const CHUA_BETA = 100.0
const CHUA_C = 14.0

# Define the Chua system differential equation
function chua!(du, u, p, t)
    du[1] = CHUA_ALPHA * (u[2] - u[1] - h(u[1]))
    du[2] = u[1] - u[2] + u[3]
    du[3] = -CHUA_BETA * u[2] - CHUA_C * u[3]
end

# Chua's system piecewise-linear function h(x)
function h(x)
    m0 = -1.0
    m1 = 1.0
    return (m1 * x + m0 * x) / 2 + (abs(m1 * x) - abs(m0 * x)) / 2
end

# Three-Scroll system parameters
const TS_A = 40.0
const TS_B = 55.0
const TS_C = 22.0

# Define the Three-Scroll system differential equation
function three_scroll!(du, u, p, t)
    du[1] = -TS_A * u[1] + u[2]
    du[2] = -TS_B * u[2] + u[3]
    du[3] = -TS_C * u[3] + u[1] * u[2]
end


# Global variables for initial conditions, time spans, and time steps
const u0 = [
    [0.1, -0.2, 0.3],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.5, 0.0, 0.0],
    [1.0, 1.0, 1.0],
]

const tspan = fill((0.0, 100.0), 6)
const dt = fill(0.001, 6)


using DifferentialEquations, JLD2, FileIO

# Save function for storing data in JLD2 format
function save_chunk(system_name, data, chunk_index, save=true)
    if save
        filename = "timeseries_prediction/data/$(system_name)/$(system_name)_data_chunk_$(chunk_index).jld2"
        @save filename data
    end
end

# Function to save the data for the specified system
function save_system_data(system_name, system_functions, chunk_size, num_chunks)
    if !(system_functions isa Tuple)
        system_functions = (system_functions,)
    end

    for (idx, system_function) in enumerate(system_functions)# Integration and save chunks
        for i in 1:num_chunks
            # Calculate the time span for the current chunk
            tstart = (i - 1) * chunk_size * dt[idx]
            tend = min(tstart + chunk_size * dt[idx], tspan[idx][2])
            tspan_chunk = (tstart, tend)

            # Solve the system for the current chunk
            prob = ODEProblem(system_function, u0[idx], tspan_chunk)
            sol = solve(prob, saveat=dt[idx])

            # Save the current chunk to a JLD2 file
            save_chunk(system_name, sol, i, true)

            # Update the initial condition for the next chunk
            u0[idx] = sol[end]
        end
    end
end

# Function to save the data for all six systems
function save_all_system_data()
    system_names = ["genesio_tesi", "rossler", "lorenz", "rossler_variant", "chua", "three_scroll"]
    system_functions = (genesio_tesi!, rossler!, lorenz!, rossler_variant!, chua!, three_scroll!)

    if !(u0 isa Vector{Vector{<:Real}})
        u0 = fill(u0, length(system_functions))
    end

    if !(tspan isa Vector{Tuple{<:Real, <:Real}})
        tspan = fill(tspan, length(system_functions))
    end

    if !(dt isa Vector{<:Real})
        dt = fill(dt, length(system_functions))
    end

    for (idx, (system_name, system_function)) in enumerate(zip(system_names, system_functions))
        save_system_data(system_name, system_function, u0[idx], tspan[idx], dt[idx], chunk_size)
    end
end

# Function to save data for selected systems
function save_selected_system_data(selected_system_names, chunk_size = 1000, num_chunks = 100)
    system_names = ["genesio_tesi", "rossler", "lorenz", "rossler_variant", "chua", "three_scroll"]
    system_functions = (genesio_tesi!, rossler!, lorenz!, rossler_variant!, chua!, three_scroll!)

    for (system_name, system_function) in zip(system_names, system_functions)
        if system_name in selected_system_names
            save_system_data(system_name, system_function, chunk_size, num_chunks)
        end
    end
end

export save_system_data, save_all_system_data, save_selected_system_data
export genesio_tesi!, rossler!, lorenz!, rossler_variant!, chua!, three_scroll!

end # module