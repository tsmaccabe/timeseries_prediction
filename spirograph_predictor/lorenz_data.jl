using Flux: DataLoader, gpu
using Random: randperm
using CUDA, JLD2, FileIO, DifferentialEquations, LinearAlgebra

function truncate_array(arr::AbstractArray{T}, dim::Int, n::Int) where T
    initial_value = 1
    dim_trunc = size(arr, dim) ÷ n * n # find the multiple of n to truncate at
    arr_trunc = selectdim(arr, dim, initial_value:dim_trunc) # truncate the array along the specified dimension
    arr_remain = selectdim(arr, dim, dim_trunc+1:size(arr, dim)) # remaining portion of the array
    truncated_arr = cat(arr_trunc, arr_remain, dims=dim) # concatenate the truncated and remaining portions
    return truncated_arr
end

function round_down_to_multiple(n, m)
    return div(n, m) * m
end

function detrend(p::Matrix{Float32})
    n = size(p, 1)
    X = hcat(ones(Float32, n), collect(1:n))
    β = X \ p
    p_detrended = p .- X * β
    return p_detrended
end

function normalize(p::Matrix{Float32}, range::Tuple{Float64, Float64})
    min_p, max_p = extrema(p)
    p_norm = (p .- min_p) ./ (max_p - min_p)
    p_norm = p_norm .* (range[2] - range[1]) .+ range[1]
    return p_norm
end

function preprocess(p::Matrix{Float32}, range::Tuple{Float64, Float64})
    p_detrended = detrend(p)
    p_normalized = normalize(p_detrended, range)
    return p_normalized
end


function lorenz_attractor(σ::Float32, ρ::Float32, β::Float32, T::Int64, dt::Float32, init_conditions::Vector{<:Real})
    function lorenz!(du, u, p, t)
        σ, ρ, β = p
        x, y, z = u

        du[1] = σ * (y - x)
        du[2] = x * (ρ - z) - y
        du[3] = x * y - β * z
    end

    # Define initial conditions and parameters
    u0 = Float32.(init_conditions)
    tspan = (0.0f0, Float32((T+10)*dt))
    q = [σ, ρ, β]

    # Create an ODEProblem and solve it
    prob = ODEProblem(lorenz!, u0, tspan, q)
    sol = solve(prob, Tsit5(), adaptive = false, dt=dt, reltol=1e-8, abstol=1e-8, saveat=dt)

    # Extract the solution at the specified time points
    sol_array = Array(sol)
    p = Matrix{Float32}(transpose(sol_array))

    # Apply preprocessing
    p_preprocessed = preprocess(p, (-1., 1.))

    return p_preprocessed
end

# Hardcoded for 3 channels, like (x(t), y(t), z(t)) for Lorenz attractor
function generate_data(p, input_length, output_length, step_size::Int)
    L = axes(p)[1][end]
    n_samples = div(L - input_length - output_length + step_size, step_size)
    X = zeros(Float32, input_length, 1, 3, n_samples)
    Y = zeros(Float32, 3 * output_length, n_samples)
    
    for i in 1:n_samples
        start_idx = (i - 1) * step_size + 1
        for j in 1:input_length
            X[j, 1, 1, i] = p[start_idx + j - 1, 1]
            X[j, 1, 2, i] = p[start_idx + j - 1, 2]
            X[j, 1, 3, i] = p[start_idx + j - 1, 3]
        end
        for j in 1:output_length
            Y[3 * (j - 1) + 1, i] = p[start_idx + input_length + j - 1, 1]
            Y[3 * (j - 1) + 2, i] = p[start_idx + input_length + j - 1, 2]
            Y[3 * (j - 1) + 3, i] = p[start_idx + input_length + j - 1, 3]
        end
    end
    
    return X, Y
end

function shuffle_batches(loaded_data::Array{Tuple{A, B}, 1}, batch_size) where {A, B}
    n_batches = div(length(loaded_data), batch_size)
    indices = randperm(n_batches)
    
    shuffled_data = copy(loaded_data)

    for i in 1:n_batches
        source_range = ((indices[i] - 1) * batch_size + 1):(indices[i] * batch_size)
        target_range = ((i - 1) * batch_size + 1):(i * batch_size)
        shuffled_data[target_range] = loaded_data[source_range]
    end

    return shuffled_data
end

function generate_dataloader(T, dt, init_conditions, input_length, output_length, stride, chunk_size, batch_size)
    data = zeros(Float32, (1, 3))
    batches_per_chunk = Int64(floor(chunk_size/batch_size))
    rs_list = generate_random_rs_list(batches_per_chunk)
    for i = 1:batches_per_chunk
        data = cat(dims = 1, data, lorenz_attractor(rs_list[i][1], rs_list[i][2], rs_list[i][3], T, dt, init_conditions))
    end
    data = data[2:end, :]
    X, Y = generate_data(data, input_length, output_length, stride)
    X = truncate_array(X, 4, chunk_size)
    Y = truncate_array(Y, 2, chunk_size)
    return DataLoader((X, Y), batchsize=batch_size, partial=false, shuffle=false)
end

# Set parameters for data generation
function generate_random_rs_list(num_vectors::Int)
    σ_range = (10, 15)
    ρ_range = (20, 30)
    β_range = (2.5, 3.0)

    rs_list = Vector{Vector{Float32}}(undef, num_vectors)
    for i in 1:num_vectors
        σ = rand(σ_range[1]:0.1f0:σ_range[2])
        ρ = rand(ρ_range[1]:0.1f0:ρ_range[2])
        β = rand(β_range[1]:0.01f0:β_range[2])
        rs_list[i] = [σ, ρ, β]
    end
    return rs_list
end

function save_dataloaders(dl::DataLoader, prefix::String, idx::Int)
    chunk = collect(dl)
    if !isempty(chunk)
        save("$(prefix)_chunk_$idx.jld2", "chunk", chunk)
    else
        println("Warning: Empty chunk encountered at index $idx. Skipping save.")
    end
end


# Load chunk
function load_chunk(filename::String, batch_size)
    loaded_data = load(filename, "chunk") |> gpu
    loaded_data = shuffle_batches(loaded_data, batch_size)
    return DataLoader(loaded_data, batchsize=batch_size, partial=false, shuffle=false)
end

function generate_and_save_data(total_samples::Int, train_fraction::Float64=0.8)
    n_samples_train = Int(floor(total_samples * train_fraction))
    n_samples_test = total_samples - n_samples_train

    n_train_chunks = Int(floor(n_samples_train / chunk_size))
    n_test_chunks = Int(floor(n_samples_test / chunk_size))

    # Generate and save training data
    for i in 1:n_train_chunks
        println("Saving training chunk $(i) of $(n_train_chunks)")
        init_conditions = [0.5f0, 0.5f0, 0.0f0]  # Random initial conditions in the range [-10, 10]
        local train_loader = generate_dataloader(T, dt, init_conditions, input_length, output_length, stride, chunk_size, batch_size)
        save_dataloaders(train_loader, "D:/timeseries_prediction/data/lorenz/train", i)
    end

    # Generate and save test data
    for i in 1:n_test_chunks
        println("Saving testing chunk $(i) of $(n_test_chunks)")
        init_conditions = [0.5f0, 0.5f0, 0.0f0]  # Random initial conditions in the range [-10, 10]
        local test_loader = generate_dataloader(T, dt, init_conditions, input_length, output_length, stride, chunk_size, batch_size)
        save_dataloaders(test_loader, "D:/timeseries_prediction/data/lorenz/val", i)
    end

    CUDA.reclaim()
    GC.gc()

    return n_train_chunks, n_test_chunks
end

drift = t -> (0f0, 0f0, 0f0)
input_length = 100
output_length = 10
n_channels = 3
input_channels = n_channels
output_channels = n_channels
batch_size = 32
chunk_size = 128 * batch_size
stride = Int64(ceil(input_length))

total_samples = 30000
dt = 0.01f0  # Time step size
T = batch_size*input_length # Timesteps per parameter set

# Set the total number of samples and call the generate_and_save_data function
n_train_chunks, n_test_chunks = generate_and_save_data(total_samples)
