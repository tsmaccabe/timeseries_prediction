using Flux: DataLoader, gpu
using Random: randperm
using CUDA, JLD2, FileIO

function truncate_array(arr::AbstractArray{T}, dim::Int, n::Int) where T
    initial_value = 1
    dim_trunc = size(arr, dim) ÷ n * n # find the multiple of n to truncate at
    arr_trunc = selectdim(arr, dim, initial_value:dim_trunc) # truncate the array along the specified dimension
    arr_remain = selectdim(arr, dim, dim_trunc+1:size(arr, dim)) # remaining portion of the array
    truncated_arr = cat(arr_trunc, arr_remain, dims=dim) # concatenate the truncated and remaining portions
    return truncated_arr
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

function spirograph(rs::Vector{<:Real}, d::Float64, n::Int64, T::Int64, drift)
    I = axes(rs)[1]
    t = range(0, stop=2π*n, length=T)
    x = zeros(Float32, length(t))
    y = zeros(Float32, length(t))
    driftvals = drift(t)
    for i in I
        R = rs[i]
        r = rs[mod1(i+1, length(I))]
        x .+= (R-r)*cos.(t) + d*cos.((R-r)/r*t) .+ driftvals[1]
        y .+= (R-r)*sin.(t) - d*sin.((R-r)/r*t) .+ driftvals[2]
    end
    p = Float32.(hcat(x, y))
    
    # Apply preprocessing
    p_preprocessed = preprocess(p, (-1., 1.))

    return p_preprocessed
end

# Hardcoded for 2 channels, like (x(t), y(t)) for spirographs
function generate_data(p, input_length, output_length, step_size::Int)
    L = axes(p)[1][end]
    n_samples = div(L - input_length - output_length + step_size, step_size)
    X = zeros(Float32, input_length, 1, 2, n_samples)
    Y = zeros(Float32, 2 * output_length, n_samples)
    
    for i in 1:n_samples
        start_idx = (i - 1) * step_size + 1
        for j in 1:input_length
            X[j, 1, 1, i] = p[start_idx + j - 1, 1]
            X[j, 1, 2, i] = p[start_idx + j - 1, 2]
        end
        for j in 1:output_length
            Y[2 * (j - 1) + 1, i] = p[start_idx + input_length + j - 1, 1]
            Y[2 * j, i] = p[start_idx + input_length + j - 1, 2]
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



function generate_dataloader(rs, d, n, T, drift, input_length, output_length, stride, chunk_size, batch_size)
    data = spirograph(rs, d, n, T, drift)
    X, Y = generate_data(data, input_length, output_length, stride)
    X = truncate_array(X, 4, chunk_size)
    Y = truncate_array(Y, 2, chunk_size)

    return DataLoader((X, Y), batchsize=batch_size, partial=false, shuffle=false)
end

# Set parameters for data generation
function generate_random_rs_list(num_vectors::Int, n_range::Tuple{Int, Int})
    rs_list = Vector{Vector{Float64}}(undef, num_vectors)
    for i in 1:num_vectors
        num_values = rand(n_range[1]:n_range[2])
        rs = rand(0.1:0.1:25, num_values)
        rs_list[i] = rs
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
    rs_range = (10, 15)
    rs_list = generate_random_rs_list(n_train_chunks, rs_range)
    for i in 1:n_train_chunks
        println("Saving training chunk $(i) of $(n_train_chunks)")
        local train_loader = generate_dataloader(rs_list[i], 1., n_revs, n_ts, drift, input_length, output_length, stride, chunk_size, batch_size)
        save_dataloaders(train_loader, "D:/timeseries_prediction/data/spirograph/train", i)
    end

    # Generate and save test data
    rs_list = generate_random_rs_list(n_test_chunks, rs_range)
    for i in 1:n_test_chunks
        println("Saving testing chunk $(i) of $(n_test_chunks)")
        local test_loader = generate_dataloader(rs_list[i], 1., n_revs, n_ts, drift, input_length, output_length, stride, chunk_size, batch_size)
        save_dataloaders(test_loader, "D:/timeseries_prediction/data/spirograph/val", i)
    end

    CUDA.reclaim()
    GC.gc()

    return n_train_chunks, n_test_chunks
end

drift = t -> (0., 0.)
input_length = 250
output_length = 50
n_channels = 2
input_channels = n_channels
output_channels = n_channels
batch_size = 20
chunk_size = 50*batch_size
stride = Int64(ceil(input_length))


total_samples = 25000
n_revs = round(Int64, total_samples/50)
n_ts = (total_samples-1)*stride + input_length

# Set the total number of samples and call the generate_and_save_data function
n_train_chunks, n_test_chunks = generate_and_save_data(total_samples)
