using Flux: DataLoader, gpu

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

function generate_data(p, input_length, output_length)
    L = axes(p)[1][end]
    X = zeros(Float32, input_length, 1, 2, L-input_length-output_length+1)
    Y = zeros(Float32, 2*output_length, L-input_length-output_length+1)
    for i in 1:L-input_length-output_length+1
        for j in 1:input_length
            X[j, 1, 1, i] = p[i+j-1, 1]
            X[j, 1, 2, i] = p[i+j-1, 2]
        end
        for j in 1:output_length
            Y[2*(j-1)+1, i] = p[i+input_length+j-1, 1]
            Y[2*j, i] = p[i+input_length+j-1, 2]
        end
    end
    return X, Y
end

function generate_dataloaders(rs, d, n, T, drift, input_length, output_length, batch_size)
    # Generate data and split into train, validation, and test sets
    data = spirograph(rs, d, n, T, drift)
    L = size(data, 1)
    train_idx = 1:Int(0.8*L)
    test_idx = Int(0.8*L)+1:Int(0.9*L)
    val_idx = Int(0.9*L)+1:L
    train_data = data[train_idx, :]
    val_data = data[val_idx, :]
    test_data = data[test_idx, :]

    # Generate X and Y arrays for train, validation, and test sets
    train_X, train_Y = generate_data(train_data, input_length, output_length)
    val_X, val_Y = generate_data(val_data, input_length, output_length)
    test_X, test_Y = generate_data(test_data, input_length, output_length)

    train_X = truncate_array(train_X, 4, batch_size)
    val_X = truncate_array(val_X, 4, batch_size)
    test_X = truncate_array(test_X, 4, batch_size)

    train_Y = truncate_array(train_Y, 2, batch_size)
    val_Y = truncate_array(val_Y, 2, batch_size)
    test_Y = truncate_array(test_Y, 2, batch_size)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y
end

function generate_all_dataloaders(rs_list, input_length, output_length, n_samples, T, batch_size)
    # Define constants
    d = 1.
    drift2(t) = (0., 0.)

    # Generate dataloaders for each set of rs values
    train_X = Array{Float32}(undef, input_length, 1, 2, 1) |> gpu
    val_X = Array{Float32}(undef, input_length, 1, 2, 1) |> gpu
    test_X = Array{Float32}(undef, input_length, 1, 2, 1) |> gpu
    train_Y = Array{Float32}(undef, 2*output_length, 1) |> gpu
    val_Y = Array{Float32}(undef, 2*output_length, 1) |> gpu
    test_Y = Array{Float32}(undef, 2*output_length, 1) |> gpu

    for rs in rs_list
        train_x, train_y, val_x, val_y, test_x, test_y = generate_dataloaders(rs, d, n_samples, T, drift2, input_length, output_length, batch_size) |> gpu
        train_X = cat(train_X, train_x, dims = 4)
        val_X = cat(val_X, val_x, dims = 4)
        test_X = cat(test_X, test_x, dims = 4)
        train_Y = cat(train_Y, train_y, dims = 2)
        val_Y = cat(val_Y, val_y, dims = 2)
        test_Y = cat(test_Y, test_y, dims = 2)
    end

    train_X = train_X[:, :, :, 2:end]
    val_X = val_X[:, :, :, 2:end]
    test_X = test_X[:, :, :, 2:end]
    train_Y = train_Y[:, 2:end]
    val_Y = val_Y[:, 2:end]
    test_Y = test_Y[:, 2:end]

    train_loader = DataLoader((train_X, train_Y), batchsize=batch_size, partial=false, shuffle=true)
    val_loader = DataLoader((val_X, val_Y), batchsize=batch_size, partial=false, shuffle=true)
    test_loader = DataLoader((test_X, test_Y), batchsize=batch_size, partial=false, shuffle=true)

    return train_loader, val_loader, test_loader
end

# Set parameters for data generation
function generate_random_rs_list(num_vectors::Int)
    rs_list = Vector{Vector{Float64}}(undef, num_vectors)
    for i in 1:num_vectors
        num_values = rand(3:12)
        rs = rand(0.1:0.1:25, num_values)
        rs_list[i] = rs
    end
    return rs_list
end


drift = t -> (0., 0.)
input_length = 500
output_length = 100
n_channels = 2
input_channels = n_channels
output_channels = n_channels
n_revs = 80
n_ts = 100*n_revs
batch_size = 16 # to keep batch data in chronological order, n_ts > batch_size*input_length

rs_list1 = generate_random_rs_list(100)
train_loader, _, _ = generate_all_dataloaders(rs_list1, input_length, output_length, n_revs, n_ts, batch_size)
rs_list2 = generate_random_rs_list(100)
_, val_loader, _ = generate_all_dataloaders(rs_list2, input_length, output_length, n_revs, n_ts, batch_size)
rs_list3 = generate_random_rs_list(100)
_, _, test_loader = generate_all_dataloaders(rs_list3, input_length, output_length, n_revs, n_ts, batch_size)
