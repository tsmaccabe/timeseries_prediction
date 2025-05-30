using JLD2, FileIO, CUDA, DifferentialEquations
using Random: shuffle, seed
using Flux: DataLoader

# Parameters
input_length = 100
output_length = 100
input_channels = 3
output_channels = 3

# Match parameters below to save_lorenz.jl if you are using both scripts
# ERR: running this alone, with data stored, doesn't work
u0 = Float32[1.0, 1.0, 1.0]
tspan = (0.0, 100.0)
dt = 0.01

chunk_size = 1000
num_chunks = ceil(Int, (tspan[2] - tspan[1]) / (chunk_size * dt))

# Load the data from the JLD2 files
function load_data(num_chunks, chunk_size)
    data = []
    for i in 1:num_chunks
        filename = "timeseries_prediction/data/rossler/rossler_data_chunk_$(i).jld2"
        chunk = load(filename, "data")
        append!(data, chunk.u)
    end
    return data
end

# Create input x and output y arrays
function create_xy(data, input_length, output_length, input_channels, output_channels)
    n_samples = length(data) - input_length - output_length + 1
    x = zeros(input_length, 1, input_channels, n_samples) |> cu
    y = zeros(output_length * output_channels, n_samples) |> cu

    for i in 1:n_samples
        x[:, 1, :, i] = hcat(data[i:i+input_length-1]...)'
        y[:, i] = reshape(hcat(data[i+input_length:i+input_length+output_length-1]...), (output_length * output_channels,))
    end

    return x, y
end

# Split the data into training, testing, and validation sets
function split_data(x, y)
    n_samples = size(x, 4)
    idx = shuffle(1:n_samples)
    
    train_ratio = 0.8
    test_ratio = 0.1

    train_end_idx = floor(Int, n_samples * train_ratio)
    test_end_idx = floor(Int, n_samples * (train_ratio + test_ratio))

    train_idx = idx[1:train_end_idx]
    test_idx = idx[train_end_idx + 1:test_end_idx]
    val_idx = idx[test_end_idx + 1:end]

    x_train = x[:, :, :, train_idx]
    y_train = y[:, train_idx]

    x_test = x[:, :, :, test_idx]
    y_test = y[:, test_idx]

    x_val = x[:, :, :, val_idx]
    y_val = y[:, val_idx]

    return x_train, y_train, x_test, y_test, x_val, y_val
end

# Load data
data = load_data(num_chunks, chunk_size)

# Create x and y arrays
x, y = create_xy(data, input_length, output_length, input_channels, output_channels)

# Split the data into training, testing, and validation sets
x_train, y_train, x_test, y_test, x_val, y_val = split_data(x, y) |> cu


# Create data loaders
batch_size = 64

train_data = DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)
val_data = DataLoader((x_val, y_val), batchsize=batch_size)
test_data = DataLoader((x_test, y_test), batchsize=batch_size)