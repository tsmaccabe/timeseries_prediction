using JLD2, FileIO, CUDA, Statistics, DifferentialEquations, Main.Attractors3Channel
using Random: shuffle, seed!
using Flux: DataLoader

# Parameters
input_length = 40
output_length = 40
input_channels = 3
output_channels = 3

if !@isdefined(chunk_size) & !@isdefined(num_chunks)
    chunk_size = 5000
    num_chunks = 10
end

# Load the data from the JLD2 files
function load_data(system_names, num_chunks, chunk_size)
    data = []
    for system_name in system_names
        for i in 1:num_chunks
            filename = "timeseries_prediction/data/$(system_name)/$(system_name)_data_chunk_$(i).jld2"
            chunk = load(filename, "data")
            append!(data, chunk.u)
        end
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
    idx = 1:n_samples
    
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

# List of system names to load data from
system_names = ["genesio_tesi", "rossler", "lorenz", "rossler_variant"]

# Load data
data = load_data(system_names, num_chunks, chunk_size)

# Create x and y arrays
x, y = create_xy(data, input_length, output_length, input_channels, output_channels)

# Split the data into training, testing, and validation sets
x_train, y_train, x_test, y_test, x_val, y_val = split_data(x, y) |> cu

# Create data loaders
batch_size = 64

train_data = DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)
val_data = DataLoader((x_val, y_val), batchsize=batch_size)
test_data = DataLoader((x_test, y_test), batchsize=batch_size)