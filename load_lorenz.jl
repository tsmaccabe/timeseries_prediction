using JLD2, FileIO

# Load the data from the JLD2 files
function load_data(system_names, num_chunks)
    if !(system_names isa Tuple)
        system_names = (system_names,)
    end

    combined_data = []
    for system_name in system_names
        data = []
        for i in 1:num_chunks
            filename = "timeseries_prediction/data/$(system_name)/$(system_name)_data_chunk_$(i).jld2"
            chunk = load(filename, "data")
            append!(data, chunk.u)
        end
        append!(combined_data, data)
    end

    return combined_data
end


# Parameters
input_length = 100
output_length = 100
input_channels = 3
output_channels = 3

# Match parameters below to save_chaos.jl if you are using both scripts
u0 = Float32[1.0, 1.0, 1.0]
tspan = (0.0, 100.0)
dt = 0.01

chunk_size = 1000
num_chunks = ceil(Int, (tspan[2] - tspan[1]) / (chunk_size * dt))


# Load data for desired systems
system_names = ("lorenz", "rossler", "genesio_tesi", "rossler_variant", "chua", "three_scroll")
combined_data = load_data(system_names, num_chunks, chunk_size)


# Shuffle the combined data
using Random: seed!, Random.shuffle!
Random.seed!(42)
shuffle!(combined_data)

# Create x and y arrays
x, y = create_xy(combined_data, input_length, output_length, input_channels, output_channels)

# Split the data into training, testing, and validation sets
x_train, y_train, x_test, y_test, x_val, y_val = split_data(x, y) |> cu

# Create data loaders
batch_size = 64

train_data = DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)
val_data = DataLoader((x_val, y_val), batchsize=batch_size)
test_data = DataLoader((x_test, y_test), batchsize=batch_size)