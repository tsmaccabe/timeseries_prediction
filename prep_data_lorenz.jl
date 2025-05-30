using Flux
using Flux.Data: DataLoader
using CUDA

# Parameters
train_split = 0.8
val_split = 0.1
test_split = 0.1

# Calculate dataset sizes
num_samples = size(X, 4)
train_size = floor(Int, num_samples * train_split)
val_size = floor(Int, num_samples * val_split)
test_size = num_samples - train_size - val_size

# Split data
X_train = X[:, :, :, 1:train_size] |> gpu
Y_train = Y[:, 1:train_size] |> gpu

X_val = X[:, :, :, (train_size + 1):(train_size + val_size)] |> gpu
Y_val = Y[:, (train_size + 1):(train_size + val_size)] |> gpu

X_test = X[:, :, :, (train_size + val_size + 1):end] |> gpu
Y_test = Y[:, (train_size + val_size + 1):end] |> gpu

# Create data loaders
batch_size = 64

train_data = DataLoader((X_train, Y_train), batchsize=batch_size, shuffle=true)
val_data = DataLoader((X_val, Y_val), batchsize=batch_size, shuffle=true)
test_data = DataLoader((X_test, Y_test), batchsize=batch_size, shuffle=true)