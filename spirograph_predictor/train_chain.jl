using Random
using Flux

function exponential_decay(initial_lr::Float64, decay_rate::Float64, epoch::Int, decay_steps::Int)
    return initial_lr * exp(-decay_rate * epoch / decay_steps)
end
function step_decay(initial_lr::Float64, decay_factor::Float64, decay_epochs::Int, epoch::Int)
    return initial_lr * decay_factor^floor((epoch-1)/decay_epochs)
end
function linear_decay(initial_lr::Float64, min_lr::Float64, epoch::Int, decay_epochs::Int)
    slope = (initial_lr - min_lr) / decay_epochs
    return max(initial_lr - slope * epoch, min_lr)
end
function inverse_decay(initial_lr::Float64, epoch::Int)
    return initial_lr / sqrt(epoch)
end
function cosine_annealing_lr(initial_lr::Float64, epoch::Int, epochs::Int)
    return initial_lr * 0.5 * (1 + cos(pi * epoch / epochs))
end

function clip_weights!(model, lower_bound, upper_bound)
    for layer in model.layers
        if isa(layer, Dense) || isa(layer, Conv)
            w = params(layer)[1] # Access the weight matrix of the layer
            w .= clamp.(w, lower_bound, upper_bound) # Clip the weights
        end
    end
end

# Create an array of zeros with the same shape and data type as the given array
function zeros_like(a::AbstractArray)
    z = CUDA.zeros(size(a))
    return z
end

# Initialize the gradients to zeros
function init_grads(model)
    return [zeros_like(p) for p in params(model)]
end

function l2_reg(model, lambda::Float64)
    l2_term = 0.0
    for layer in model.layers
        if isa(layer, Dense) || isa(layer, Conv)
            l2_term += sum(params(layer)[1] .^ 2)
        end
    end
    return 0.5 * lambda * l2_term
end

function spirograph(rs::Vector{<:Real}, d::Float64, n::Int64, T::Int64, drift)
    I = axes(rs)[1]
    t = range(0, stop=2π*n, length=T)
    x = zeros(length(t))
    y = zeros(length(t))
    driftvals = drift(t)
    for i in I
        R = rs[i]
        r = rs[mod1(i+1, length(I))]
        x .+= (R-r)*cos.(t) + d*cos.((R-r)/r*t) .+ driftvals[1]
        y .+= (R-r)*sin.(t) - d*sin.((R-r)/r*t) .+ driftvals[2]
    end
    return (x, y)
end

function create_spirograph_data(param_set, data_size)
    rs, d, n, T, drift = param_set
    data = spirograph(rs, d, n, T, drift)
    return data
end

function create_xy(data, input_length, output_length, input_channels, output_channels)
    # Create input x and output y arrays
    x, y = data
    return x, y
end

function create_data_loader(params, data_size, input_length, output_length, input_channels, output_channels)
    d = 1.0
    n = 2000
    T = 10 * n
    drift = t -> (0, 0)  # No drift

    # Generate data using the given parameter set
    data = create_spirograph_data((params, d, n, T, drift), data_size)

    # Create input x and output y arrays
    x, y = create_xy(data, input_length, output_length, input_channels, output_channels)

    # Generate the train and test datasets
    train_data = DataLoader((x, y), batchsize=batch_size, shuffle=true)

    return train_data
end

# Define the 1D convolutional neural network architecture
function build_cnn(input_length::Int, input_channels::Int, output_length::Int)
    ch = Chain(
        x -> reshape(x, input_length, 1, input_channels, :),
        Conv((5,1), input_channels => 12, pad=SamePad(), relu),
        BatchNorm(12),
        Conv((4,1), 12 => 24, pad=SamePad(), relu),
        BatchNorm(24),
        MaxPool((3,1), pad=SamePad()),
        x -> x[:, 1, :, :], # Remove second dimension
        x -> permutedims(x, (2, 1, 3)), # Transpose dimensions to have the time dimension first
        GRU(24, 24),
        Flux.flatten,
        Dense(168, output_length*input_channels)
    )
    return ch
end

function sequential_training(param_sets, train_data_fn, args)
    initial_lr, min_lr, epochs, patience, data_size, input_length, output_length, input_channels, output_channels = args
    results = Dict()

    for params in param_sets
        println("Training with parameters: $params")

        # Create DataLoader for the current set of input parameters
        train_data = train_data_fn(params, data_size, input_length, output_length, input_channels, output_channels)

        # Initialize variables for early stopping
        best_loss = Inf
        num_bad_epochs = 0

        # Train Loop
        loss_train = []
        loss_test = []

        opt = ADAM(initial_lr)

        for epoch in 1:epochs
            # Update learning rate
            opt.eta = linear_decay(initial_lr, min_lr, epoch, epochs)

            epoch_loss = 0.0
            num_batches = 0
            for (x, y) in train_data
                batch_loss = loss(x, y)
                epoch_loss += batch_loss
                num_batches += 1

                # Compute gradients for current mini-batch
                gradients = gradient(() -> loss(x, y), params(model))

                Flux.Optimise.update!(opt, params(model), gradients) # Update model weights using gradients
                clip_weights!(model, -5., 5.) # Clip weights
            end

            epoch_loss /= num_batches
            push!(loss_train, epoch_loss)
            test_loss = loss(x_test, y_test)
            push!(loss_test, test_loss)
            println("Epoch $epoch, bad $num_bad_epochs - Loss: $epoch_loss - Test Loss: $test_loss")

            if test_loss < best_loss
                best_loss = test_loss
                num_bad_epochs = 0
            else
                num_bad_epochs += 1
                if num_bad_epochs >= patience
                    println("Stopping early")
                    break
                end
            end
        end

        # Log the results for the current set of input parameters
        results[params] = (loss_train=loss_train, loss_test=loss_test)

        # Reset the model before the next iteration
        Flux.reset!(model)
    end

    return results
end

# Build the model
model = build_cnn(input_length, input_channels, output_length) |> gpu

num_sets = 10 # The number of parameter sets you want to create
param_sets = [create_random_param_set() for _ in 1:num_sets]

# Define the loss function and optimizer
loss(x, y) = Flux.mse(model(x), y) |> gpu
# Custom training loop with loss plotting
loss_train = Float64[]
loss_test = Float64[]

epochs = 100

initial_lr = 1e-2
min_lr = 1e-3
decay_rate = 0.75
decay_steps = epochs

batch_size = 64
input_length = 20
output_length = 20
input_channels = 2
output_channels = 2

patience = 10
data_size = 1000

# Wrap up simpler inputs into an argument tuple
args = (initial_lr, min_lr, epochs, patience, data_size, input_length, output_length, input_channels, output_channels)

# Train the model sequentially with different parameter sets
results = sequential_training(param_sets, create_data_loader, args)