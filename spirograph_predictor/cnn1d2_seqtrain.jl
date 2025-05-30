using Flux, CUDA,  Plots, DifferentialEquations
using Flux: params
using Plots: plot, plot!, scatter, scatter!

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

function train(model, train_loaders, test_loaders, epochs, initial_lr, min_lr, patience)
    # Define the loss function and optimizer
    loss(x, y) = Flux.mse(model(x), y) |> gpu
    opt = ADAM(initial_lr)

    # Custom training loop with loss plotting
    loss_train = Float64[]
    loss_test = Float64[]

    best_loss = Inf
    num_bad_epochs = 0

    for epoch in 1:epochs
        # Update learning rate
        opt.eta = linear_decay(initial_lr, min_lr, epoch, epochs)

        epoch_loss = 0.0
        global num_batches = 0
        
        # Train on each dataloader sequentially
        for train_loader in train_loaders
            for (x, y) in train_loader
                batch_loss = loss(x, y)
                epoch_loss += batch_loss
                num_batches += 1

                # Compute gradients for current mini-batch
                gradients = gradient(() -> loss(x, y), params(model))

                Flux.Optimise.update!(opt, params(model), gradients) # Update model weights using gradients
                clip_weights!(model, -5., 5.) # Clip weights
            end
        end
        
        epoch_loss /= num_batches
        push!(loss_train, epoch_loss)

        epoch_loss = 0.0
        global num_batches = 0

        # Train on each dataloader sequentially
        for test_loader in test_loaders
            for (x, y) in test_loader
                batch_loss = loss(x, y)
                epoch_loss += batch_loss
            end
        end
        
        # Compute test Loss
        epoch_loss /= num_batches
        push!(loss_test, epoch_loss)
        println("Epoch $epoch, bad $num_bad_epochs - Loss: $epoch_loss - Test Loss: $test_loss")

        # Check for early stopping
        global best_loss = best_loss
        global num_bad_epochs = num_bad_epochs
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
    
    Flux.reset!(model)
    val_loss = loss(x_val, y_val)
    println("Validation Loss: ", val_loss)
end

# Define the 1D convolutional neural network architecture
function build_cnn(input_length::Int, input_channels::Int, output_length::Int)
    ch = Chain(
        x -> reshape(x, input_length, 1, input_channels, size(x, 4)),
        Conv((5,1), input_channels => 12, pad=SamePad(), elu),
        BatchNorm(12, relu),
        Conv((4,1), 12 => 24, pad=SamePad(), elu),
        BatchNorm(24, relu),
        MaxPool((3,1), pad=SamePad()),
        x -> x[:, 1, :, :], # Remove second dimension
        x -> permutedims(x, (2, 1, 3)), # Transpose dimensions to have the time dimension first
        GRU(24, 24),
        Flux.flatten,
        Dense(168, output_length*input_channels)
    )
    return ch
end

# Build the model
model = build_cnn(input_length, input_channels, output_length) |> gpu

epochs = 100
patience = 10
initial_lr = 1e-2
min_lr = 1e-3

train(model, train_loaders, test_loaders, epochs, initial_lr, min_lr, patience)