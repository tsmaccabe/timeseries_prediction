using Flux, CUDA, Random, Plots, DifferentialEquations, Printf
using Flux: params, RNN, GRU, LSTM
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


function clip_weights!(layer::Union{Dense, Conv}, clip_value::Real)
    for param in params(layer)
        clamp!(param, -clip_value, clip_value)
    end
end

function clip_weights!(layer::UnionAll, clip_value::Real)
    if layer <: Union{RNN, LSTM, GRU}
        for param in params(layer)
            clamp!(param, -clip_value, clip_value)
        end
    end
end

function clip_weights!(model, clip_value::Real)
    for layer in model.layers
        if layer isa Union{Dense, Conv}
            clip_weights!(layer, clip_value)
        elseif layer isa UnionAll
            clip_weights!(typeof(layer), clip_value)
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
        if isa(layer, Dense) || isa(layer, Conv) || isa(layer, Flux.Recur)
            l2_term += sum(params(layer)[1] .^ 2)
        end
    end
    return 0.5 * lambda * l2_term
end

function train!(model, train_loader, val_loader, test_loader, opt, epochs, initial_lr, min_lr, patience)
    train_losses = Float32[] |> gpu
    test_losses = Float32[] |> gpu
    best_loss = Inf
    num_bad_epochs = 0

    t0 = time()

    for epoch in 1:epochs
        # Update learning rate
        opt.eta = linear_decay(initial_lr, min_lr, epoch, epochs)

        epoch_loss = 0.0f0 |> gpu
        global num_batches = 0
        for batch in train_loader
            x, y = batch

            batch_loss = loss(x, y)
            epoch_loss += batch_loss
            num_batches += 1

            # Compute gradients for current mini-batch
            gradients = gradient(() -> loss(x, y), params(model))

            Flux.Optimise.update!(opt, params(model), gradients) # Update model weights using gradients
            clip_weights!(model,  5.) # Clip weights

            Flux.reset!(model)
        end
        epoch_loss /= num_batches
        push!(train_losses, epoch_loss)
    
        test_loss = 0.0 |> gpu
        global num_batches = 0
        for batch in test_loader
            x, y = batch

            batch_loss = loss(x, y)
            test_loss += batch_loss
            num_batches += 1

            Flux.reset!(model)
        end
        test_loss /= num_batches
        push!(test_losses, test_loss)
    
        elapsed_time = time() - t0
        
        hours, rem = divrem(elapsed_time, 3600)
        minutes, seconds = divrem(rem, 60)
        
        if test_loss < best_loss
            best_loss = test_loss
            num_bad_epochs = 0
        else
            num_bad_epochs += 1
        end

        print(@sprintf("Elapsed time: %02d:%02d:%02d | ", hours, minutes, seconds))
        println(@sprintf("Epoch %d, bad %d - Loss: %.4f - Test Loss: %.4f", epoch, num_bad_epochs, epoch_loss, test_loss))
        if num_bad_epochs >= patience
            println("Stopping early")
            break
        end
    end

    return model, train_losses, test_losses
end

# Define the 1D convolutional neural network architecture
function build_cnn(input_length::Int, input_channels::Int, output_length::Int)
    ch = Chain(
        #x -> reshape(x, input_length, 1, input_channels, size(x, 4)),
        Conv((10,1), input_channels => 4, pad=SamePad(), selu),
        MaxPool((2,1), pad=SamePad()),
        Conv((5,1), 4 => 8, pad=SamePad(), selu),
        MaxPool((2,1), pad=SamePad()),
        x -> dropdims(x, dims=2), # Remove second dimension
        x -> permutedims(x, (2, 1, 3)), # Transpose dimensions to have the time dimension first
        GRU(8, 4),
        Dropout(0.5),
        GRU(4, 4),
        Dropout(0.5),
        Flux.flatten,
        Dense(52, 52, selu),
        Dropout(0.5),
        Dense(52, output_length*input_channels, selu),
        Dropout(0.5),
        Flux.Scale(output_length*input_channels)
    ) |> gpu
    return ch
end


function causal_conv(kernel_size, in_channels, out_channels, dilation)
    padding = (dilation * (kernel_size - 1), 0)
    return Conv((kernel_size, 1), in_channels => out_channels, pad=padding)
end

function tcn_block(in_channels, out_channels, kernel_size, dilation, dropout_rate)
    conv1 = causal_conv(kernel_size, in_channels, out_channels, dilation)
    conv2 = causal_conv(kernel_size, out_channels, out_channels, dilation)
    dropout_layer = Dropout(dropout_rate)
    #skip_layer = in_channels == out_channels ? identity : Conv((1, 1), in_channels => out_channels)

    return Chain(conv1, dropout_layer, relu, conv2, dropout_layer, relu)
end

function build_tcn(input_shape, output_length, input_length, num_blocks, kernel_size, dropout_rate)
    model = Chain(
        tcn_block(input_shape[1], input_length, kernel_size, 1, dropout_rate),
        [tcn_block(input_length, input_length, kernel_size, 2^i, dropout_rate) for i in 1:num_blocks-1]...,
        GlobalMeanPool(),
        Flux.flatten,
        Dense(input_length, output_length),
        Flux.Scale(output_length)
    )
    return model
end

# Build the model
ts_size = (2, n_ts)
n_blocks = 4
kernel_size = 10
model = build_tcn(ts_size, output_length, n_blocks, kernel_size, input_length, 0.3) |> gpu

# Define the loss function and optimizer
loss(x, y) = Flux.mse(model(x), y) |> gpu

epochs = 25

initial_lr = 1exp(-8)
min_lr = 1exp(-8)

decay_rate = 0.75
decay_steps = epochs

#opt = ADAM(initial_lr)

best_loss = Inf
patience = 6
num_bad_epochs = 0

model, train_losses, test_losses = train!(model, train_loader, test_loader, val_loader, opt, epochs, initial_lr, min_lr, patience)

val_loss = 0.0f0 |> gpu
num_batches = 0
for batch in test_loader
    x, y = batch

    batch_loss = loss(x, y)
    global val_loss += batch_loss
    global num_batches += 1

    Flux.reset!(model)
end
val_loss /= num_batches

# Plot the loss values
pltl = plot(train_losses, xlabel="Epoch", ylabel="Loss", label="Training Loss", color = :blue, legend = :topleft)
plot!(pltl, test_losses, label = "Test Loss", color = :green)
display(pltl)