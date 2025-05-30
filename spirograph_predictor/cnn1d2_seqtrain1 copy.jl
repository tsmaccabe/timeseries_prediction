using Flux, CUDA, Random, Plots, DifferentialEquations
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

function compute_losses(data_loaders)
    losses = Float32[]

    for data_loader in data_loaders
        loader_loss = 0.0
        num_batches = 0
        
        for (x, y) in data_loader
            batch_loss = loss(x, y)
            loader_loss += batch_loss
            num_batches += 1
        end
        
        avg_loss = loader_loss / num_batches
        push!(losses, avg_loss)
    end

    return losses
end

function train!(model, train_data_loaders, test_data_loaders, opt, epochs, cycles, initial_lr, min_lr, patience)
    loss_train = Float64[]
    loss_test = Float64[]
    best_loss = Inf
    num_bad_epochs = 0

    for cycle = 1:cycles
        i = 0
        I = randperm(axes(train_data_loaders)[1][end])
        for train_data in train_data_loaders[I]
            i = i + 1
            best_loss = Inf
            num_bad_epochs = 0
            println("Cycle $cycle, System #$i: ")
            test_data_loaders_sh = test_data_loaders[I]
            test_data = test_data_loaders_sh[i]
            for epoch in 1:epochs
                # Update learning rate

                epoch_loss = 0.0
                global num_batches = 0
                for (x, y) in train_data
                    batch_loss = loss(x, y)
                    epoch_loss += batch_loss
                    num_batches += 1
                    
                    # Compute gradients for current mini-batch
                    gradients = gradient(() -> loss(x, y), params(model))

                    Flux.Optimise.update!(opt, params(model), gradients) # Update model weights using gradients
                    clip_weights!(model,  5.) # Clip weights
                end
                Flux.reset!(model)

                epoch_loss /= num_batches
                push!(loss_train, epoch_loss)

                x_test, y_test = first(test_data)
                #println(size(test_loss), ", ", size(x_test), size(y_test))
                test_loss = loss(x_test, y_test)
                push!(loss_test, test_loss)
                println("Cycle $cycle, Epoch $epoch, bad $num_bad_epochs - Loss: $epoch_loss - Test Loss: $test_loss")

                if test_loss < best_loss
                    best_loss = test_loss
                    num_bad_epochs = 0
                else
                    num_bad_epochs += 1
                    if num_bad_epochs >= patience
                        num_bad_epochs = 0
                        println("Cycle $cycle, Epoch $epoch, Stopping early")
                        break
                    end
                end
            end
        end

        val_losses = compute_losses(val_loaders)
        println("Cycle $cycle Validation Losses By System: ", val_losses)    
    end

    return model, loss_train, loss_test
end

# Define the 1D convolutional neural network architecture
function build_cnn(input_length::Int, input_channels::Int, output_length::Int)
    ch = Chain(
        x -> reshape(x, input_length, 1, input_channels, size(x, 4)),
        Conv((5,1), input_channels => 4, pad=SamePad(), leakyrelu),
        Conv((4,1), 4=> 8, pad=SamePad(), leakyrelu),
        MaxPool((2,1), pad=SamePad()),
        x -> x[:, 1, :, :], # Remove second dimension
        x -> permutedims(x, (2, 1, 3)), # Transpose dimensions to have the time dimension first
        GRU(8, 4),
        Flux.flatten,
        Dense(40, output_length*input_channels, leakyrelu),
        Dense(output_length*input_channels, output_length*input_channels, leakyrelu),
        Dense(output_length*input_channels, output_length*input_channels, leakyrelu),
        Dense(output_length*input_channels, output_length*input_channels, leakyrelu),
        Flux.Scale(output_length*input_channels)
    )
    return ch
end
# Build the model
model = build_cnn(input_length, input_channels, output_length) |> gpu

# Define the loss function and optimizer
loss(x, y) = Flux.mse(model(x), y) |> gpu
# Custom training loop with loss plotting
loss_train = Float64[]
loss_test = Float64[]

epochs = 8
cycles = 20

initial_lr = 1e-4
min_lr = 5e-6
decay_rate = 0.75
decay_steps = epochs

opt = ADAM(initial_lr)

best_loss = Inf
patience = 5
num_bad_epochs = 0

model, loss_train, loss_test = train!(model, train_loaders, test_loaders, opt, epochs, cycles, initial_lr, min_lr, patience)

val_losses = compute_losses(val_loaders)
println("Validation Losses By System #: ", val_losses)

# Plot the loss values
pltl = plot(loss_train[20:end], xlabel="Epoch", ylabel="Loss", label="Training Loss", legend=:topleft)
plot!(loss_test[20:end], label = "Test Loss")


function predict(model, x, output_length, output_channels)
    prediction = model(x)
    return reshape(prediction, (output_length, output_channels))
end

# Choose a random sample from the validation set
sample_index = rand(1:size(x_val, 4))

# Get the input and ground truth output for the chosen sample
input_sample = x_val[:, :, :, sample_index]
ground_truth_output = y_val[:, sample_index]

# Get the model's prediction for the chosen sample
model_prediction = predict(model, input_sample, output_length, output_channels)

# Reshape ground truth output for plotting
ground_truth_output = reshape(ground_truth_output, (output_length, output_channels))

# Plot the ground truth and model's prediction
colors = [:blue, :red, :green]

plt1 = plot()
scatter!(plt1, 1:output_length, ground_truth_output[:, 1], label="x_true", color=colors[1], xlabel="Time", ylabel="Value", title="Validation Sample and Model Prediction", markershape=:cross)
scatter!(plt1, 1:output_length, model_prediction[:, 1], linestyle=:dash, label="x_pred", markershape=:circle, color=colors[1])

scatter!(plt1, 1:output_length, ground_truth_output[:, 2], label="y_true", color=colors[2], markershape=:cross)
scatter!(plt1, 1:output_length, model_prediction[:, 2], linestyle=:dash, label="y_pred", markershape=:circle, color=colors[2])

scatter!(plt1, 1:output_length, ground_truth_output[:, 3], label="z_true", color=colors[3], markershape=:cross)
scatter!(plt1, 1:output_length, model_prediction[:, 3], linestyle=:dash, label="z_pred", markershape=:circle, color=colors[3])


# Calculate the true and predicted delay embeddings for u[1]
embedding_dimension = 2
embedding_delay = 5

num_embedding_points = output_length - (embedding_dimension - 1) * embedding_delay

true_embedding = zeros(num_embedding_points, embedding_dimension)
pred_embedding = zeros(num_embedding_points, embedding_dimension)

for i in 1:num_embedding_points
    true_embedding[i, 1] = ground_truth_output[i, 1]
    true_embedding[i, 2] = ground_truth_output[i + embedding_delay, 1]

    pred_embedding[i, 1] = model_prediction[i, 1]
    pred_embedding[i, 2] = model_prediction[i + embedding_delay, 1]
end

# Plot the true and predicted delay embeddings for u[1]
plt2 = scatter(pred_embedding[:, 1], pred_embedding[:, 2], label="Predicted", color=:red, markershape=:circle, xlabel="u1(t)", ylabel="u1(t+τ)", title="True and Predicted Delay Embeddings")
scatter!(plt2, true_embedding[:, 1], true_embedding[:, 2], label="True", color=:blue, markershape=:cross)

# Set plot sizes
width = 800
height = 400
square_size = 800

# Set the size of each plot
plot!(pltl, size=(width, height))
plot!(plt1, size=(width, height))
plot!(plt2, size=(square_size, square_size))

# Stack plots vertically
plt_all = plot(pltl, plt1, plt2, layout=(3, 1), size=(width, height * 2 + square_size))

display(plt_all)