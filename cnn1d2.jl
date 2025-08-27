using Flux
using Flux: params, Recur
using DifferentialEquations
using Plots
using CUDA


rankup4(x) = reshape(x, :, size(x, 2), size(x, 3), size(x, 4))
rankup2(x) = reshape(x, :, size(x, 2))
function exponential_decay(initial_lr::Float64, decay_rate::Float64, epoch::Int, decay_steps::Int)
    return initial_lr * exp(-decay_rate * epoch / decay_steps)
end

# Define the 1D convolutional neural network architecture
function build_cnn(input_size::Int, input_channels::Int, output_size::Int)
    ch = Chain(
        x -> rankup4(x),    
        Conv((12,1), input_channels => 16, pad=SamePad(), relu),
        MaxPool((2,1), pad=SamePad()),
        Conv((6,1), 16 => 32, pad=SamePad(), relu),
        MaxPool((2,1), pad=SamePad()),
        Conv((3,input_channels), 32 => 64, pad=SamePad(), relu),
        MaxPool((2,1), pad=SamePad()),
        x -> reshape(x, :, size(x, 4)),
        Dense(832, 128, relu),
        Dropout(0.5),
        Dense(128, output_size*output_channels, relu),
        Dense(output_size*output_channels, output_size*output_channels),
        Flux.Scale(output_size*output_channels)
    )
    return ch
end


# Define the Lorenz equations
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# Set the initial conditions and parameters
u0 = [1.0, 1.0, 1.0]
tspan = (5, 7)
p = [10.0, 28.0, 8.0/3.0]

# Solve the differential equation
prob = ODEProblem(lorenz!, u0, tspan, p)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=0.001)


# Constants
input_channels = 3
input_size = 100  # Length of the input time series
output_channels = input_channels
output_size = 100 # Number of predicted time steps

# Create the training dataset
time_steps = length(sol.t)
X = Array(sol.u)
X = reshape(hcat(X...), (time_steps, input_channels)) # Shape: (time_steps, 3)

n = input_size + output_size

# Generate input and output sequences for the dataset
x_data = [Float32.(X[i:i+input_size-1, :]) for i in 1:time_steps-n]
y_data = [Float32.(X[i+input_size:i+n-1, :]) for i in 1:time_steps-n]

# Convert the data to the required format
x_train = cat([reshape(x, input_size, 1, input_channels) for x in x_data]..., dims=4) |> gpu  # Shape: (input_size, 1, input_channels, num_samples)
y_train = hcat([reshape(y, output_size*output_channels) for y in y_data]...) |> gpu # Shape: (output_size, num_samples)

n_train = size(x_train, 4)

# Check the shapes
println(size(x_train))
println(size(y_train))

# Build the model
model = build_cnn(input_size, input_channels, output_size) |> gpu

# Define the loss function and optimizer
loss(x, y) = Flux.mse(model(x), y) |> gpu

data = Flux.Data.DataLoader((x_train, y_train), batchsize=10, shuffle=true)

# Custom training loop with loss plotting
loss_values = Float64[]

epochs = 100

initial_lr = 0.004
decay_rate = 2.
decay_steps = epochs

opt = ADAM(0.001)

for epoch in 1:epochs
    # Update learning rate
    opt.eta = exponential_decay(initial_lr, decay_rate, epoch, decay_steps)

    epoch_loss = 0.0
    num_batches = 0
    
    for (x, y) in data
        batch_loss = loss(x, y)
        epoch_loss += batch_loss
        num_batches += 1
        Flux.train!(loss, params(model), [(x, y)], opt)
    end
    
    epoch_loss /= num_batches
    push!(loss_values, epoch_loss)
    println("Epoch $epoch - Loss: $epoch_loss")
end

# Plot the loss values
pltl = plot(loss_values, xlabel="Epoch", ylabel="Loss", label="Training Loss", legend=:topleft)



# 2D delay plot for the first variable of the Lorenz attractor
embedding_dimension = 2
embedding_delay = 5

x_data_array = reshape(hcat(x_data...), output_size, 1, output_channels, n_train) |> cpu

x_test = reshape(x_data_array[:, 1, :, 1], input_size, 1, input_channels, 1)
x1_test_vec = vec(x_test[:, 1, 1, 1])
y_test = x_data_array[:, 1, 1, input_size+1]
ybar_test = model(x_test |> gpu) |> cpu
t = sol.t[1:input_size]



x1_embedded = [x_test[i:i+embedding_dimension*embedding_delay-embedding_delay] for i in 1:input_size-(embedding_dimension-1)*embedding_delay]
x1_embedded_array = hcat(x1_embedded...)

x1_pred_embedded = [ybar_test[i:i+embedding_dimension*embedding_delay-embedding_delay] for i in 1:input_size-(embedding_dimension-1)*embedding_delay]
x1_pred_embedded_array = hcat(x1_pred_embedded...)

x1_t1 = x1_embedded_array[1, :]
x1_t2 = x1_embedded_array[embedding_delay+1, :]

x1_pred_t1 = x1_pred_embedded_array[1, :]
x1_pred_t2 = x1_pred_embedded_array[embedding_delay+1, :]


# 2D delay plot for the original Lorenz attractor X₁
plt1 = plot(x1_t1, x1_t2, xlabel="X₁(t)", ylabel="X₁(t+τ)", label="Lorenz Attractor X₁", legend=:topleft)

# 2D delay plot for the predicted Lorenz attractor X₁
plot!(plt1, x1_pred_t1, x1_pred_t2, label="Predicted Lorenz Attractor X₁")


# Original Lorenz attractor X₁ trajectory
original_t = sol.t[1:input_size]
plt2 = plot(original_t, x1_test_vec, xlabel="Time", ylabel="X₁", label="Lorenz Attractor X₁", legend=:topleft)

# Predicted Lorenz attractor X₁ trajectory
pred_t = original_t
plot!(plt2, pred_t, y_test, label="Predicted Lorenz Attractor X₁")

display(plt1)
display(plt2)
display(pltl)