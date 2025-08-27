using DifferentialEquations: ODEProblem, solve, Tsit5
using Flux
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
        Dense(192, 128, relu),
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
tspan = (0, 5)
p = [10.0, 28.0, 8.0/3.0]

# Solve the differential equation
prob = ODEProblem(lorenz!, u0, tspan, p)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=0.01)

# Constants
input_channels = 3
input_size = 20  # Length of the input time series
output_channels = input_channels
output_size = 20 # Number of predicted time steps

# Create the training dataset
time_steps = length(sol.t)
X = Array(sol.u)
X = reshape(hcat(X...), (time_steps, input_channels)) # Shape: (time_steps, 3)

# Add stride parameter for controlling sequence overlap
stride = 1
sol_u_downsample = sol.u[1:stride:end]

u1 = [u[1] for u in sol_u_downsample];
u2 = [u[2] for u in sol_u_downsample];
u3 = [u[3] for u in sol_u_downsample];

u = hcat(u1, u2, u3);

n = input_size + output_size

# Generate input and output sequences for the dataset with stride
x_data = [Float32.(u[i:i+input_size-1, :]) for i in 1:stride:time_steps-n]
y_data = [Float32.(u[i+input_size:i+n-1, :]) for i in 1:stride:time_steps-n]

n_samples = size(x_data, 4)

# Convert the data to the required format (unchanged)
x_train = cat([reshape(x, input_size, 1, input_channels) for x in x_data]..., dims=4) |> gpu
y_train = hcat([reshape(y, output_size*output_channels) for y in y_data]...) |> gpu

n_train = size(x_train, 4)

# Check the shapes
println(size(x_train))
println(size(y_train))

# Build the model
model = build_cnn(input_size, input_channels, output_size) |> gpu

# Define the loss function and optimizer
loss(m, x, y) = Flux.mse(m(x), y) |> gpu

data = Flux.DataLoader((x_train, y_train), batchsize=10, shuffle=true)

# Custom training loop with loss plotting
loss_values = Float64[]

epochs = 100

initial_lr = 0.001
decay_rate = 2.
decay_steps = epochs

opt = Flux.setup(ADAM(initial_lr), model)

for epoch in 1:epochs

    epoch_loss = 0.0
    num_batches = 0
    
    for datum in data
        batch_loss = loss(model, datum...)
        epoch_loss += batch_loss
        num_batches += 1
        ∂L∂m = gradient(loss, model, datum...)[1]
        Flux.update!(opt, model, ∂L∂m)
    end
    
    Flux.adjust!(opt, exponential_decay(initial_lr, decay_rate, epoch, decay_steps))

    epoch_loss /= num_batches
    push!(loss_values, epoch_loss)
    println("Epoch $epoch - Loss: $epoch_loss")
end

# Plot the loss values
pltl = plot(loss_values, xlabel="Epoch", ylabel="Loss", label="Training Loss", legend=:topleft)

# Get test sample
test_idx = 1
x_test_data = x_data[test_idx]
y_test_data = y_data[test_idx]

# Prepare input for model
x_test_tensor = reshape(x_test_data, input_size, 1, input_channels, 1) |> gpu

# Get prediction and reshape correctly
y_pred_flat = model(x_test_tensor) |> cpu

# Reshape treating channels as separate blocks
y_pred_reshaped = zeros(Float32, output_size, output_channels)
for ch in 1:output_channels
    start_idx = (ch-1) * output_size + 1
    end_idx = ch * output_size
    y_pred_reshaped[:, ch] = y_pred_flat[start_idx:end_idx]
end

# Time vectors
t_input = range(0, length=input_size, step=0.001)  
t_output = range(t_input[end] + 0.001, length=output_size, step=0.001)

# Single plot for X₁
plt1 = plot(t_input, x_test_data[:, 1], label="Input X₁", linewidth=2, color=:blue)
plot!(plt1, t_output, y_test_data[:, 1], label="Actual X₁", linewidth=2, color=:green)
plot!(plt1, t_output, y_pred_reshaped[:, 1], label="Predicted X₁", linewidth=2, color=:red, linestyle=:dash)
vline!([t_input[end]], label="Prediction Start", color=:gray, linestyle=:dot, linewidth=1)
xlabel!("Time")
ylabel!("X₁")
title!("Lorenz System X₁: Actual vs Predicted")

display(plt1)

println("First 5 timesteps of actual X₁: ", y_test_data[1:5, 1])
println("First 5 timesteps of predicted X₁: ", y_pred_reshaped[1:5, 1])

println("Last input X₁ value: ", x_test_data[end, 1])
println("First output X₁ value (actual): ", y_test_data[1, 1])
println("First output X₁ value (predicted): ", y_pred_reshaped[1, 1])

#=
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
x1_t2 = x1_embedded_array[2, :]
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
=#