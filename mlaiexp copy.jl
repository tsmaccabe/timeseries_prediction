using Flux, Plots

# Define the size of the input and output layers
input_size = 150
output_size = 75

# Define the size of the hidden layer
hidden_size = 100

# Define the model architecture
model = Chain(
    Dense(input_size, hidden_size, relu),
    Dense(hidden_size, output_size)
)

# Initialize the model parameters
params = Flux.params(model)

# Define the loss function
loss(x, y) = Flux.mse(model(x), y)

# Generate a larger dataset and separate into training and test sets
N = 1000
x_train = rand(input_size, N)
y_train = rand(output_size, N)
x_test = rand(input_size, N÷10)
y_test = rand(output_size, N÷10)

# Define the batch size
batch_size = 10

# Train the model using minibatches and record the loss progress
opt = ADAM()
T = 100
test_losses = zeros(Float64, T)
for i in 1:T
    # Shuffle the training data for each epoch
    shuffle!(x_train)
    shuffle!(y_train)
    # Train on minibatches
    batch_loss = 0.0
    for j in 1:N÷batch_size
        # Extract a minibatch
        batch_start = (j-1)*batch_size + 1
        batch_end = j*batch_size
        x_batch = x_train[:, batch_start:batch_end]
        y_batch = y_train[:, batch_start:batch_end]
        # Update the model parameters using the minibatch
        Flux.train!(loss, params, [(x_batch, y_batch)], opt)
        # Evaluate the model on the test set
        test_losses[i] = loss(x_test, y_test)
    end
end

# Plot the loss progress
plot(test_losses, xlabel="Iteration", ylabel="Loss", title="Test Loss Progress")