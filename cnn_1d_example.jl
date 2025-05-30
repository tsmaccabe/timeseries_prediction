using Flux

include(string(pwd(), "\\systems\\dynamic_systems\\double_pendulum_example.jl"))

int_div(a, b) = Int64(floor(Float64(a)/Float64(b)))

function objective(x, tspan)
    sol = solve_double_pendulum(x, tspan)
    println("objective eval")
    return Float32(sol[1, end])
end

function vecNd_data(objective, tspan, n, N, n_out = 1, w = 1)
    X = Array(w*rand(Float32, N, n))
    Data = Array{Tuple}(undef, n)
    for i = 1:n
        y = objective(X[:, i], tspan)
        Data[i] = (X[:, i], y)
    end
    return Data, X, Y
end

function dblpend_tsdata_rand(n, tI, tspan = (0., 10.), p = (0., pi/4))
    N = 4
    M = 4

    tL = lastindex(tI)

    w = p[2] - p[1]
    xmin = p[1]

    X0 = xmin .+ w*rand(Float32, 1, N, 1, n)
    X = zeros(Float32, tL, N, 1, n)
    X[1, :, 1, :] = X0
    Y = zeros(Float32, M, n)
    Data = Tuple[]
    for i = 1:n
        x0 = X0[1, :, 1, i]
        x, y = solve_double_pendulum(x0, tI, tspan)
        X[:, :, 1, i] = x
        Y[:, i] = y
        push!(Data, (x, y))
    end

    return Data, X, Y
end


# Get/generate data set

D = 4 # phase space dimension of double pendulum
tI = Vector(0:0.01:5)
tL = lastindex(tI)
tspan = (0., 6.)
x0_rng = (0., pi/2)

nt = 1500
data_train, Xt, Yt = dblpend_tsdata_rand(nt, tI, tspan, x0_rng) # 4-channel data

ns = 1000
data_test, Xs, Ys = dblpend_tsdata_rand(ns, tI, tspan, x0_rng) # 4-channel data


# Create model

datasize = (tL, D, 1)
drop1 = 0.1
drop2 = 0.05
Model = Chain(
    # Pad rank of inputs of rank < 4 up to 4
    x -> reshape(x, size(x, 1), size(x, 2), size(x, 3), size(x, 4)),
    Conv((30, 1), 1 => 6, pad = (1, 1), relu),
    MaxPool((2, 1)),
    Conv((10, 2), 6 => 12, pad = (1, 1), relu),
    Conv((10, 2), 12 => 16, pad = (1, 1), relu),
    MaxPool((2, 1)),
    Flux.flatten,
    Dropout(drop1),
    Dense(14208, 32, relu),
    Dropout(drop2),
    Dense(32, 16, relu),
    Dense(16, 12, relu),
    Dense(12, 4)
)

# Define the loss function and optimizer

loss(X, Y) = Flux.mae(Model(X), Y)
P = Flux.params(Model)
opt_state = Flux.RMSProp()


# Train the model for 20 epochs

Flux.testmode!(Model, true)
epochs = 60
for i in 1:epochs
  println(i/epochs)
  Flux.train!(loss, P, data_train, opt_state)
  if mod(i, int_div(epochs, 10)) == 0
    println("Test loss: ", loss(Xs, Ys))
  end
end
Flux.testmode!(Model, false)

tol = 10^-2
Ysbar = Model(Xs)
residue =  Ys - Ysbar
hits = residue[1, :] .< tol
hitrate = sum(hits)/length(hits)
println("Ybar: ", Model(Xs))
println("Residue: ", residue)
println("Threshold hits: ", hits)
println("Hit rate: ", hitrate)