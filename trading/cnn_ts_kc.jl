using Flux
using LinearAlgebra

include(string(pwd(), "\\timeseries_prediction\\trading\\label_ohlc.jl"))

int_div(a, b) = Int64(floor(Float64(a)/Float64(b)))


# Get/generate data set

Data_ohlc = Array{Float32}[]
path = string(datadir, "\\ohlc1BTC-USDT-200k-0os-2023-03-09T08%3A30%3A23.083.jld")
ohlc_dict1 = load(path)
push!(Data_ohlc, ohlc_dict1["ohlc"])

path = string(datadir, "\\ohlc1BTC-USDT-50k-200kos-2023-03-09T08%3A34%3A48.649.jld")
ohlc_dict1 = load(path)
push!(Data_ohlc, ohlc_dict1["ohlc"])


n_data = 10000
n_candles_datum = 240
training_part = 0.5
predict_part = 1 - training_part

n_candles_predict = Int(ceil(predict_part*n_candles_datum))
n_candles_train = n_candles_datum - n_candles_predict


xW = 4
xL = n_candles_train

tp_rel = 1.01
sl_rel = 0.99


X = zeros(Float32, xL, xW, 1, 0)
Y = zeros(Float32, 0)
Data, X, Y = label_ohlc(Data_ohlc[1], tp_rel, sl_rel, n_data, n_candles_datum, training_part)

Data = Data[randperm(n_data)]
X = X[:, :, 1:1, randperm(n_data)]
Y = Y[randperm(n_data)]


train_test_ratio = 0.85
n_train = Int(ceil(train_test_ratio*n_data))
n_test = Int(floor((1-train_test_ratio)*n_data))


data_train = Data[1:n_train]
Xt = X[:, :, 1:1, 1:n_train]
Yt = Y[1:n_train]

data_test = Data[n_data - n_test:n_data]
Xs = X[:, :, 1:1, n_data - n_test:n_data]
Ys = Y[n_data - n_test:n_data]


n_data_v = 500
Xv = zeros(Float32, xL, xW, 1, 0)
Yv = zeros(Float32, 0)
Data_validation, Xv, Yv = label_ohlc(Data_ohlc[2], tp_rel, sl_rel, n_data_v, n_candles_datum, training_part)


# Create model

datasize = (xL, xW, 1)
drop1 = 0.
drop2 = 0.1
drop3 = 0.1
Model = Chain(
    # Pad rank of inputs of rank < 4 up to 4
    x -> reshape(x, size(x, 1), size(x, 2), size(x, 3), size(x, 4)),
    Conv((10, 1), 1 => 6, pad = (1, 0), relu),
    Dropout(drop1),
    Conv((10, 4), 6 => 16, pad = (1, 0), relu),
    Dropout(drop1),
    Conv((10, 1), 16 => 32, pad = (1, 0), relu),
    Dropout(drop1),
    MaxPool((2, 1)),
    Conv((10, 1), 32 => 20, pad = (1, 0), relu),
    Dropout(drop1),
    MaxPool((2, 1)),
    Flux.flatten,
    Dense(420, 64, relu),
    Dropout(drop1),
    Dense(64, 16, relu),
    Dropout(drop1),
    Dense(16, 1, sigmoid)
)


# Define the loss function and optimizer

P = Flux.params(Model)
loss(X, Y) = Flux.binary_focal_loss(vec(Model(X)), Y)
opt_state = Flux.Descent()


# Train the model

Flux.testmode!(Model, true)
epochs = 10
for i in 1:epochs
  println(i/epochs)
  Flux.train!(loss, P, data_train, opt_state)
  if mod(i, int_div(epochs, 10)) == 0
    println("Test loss: ", loss(Xs, Ys))
  end
end
Flux.testmode!(Model, false)


println("Validation Loss: ", loss(Xv, Yv))
#=tol = 10^-2
Ysbar = Model(Xs)
residue =  Ys - Ysbar
hits = residue[1, :] .< tol
hitrate = sum(hits)/length(hits)
println("Ybar: ", Model(Xs))
println("Residue: ", residue)
println("Threshold hits: ", hits)
println("Hit rate: ", hitrate)=#