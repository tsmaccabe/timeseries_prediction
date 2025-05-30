using Random

include(string(pwd(), "\\timeseries_prediction\\trading\\load_ohlc.jl"))

function label_ohlc(ohlc, tp_rel, sl_rel, n_data, n_candles_datum, training_part = 0.5)
    n_candles = size(ohlc, 1)

    predict_part = 1 - training_part
    n_candles_predict = Int(ceil(predict_part*n_candles_datum))
    n_candles_train = n_candles_datum - n_candles_predict

    xW = 4
    xL = n_candles_train

    long = true

    # x(t) format for conv net:
    #     ohlc

    X = zeros(Float32, xL, xW, 1, n_data)
    Y = zeros(Float32, n_data)
    Data = Tuple[]
    for i = 1:n_data
        i_rand = rand(1:n_candles - n_candles_datum)
        I_rand = i_rand:i_rand .+ n_candles_train - 1
        X[:, :, 1, i] = ohlc[I_rand, :]

        I_pred = I_rand[end] + 1:i_rand + n_candles_datum - 1
        Y_ts = ohlc[I_pred, :]

        recent_val = X[end, 4, 1, i]
        hit_tp = any(Y_ts[:, 2] .> tp_rel*recent_val)
        hit_sl = any(Y_ts[:, 3] .< sl_rel*recent_val)
        if hit_tp
            Y[i] = Int(long)
            continue
        end
        if hit_sl
            Y[i] = Int(!long)
            continue
        end

        push!(Data, (X[:, :, :, i], Y[i]))
    end

    return Data, X, Y
end

n_data = 1000
n_candles_datum = 600
training_part = 0.5
predict_part = 1 - training_part

n_candles_predict = Int(ceil(predict_part*n_candles_datum))
n_candles_train = n_candles_datum - n_candles_predict

xW = 4
xL = n_candles_train

tp_rel = 1.015
sl_rel = 0.99

X = zeros(Float32, xL, xW, 1, 0)
Y = zeros(Float32, 0)
Data = Tuple[]
for ohlc in Data_ohlc
    d, x, y = label_ohlc(ohlc, tp_rel, sl_rel, n_data_each, n_candles_datum, training_part)
    println(size(d))
    global Data = cat(Data, d, dims = 1)
    global X = cat(X, x, dims = 4)
    global Y = cat(Y, y, dims = 1)
end