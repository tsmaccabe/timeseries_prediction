using JLD

datadir = string(pwd(), "\\timeseries_prediction\\trading\\data")

Data_ohlc = Array{Float32}[]
path = string(datadir, "\\ohlc1BTC-USDT2023-03-09T08%3A09%3A13.631.jld")
ohlc_dict1 = load(path)
push!(Data_ohlc, ohlc_dict1["ohlc"])
#=

path = string(datadir, "\\ohlc1BTC-USDT2023-03-07T20%3A07%3A51.870.jld")
ohlc_dict1 = load(path)
push!(Data_ohlc, ohlc_dict1["ohlc"])


path = string(datadir, "\\ohlc1ETH-USDT2023-03-07T19%3A56%3A37.375.jld")
ohlc_dict1 = load(path)
push!(Data_ohlc, ohlc_dict1["ohlc"])


path = string(datadir, "\\ohlc1ETH-USDT2023-03-07T20%3A09%3A06.435.jld")
ohlc_dict1 = load(path)
push!(Data_ohlc, ohlc_dict1["ohlc"])


path = string(datadir, "\\ohlc1MATIC-USDT2023-03-07T19%3A58%3A10.854.jld")
ohlc_dict1 = load(path)
push!(Data_ohlc, ohlc_dict1["ohlc"])


path = string(datadir, "\\ohlc1MATIC-USDT2023-03-07T20%3A10%3A24.830.jld")
ohlc_dict1 = load(path)
push!(Data_ohlc, ohlc_dict1["ohlc"])


path = string(datadir, "\\ohlc1SOL-USDT2023-03-07T19%3A59%3A38.386.jld")
ohlc_dict1 = load(path)
push!(Data_ohlc, ohlc_dict1["ohlc"])


path = string(datadir, "\\ohlc1SOL-USDT2023-03-07T20%3A06%3A33.553.jld")
ohlc_dict1 = load(path)
push!(Data_ohlc, ohlc_dict1["ohlc"])
=#