using Random
using JLD
using HTTP
using Dates
using JSON

function timestamp(datetime)
    floor(Int64, datetime2unix(datetime))
end

function get_candle_batch(symbol, n_candles, candle_size, candle_offset = 0)
    candles = Vector{Float32}[]

    # TODO: hour, day, wk candles
    n_batch_reqlimit = [1 3 5 10 15 30; 400 133 80 36 25 12]
    i_batch_reqlimit = n_batch_reqlimit[1, :] .== candle_size
    n_candles_batch = n_batch_reqlimit[2, i_batch_reqlimit][1]

    if n_candles < n_candles_batch
        batch_s = n_candles*candle_size*60
    else
        batch_s = n_candles_batch*candle_size*60
    end

    batches = Int(ceil(n_candles/n_candles_batch))
    now_s = timestamp(now())
    for i_batch in 1:batches
        println("Download Progress: ", i_batch/batches)

        offset_s = candle_offset*candle_size*60

        start_s = now_s - offset_s - (i_batch + 1)*batch_s + 1
        end_s = now_s - offset_s - i_batch*batch_s
        println(start_s, ", ", end_s)

        url = string("https://api.kucoin.com/api/v1/market/candles?type=", candle_size, "min&symbol=", symbol, "&startAt=", start_s, "&endAt=", end_s)
        candles_response = HTTP.get(url)
        candles_string = String(candles_response.body)
        candles_dict = JSON.parse(candles_string)
        TMR_flag = true
        while TMR_flag == true
            candles_response = HTTP.get(url)
            candles_string = String(candles_response.body)
            candles_dict = JSON.parse(candles_string)

            TMR_flag = candles_dict.count != 2
            if TMR_flag
                println("Request delayed")
                sleep(1)
            end
        end

        candles_packed = candles_dict["data"]
        for i_candle in 1:lastindex(candles_packed)
            candle_strings = candles_packed[i_candle]
            candle = parse.(Float32, candle_strings)
            push!(candles, candle)
        end
        
    end

    return reverse(candles)
end


function get_ohlc(symbol, n_candles, candle_size, candle_offset = 0)
    candles_vec = get_candle_batch(symbol, n_candles, candle_size, candle_offset)

    # x(t) format for conv net:
    #     ovhvlvcv
    # should ensure that volume-candle component relationships are considered
    xW = 4
    xL = n_candles

    ohlc = zeros(Float32, xL, xW)
    for i = 1:xL
        ohlc[i, :] = candles_vec[i][2:5]
    end

    ohlc = zeros(Float32, xL, xW)
    for i = 1:xL
        ohlc[i, :] = candles_vec[i][2:5]
    end

    return ohlc
end

symbol = "BTC-USDT"
n_candles = 50000
candle_size = 1
candle_offset = 200000
ohlc = get_ohlc(symbol, n_candles, candle_size, candle_offset)
println(size(ohlc))

namedata = HTTP.escapeuri(string(now()))
save(string(pwd(), "\\timeseries_prediction\\trading\\data\\ohlc", candle_size, symbol, namedata,".jld"), "ohlc", ohlc)

