using Dates
using Statistics
using Flux
using PyCall
using Random

pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
reqs = pyimport("api_calls.api_requests")

# TODO: -Make OHLC acquisition from a time period a self-contained function
#       -Make feature vector acquisition from OHLC a self-contained function

SECONDS_IN_DAY = 86400
S = SECONDS_IN_DAY
MINUTES_IN_DAY = 1440
M = MINUTES_IN_DAY

function timestamp(datetime)
    floor(Int64, datetime2unix(datetime))
end

function get_candles(symbol, candle_type, start_ts, end_ts)
    #n_candles = floor(Int64, (end_ts - start_ts)/60)
    start_ts_str = string(start_ts)
    end_ts_str = string(end_ts)
    response = []
    new_response = []
    TMR_flag = true
    while TMR_flag == true
        new_response = []
        append!(new_response, reqs.get_candles(symbol, candle_type, start_ts_str, end_ts_str))
        TMR_flag = (new_response[1][2] == "Too Many Requests")
        if TMR_flag
            println("Request delayed")
            sleep(1)
        end
    end
    append!(response, new_response)
    return parse.(Float64, response[1][2])
end

function get_1m_ohlc(symbol, start_ts, end_ts)
    # Break up time interval into 1-day periods

    n_days = round(Int64, (end_ts-start_ts)/SECONDS_IN_DAY)
    intervals = zeros(Int64, n_days, 2)
    intervals[1, 1] = start_ts
    intervals[1, 2] = end_ts - 1
    for i = 2:n_days
        intervals[i, 1] = intervals[i-1, 2] + 1
        intervals[i, 2] = intervals[i, 1] + SECONDS_IN_DAY - 1
    end

    # Request candle data (1m candles)
    
    #candles = get_candles(symbol, candle_type, start_ts, start_ts + SECONDS_IN_DAY)
    candles = Matrix{Float64}(undef, n_days*M, 7) # 7 is the number of elements for each candle in the exchange's response
    ohlc = Matrix{Float64}(undef, n_days*M, 4)
    for i = 1:n_days
        int_start_m = (i-1)*M + 1
        int_end_m = i*M
        I = int_start_m:int_end_m
        candles_this = get_candles(symbol, "1min", start_ts + (i-1)*SECONDS_IN_DAY, start_ts + i*SECONDS_IN_DAY)
        candles[I, :] = candles_this
        ohlc[I, :] = hcat(candles[I, 2], candles[I, 4], candles[I, 5], candles[I, 3])
        println("Data request progress: ", i/n_days)
    end
    return ohlc
end

function download_data(symbol, start_ts, end_ts)
    ohlc = get_1m_ohlc(symbol, start_ts, end_ts)
    return ohlc
end
function download_data(P_req)
    symbol, start_ts, end_ts = P_req
    ohlc = get_1m_ohlc(symbol, start_ts, end_ts)
    return ohlc
end

### General parameters

### Training set paramters

# Request parameters
symbol = "BTC-USDT"

n_days = 100
n_mins = n_days*M
length_s = n_days*SECONDS_IN_DAY
length_m = floor(Int64, length_s/60)
start_prior_s = 110*SECONDS_IN_DAY

now_ts = timestamp(now())
start_ts = now_ts - start_prior_s
end_ts = start_ts + length_s

P_req_t = (symbol, start_ts, end_ts)

### Validation set parameters

# Request parameters
symbol = "BTC-USDT"

n_days = 20
n_mins = n_days*M
length_s = n_days*SECONDS_IN_DAY
length_m = floor(Int64, length_s/60)
start_prior_s = 160*SECONDS_IN_DAY

now_ts = timestamp(now())
start_ts = now_ts - start_prior_s
end_ts = start_ts + length_s

P_req_v = (symbol, start_ts, end_ts)

### Get training & validation sets
println("Getting training set")
ohlc_t = download_data(P_req_t)
println("Getting validation set")
ohlc_v = download_data(P_req_v)

