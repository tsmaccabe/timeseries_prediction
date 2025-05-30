using Plots
using Random
using Flux.Data: DataLoader
using CUDA
using Random: shuffle!

function spirograph_dataloader(rs_vals::Vector{<:Vector{<:Real}}, d::Float64, n::Int64, T::Int64, drift, train_pct=0.8, test_pct=0.1, batchsizes=[32, 16, 16])
    dataloaders = []
    i = 0
    for rs in rs_vals
        I = axes(rs)[1]
        t = range(0, stop=2π*n, length=T)
        x = zeros(length(t))
        y = zeros(length(t))
        driftvals = drift(t)
        for i in I
            R = rs[i]
            r = rs[mod1(i+1, length(I))]
            x .+= (R-r)*cos.(t) + d*cos.((R-r)/r*t) .+ driftvals[1]
            y .+= (R-r)*sin.(t) - d*sin.((R-r)/r*t) .+ driftvals[2]
        end
        data = reshape(cat(x, y, dims=1), (T, 1, 2, length(I)))
        n_samples = size(data, 1)
        train_end = Int(round(n_samples * train_pct))
        test_end = train_end + Int(round(n_samples * test_pct))
        indices = shuffle(collect(1:n_samples))
        train_indices = indices[1:train_end]
        test_indices = indices[train_end+1:test_end]
        val_indices = indices[test_end+1:end]
        train_data = data[train_indices, :, :, :]
        test_data = data[test_indices, :, :, :]
        val_data = data[val_indices, :, :, :]
        batchsizes = batchsizes[1:min(3, length(batchsizes))]
        train_loader = DataLoader(train_data, batchsize=batchsizes[1], shuffle=true)
        test_loader = DataLoader(test_data, batchsize=batchsizes[2], shuffle=true)
        val_loader = DataLoader(val_data, batchsize=batchsizes[3], shuffle=true)
        push!(dataloaders, (train_loader, test_loader, val_loader))
    end
    return dataloaders
end

# Example usage (continued)
rs_vals = [[1.0, 2.0, 3.0], [0.5, 1.5, 2.5, 3.5], [1.0, 3.0, 5.0, 7.0], [5.5, pi, 2.]]
d = 0.5
n = 10000
T = 10*n
#drift(t) = [0.01*sin.(0.1*t), 0.02*cos.(0.2*t)]

dataloaders = spirograph_dataloader(rs_vals, d, n, T, drift, 0.8, 0.1, [32, 16, 16])

training_loaders = [dataloaders[:][1]...]

# Access the training data for the first rs value
for (x, _) in dataloaders[1][1]
    # Process training data here
end

# Access the testing data for the first rs value
for (x, _) in dataloaders[1][2]
    # Process testing data here
end

# Access the validation data for the first rs value
for (x, _) in dataloaders[1][3]
    # Process validation data here
end