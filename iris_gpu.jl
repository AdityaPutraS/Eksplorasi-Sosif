using Statistics
using Plots
using Flux, LinearAlgebra
using Random
using CuArrays
CuArrays.allowscalar(false)

function partitionTrainTest(data, at = 0.7)
    n = size(data)[1]
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at * n))
    test_idx = view(idx, (floor(Int, at * n)+1):n)
    data[train_idx, :], data[test_idx, :]
end


Flux.Data.Iris.load()
iris_x = Flux.Data.Iris.features()
iris_x = Flux.normalise(iris_x, dims = 2)
iris_y = Flux.Data.Iris.labels()
iris_y = Flux.onehotbatch(iris_y, sort(unique(iris_y)))

train_indices = [1:3:150; 2:3:150]
test_indices = 3:3:150
# Data
test_x, test_y = iris_x[:, test_indices] |> gpu, iris_y[:, test_indices] |> gpu

train_x, train_y = iris_x[:, train_indices] |> gpu, iris_y[:, train_indices] |> gpu

train_iterator = Iterators.repeated((train_x, train_y), 1)

# Pembuatan model
model = Chain(Dense(4, 4, σ), Dense(4, 3), softmax) |> gpu
# Training Parameter
loss(x, y) = Flux.crossentropy(model(x), y)
accuracy(x, y) = mean(Flux.onecold(model(x) |> cpu) .== Flux.onecold(y |> cpu))
opt = Descent(0.5)
test_loss_history = []
train_loss_history = []
test_acc_history = []
train_acc_history = []
cb = function ()
    test_loss = loss(test_x, test_y)
    test_acc = accuracy(test_x, test_y)
    train_loss = loss(train_x, train_y)
    train_acc = accuracy(train_x, train_y)
    push!(test_loss_history, test_loss)
    push!(train_loss_history, train_loss)
    push!(test_acc_history, test_acc)
    push!(train_acc_history, train_acc)
    println("Train Loss : ", train_loss, " , Train Accuracy : ", train_acc)
    println("Test Loss : ", test_loss, " , Test Accuracy : ", test_acc)
end

# Training
@time @Flux.epochs 110 Flux.train!(
    loss,
    params(model),
    train_iterator,
    opt,
    cb = Flux.throttle(cb, 1),
)

plot(train_loss_history)
plot!(train_acc_history)

plot!(test_loss_history)
plot!(test_acc_history)
