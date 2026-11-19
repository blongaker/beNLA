using JLD2

@doc raw"""
Loads the entire SUSY dataset.
"""
function load_entire_susy_dataset()
    f = jldopen("data/SUSY.jld2", "r")
    return f["X"], f["y"]
end

# TODO: include only loading a subset of the data.

function load_entire_mnist_dataset()
    f = jldopen("data/MNIST.jld2", "r")
    return f["X_train"], f["y_train"], f["X_test"], f["y_test"]
end

function load_pendigits_dataset()
    f = jldopen("data/pendigits.jld2", "r")
    return f["X_train"], f["y_train"], f["X_test"], f["y_test"]
end
