include("kernel_matrix.jl")
include("kernels.jl")
include("krr.jl")
include("../data/load_data.jl")
using StatsBase


function main()
    @info "Loading pendigits dataset..."
    X_train, y_train, X_test, y_test = load_pendigits_dataset()
    N_train = size(X_train, 2); N_test = size(X_test, 2)

    one_hot = zeros(N_train, 10)
    for i in 1:N_train
        one_hot[i, Int64(y_train[i]) + 1] = 1.0  # +1 because Julia is 1-indexed, digits are 0-indexed
    end

    @info "Beginning KRR problem solve..."
    krr = solve_full_krr_problem(X_train, one_hot, rbf_kernel, 1.0)

    @info "KRR problem solve complete. Evaluating performance..."
    regression = predict(krr, X_test) # This is N_test * 10 matrix
    y_model = [c[2] for c in argmax(regression, dims=2)] .- 1.0
    accuracy = sum(y_model .== y_test) / N_test * 100
    println("Accuracy: $(round(accuracy, digits=2))%")
end

main()