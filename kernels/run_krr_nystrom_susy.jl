include("kernel_matrix.jl")
include("kernels.jl")
include("krr.jl")
include("../data/load_data.jl")
using StatsBase


function main()
    @info "Loading SUSY dataset..."
    X, y = load_entire_susy_dataset()
    y = (y .* 2.0) .- 1.0
    d, N = X.size
    N_train = Int64(1.0e5); N_test = Int64(5.0e4)
    X_train = X[:,1:N_train]; y_train = y[1:N_train]
    X_test = X[:,(N_train+1):(N_train+N_test)]; y_test = y[(N_train+1):(N_train+N_test)]

    @info "Sampling centers..."
    M = Int64(1e3)
    center_inds = sample(1:N_train, M; replace=false)

    @info "Beginning KRR problem solve..."
    krr = solve_nystrom_krr_problem(X_train, y_train, rbf_kernel, 1e-6, center_inds)

    @info "KRR problem solve complete. Evaluating performance..."
    regression = BeNLA.predict(krr, X_test)
    y_model = sign.(regression)
    accuracy = sum(y_model .== y_test) / N_test * 100
    println("Accuracy: $(round(accuracy, digits=2))%")
end

main()