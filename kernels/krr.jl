include("kernel_matrix.jl")
using LinearAlgebra
using IterativeSolvers


struct KernelRidgeRegressor
    alpha::Array{AbstractFloat}
    kernel_function::Function
    centers::Array
end

@doc raw"""
    solve_full_krr_problem(x::Array, y::Array, kernel::Function, reg::AbstractFloat)

Solves the full KRR problem: (Kₙₙ + λI)α = y.

Returns α, which defines a regression model via
$$\hat{f}(x) = \sum_{i=1}^N \alpha_i k(x, x_i).$$
"""
function solve_full_krr_problem(
    x::Array,
    y::Array,
    kernel::Function,
    reg::AbstractFloat;
    solve_strategy::String = "chol",
    preconditioner = UniformScaling(1.0)
)

    # Input checking (ignore for now)
    d, N = x.size

    # Build kernel matrix
    kernelmat = KernelMatrix(kernel, x)
    K = build_kernel_matrix(kernelmat) + UniformScaling(reg)

    # Solve, based on strategy specified by user
    alpha = zeros(N)
    if solve_strategy === "chol"
        alpha = _cholesky_solve(K, y)
    elseif solve_strategy === "cg"
        alpha = _cg_solve(K, y; precond=preconditioner)
    else
        @warn "Solve strategy not implemented. Setting regressor to the zero function."
    end

    krr = KernelRidgeRegressor(alpha, kernel, x)
    return krr
    
end


function solve_nystrom_krr_problem(
    x::Array,
    y::Array,
    kernel::Function,
    reg::AbstractFloat,
    centers::Vector{<:Integer};
    solve_strategy::String = "chol",
    preconditioner = UniformScaling(1.0)
)
    # Input checking (ignore for now)
    d, N = x.size
    M = length(centers)

    # Build kernel matrix
    @info "Evaluating kernel entries"
    kernelmat = KernelMatrix(kernel, x)
    Kₙₘ = build_kernel_submatrix(kernelmat, 1:N, centers)
    Kₘₘ = build_kernel_submatrix(kernelmat, centers, centers)
    @info "Building matrices"
    K = (Kₙₘ' * Kₙₘ) .+ (reg * Kₘₘ)
    b = Kₙₘ' * y

    # Solve, based on strategy specified by user
    @info "Starting Cholesky solve"
    alpha = zeros(M)
    if solve_strategy === "chol"
        alpha = _cholesky_solve(K, b)
    elseif solve_strategy === "cg"
        alpha = _cg_solve(K, b; precond=preconditioner)
    else
        @warn "Solve strategy not implemented. Setting regressor to the zero function."
    end

    krr = KernelRidgeRegressor(alpha, kernel, x[:,centers])
    return krr

end


function predict(krr::KernelRidgeRegressor, x_test::Array)
    d1, N_test = x_test.size
    d2, N_train = krr.centers.size
    kernel_evals = zeros(N_test, N_train)
    for i=1:N_test
        for j=1:N_train
            kernel_evals[i,j] = krr.kernel_function(x_test[:,i], krr.centers[:,j])
        end
    end
    return kernel_evals * krr.alpha
end

# Solves the linear system using a Cholesky factorization.
# Assumes A is positive definite, which is guaranteed for kernel matrices def. by pd kernels.
function _cholesky_solve(A, b)
    L = cholesky(A)
    return L \ b
end

function _cg_solve(A, b; precond=UniformScaling(1.0))
    return cg(A, b, Pl=precond)
end
