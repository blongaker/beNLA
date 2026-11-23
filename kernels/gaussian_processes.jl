# A Gaussian process has the property that for a finite collection of points,
# the points follow a Gaussian distribution. Below we sample random functions
# for the squared exponential kernel function.

include("kernels.jl")
include("kernel_matrix.jl")

using LinearAlgebra
using Plots


# Variables
σ = 1.0
x_train = [-4.0, -2.0, 1.0, 3.0, 4.0]
y_train = [-2.0,  1.0, -1.0, 1.0, -1.0]
K(x,y) = rbf_kernel(x, y; σ=σ)

build_covariance_matrix(x1, x2, kernel) = [kernel(a, b) for a in x1, b in x2]

# Condition to build new distribution of functions
function gaussian_process_regression(x_train, y_train, x_test, K)
    # Build covariance matrices
    K11 = build_covariance_matrix(x_train, x_train, K)
    K21 = build_covariance_matrix(x_test,  x_train, K)
    K12 = build_covariance_matrix(x_train, x_test,  K)
    K22 = build_covariance_matrix(x_test,  x_test,  K)

    # Eigendecomposition of the covariance matrix of the data
    D, U = eigen(Symmetric(K11))
    Dinv = Diagonal(D.^(-1))

    # Mean and covariance of the conditioned multivariate Gaussian
    mean = K21 * (U * (Dinv * (U' * y_train)))
    cov  = K22 - K21 * (U * (Dinv * (U' * K12)))

    return mean, cov
end

# Sample from N(mean, cov)
function sample_mvnormal(mean, cov)
    D, U = eigen(Symmetric(cov))
    return mean + U * (sqrt.(max.(D, 0)) .* randn(length(mean)))
end


### Plot 1: Generate standard N(0,K) Gaussian process (no conditioning)
N = 200
xgrid = range(-5, 5, length=N)
mean = zeros(N)
cov = build_covariance_matrix(xgrid, xgrid, K)

M = 5 # number of trajectories
p1 = plot(title="Gaussian Process before conditioning", ylims=(-3, 3), legend=false)

# Entrywise variance (±2 standard deviations)
std = sqrt.(max.(diag(cov), 0))
plot!(p1, xgrid, mean, ribbon=2 .* std, fillalpha=0.25, fillcolor=:gray, color=:black)

# Sample a few times
for k in 1:M
    plot!(p1, xgrid, sample_mvnormal(mean, cov), linestyle=:solid, linewidth=1.5)
end



### Plot 2: Gaussian process with conditioning
mean, cov = gaussian_process_regression(x_train, y_train, xgrid, K)

p2 = plot(title="Gaussian Process after conditioning", ylims=(-3, 3), legend=false)

# Entrywise variance (±2 standard deviations)
std = sqrt.(max.(diag(cov), 0))
plot!(p2, xgrid, mean, ribbon=2 .* std, fillalpha=0.25, fillcolor=:gray, color=:black, linewidth=1.5)

# Do a few samples
for k in 1:M
    plot!(p2, xgrid, sample_mvnormal(mean, cov), linestyle=:solid, linewidth=1.5)
end
scatter!(p2, x_train, y_train, color=:black)

# Plot
fig = plot(p1, p2, layout=(1, 2), size=(1200, 500))
display(fig)
savefig(fig, "kernels/gaussian_processes.png")