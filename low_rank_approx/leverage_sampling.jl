include("../kernels/kernel_matrix.jl")
include("../kernels/kernels.jl")
include("../data/load_data.jl")
using Plots
using StatsBase

"""
To examine the effectiveness of leverage score sampling, we approximate a kernel matrix
from points in 1D, using a non-uniform distribution of points on the interval.
In particular, let's use Chebyshev points on the interval [-1, 1].

We take data points __non-uniformly__ sampled, centered at the origin.
Since data near the center is more sparse, these points should be more likely to be sampled using leverage scores.
"""

N = 500
# Generate Chebyshev interpolation points on [-1, 1]
θ = range(-pi, pi, N)
X = cos.(θ)
# Cluster even more on the edges
α = 1e-4
X = (asin.(α * cos.(θ))) ./ asin(α)
# Generate uniform interpolation points on [-1, 1]
X = range(-1, 1, N)

# Build kernel matrix
kernelmat = KernelMatrix(rbf_kernel, X')
println("Evaluating kernel entries...")
K = build_kernel_matrix(kernelmat)
λ = 1e-8

# Compute leverage scores
println("Computing eigenvalue decomposition of kernel matrix...")
F = eigen(K)
println("Computing leverage scores...")
D_mod = F.values ./ (F.values .+ λ)
ridge_lev_scores = []
for i=1:N
    Ui = F.vectors[i,:]
    lev = Ui' * Diagonal(D_mod) * Ui
    push!(ridge_lev_scores, lev)
end

eff_rank = sum(ridge_lev_scores)
println("Effective rank of kernel matrix: $eff_rank")
println("N: $N")

# Plot eigenvalue decay of kernel matrix (is there any low rank structure?)
# plot(1:N, abs.(F.values), yaxis=:log10, title="Eigenvalue decay of kernel matrix")

# Plot leverage scores
plot(X, ridge_lev_scores, title="Leverage scores")
scatter!(X, zeros(N), markersize=5)


# Using these leverage scores, compute Nystrom approximation
lev_probs = ridge_lev_scores ./ eff_rank
unif_probs = ones(N) ./ N
k = 30
nystrom_pts_lev = sample(1:N, weights(lev_probs), k)
nystrom_pts_unif = sample(1:N, weights(unif_probs), k)
K_sub_lev = K[nystrom_pts_lev,nystrom_pts_lev]
K_sub_unif = K[nystrom_pts_unif,nystrom_pts_unif]
K_nys_lev = K[:,nystrom_pts_lev] * inv(K_sub_lev) * K[nystrom_pts_lev,:]
K_nys_unif = K[:,nystrom_pts_unif] * inv(K_sub_unif) * K[nystrom_pts_unif,:]
lev_err = norm(K - K_nys_lev) / norm(K)
unif_err = norm(K - K_nys_unif) / norm(K)
println("Leverage score sampling error: $lev_err")
println("Uniform sampling error: $unif_err")
