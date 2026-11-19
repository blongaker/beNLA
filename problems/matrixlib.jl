using LinearAlgebra
using StatsBase
include("../kernels/kernels.jl")

@doc raw"""
    far_field_interaction_matrix(R::Float64)

Computes a matrix that encodes far field interactions.

We randomly sample points in a source box and a target box, separated by distance R.
The (i,j) entry of the matrix is the interaction of the xᵢ source with the yⱼ target.
If R is sufficiently large, this matrix will have numerically low rank.
"""
function far_field_interaction_matrix(R::Float64, N::Integer)

    # Sample points in the source box [0,1]².
    source = rand(Float64, (2, N))

    # Sample points in the target box [R,R+1]×[0,1].
    target = rand(Float64, (2, N))
    target[1,:] = target[1,:] .+ R

    # Compute kernel matrix
    K = zeros(N,N)
    for i=1:N
        for j=1:N
            K[i,j] = laplace_2d_kernel(source[:,i], target[:,j])
        end
    end

    return K

end