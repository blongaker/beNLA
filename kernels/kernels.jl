using LinearAlgebra

function rbf_kernel(x::Vector{Float64}, y::Vector{Float64}; σ=1.0)
    return exp(sum(abs2, x.-y) / (-2*σ^2))
end

function rbf_kernel(x::Float64, y::Float64; σ=1.0)
    return exp(sum(abs2, x-y) / (-2*σ^2))
end

function laplace_2d_kernel(x::Vector{Float64}, y::Vector{Float64})
    return log(norm(x-y)) / (2*pi)
end

function brownian_motion_kernel(x::Float64, y::Float64)
    return min(x,y)
end
