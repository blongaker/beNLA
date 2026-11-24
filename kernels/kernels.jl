using LinearAlgebra
using SpecialFunctions

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

# Taken from Wikipedia
function matern_12_kernel(x::AbstractFloat, y::AbstractFloat; ρ::AbstractFloat=1.0)
    d = abs(x-y)
    return exp(-d/ρ)
end

function matern_32_kernel(x::AbstractFloat, y::AbstractFloat; ρ::AbstractFloat=1.0)
    d = abs(x-y)
    arg = sqrt(3)*d/ρ
    return (1 + arg) * exp(-arg)
end

function matern_52_kernel(x::AbstractFloat, y::AbstractFloat; ρ::AbstractFloat=1.0)
    d = abs(x-y)
    arg = sqrt(5)*d/ρ
    return (1 + arg + arg^2/3) * exp(-arg)
end

function rational_quadratic_kernel(x::AbstractFloat, y::AbstractFloat; α::AbstractFloat=1.0, ℓ::AbstractFloat=1.0)
    d = abs(x-y)
    return (1 + d^2/(2*α*ℓ^2))^(-α)
end

function periodic_kernel(x::AbstractFloat, y::AbstractFloat; p::AbstractFloat=1.0, ℓ::AbstractFloat=1.0)
    d = abs(x-y)
    return exp((-2/ℓ^2) * sin(pi*d/p)^2)
end

function linear_kernel(x::AbstractFloat, y::AbstractFloat; c::AbstractFloat=0.0)
    return (x-c) * (y-c)
end
