using LinearAlgebra
using StatsBase
using FFTW
using Hadamard
using SparseArrays


@doc raw"""
    gaussian_matvec(A::Array)

Computes a matvec of A with a random Gaussian vector ω, which has N(0,1) entries.
"""
function gaussian_matvec(A::Array)
    m, n = A.size
    ω = randn(n)
    return A * ω
end


@doc raw"""
    gaussian_sketch(A::Array)

Computes the sketch Y=AΩ, where Ω is n×k and has N(0,1) entries.
"""
function gaussian_sketch(A::Array, k::Integer)
    m, n = A.size
    Ω = randn(n, k) / sqrt(k)
    return A * Ω
end


@doc raw"""
    SRFT(A::Array, k::Integer)

Sketches A from the right via a subsampled randomized Fourier transform (SRFT).

In other words, we compute AΘ, where Θ=sqrt(n/k)*DFS, where:
- D is an n×n diagonal matrix which has uniform ±1 values
- F is an n×n Fourier transform matrix (computed using FFT)
- S is an n×k matrix that samples k columns uniformly (replacement?)

To save memory, we apply the transforms in place.
"""
function SRFT(A::Array, k::Integer)
    m, n = A.size

    # Apply D via random diagonal matrix
    D_diag = rand((-1.0, 1.0), n)
    D = Diagonal(D_diag)
    A = A * D

    # Apply F via FFT to each row
    A = fft(A, 2) ./ sqrt(n)

    # Apply S by randomly sampling columns
    s = sample(1:n, k; replace=false)
    return A[:,s] .* sqrt(n/k)

end


@doc raw"""
    SRHT(A::Array, k::Integer)

Sketches A from the right via a subsampled randomized Hadamard transform (SRHT).

In other words, we compute AΘ, where Θ=sqrt(n/k)*DHS, where:
- D is an n×n diagonal matrix which has uniform ±1 values
- H is an n×n Hadamard transform matrix (applies to each row)
- S is an n×k matrix that samples k columns uniformly, without replacement

To save memory, we apply the transforms in place.
"""
function SRHT(A::Array, k::Integer)
    m, n = A.size

    # Apply D via random diagonal matrix
    D_diag = rand((-1.0, 1.0), n)
    D = Diagonal(D_diag)
    A = A * D

    # Apply H via fast Hadamard transform
    # Note that we need each row to have 2ⁿ entries. We must pad with zeros.
    n_pad = nextpow(2, n)
    if n_pad != n
        A_padded = zeros(eltype(A), m, n_pad)
        A_padded[:,1:n] = A
    else
        A_padded = A
    end
    A = fwht(A_padded, 2) .* sqrt(n)

    # Apply S by randomly sampling columns
    s = sample(1:n, k; replace=false)
    return sqrt(n/k) .* A[:,s]

end



@doc raw"""
    sparse_sign_embedding(A::Array, k::Integer, ζ::Integer)

Sketches A from the right via a sparse sign embedding.

Each row has a fixed number of nonzero entries, defined by r. For each row, sample r indices from [k].
Then each entry takes ±1 with equal probability.

This results in a sparse matrix-matrix multiply, and the sketch costs O(ζmn), rather than O(mn*log(n)) for SRFT/SRHT.
"""
function sparse_sign_embedding(A::Array, k::Integer, ζ::Integer)
    m, n = A.size
    I = zeros(ζ * n); J = zeros(ζ * n);
    V = rand((-1.0, 1.0), ζ * n)

    # Loop through rows; sample ζ indices for each
    for i=1:n
        s = ζ * (i-1) + 1
        e = ζ * i
        I[s:e] .= i
        J[s:e] = sample(1:k, ζ; replace=false)
    end

    Θ = sparse(I, J, V)
    return A * Θ ./ sqrt(ζ)
end
